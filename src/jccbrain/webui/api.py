"""前端可调用的 API；仅公开带统一结果封装的方法。"""
from __future__ import annotations

import logging
import threading
from time import perf_counter
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

from jccbrain import __version__
from jccbrain.data.db import GameDB, enabled_comp_sources
from jccbrain.data.icons import IconCache
from jccbrain.data.official import INGAME_SEASONS
from jccbrain.engine.lifecycle import has_live_hud
from jccbrain.engine.llm import LLMClient, PRESETS, apply_preset
from jccbrain.engine.odds import PoolState, any_hit_per_roll, hit_odds, simulate_rolldown
from jccbrain.modes import AssistantMode, AutoRollMode, AutopilotMode, ModeRunner, RollTarget
from jccbrain.modes.scout import own_view
from jccbrain.core.models import Phase
from jccbrain.core import comp_presets
from jccbrain.engine.advisor import Advice, BuyProjection
from jccbrain.core.buy_source import source_matches_state, buy_order
from jccbrain.webui import macos, serialize, settings_schema
from jccbrain.webui.events import WebSink, _log_entry
from jccbrain.webui.population_preview import population_preview

if TYPE_CHECKING:
    from jccbrain.context import AppContext
    from jccbrain.webui.events import EventBus

log = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


def expose(method: Callable[P, T]) -> Callable[P, dict[str, Any]]:
    """把返回值和异常转换为前端统一使用的结果结构。"""
    @wraps(method)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        started = perf_counter()
        try:
            return {"ok": True, "data": method(*args, **kwargs)}
        except Exception as exc:
            log.exception("界面 API %s 调用失败", method.__name__)
            return {"ok": False, "error": str(exc) or "请求失败，请查看日志"}
        finally:
            elapsed = (perf_counter() - started) * 1000
            if elapsed > 300:
                log.warning('界面 API %s 耗时 %.0fms', method.__name__, elapsed)

    return wrapped


_MODE_TYPES = (AssistantMode, AutoRollMode, AutopilotMode)
_POINTS = [
    ('r1c1', '棋盘：最前排 最左格'), ('r1c7', '棋盘：最前排 最右格'),
    ('r4c1', '棋盘：最后排 最左格'), ('r4c7', '棋盘：最后排 最右格'),
    ('bench0', '备战席：第 1 格'), ('bench8', '备战席：第 9 格'),
    ('refresh_button', '刷新按钮'), ('xp_button', '购买经验按钮'),
    ('player0', '玩家栏：第 1 个头像'), ('player7', '玩家栏：第 8 个头像'),
]


class Api:
    def __init__(self, ctx: AppContext, bus: EventBus, *, hud=None):
        self._ctx, self._bus = ctx, bus
        self._lock = threading.RLock()
        # 命令串行化；等待模式线程退出时不持有 ctx 锁，允许 finished 回调读取状态。
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._window = None
        self._last_state = None
        self._last_advice = None
        self._last_advice_ai_source = None
        self._preview_state = None
        self._buy_projection = None
        self._ai_projection = None
        self._ai_projection_lock = threading.RLock()
        self._projection_lock = threading.RLock()
        self._hud = hud
        self._hud_emit_lock = threading.RLock()
        self._last_scout_notice = None
        self._drag_panel = None
        self._window_drag = macos.NativeDragRegions(self._drag_window)
        self._finished_mode = None
        self._syncing = False
        self._sync_thread = None
        self._closing = threading.Event()
        self._runner = ModeRunner(ctx)
        self._icons = IconCache(self._db, lambda keys: self._bus.emit('icons_ready', keys))
        self._sink = WebSink(bus, self._db, on_state=self._on_state, on_advice=self._on_advice,
                             on_buy_projection=self._on_buy_projection, on_ai_projection=self._on_ai_projection,
                             on_finished=self._on_finished, status_getter=self._status_data,
                             hud_callback=self._emit_hud, hud_visible=self._scout_hud_visible, toast_capable=self._scout_toast_capable)
        if hud is not None:
            hud.set_refresh(self._refresh_hud)
            if hasattr(hud, 'set_scout_notice'):
                hud.set_scout_notice(self._scout_notice_shown, self._scout_notice_cancel)

    def _scout_toast_capable(self):
        return bool(not self._closing.is_set() and self._hud is not None
                    and hasattr(self._hud, "notice_capable") and self._hud.notice_capable())

    def _scout_hud_visible(self, window=None):
        return bool(not self._closing.is_set() and self._hud is not None
                    and hasattr(self._hud, 'notice_ready') and self._hud.notice_ready(window))

    def _scout_notice_shown(self, notice_id, window):
        mode = self._runner.mode
        return bool(mode and getattr(mode, 'scout', None) and self._scout_hud_visible(window)
                    and mode.scout.notice_shown(notice_id, window))

    def _scout_notice_cancel(self, notice_id, window):
        mode = self._runner.mode
        return bool(mode and getattr(mode, 'scout', None) and self._scout_hud_visible(window)
                    and mode.scout.cancel_notice(notice_id))

    def _db(self):
        with self._lock:
            return self._ctx.db

    def _buy_epoch_current(self, source):
        mode = self._runner.mode
        return (not source or isinstance(mode, AssistantMode) and source.session == mode.buy_session
                and source.input_epoch == mode.buy_input_epoch)

    def _accept_buy_source(self, source, *, clearing=False):
        if not source:
            return True
        mode = self._runner.mode
        if not self._buy_epoch_current(source):
            return False
        if clearing:
            return True  # cached invalidation metadata needs no selection/native/OCR lock
        if mode.stopped and (source.eligible or source.stage is not None):
            return False
        return (source.generation == mode.lifecycle.generation
                and source.selection == self._ctx.assistant_selection()['revision']
                and source.season == self._ctx.db.season_id)

    @staticmethod
    def _behind_buy_barrier(source, barrier):
        if not source or barrier is None:
            return False
        return (buy_order(source) < buy_order(barrier.source)
                or buy_order(source) == buy_order(barrier.source)
                and not barrier.source.eligible and source.eligible)

    def _on_state(self, state):
        source = getattr(state, 'buy_source', None)
        if not self._accept_buy_source(source):
            return False
        with self._state_lock:
            with self._projection_lock:
                if not self._buy_epoch_current(source) or self._behind_buy_barrier(source, self._buy_projection):
                    return False
                previous = getattr(self._last_state, 'buy_source', None)
                if source and previous and buy_order(source) < buy_order(previous):
                    return False
                self._last_state = state
                if source:
                    current = self._buy_projection
                    if current is None or not current.valid or not source_matches_state(current.source, state):
                        self._buy_projection = BuyProjection(source)
                        self._bus.emit('buy_projection', serialize.buy_projection(self._buy_projection))
                elif not isinstance(self._runner.mode, AssistantMode):
                    self._buy_projection = None

    def _on_buy_projection(self, projection):
        source = projection.source
        clearing = not projection.valid and not source.eligible
        if not self._accept_buy_source(source, clearing=clearing):
            return False
        if clearing:
            # This lock is never held across state copies, compute, selection or
            # native calls: read/OCR and ordinary callbacks cannot block revocation.
            with self._projection_lock:
                if not self._buy_epoch_current(source) or self._behind_buy_barrier(source, self._buy_projection):
                    return False
                self._buy_projection = projection
            return
        with self._state_lock:
            with self._projection_lock:
                if not self._buy_epoch_current(source) or self._behind_buy_barrier(source, self._buy_projection):
                    return False
                current = self._buy_projection
                if current and current.valid and not projection.valid and source.eligible and buy_order(source) == buy_order(current.source):
                    return False  # late provisional clear cannot downgrade a confirmed same-frame buy
                if projection.valid and (self._last_state is None or not source_matches_state(source, self._last_state)):
                    return False
                self._buy_projection = projection

    def _on_ai_projection(self, projection):
        from jccbrain.core.ai_source import ai_order
        mode = self._runner.mode
        if not isinstance(mode, AssistantMode) or projection.source.session != mode.buy_session:
            return False
        # No ordinary API/state/native lock on the immediate revocation path.
        with self._ai_projection_lock:
            current = mode._ai.projection
            if ai_order(projection.source) != ai_order(current.source):
                return False
            if self._ai_projection and ai_order(projection.source) < ai_order(self._ai_projection.source):
                return False
            self._ai_projection = projection

    def _on_advice(self, advice):
        if advice.selection and advice.selection['revision'] != self._ctx.assistant_selection()['revision']:
            return False
        if not self._accept_buy_source(advice.buy_source):
            return False
        with self._state_lock:
            current = getattr(self._last_state, 'buy_source', None)
            if advice.buy_source and (current is None or buy_order(advice.buy_source) != buy_order(current)):
                return False
            preview = deepcopy(self._last_state)
            with self._projection_lock:
                if not self._buy_epoch_current(advice.buy_source) or self._behind_buy_barrier(advice.buy_source, self._buy_projection):
                    return False
                self._last_advice = advice
                mode = self._runner.mode
                ai_source = (mode._ai_frames.get((advice.buy_source.input_epoch, advice.buy_source.frame))
                             if isinstance(mode, AssistantMode) and advice.buy_source else None)
                self._last_advice_ai_source = ai_source
                self._preview_state = preview  # always paired with full advice, never repaired by a newer raw

        if advice.target is not None and not self._closing.is_set():
            with self._lock:
                self._icons.prefetch(advice.target.comp, advice.board)
        return ai_source

    def _serialized_advice(self, advice):
        with self._state_lock:
            source = self._last_advice_ai_source if advice is self._last_advice else None
        return serialize.advice_with_ai(advice, source)

    @expose
    def icons(self, keys):
        with self._lock:
            return self._icons.get(keys)

    def _emit_hud(self):
        with self._hud_emit_lock:
            self._emit_hud_serial()

    def _emit_hud_serial(self):
        if self._closing.is_set() or self._hud is None:
            return
        mode = self._runner.mode
        scout = getattr(mode, 'scout', None)
        notice = scout.status['notice'] if scout else None
        previous_notice, self._last_scout_notice = self._last_scout_notice, notice
        if hasattr(self._hud, 'set_notice'):
            self._hud.set_notice(notice)
        if not notice and not previous_notice and not any(self._hud.status.values()):
            return
        with self._lock:
            with self._state_lock:
                state, advice, projection = self._last_state, self._last_advice, self._buy_projection
            payload = serialize.hud(state, advice, self._ctx.db, self._ctx.layout, self._ctx.settings.hud, projection=projection)
            payload['scout_notice'] = notice
        self._bus.emit('hud', payload)

    def _refresh_hud(self):
        self._emit_hud()
        with self._state_lock:
            advice = self._last_advice
            projection = self._buy_projection
        if projection is not None:
            self._bus.emit('buy_projection', serialize.buy_projection(projection))
        if advice is not None and not self._closing.is_set():
            self._bus.emit('advice', self._serialized_advice(advice))

    def _on_finished(self, mode, reason, expected=None):
        # ModeRunner 在调用 sink 时线程尚未退出，但状态应已经显示空闲。
        current = self._runner.mode
        if current and current.key == mode and (expected is None or current is expected):
            self._finished_mode = current

    def _status_data(self):
        with self._lock:
            ctx = self._ctx
            with self._state_lock:
                projection = self._buy_projection
            mode = self._runner.mode
            stopping = bool(getattr(self._runner, 'stopping', False))
            running = self._runner.running and mode is not None and (stopping or mode is not self._finished_mode)
            rules = ctx.rules
            source = rules.borrowed_from_season
            source_name = ctx.db.catalog_name(source) if source else f'S{rules.borrowed_from}'
            preset = PRESETS.get(ctx.settings.llm.preset)
            # 状态显示用户选择的原始记录；推荐/执行仍通过 enabled_comp 的默认关联解析。
            target = ctx.enabled_comp(ctx.effective_lock_id() or '', raw=True)
            default = ctx.enabled_comp(ctx.settings.assistant.default_lock_comp_id, raw=True)
            return {
                'mode': {'key': mode.key, 'title': mode.title} if running else None,
                'stopping': stopping,
                'autoroll_task': mode.status if running and isinstance(mode, AutoRollMode) else None,
                'autopilot': mode.status if running and isinstance(mode, AutopilotMode) else None,
                'augment_selection': ctx.settings.autopilot.augment_selection,
                'feature_settings': self._feature_values(),
                'assistant_selection': ctx.assistant_selection(),
                'buy_projection': serialize.buy_projection(projection) if projection else None,
                'ai_projection': serialize.ai_projection(self._ai_projection) if self._ai_projection else None,
                'modes': [{'key': cls.key, 'title': cls.title, 'acts': cls.acts} for cls in _MODE_TYPES],
                'season': {'id': ctx.db.season_id, 'label': ctx.db.season.label,
                           'rules_note': f'规则表沿用{source_name}，未核实' if rules.borrowed else ''},
                'llm': {'enabled': ctx.settings.llm.enabled, 'preset': ctx.settings.llm.preset,
                        'preset_title': preset.title if preset else '自定义', 'model': ctx.settings.llm.model},
                'dry_run': ctx.settings.actuator.dry_run, 'vision_backend': ctx.settings.vision.backend,
                'sources': self._source_flags(),
                'comp_source_notice': ('三个数据源都已关闭，没有可推荐的阵容'
                                       if not enabled_comp_sources(ctx.settings.data) else
                                       '当前赛季暂不支持微社区阵容' if ctx.db.season_id not in INGAME_SEASONS
                                       and enabled_comp_sources(ctx.settings.data) == {'ingame'} else ''),
                'db_empty': ctx.db.is_empty, 'game_version': ctx.db.game_version, 'syncing': self._syncing,
                'lock': {**ctx.comp_identity(target.id),
                         'source': 'default' if ctx.game_lock_id is None else 'game'} if target else None,
                'default_lock': ctx.comp_identity(default.id) if default else None,
                'watch': [ctx.comp_identity(cid) for cid in ctx.game_watch_ids],
                'hero_costs': {hero.name: hero.cost for hero in ctx.db.pool_heroes()},
                'augments_taken': list(ctx.game_augments),
                'scout': mode.scout.status if running and getattr(mode, 'scout', None) else None,
                'scout_unavailable_reason': (mode.scout.request_problem(self._last_state)
                    if running and getattr(mode, 'scout', None) else '请先启动辅助大脑'),
                'economy_preference': ctx.settings.assistant.economy_preference,
                'hud': self._hud.status if self._hud else {'overlay': False, 'panel': False},
            }

    def _emit_status(self):
        result = self._status_data()
        if not self._closing.is_set():
            self._bus.emit('status', result)
        return result

    def _emit_lock_changed(self):
        status = self._status_data()
        if not self._closing.is_set():
            self._bus.emit('lock_changed', {key: status[key] for key in ('lock', 'default_lock')})
            self._bus.emit('status', status)
        return status

    def _default_comp(self):
        ctx = self._ctx
        for id in (ctx.effective_lock_id(), ctx.settings.assistant.default_lock_comp_id):
            comp = ctx.enabled_comp(id) if id else None
            if comp is not None:
                return comp
        return next(iter(ctx.enabled_comps()), None)

    def _selection_updated(self):
        """绝不持api锁等待mode锁；mode的sink会反向取api锁。"""
        mode = self._runner.mode
        if isinstance(mode, AssistantMode):
            mode.refresh_selection()
        else:
            with self._state_lock:
                old = self._last_advice
            scope = self._ctx.assistant_selection()
            if old and old.selection and old.selection['revision'] != scope['revision']:
                self._sink.on_advice(Advice(stage=None, phase=Phase.UNKNOWN,
                    headline='预选范围已更新，等待下一帧有效状态', selection=scope))

    def _comp_presets_data(self):
        ctx = self._ctx
        value, error = comp_presets.read(ctx.settings.assistant.comp_presets, ctx.db.season_id)
        catalog = ctx.enabled_comp_catalog()
        rows = [{'id': 'all', 'name': '全部阵容', 'members': [c.id for c in catalog], 'builtin': True},
                *[{**r, 'builtin': False} for r in value['configs']]]
        for row in rows:
            resolved, unavailable = set(), []
            for cid in row['members']:
                c = ctx.enabled_comp(cid)
                if c and c.heroes:
                    resolved.add(c.id)
                else:
                    original = ctx.db.comp(cid, raw=True)
                    unavailable.append({'id': cid, 'name': original.name if original else cid})
            row.update(selected_count=len(row['members']), effective_count=len(resolved), unavailable=unavailable)
        return {'season_id': ctx.db.season_id, 'active_id': value['selected'], 'configs': rows,
                'error': error, 'scope': ctx.assistant_selection(),
                'catalog': [{**serialize.comp_row(c), 'heroes': [h.name for h in c.heroes]} for c in catalog]}

    @expose
    def comp_presets(self):
        with self._lock, self._ctx.selection_lock:
            return self._comp_presets_data()

    def _edit_comp_preset(self, season_id, action, config_id=None, name=None, members=None):
        with self._operation_lock:
            with self._lock, self._ctx.selection_lock:
                ctx = self._ctx
                if not isinstance(season_id, str) or season_id != ctx.db.season_id:
                    raise ValueError('赛季已变化，请重新加载预选配置')
                raw, saved_id = comp_presets.edit(ctx.settings.assistant.comp_presets, season_id, action,
                    config_id=config_id, name=name, members=members,
                    known_ids={c.id for c in ctx.db.comps_catalog()})
                if raw != ctx.settings.assistant.comp_presets:
                    staged = deepcopy(ctx.settings)
                    staged.assistant.comp_presets = raw
                    staged.save()
                    ctx.settings.assistant.comp_presets = raw
                result = {**self._comp_presets_data(), 'saved_id': saved_id}
            self._selection_updated()
            self._emit_status()
            return result

    @expose
    def save_comp_preset(self, season_id, config_id, name, members):
        return self._edit_comp_preset(season_id, 'create' if config_id is None else 'update', config_id, name, members)

    @expose
    def select_comp_preset(self, season_id, config_id):
        return self._edit_comp_preset(season_id, 'select', config_id)

    @expose
    def delete_comp_preset(self, season_id, config_id):
        return self._edit_comp_preset(season_id, 'delete', config_id)

    def _log(self, text, level='info'):
        self._bus.emit('log', _log_entry(text, level))

    def _stop_runner(self):
        with self._lock:
            self._ctx.guard.stop('手动停止')
        self._runner.stop()
        if self._runner.running:
            self._emit_status()
            raise ValueError('旧模式仍在停止中（等待线程退出超时），请稍后再次停止；当前设置未应用，也未启动新模式')

    def _shutdown(self):
        self._closing.set()
        self._icons.close()
        self._window_drag.close()
        with self._operation_lock:
            self._runner.stop()
            if self._hud is not None:
                self._hud.close()

    @staticmethod
    def _integer(value, label, minimum, maximum):
        if isinstance(value, bool):
            raise ValueError(f'「{label}」必须是整数')
        try:
            number = int(value)
            if float(value) != number or not minimum <= number <= maximum:
                raise ValueError()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'「{label}」必须是 {minimum} 到 {maximum} 的整数') from exc
        return number

    @staticmethod
    def _boolean(value):
        if not isinstance(value, bool):
            raise ValueError('开关值必须是布尔值')
        return value

    @expose
    def set_window(self, window):
        self._window = window
        events = getattr(window, 'events', None)
        if events is not None:
            events.shown += self._window_drag.install
            # closing 在 AppKit 主线程同步执行，先移除监听再关闭最后一个窗口。
            if hasattr(events, 'closing'):
                events.closing += self._window_drag.close

    def _drag_window(self, key):
        if self._closing.is_set():
            return None
        if key == 'main':
            return self._window
        if key == 'hud_panel' and self._hud is not None:
            return self._hud._drag_window()
        return None

    @expose
    def window_drag_regions(self, key, regions, exclusions):
        self._window_drag.update(key, regions, exclusions)

    @expose
    def app_info(self):
        with self._lock:
            return {'version': __version__, 'season': self._ctx.db.season.label,
                    'dry_run': self._ctx.settings.actuator.dry_run}

    @expose
    def status(self):
        return self._status_data()

    def _toggle_hud(self, kind, on):
        on = self._boolean(on)
        with self._operation_lock:
            if self._hud is None or self._closing.is_set():
                raise ValueError('HUD 窗口尚未就绪')
            previous = self._hud.status[kind]
            if kind == 'overlay':
                info = self._ctx.locator.find(0) if on else None
                if on and info is None:
                    raise ValueError(f'找不到游戏窗口（进程名包含「{self._ctx.settings.actuator.window_owner}」）')
                self._hud.toggle_overlay(on, info)
            else:
                self._hud.toggle_panel(on)
                panel = self._hud._drag_window() if on else None
                if panel is not None and panel is not self._drag_panel:
                    # 概率窗口只额外暴露拖动区域上报，不开放业务方法。
                    panel.expose(self.window_drag_regions)
                    self._drag_panel = panel
            try:
                if on and not self._runner.running:
                    self._start_mode('assistant')
            except Exception:
                if kind == 'overlay':
                    self._hud.toggle_overlay(previous)
                else:
                    self._hud.toggle_panel(previous)
                self._emit_status()
                raise
            self._refresh_hud()
            mode = self._runner.mode
            if mode and getattr(mode, 'scout', None):
                mode.scout.hud_changed()
            return self._emit_status()

    @expose
    def hud_toggle(self, on):
        return self._toggle_hud('overlay', on)

    @expose
    def hud_panel_toggle(self, on):
        return self._toggle_hud('panel', on)

    def _start_mode(self, key, options=None, stop_current=False):
        with self._operation_lock:
            cls = next((cls for cls in _MODE_TYPES if cls.key == key), None)
            if cls is None:
                raise ValueError(f'未知的模式：{key}')
            self._boolean(stop_current)
            if self._runner.running:
                if getattr(self._runner, 'stopping', False):
                    self._emit_status()
                    raise ValueError('旧模式仍在停止中，请等待退出或再次停止')
                if self._runner.mode and self._runner.mode.key == key and not stop_current:
                    return self._emit_status()
                if not stop_current:
                    raise ValueError('已有运行模式，请确认停止并启动')
                self._stop_runner()
            # 停止超时后旧线程可能晚到；结束回调必须绑定本次模式实例。
            sink = WebSink(self._bus, self._db, on_state=self._on_state, on_advice=self._on_advice,
                           on_buy_projection=self._on_buy_projection, on_ai_projection=self._on_ai_projection,
                           hud_callback=self._emit_hud, status_getter=self._status_data,
                           hud_visible=self._scout_hud_visible, toast_capable=self._scout_toast_capable,
                           on_finished=lambda mode_key, reason: self._on_finished(mode_key, reason, mode))
            with self._lock:
                if self._ctx.db.is_empty:
                    raise ValueError('数据库为空，请先同步数据')
                if cls is AutoRollMode:
                    opts = options or {}
                    targets = opts.get('targets') or []
                    if not targets:
                        raise ValueError('没有勾选目标卡')
                    picked = []
                    for target in targets:
                        if not self._ctx.db.hero(target['name']):
                            raise ValueError(f'没有英雄「{target["name"]}」')
                        picked.append(RollTarget(target['name'], self._integer(target['star'], '目标星级', 1, 3)))
                    keep = self._integer(opts.get('keep', self._ctx.settings.autoroll.keep_gold), '保留金币', 0, 100)
                    maximum = self._integer(opts.get('max_rolls', self._ctx.settings.autoroll.max_rolls), '最多刷新次数', 1, 200)
                    mode = cls(self._ctx, sink, picked, keep, maximum)
                else:
                    mode = cls(self._ctx, sink)
            self._sink = sink
            problems = self._runner.start(mode)
            if problems:
                raise ValueError('无法启动：\n' + '\n'.join(problems))
            self._log(f'启动 {mode.title}')
            return self._emit_status()

    @expose
    def start_mode(self, key, options=None, stop_current=False):
        return self._start_mode(key, options, stop_current)

    @expose
    def stop_mode(self):
        with self._operation_lock:
            self._stop_runner()
            return self._emit_status()

    @expose
    def scout_now(self):
        with self._lock:
            mode = self._runner.mode
            if not self._runner.running or not isinstance(mode, AssistantMode) or not mode.scout:
                raise ValueError('请先启动辅助大脑')
            with self._state_lock:
                state = self._last_state
            problem = mode.scout.request_problem(state)
            if problem:
                raise ValueError(problem)
            mode.scout.request()
            return mode.scout.status

    @expose
    def scout_stop(self):
        mode = self._runner.mode
        if mode and getattr(mode, 'scout', None):
            mode.scout.stop()
        return self._emit_status()

    @expose
    def autopilot_resume(self, token):
        with self._operation_lock:
            mode = self._runner.mode
            if (not self._runner.running or not isinstance(mode, AutopilotMode)
                    or getattr(self._runner, 'stopping', False)):
                raise ValueError('托管未运行或正在停止，不能继续')
            mode.resume(token)
            return self._emit_status()

    def _autoroll_comp(self):
        # 本局有效锁定（未初始化时为默认锁定）优先；显式自动不会退回长期默认。
        ctx = self._ctx
        locked = ctx.effective_lock_id()
        if locked:
            return ctx.enabled_comp(locked)
        with self._state_lock:
            state, advice = self._last_state, self._last_advice
        if (state is None or not own_view(state) or state.players_stale or state.units_stale
                or state.stage_tuple is None or state.phase not in (Phase.PLANNING, Phase.COMBAT, Phase.AUGMENT)
                or advice is None or advice.target is None or advice.stage != state.stage):
            return None
        return ctx.enabled_comp(advice.target.comp.id)

    @expose
    def autoroll_defaults(self):
        with self._lock:
            cfg = self._ctx.settings.autoroll
            comp = self._autoroll_comp()
            heroes = [{'name': hero.name, 'cost': hero.cost, 'star': hero.star_target or cfg.target_star,
                       'checked': bool(hero.star_target or hero.is_carry or hero.is_subcarry)}
                      for hero in (comp.heroes if comp else []) if self._ctx.db.hero(hero.name)]
            return {'comp_name': comp.name if comp else None, 'keep': cfg.keep_gold, 'max_rolls': cfg.max_rolls,
                    'dry_run': self._ctx.settings.actuator.dry_run, 'heroes': heroes, 'all_heroes': self._hero_options()}

    @expose
    def settings_schema(self):
        with self._lock:
            return settings_schema.schema(self._ctx.settings, self._ctx.db)

    @expose
    def save_settings(self, values, stop_current=False):
        return self._save_settings(values, stop_current)

    def _source_flags(self):
        return {source: getattr(self._ctx.settings.data, 'source_' + source)
                for source in ('official', 'ingame', 'dataj')}

    def _save_settings(self, values, stop_current=False):
        self._boolean(stop_current)
        with self._operation_lock:
            with self._lock:
                previous_hud = asdict(self._ctx.settings.hud)
                previous_default = self._ctx.settings.assistant.default_lock_comp_id
                previous_sources = self._source_flags()
                staged = settings_schema._prepare(self._ctx.settings, values, self._ctx.db)
                if asdict(staged) == asdict(self._ctx.settings):
                    return self._emit_status()
                source_only = set(values) == {'data'} and set(values['data']) <= {'source_official', 'source_ingame', 'source_dataj'}
                if self._runner.running and not source_only and not stop_current:
                    raise ValueError('保存这些设置需要停止当前模式，请确认停止并应用')
            if not source_only:
                self._stop_runner()
            with self._lock:
                settings_schema.apply(self._ctx.settings, values, self._ctx.db)
                hud_changed = previous_hud != asdict(self._ctx.settings.hud)
                lock_changed = previous_default != self._ctx.settings.assistant.default_lock_comp_id
                sources_changed = previous_sources != self._source_flags()
                source_only = set(values) == {'data'} and set(values['data']) <= {'source_official', 'source_ingame', 'source_dataj'}
            if not source_only:
                with self._lock:
                    self._ctx.reset_runtime()
            if sources_changed:
                with self._lock:
                    messages = self._ctx.refresh_comp_sources()
                    if previous_default and not self._ctx.enabled_comp(previous_default):
                        messages.append('阵容来源已关闭，默认锁定保留但暂不可用：' + previous_default)
                    mode = self._runner.mode
                    if mode and not isinstance(mode, AssistantMode):
                        mode.tracker.reset()
                    with self._state_lock:
                        state = self._last_state
                if isinstance(mode, AssistantMode):
                    self._selection_updated()
                elif state is not None:
                    selection = self._ctx.assistant_selection()
                    advice = (mode.compute_advice(state) if mode else
                              self._ctx.advisor.advise(state, selection['lock_id'],
                                                       watch_ids=selection['watch_ids'],
                                                       augments_taken=self._ctx.game_augments,
                                                       selection=selection))
                    self._sink.on_advice(advice)
                for message in messages:
                    self._log(message)
                self._bus.emit('data_changed', {})
                status = self._status_data()
                self._bus.emit('watch_changed', {'watch': status['watch']})
                lock_changed = True
            self._log('设置已保存')
            if hud_changed:
                self._emit_hud()
            return self._emit_lock_changed() if lock_changed else self._emit_status()

    def _feature_values(self):
        return {section: {key: getattr(getattr(self._ctx.settings, section), key) for key in keys}
                for section, keys in settings_schema.FEATURE_FIELDS.items()}

    def _feature_settings_data(self):
        values = self._feature_values()
        choices = next(section for section in settings_schema.schema(self._ctx.settings, self._ctx.db, full=True)
                       if section['key'] == 'assistant')['fields']
        return {'values': values, 'comp_options': next(f['choices'] for f in choices if f['key'] == 'default_lock_comp_id'),
                'scout_rounds': list(self._ctx.settings.scout.rounds)}

    @expose
    def feature_settings(self):
        """功能页只读自己的字段，绝不夹带连接设置和密钥。"""
        with self._lock:
            return self._feature_settings_data()

    @expose
    def save_feature_settings(self, values):
        with self._operation_lock:
            with self._lock:
                if not isinstance(values, dict) or any(section not in settings_schema.FEATURE_FIELDS
                        or not isinstance(edits, dict) or set(edits) - settings_schema.FEATURE_FIELDS[section]
                        for section, edits in values.items()):
                    raise ValueError('功能设置只能更新所属字段')
                staged = settings_schema._prepare(self._ctx.settings, values, self._ctx.db)
                scout_changed = asdict(staged.scout) != asdict(self._ctx.settings.scout)
                if asdict(staged) == asdict(self._ctx.settings):
                    return self._feature_settings_data()
                # 长期默认只供下一局；当前模式尚未初始化本局方向时也保留原方向。
                if (staged.assistant.default_lock_comp_id != self._ctx.settings.assistant.default_lock_comp_id
                        and self._runner.running and self._ctx.game_lock_id is None):
                    self._ctx.game_lock_id = self._ctx.effective_lock_id() or ''
                settings_schema.apply(self._ctx.settings, values, self._ctx.db)
                result = self._feature_settings_data()
            mode = self._runner.mode
            if scout_changed and mode and getattr(mode, 'scout', None):
                mode.scout.invalidate_notice('侦察设置已更改，下次调度生效', stop=True)
            self._emit_lock_changed()
            self._emit_hud()
            self._log('功能设置已保存，不会自动启动')
            return result

    @expose
    def set_dry_run(self, on, stop_current=False):
        on = self._boolean(on)
        self._boolean(stop_current)
        with self._operation_lock:
            if on == self._ctx.settings.actuator.dry_run:
                return self._emit_status()
            # 辅助大脑也持有侦察执行器，因此任何运行模式都必须显式停止。
            if self._runner.running:
                if not stop_current:
                    raise ValueError('切换全局演练需要停止当前模式，请确认停止并应用')
                self._stop_runner()
            with self._lock:
                self._ctx.settings.actuator.dry_run = on
                self._ctx.settings.save()
                self._ctx.reset_runtime()
            return self._emit_status()

    @expose
    def set_llm_enabled(self, on):
        on = self._boolean(on)
        with self._lock:
            cfg = self._ctx.settings.llm
            if on and not (cfg.api_key and cfg.model):
                raise ValueError('还没填 API Key / 模型：设置 → 大模型')
            cfg.enabled = on
            self._ctx.settings.save()
            self._ctx.reset_llm()
        return self._emit_status()

    @expose
    def set_llm_preset(self, key):
        with self._lock:
            if key not in PRESETS:
                raise ValueError(f'未知的大模型档位：{key}')
            preset = apply_preset(self._ctx.settings.llm, key)
            self._ctx.settings.save()
            self._ctx.reset_llm()
        self._log(f'大模型切换到「{preset.title}」：{preset.model}，平时 {preset.effort}，海克斯 {preset.effort_augment}（{preset.note}）')
        return self._emit_status()

    @expose
    def llm_presets(self):
        with self._lock:
            return [{'key': p.key, 'title': p.title, 'model': p.model, 'effort': p.effort,
                     'effort_augment': p.effort_augment, 'note': p.note,
                     'active': p.key == self._ctx.settings.llm.preset} for p in PRESETS.values()]

    @expose
    def test_llm(self, values=None):
        with self._lock:
            cfg = (settings_schema._prepare(self._ctx.settings, {'llm': values.get('llm', {})}, self._ctx.db).llm
                   if values is not None else deepcopy(self._ctx.settings.llm))
        client = LLMClient(cfg)
        try:
            return client.ping()
        finally:
            client._http.close()

    @expose
    def seasons(self):
        with self._lock:
            db = self._ctx.db
            synced = db.synced_seasons()
            catalog = list(db.catalog())
            if db.season_id not in {season.id for season in catalog}:
                catalog.append(db.season)
            return [{'id': season.id, 'label': season.label,
                     'game_version': (synced.get(season.id) or {}).get('game_version'),
                     'has_dataj': bool(season.dataj_set), 'has_ingame': season.id in INGAME_SEASONS,
                     'current': season.id == db.season_id} for season in catalog]

    @expose
    def switch_season(self, id):
        with self._operation_lock:
            with self._lock:
                if self._syncing:
                    raise ValueError('正在同步数据，完成后再切换')
            self._stop_runner()
            with self._lock:
                self._ctx.switch_season(id)
                with self._state_lock:
                    self._last_state = None
                    self._last_advice = None
                result = {'needs_sync': self._ctx.db.is_empty}
            self._bus.emit('data_changed', {})
            self._emit_hud()
            self._emit_status()
            return result

    @expose
    def set_source(self, which, on):
        on = self._boolean(on)
        if which not in ('official', 'ingame', 'dataj'):
            raise ValueError(f'未知的数据源：{which}')
        with self._operation_lock:
            self._save_settings({'data': {'source_' + which: on}})
            with self._lock:
                return {**self._source_flags(), 'message': self._status_data()['comp_source_notice']}

    @expose
    def sync(self, details, all_seasons=False):
        details, all_seasons = self._boolean(details), self._boolean(all_seasons)
        with self._operation_lock:
            with self._lock:
                if self._syncing:
                    raise ValueError('正在同步')
                if self._closing.is_set():
                    raise ValueError('窗口正在关闭')
                if not enabled_comp_sources(self._ctx.settings.data):
                    raise ValueError('至少要选一个数据源')
                path, season_id = self._ctx.db.path, self._ctx.db.season_id
                self._syncing = True
                self._sync_thread = threading.Thread(target=self._sync_worker, args=(path, season_id, details, all_seasons),
                                                     name='webui-sync', daemon=True)
            self._emit_status()
            log.info('后台同步开始：%s', '全部赛季' if all_seasons else season_id)
            self._sync_thread.start()
        return {'started': True}

    def _sync_worker(self, path, season_id, details, all_seasons):
        def progress(message):
            if not self._closing.is_set():
                self._bus.emit('sync_progress', {'message': message})
                self._log(message)

        try:
            db = GameDB(path, season_id)
            try:
                if all_seasons:
                    done = self._ctx.sync_all_seasons(details, progress, True, db=db)
                    message = ('已同步 ' + str(len(done)) + ' 个赛季：' + '、'.join(db.catalog_name(s) for s in done)
                               if done else '所有赛季都是最新的')
                else:
                    stats = self._ctx.sync(details, progress, db=db)
                    source = '官方资料库' if stats['data_source'] == 'official' else 'dataj'
                    message = (f"同步完成：{stats['season']}，版本 {stats['game_version']}（基础数据：{source}），"
                               f"{stats['heroes']} 个英雄，{stats['guides']} 套官方攻略，{stats['comps']} 套 dataj 阵容")
                if not self._closing.is_set():
                    with self._lock:
                        # 同步连接已恢复当前赛季并加载缓存；不在界面锁内重复解析。
                        self._ctx.reload_data(db)
                        if not self._runner.running:
                            self._ctx.reset_runtime()
            finally:
                db.close()
            ok = True
        except Exception as exc:
            ok, message = False, f'同步失败：{exc}'
        finally:
            with self._lock:
                self._syncing = False
        if not self._closing.is_set():
            self._log(message, 'info' if ok else 'warning')
            self._bus.emit('sync_done', {'ok': ok, 'message': message})
            if ok:
                self._bus.emit('data_changed', {})
                self._selection_updated()
            self._emit_status()

    @expose
    def db_stats(self):
        with self._lock:
            return {'stats': self._ctx.db.stats(), 'pool_count_by_cost': self._ctx.db.pool_count_by_cost()}

    @expose
    def comps(self, source='all'):
        with self._lock:
            db = self._ctx.db
            stats = db.stats()
            return {'info': {'game_version': stats['game_version'],
                             'official_count': sum('web' in c.channels for c in db.official_comps.values()),
                             'ingame_count': sum('ingame' in c.channels for c in db.official_comps.values()),
                             'dual_channel_count': sum({'web', 'ingame'} <= set(c.channels) for c in db.official_comps.values()),
                             'catalog_count': len(db.comps_catalog()), 'dataj_count': stats['comps'],
                             'synced_at': stats['synced_at'],
                             'source_notice': '当前赛季暂不支持微社区阵容'
                             if source == 'ingame' and db.season_id not in INGAME_SEASONS else ''},
                    'rows': [serialize.comp_row(comp) for comp in self._ctx.enabled_comp_catalog(source)]}

    @expose
    def comp_detail(self, id):
        with self._lock:
            comp = self._ctx.enabled_comp(id, raw=True)
            if comp is None:
                raise ValueError('找不到阵容')
            return serialize.comp_detail(comp, self._ctx.db)

    @expose
    def lock_comp_for_game(self, id):
        with self._lock:
            if not isinstance(id, str):
                raise ValueError('阵容 id 必须是文本')
            comp = self._ctx.enabled_comp(id) if id else None
            if id and comp is None:
                raise ValueError('找不到阵容')
            self._ctx.game_lock_id = id
        self._selection_updated()
        self._log('本局锁定：' + (comp.name if comp else '自动'))
        return self._emit_lock_changed()

    @expose
    def set_watch_comps(self, ids):
        with self._lock:
            if not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids):
                raise ValueError('关注阵容必须是 id 文本列表')
            ids = list(dict.fromkeys(ids))
            if len(ids) > 4:
                raise ValueError('本局最多关注 4 个阵容')
            # 来源关闭或资料消失不删除旧关注；允许逐项维护，不能新增不可用项。
            existing = set(self._ctx.game_watch_ids)
            if any(cid not in existing and self._ctx.enabled_comp(cid) is None for cid in ids):
                raise ValueError('找不到关注阵容')
            self._ctx.game_watch_ids = ids
        self._selection_updated()
        status = self._status_data()
        self._bus.emit('watch_changed', {'watch': status['watch']})
        self._bus.emit('status', status)
        return status

    @expose
    def set_default_lock_comp(self, id):
        with self._lock:
            if not isinstance(id, str):
                raise ValueError('阵容 id 必须是文本')
            comp = self._ctx.enabled_comp(id) if id else None
            if id and comp is None:
                raise ValueError('找不到阵容')
            if self._runner.running and self._ctx.game_lock_id is None:
                self._ctx.game_lock_id = self._ctx.effective_lock_id() or ''
            self._ctx.settings.assistant.default_lock_comp_id = id
            self._ctx.settings.save()
        self._selection_updated()
        self._log('默认锁定阵容：' + (comp.name if comp else '自动'))
        return self._emit_lock_changed()

    @expose
    def db_list(self, kind, query=''):
        with self._lock:
            return serialize.db_rows(self._ctx.db, kind, query)

    def _hero_options(self):
        return [{'name': hero.name, 'cost': hero.cost}
                for hero in sorted(self._ctx.db.pool_heroes(), key=lambda hero: (hero.cost, hero.name))]

    @expose
    def hero_options(self):
        with self._lock:
            return self._hero_options()

    def _odds_input(self, rows, level, use_live):
        level = self._integer(level, '等级', 1, 10)
        self._boolean(use_live)
        with self._state_lock:
            owned = self._last_state.owned_copies() if use_live and self._last_state is not None else Counter()
        cleaned = {}
        for row in rows:
            name = row['name']
            if not self._ctx.db.hero(name):
                continue
            need = self._integer(row['need'], '需要张数', 0, 30)
            others = self._integer(row.get('others', 0), '别人拿走', 0, 30)
            cleaned[name] = {'name': name, 'need': need, 'others': others}
        taken = Counter(owned)
        for name, row in cleaned.items():
            taken[name] += row['others']
            self._ctx.extra_taken[name] = row['others']
        return list(cleaned.values()), level, owned, PoolState.from_db(self._ctx.db, self._ctx.rules, taken)

    @expose
    def odds_calc(self, rows, level, use_live=True):
        with self._lock:
            rows, level, owned, pool = self._odds_input(rows, level, use_live)
            output, names = [], []
            for row in rows:
                have = owned.get(row['name'], 0)
                need = max(row['need'] - have, 0)
                odds = hit_odds(pool, row['name'], level, max(need, 1))
                if odds is None:
                    continue
                output.append({**row, 'cost': odds.cost, 'have': have, 'remaining': odds.remaining,
                               'slot_prob': odds.slot_prob, 'per_roll': odds.per_roll,
                               'rolls_50': odds.rolls_for_50 if need else None,
                               'gold_50': odds.gold_for_50 if need else None,
                               'rolls_80': odds.rolls_for_80 if need else None,
                               'gold_80': odds.gold_for_80 if need else None, 'done': need == 0})
                if need:
                    names.append(row['name'])
            return {'owned': dict(owned), 'rows': output, 'any_hit': any_hit_per_roll(pool, names, level) if names else None}

    @expose
    def odds_simulate(self, rows, level, gold, keep, use_live=True, sims=3000):
        with self._lock:
            rows, level, owned, pool = self._odds_input(rows, level, use_live)
            gold = self._integer(gold, '金币', 0, 200)
            keep = self._integer(keep, '保留金币', 0, 100)
            sims = self._integer(sims, '模拟次数', 1, 100000)
            targets = {row['name']: row['need'] - owned.get(row['name'], 0) for row in rows
                       if row['name'] in pool.heroes and row['need'] > owned.get(row['name'], 0)}
        if not targets:
            return None
        result = simulate_rolldown(pool, targets, gold, level, keep, sims=sims)
        return {key: getattr(result, key) for key in ('all_hit_rate', 'hit_rate', 'avg_rolls', 'avg_gold_left')}

    @expose
    def odds_import_target(self):
        with self._lock:
            comp = self._default_comp()
            return {'comp_name': comp.name if comp else None,
                    'rows': [{'name': hero.name, 'need': 9 if hero.star_target == 3 else 3,
                              'others': self._ctx.extra_taken.get(hero.name, 0)} for hero in (comp.heroes if comp else [])]}

    @expose
    def live_basics(self):
        with self._state_lock:
            state = self._last_state
            return {'level': state.level, 'gold': state.gold} if state is not None else None

    @expose
    def population_preview(self, population, target_id, season_id, selection_revision=None):
        if population is not None and (type(population) is not int or not 1 <= population <= 10):
            raise ValueError('预览人口必须是 1 到 10 的整数，或自动')
        if not isinstance(target_id, str) or not isinstance(season_id, str):
            raise ValueError('阵容和赛季标识必须是文字')
        with self._lock:
            if season_id != self._ctx.db.season_id:
                raise ValueError('预览上下文已变化，请重试')
            with self._state_lock:
                current, advice = self._last_state, self._last_advice
                state = self._preview_state
                if (current is None or state is None or advice is None or advice.target is None
                        or current.phase == Phase.ENDED or not has_live_hud(current)
                        or (current.stage, current.phase) != (state.stage, state.phase)):
                    return None
                if advice.target.comp.id != target_id:
                    raise ValueError('预览主目标已变化，请重试')
                if selection_revision is not None and selection_revision != advice.selection.get('revision'):
                    raise ValueError('预选范围已变化，请重试')
                if advice.selection and advice.selection['revision'] != self._ctx.assistant_selection()['revision']:
                    raise ValueError('预选范围已变化，请重试')
                snapshot, frozen_advice = deepcopy(state), deepcopy(advice)
            pool = self._ctx.advisor.pool(snapshot, dict(self._ctx.extra_taken))
            result = population_preview(snapshot, frozen_advice, population, self._ctx.db, pool)
            with self._state_lock:
                if (current is not self._last_state or advice is not self._last_advice or self._closing.is_set()
                        or advice.selection and advice.selection['revision'] != self._ctx.assistant_selection()['revision']):
                    raise ValueError('预览上下文已变化，请重试')
            return {**result, 'target_id': target_id, 'season_id': season_id,
                    'selection_revision': advice.selection.get('revision')}

    @expose
    def check_vision(self):
        with self._lock:
            self._ctx.vision.start()
            state = self._ctx.vision.read()
            return {'summary': state.summary(), 'shop': state.shop_names}

    @expose
    def check_executor(self):
        from jccbrain.actuator.executor import MacExecutor

        with self._lock:
            ctx = self._ctx
            executor = MacExecutor(ctx.layout, ctx.guard, ctx.settings.actuator, ctx.locator)
            problems = executor.preflight()
            win = ctx.locator.find(0)
            info = f'游戏窗口：{win.width:.0f}×{win.height:.0f} @ ({win.x:.0f}, {win.y:.0f})' if win else '游戏窗口：未找到'
            return {'window': info, 'calibrated': ctx.layout.board_calibrated, 'problems': problems}

    def _choose_file(self, replay=False):
        import webview

        if self._window is None:
            raise ValueError('窗口尚未就绪')
        paths = self._window.create_file_dialog(webview.FileDialog.OPEN,
                                               file_types=('JSONL (*.jsonl;*.json)',) if replay else tuple())
        return str(paths[0]) if paths else None

    @expose
    def browse_file(self):
        return self._choose_file()

    @expose
    def choose_replay(self):
        path = self._choose_file(replay=True)
        if path is None:
            return None
        with self._operation_lock:
            self._stop_runner()
            with self._lock:
                self._ctx.settings.vision.backend = 'replay'
                self._ctx.settings.vision.replay_file = path
                self._ctx.settings.save()
                self._ctx.reset_runtime()
            self._start_mode('assistant')
        return path

    def _units(self):
        with self._state_lock:
            state = self._last_state
        if state is None:
            raise ValueError('还没有画面，先启动辅助大脑')
        return {**{f'bench:{u.slot}': u for u in state.bench}, **{f'board:{i}': u for i, u in enumerate(state.board)}}

    @expose
    def label_units_info(self):
        with self._lock:
            units = []
            for key, unit in self._units().items():
                where = (f'备战席 {unit.slot + 1}' if unit.slot is not None else '备战席') if unit.zone == 'bench' else (
                    f'棋盘 ({unit.x:.2f}, {unit.y:.2f})' if unit.x is not None and unit.y is not None else '棋盘（坐标未知）')
                units.append({'key': key, 'where': where, 'star': unit.star, 'name': unit.name, 'known': unit.known})
            return {'units': units, 'heroes': [hero['name'] for hero in self._hero_options()]}

    @expose
    def label_units(self, assignments):
        with self._lock:
            units = self._units()
            bench, board = {}, []
            for assignment in assignments:
                unit = units.get(assignment['key'])
                name = assignment['name'].strip()
                if unit is None or not name or name == unit.name or not self._ctx.db.hero(name):
                    continue
                if unit.zone == 'bench' and unit.slot is not None:
                    bench[unit.slot] = name
                elif unit.x is not None:
                    board.append({'x': unit.x, 'y': unit.y, 'name': name})
            return self._ctx.vision.label_units(bench, board) if bench or board else []

    @expose
    def calibrate_points(self):
        return [{'key': key, 'label': label} for key, label in _POINTS]

    @expose
    def calibrate_capture(self):
        from jccbrain.actuator.window import mouse_location

        with self._lock:
            window = self._ctx.locator.find(0)
            if window is None:
                raise ValueError('找不到游戏窗口')
            x, y = window.to_relative(*mouse_location())
            return {'x': round(x, 4), 'y': round(y, 4)}

    @expose
    def calibrate_save(self, points):
        with self._operation_lock:
            if self._runner.running:
                raise ValueError('当前模式仍在运行或停止中，请先停止再保存标定')
            return self._calibrate_save(points)

    def _calibrate_save(self, points):
        import math

        with self._lock:
            converted = {}
            for key, label in _POINTS:
                try:
                    point = points[key]
                    if len(point) != 2:
                        raise ValueError()
                    x, y = float(point[0]), float(point[1])
                    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y)):
                        raise ValueError()
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f'「{label}」必须提供 0 到 1 之间的两个坐标') from exc
                converted[key] = (x, y)
            layout = deepcopy(self._ctx.layout)
            layout.board_corners = {key: converted[key] for key in ('r1c1', 'r1c7', 'r4c1', 'r4c7')}
            layout.board_calibrated = True
            layout.bench_x0, layout.bench_y = converted['bench0']
            layout.bench_step = round((converted['bench8'][0] - converted['bench0'][0]) / 8, 4)
            layout.refresh_button = converted['refresh_button']
            layout.xp_button = converted['xp_button']
            a, b = converted['player0'], converted['player7']
            layout.player_slots = [(a[0] + (b[0] - a[0]) * i / 7, a[1] + (b[1] - a[1]) * i / 7) for i in range(8)]
            layout.player_slots_calibrated = True
            layout.save()
            self._ctx.reset_runtime()
            return None
