"""辅助大脑：实时观察并给出建议；侦察需单独开启或手动触发。

画面每 0.4 秒左右读一帧，但给人看的建议要稳：
- 识别结果先过 StateStabilizer（读不到的沿用、单帧误读不采用）；
- 建议内容连续两帧一致才推给界面，内容没变不重推；回合/阶段切换时立即推；
- 大模型每回合问一次，回复按回合保存，这一回合之后的每次推送都带上，直到进入下一回合。
"""
from __future__ import annotations

import logging
import json
import threading
from dataclasses import replace, astuple
from collections.abc import Callable

from jccbrain.core.models import GameState, Phase
from jccbrain.core.buy_source import BuySource, new_buy_session, buy_fingerprint, buy_eligible
from jccbrain.engine.passive_scout import PassiveScout, STABLE_THRESHOLD
from jccbrain.engine.advisor import Advice, BuyProjection
from jccbrain.engine.lifecycle import AssistantLifecycle, has_live_hud
from jccbrain.engine.augments import recommended_augment
from jccbrain.engine.llm import LLMReply, build_context
from jccbrain.core.ai_source import raw_semantics, ai_scene, context_semantics
from jccbrain.modes.ai_display import AiDisplay
from jccbrain.engine.stabilizer import StateStabilizer
from jccbrain.modes.base import Mode
from jccbrain.modes.buy_activity import BuyActivityObserver
from jccbrain.modes.scout import ScoutRunner, own_view
from jccbrain.modes.unit_learning import UnitLearningRunner
from jccbrain.engine.scout import ScoutShot

log = logging.getLogger(__name__)


def advice_signature(adv: Advice) -> tuple:
    """决定界面要不要刷新的内容摘要（不含每帧都会微动的提示文字）。"""
    board = adv.board
    return (
        json.dumps(adv.opponents, sort_keys=True, ensure_ascii=False),
        adv.stage, adv.phase, adv.headline, adv.locked, adv.comp_status, adv.switch_reason,
        tuple((b.slot, b.name) for b in adv.buys),
        tuple((s.slot, s.name) for s in adv.sells),
        (adv.econ.action, adv.econ.keep_gold, adv.econ.xp_buys, adv.econ.roll_budget, adv.econ.reason) if adv.econ else None,
        tuple((m.comp.id, m.seen, m.seen_carry, m.owned_copies) for m in adv.comps),
        tuple((t.name, t.need) for t in adv.targets),
        tuple((a.name, a.score) for a in adv.augments),
        tuple(adv.augments_taken), adv.ai_augment,
        json.dumps(adv.traits_view, sort_keys=True, ensure_ascii=False),
        (tuple(sorted(adv.held_items['components'].items())), tuple(adv.held_items['completed']),
         tuple(adv.held_items['emblems'])),
        json.dumps(adv.item_plan, sort_keys=True, ensure_ascii=False),
        (board.source, board.population, board.next_hint,
         tuple((c.row, c.col, c.name, c.owned_star, c.on_board, c.item, c.is_carry, tuple(c.items), c.role,
                tuple((d['name'], tuple(d['components']), tuple(d['held']), d['ready']) for d in c.item_details))
               for c in board.cells)) if board else None,
    )


class _PublicationStop(threading.Event):
    """直接 stop_event.set() 也与检查/发布线性化；不等待网络线程。"""
    def __init__(self, lock, on_stop: Callable[[], None], fast_stop=lambda: None):
        super().__init__()
        self._publication_lock = lock
        self._on_stop = on_stop
        self._fast_stop = fast_stop
        self._set_lock = threading.Lock()

    def set(self) -> None:
        with self._set_lock:
            if self.is_set():
                return
            super().set()  # stop/epoch fence must not wait for compute or native callbacks
        self._fast_stop()
        with self._publication_lock:
            self._on_stop()


class AssistantMode(Mode):
    key = "assistant"
    title = "辅助大脑"
    acts = False

    def __init__(self, ctx, sink, *, buy_monitor=None):
        super().__init__(ctx, sink)
        self.stabilizer = StateStabilizer()
        book = getattr(ctx.advisor, "scout", None)
        self.passive_scout = PassiveScout(book) if book else None
        self.scout = ScoutRunner(ctx, sink, lambda: self.stopped) if book else None
        self.learning = UnitLearningRunner(ctx, sink, lambda: self.stopped, scout=self.scout)
        self._lock = threading.RLock()
        self.lifecycle = AssistantLifecycle()
        self.stop_event = _PublicationStop(self._lock, self._invalidate_stop, self._stop_buy_input)
        self._last_key = self._shown = self._pending = None
        self._latest: Advice | None = None
        self._llm_input_epoch = None
        self._llm_options: tuple[str, ...] | None = None
        self._llm: tuple[str, LLMReply] | None = None     # (回合, 回复)
        self._selection_revision = None
        self.buy_session = new_buy_session()
        self._buy_frame = 0
        self._buy_pending = self._buy_identity = self._buy_shown = None
        self._buy_lock = threading.RLock()
        self._input_epoch = 0
        self._full_epoch = 0
        self._full_epoch_pending = None
        self._input_blocked = False
        self._last_raw_buy = None
        self._buy_monitor = buy_monitor
        self._buy_observer = None
        self._buy_observer_token = None
        self._buy_last_source = self._buy_source(self.lifecycle.neutral_state(), eligible=False)
        self._ai = AiDisplay(self.buy_session, lambda p: getattr(self.sink, "on_ai_projection", lambda p: None)(p))
        self._ai_frames = {}
        self._llm_source = None

    @property
    def buy_input_epoch(self):
        return self._input_epoch

    @property
    def buy_input_status(self):
        return self._buy_observer.status if self._buy_observer else {'available': False, 'reason': 'offline_raw_first'}

    def buy_read_token(self):
        with self._buy_lock:
            return self._input_epoch

    def _buy_source(self, st, *, eligible=None, read_token=None):
        selection = self.ctx.assistant_selection() if hasattr(self.ctx, 'assistant_selection') else {'revision': ''}
        with self._buy_lock:
            token = self._input_epoch if read_token is None else read_token
            usable = buy_eligible(st) if eligible is None else eligible
            usable = usable and token == self._input_epoch and not self._input_blocked and not self.stopped
            source = BuySource(self.buy_session, self._buy_frame, self.lifecycle.generation,
                               selection['revision'], self.ctx.db.season_id, st.stage, st.phase.value,
                               buy_fingerprint(st), bool(usable), input_epoch=token)
            if token == self._input_epoch:
                self._buy_last_source = source
            return source

    def _clear_buys(self):
        with self._buy_lock:
            self._buy_frame += 1
            self._buy_pending = self._buy_shown = None
        source = self._buy_source(self.lifecycle.neutral_state(), eligible=False)
        self.sink.on_buy_projection(BuyProjection(source))

    def _input_changed(self, reason, blocked, detected_at=None, *, token=None, stopping=False):
        with self._buy_lock:
            if not stopping and (self.stopped or token is not None and token is not self._buy_observer_token):
                return
            self._input_epoch += 1
            self._full_epoch_pending = None
            self._input_blocked = blocked
            self._buy_frame += 1
            self._buy_pending = self._buy_shown = None
            source = replace(self._buy_last_source, frame=self._buy_frame,
                             input_epoch=self._input_epoch, eligible=False)
            self._buy_last_source = source
        # No mode lock, selection lookup, OCR or native HUD call on this path.
        self.sink.on_buy_projection(BuyProjection(source))
        self._ai.revoke(input_epoch=source.input_epoch)

    def _start_buy_input(self):
        monitor = self._buy_monitor
        if monitor is None:
            from jccbrain.vision.pyeye import PyEyeProvider
            if self.ctx.settings.vision.backend != 'pyeye' or not isinstance(self.ctx.vision, PyEyeProvider):
                return
        token = object()
        self._buy_observer_token = token
        self._input_changed('observer_start', True, token=token)
        previous = [None]
        def observed(reason, blocked, at):
            self._input_changed(reason, blocked, at, token=token)
            if reason != previous[0] and token is self._buy_observer_token and not self.stopped:
                previous[0] = reason
                self.sink.on_log('买卡输入观察：' + reason + ('；回退raw首帧保护' if reason.startswith('fallback:') else ''))
        observer = BuyActivityObserver(getattr(self.ctx, 'locator', None), observed, monitor=monitor)
        self._buy_observer = observer
        observer.start()

    def _stop_buy_input(self):
        self._buy_observer_token = None
        self._input_changed('stop', True, stopping=True)
        if self._buy_observer:
            self._buy_observer.request_stop()

    @staticmethod
    def _buy_decision(adv):
        # Seen/owned counters and equipment are not strategy identity. A target,
        # candidate ordering, lock or economic decision change waits for full advice.
        return json.dumps([adv.stage, adv.phase.value, adv.locked, adv.comp_status,
                           adv.switch_reason, [m.comp.id for m in adv.comps],
                           [adv.econ.action.value, adv.econ.keep_gold, adv.econ.xp_buys,
                            adv.econ.roll_budget, adv.econ.reason] if adv.econ else None],
                          ensure_ascii=False)

    def compute_advice(self, state: GameState, *, new_game: bool = False) -> Advice:
        with self._lock:
            return super().compute_advice(state, new_game=new_game)

    def _selection_changed(self, selection: dict) -> None:
        revision = selection['revision']
        if revision != self._selection_revision:
            if self._selection_revision is not None:
                self.tracker.reset()
                self.lifecycle.invalidate()
                self._clear_llm()
                self._last_key = self._shown = self._pending = None
                self._buy_identity = None
            self._selection_revision = revision
            self._clear_buys()

    def refresh_selection(self) -> bool:
        # API调用者必须先释放api锁；模式回调会反向进入sink/api。
        with self._lock:
            selection = self.ctx.assistant_selection()
            if selection['revision'] == self._selection_revision:
                return False
            self._selection_changed(selection)
            headline = ('辅助已停止' if self.stopped else self.lifecycle.headline
                        if not self.lifecycle.active else '预选范围已更新，等待下一帧有效状态')
            self.sink.on_advice(Advice(stage=None, phase=self.lifecycle.neutral_state().phase,
                                      headline=headline, selection=selection))
            return True

    def _clear_llm(self) -> None:
        if hasattr(self, "_ai"):
            self._ai.revoke(input_epoch=self._input_epoch)
        self._llm_source = None
        self._latest = None
        self._llm = None
        self._llm_input_epoch = None
        self._llm_options = None

    def _reset_learning(self) -> None:
        # Main-loop lifecycle boundary: discard pending confirmations before the
        # next ordinary read can configure learning for a new scene/game.
        self.learning.reset_confirmations()

    def _invalidate_stop(self) -> None:
        self.learning.stop()  # reaches OCR/promotion cancellation without waiting for it
        if self.scout:
            self.scout.stop()
        self.lifecycle.stop()
        self._clear_buys()
        self._clear_llm()
        # 包括捕获失败/回放EOF。只中止网络回填仍会让界面保留旧建议。
        clean = self.lifecycle.neutral_state()
        clean.buy_source = self._buy_source(clean, eligible=False)
        self.sink.on_state(clean)
        self.sink.on_advice(Advice(stage=None, phase=Phase.UNKNOWN, headline='辅助已停止', buy_source=clean.buy_source))

    def _on_new_game(self) -> None:
        self._reset_learning()
        if self.scout:
            self.scout.invalidate_notice('新对局，旧侦察预告已取消')
        self.lifecycle.invalidate()
        self._clear_buys()
        self._clear_llm()
        self._last_key = self._shown = self._pending = None

    def loop(self) -> str:
        try:
            self._clear_buys()
            self._start_buy_input()
            while not self.stopped:
                read_token = self.buy_read_token()
                raw = self.ctx.vision.read(fresh=True)
                accepted = self._process_frame(raw, read_token=read_token)
                if accepted is None:
                    continue
                st, adv, generation = accepted
                if self.scout and self.scout.maybe_run(st):
                    continue
                # Only raw carries the current scene/timer, never stabilized
                # values. A transaction returns to a normal read before LLM.
                self.learning.scout = self.scout
                if self.learning.maybe_run(raw):
                    continue
                with self._lock:
                    ask = (not self.stopped and self.lifecycle.active
                           and generation == self.lifecycle.generation and self.ctx.llm.ready
                           and self.ctx.llm.should_call(st, adv))
                if ask:
                    threading.Thread(target=self._ask_llm, args=(st, adv, generation), daemon=True).start()
            return "已停止"
        finally:
            # 包括回放EOF和视觉异常，旧请求都不能在模式退出后回填。
            try:
                self.stop_event.set()
            finally:
                if self._buy_observer:
                    if not self._buy_observer.close():
                        self.sink.on_log("买卡输入观察停止超时；旧监听回调已撤销授权")
                self.learning.close()

    def _process_frame(self, raw: GameState, *, read_token=None) -> tuple[GameState, Advice, int] | None:
        with self._lock:
            if self.stopped:
                return None
            read_token = self.buy_read_token() if read_token is None else read_token
            raw_ai = raw_semantics(raw, observed=True)
            self._ai.observe_raw(raw_ai, usable=ai_scene(raw) and own_view(raw), input_epoch=read_token)
            raw_buy = buy_fingerprint(raw, observed=True)
            with self._buy_lock:
                raw_dependency = (raw.stage, raw.phase, raw_buy)
                changed_raw = self._last_raw_buy is not None and raw_dependency != self._last_raw_buy
                self._last_raw_buy = raw_dependency
            if changed_raw or not buy_eligible(raw):
                # FIRST raw dependency change, before scout/HUD/stabilizer callbacks.
                self._clear_buys()
            if self.scout:
                self.scout.observe(raw)
                if not has_live_hud(raw):
                    self.scout.invalidate_notice('等待有效对局画面，侦察预告已取消')
            if (raw.phase == Phase.SPECTATE or not own_view(raw)
                    or (self.scout and self.scout.paused
                        and raw.phase not in (Phase.PLANNING, Phase.COMBAT, Phase.AUGMENT, Phase.ENDED)
                        and has_live_hud(raw))):
                if (self.scout and not own_view(raw) and has_live_hud(raw)
                        and raw.phase != Phase.ENDED and (raw.board or raw.traits)):
                    self.scout.book.add(ScoutShot.from_dict(raw.raw, slot=(raw.viewing or {}).get('slot'),
                                                          name=(raw.viewing or {}).get('name'), round=raw.stage))
                self._clear_buys()
                return None
            previous = self.lifecycle.generation
            new_game = self.lifecycle.observe(raw)
            changed = self.lifecycle.generation != previous
            if changed:
                self._buy_identity = None
                self._clear_buys()
                if not new_game:
                    self._reset_learning()
                self.stabilizer.reset()
                self._clear_llm()
                self._last_key = self._shown = self._pending = None
            if not self.lifecycle.active:
                clean = self.lifecycle.neutral_state()
                clean.buy_source = self._buy_source(clean, eligible=False, read_token=read_token)
                self.sink.on_state(clean)
                adv = Advice(stage=None, phase=clean.phase, headline=self.lifecycle.headline, buy_source=clean.buy_source)
                if changed or self._latest is None or self._latest.headline != adv.headline:
                    self._latest = adv
                    self.sink.on_advice(adv)
                    self.sink.on_log(adv.lines()[0])
                return None
            if self.scout:
                self.scout.mark_own()
            self._observe_opponent(raw)
            st = GameState.from_dict(self.stabilizer.update(raw.raw)) if raw.raw else raw
            if hasattr(self.ctx, 'assistant_selection'):
                self._selection_changed(self.ctx.assistant_selection())
            with self._buy_lock:
                self._buy_frame += 1
            raw_matches = raw_buy == buy_fingerprint(st)
            st.buy_source = self._buy_source(st, eligible=buy_eligible(raw) and buy_eligible(st) and raw_matches,
                                             read_token=read_token)
            with self._buy_lock:
                changed_shown = bool(self._buy_shown and self._buy_shown[2] != st.buy_source.fingerprint)
                if changed_shown:
                    self._buy_shown = None
            if changed_shown:
                self.sink.on_buy_projection(BuyProjection(st.buy_source))
            self.sink.on_state(st)
            adv = self.compute_advice(st, new_game=new_game)
            if self.stopped:
                return None
            source = self._buy_source(st, eligible=buy_eligible(raw) and buy_eligible(st) and raw_matches,
                                      read_token=read_token)
            if source != st.buy_source:
                with self._buy_lock:
                    if read_token == self._input_epoch and not self.stopped:
                        # compute may start a new game and publish its clear.
                        # This new state must follow that clear in publication
                        # order, rather than overriding an explicit same-frame fence.
                        self._buy_frame += 1
                source = self._buy_source(st, eligible=buy_eligible(raw) and buy_eligible(st) and raw_matches,
                                          read_token=read_token)
                st.buy_source = source
                self.sink.on_state(st)
            decision = self._buy_decision(adv)
            adv.buy_source = replace(st.buy_source, decision=decision)
            # LLM context is computed before cached AI is attached, avoiding
            # feedback from the previous response into semantic identity.
            ai_source = self._ai.confirm(raw_semantics(st), context_semantics(build_context(st, adv)), st.buy_source,
                usable=ai_scene(st) and own_view(st) and self.lifecycle.active
                and not self._input_blocked and read_token == self._input_epoch)
            self._attach_llm(adv)
            self._ai_frames[(adv.buy_source.input_epoch, adv.buy_source.frame)] = ai_source
            if len(self._ai_frames) > 512:
                self._ai_frames.pop(next(iter(self._ai_frames)))
            self._latest = adv
            if read_token != self._input_epoch:
                return st, adv, self.lifecycle.generation
            sig = advice_signature(adv)
            key = (st.stage, st.phase)
            with self._buy_lock:
                refresh_pair = self._full_epoch != read_token and self._full_epoch_pending == (read_token, sig)
            publish_full = key != self._last_key or (sig != self._shown and sig == self._pending) or refresh_pair
            if publish_full:
                self.sink.on_advice(adv)
            with self._buy_lock:
                # Input may arrive while the ordinary callback waits for native
                # UI work. Never mark its now-rejected old advice as shown.
                if read_token != self._input_epoch or self.stopped:
                    return st, adv, self.lifecycle.generation
                new_key = key != self._last_key
                if publish_full:
                    self._buy_identity = decision
                    self._full_epoch = read_token
                    self._shown = sig
                    self._last_key = key
                self._pending = sig
                self._full_epoch_pending = (read_token, sig)
            if publish_full and new_key:
                self.sink.on_log(adv.lines()[0])
            with self._buy_lock:
                candidate = (st.buy_source.generation, st.buy_source.selection, st.buy_source.fingerprint,
                             decision, tuple(astuple(b) for b in adv.buys), st.buy_source.season, read_token)
                usable = st.buy_source.eligible and read_token == self._input_epoch and not self.stopped
                publication = None
                if not usable or decision != self._buy_identity:
                    self._buy_shown = None
                    publication = BuyProjection(replace(adv.buy_source, eligible=False) if not usable else adv.buy_source)
                elif candidate == self._buy_pending and candidate != self._buy_shown:
                    publication = BuyProjection(adv.buy_source, tuple(replace(b) for b in adv.buys), True)
                    self._buy_shown = candidate
                self._buy_pending = candidate if usable else None
            if publication is not None:
                self.sink.on_buy_projection(publication)
            return st, adv, self.lifecycle.generation

    def _observe_opponent(self, raw: GameState) -> None:
        if not self.passive_scout:
            return
        if not self.passive_scout.needs_image(raw):
            self.passive_scout.observe(raw, None)
            return
        try:
            import cv2
            import numpy as np
            # 现有只读截图接口返回刚读完的画面；0保持原尺寸，避免二次采集。
            data = self.ctx.vision.screenshot(width=0)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            cfg = getattr(self.ctx.settings, 'scout', None)
            threshold = getattr(cfg, 'stable_threshold', STABLE_THRESHOLD)
            self.passive_scout.observe(raw, image, threshold)
        except Exception:
            log.exception('被动侦察读取失败，跳过此帧')

    def _attach_llm(self, adv: Advice) -> None:
        """把本回合的大模型意见挂到建议上；进入下一回合后不再显示上一回合的。"""
        if (not self._ai.current(self._llm_source) or not self._llm or self._llm[0] != adv.stage
                or getattr(self, '_llm_input_epoch', None) is not None and self._llm_input_epoch != self._input_epoch):
            return
        if adv.augments and self._llm_options != tuple(a.name for a in adv.augments):
            return  # 选项刷新后，旧选项的AI回复不可标记到新卡片。
        reply = self._llm[1]
        if adv.selection and reply.switch_comp and reply.switch_comp not in {r['name'] for r in adv.selection['allowed_comps']}:
            return
        adv.ai_augment = recommended_augment(reply.headline, [a.name for a in adv.augments]) if adv.augments else None
        adv.llm_text = reply.headline + ("：" + "；".join(reply.actions) if reply.actions else "") + \
            (f"（{reply.reason}）" if reply.reason else "")
        allowed = adv.comp_status != '关注' or reply.switch_comp in {m.name for m in adv.comps}
        if allowed and not adv.locked and reply.switch_comp and adv.target and reply.switch_comp != adv.target.name:
            suffix = '（已定型，仅供参考）' if adv.comp_status == '定型' else ''
            adv.notes.append(f"AI 建议转：{reply.switch_comp}{suffix}")

    def _ask_llm(self, st: GameState, adv: Advice, generation: int | None = None) -> None:
        """网络调用不持发布锁；代次检查与最终推送在同一临界区完成。"""
        with self._lock:
            generation = self.lifecycle.generation if generation is None else generation
            if (self.stopped or not self.lifecycle.active or generation != self.lifecycle.generation
                    or self.scout and self.scout.paused):
                return
            if adv.buy_source and adv.buy_source.input_epoch != self._input_epoch:
                return
            binding = self._ai_frames.get((adv.buy_source.input_epoch, adv.buy_source.frame)) if adv.buy_source else None
            if not self._ai.current(binding):
                return
            revision = adv.selection.get('revision')
            if revision and revision != self.ctx.assistant_selection()['revision']:
                return
        try:
            reply = self.ctx.llm.advise(st, adv)
        except Exception as e:  # noqa: BLE001 - 网络错误只记录
            with self._lock:
                if not self.stopped and self.lifecycle.active and generation == self.lifecycle.generation:
                    self.sink.on_log(f"大模型请求失败：{e}")
            return
        with self._lock:
            if (self.stopped or not self.lifecycle.active or generation != self.lifecycle.generation
                    or self.scout and self.scout.paused):
                return
            if revision and revision != self.ctx.assistant_selection()['revision']:
                return
            if adv.buy_source and adv.buy_source.input_epoch != self._input_epoch:
                return
            if not self._ai.current(binding):
                return
            latest = self._latest
            options = tuple(a.name for a in adv.augments)
            if latest is None or latest.stage != st.stage or tuple(a.name for a in latest.augments) != options:
                self.sink.on_log(f"AI（{reply.latency_s:.1f}s，已过回合不显示）：{reply.headline}")
                return
            self._llm_source = binding
            self._llm = (st.stage, reply)
            self._llm_input_epoch = adv.buy_source.input_epoch if adv.buy_source else self._input_epoch
            self._llm_options = options or None
            latest = replace(latest, notes=list(latest.notes), ai_augment=None)
            self._attach_llm(latest)
            if not self._ai.reply(binding, latest.llm_text or "", latest.ai_augment, options, buy_frame=adv.buy_source.frame):
                return
            self.sink.on_log(f"AI（{reply.latency_s:.1f}s）：{latest.llm_text}")
            if not (self.scout and self.scout.paused):
                self.sink.on_advice(latest)
