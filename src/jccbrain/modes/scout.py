"""快速扫描只做受控点击和截图；独立读取器在后台消费原图。"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite
from uuid import uuid4

import numpy as np

from jccbrain.actuator.actions import ClickPoint, RestoreMouse
from jccbrain.actuator.safety import AbortError
from jccbrain.actuator.executor import MacExecutor
from jccbrain.core.models import GameState, Phase
from jccbrain.modes.input_activity import InputActivity
from jccbrain.core.paths import var_dir
from jccbrain.engine.scout import ScoutShot

log = logging.getLogger('jccbrain.scout')

# 1774×1136实测图（含标题栏）核对：左栏图标/人数约x=.05～.17、y=.11～.70，
# 右栏高亮/头像约x=.85～.99、y=.08～.78。保留边缘覆盖刷新，不只看棋盘背景。
SCOUT_REGIONS = {'board': (.15, .12, .80, .78), 'traits': (.00, .10, .18, .72),
                 'players': (.84, .08, 1.0, .80)}
TRAITS_REFRESH_THRESHOLD = 1.0  # 0～255平均像素差；只排除未刷新，不能代替后台羁绊校验。
NOTICE_SECONDS = 3.0
NOTICE_ACK_TIMEOUT = 5.0


@dataclass
class ScoutCapture:
    image: np.ndarray
    loaded: bool
    timing: dict


def thumbnails(image):
    small = image[::8, ::8].astype(np.float32)
    h, w = small.shape[:2]
    return {key: small[int(h*y0):int(h*y1), int(w*x0):int(w*x1)]
            for key, (x0, y0, x1, y1) in SCOUT_REGIONS.items()}


def thumbnail(image):
    return thumbnails(image)['board']


def difference(a, b):
    return float(np.abs(a - b).mean()) if a.shape == b.shape and a.size else float('inf')


def trait_signature(traits):
    return tuple(sorted([(t.name, t.count, t.active, t.next) for t in traits],
                        key=lambda row: tuple((value is not None, value if value is not None else empty)
                                              for value, empty in zip(row, ('', 0, False, 0)))))


def validate_shot(shot, target, previous_traits=(), previous_slot=None):
    """保留原识别数据供审计；质量标记决定后续棋盘能否用于推断。"""
    if (shot.slot is not None and shot.slot != target.slot) or (shot.name and target.name and shot.name != target.name):
        shot.flags.append('viewing_mismatch')
    signature = trait_signature(shot.traits)
    if signature and target.slot != previous_slot and signature == previous_traits:
        shot.flags.append('traits_same_as_prev')
    if sum(t.count or 0 for t in shot.traits) >= 3 and not shot.board:
        shot.flags.append('board_not_loaded')
    shot.flags = list(dict.fromkeys(shot.flags))
    return shot


def own_view(state):
    viewing = state.viewing
    if not viewing or all(viewing.get(k) is None for k in ('slot', 'name')):
        return True
    return any(p.is_self and ((p.slot is not None and viewing.get('slot') == p.slot)
                             or (p.name and viewing.get('name') == p.name)) for p in state.players)


class ScoutRunner:
    def __init__(self, ctx, sink, stopped=lambda: False, *, capture=None, reader_factory=None,
                 probe=None, clock=time.monotonic, sleep=time.sleep, input_monitor=None):
        self.ctx, self.sink, self.stopped = ctx, sink, stopped
        self.book = ctx.advisor.scout
        self._capture = capture
        self._simulated_frames = capture is not None
        self._reader_factory = reader_factory or self._make_reader
        self._probe = probe or self._timer_probe
        self.clock, self.sleep = clock, sleep
        self.activity = InputActivity(getattr(ctx, "locator", None), monitor=input_monitor, clock=clock)
        self._lock = threading.RLock()
        self._pending = threading.Event()
        self._cancel = threading.Event()
        self._phase_abort = threading.Event()
        self._thread = None
        self._notice_state = None
        self._latest_state = None
        self._input_reason = None
        self._auto_permit = None
        self._touched = False
        self._awaiting_own = False
        self._attempted = set()
        self._generation = self.book.started
        self._accept_retries = False
        self._retry_targets = deque()
        self._retried_slots = set()
        self._status = {'scanning': False, 'processing': False, 'pending': False,
                        'captured': 0, 'total': 0, 'processed': 0, 'recognized': 0,
                        'interruption_s': 0, 'background_s': 0, 'retries': 0, 'reason': '', 'notice': None}

    def _notice(self, text):
        log.info(text)
        self.sink.on_log(text)

    def _remaining_timer(self):
        return min(self._initial.timer - (self.clock() - self._started),
                   self._timer_value - (self.clock() - self._timer_at))

    def _request_retry(self, target):
        with self._lock:
            if (self._accept_retries and target.slot not in self._retried_slots
                    and self._remaining_timer() >= max(5, self.ctx.settings.scout.min_timer / 2)
                    and not self._cancel.is_set() and not self._phase_abort.is_set() and not self.stopped()):
                self._retried_slots.add(target.slot)
                self._retry_targets.append(target)

    def _next_retry(self):
        with self._lock:
            if self._retry_targets and self._remaining_timer() >= max(5, self.ctx.settings.scout.min_timer / 2):
                return self._retry_targets.popleft()
            self._accept_retries = False
            return None

    @property
    def status(self):
        with self._lock:
            result = dict(self._status)
            result['notice'] = dict(result['notice']) if result['notice'] else None
            return result

    @property
    def scanning(self):
        return self.status['scanning']

    @property
    def paused(self):
        return self.scanning or self._awaiting_own

    def mark_own(self):
        self._awaiting_own = False

    @property
    def busy(self):
        s = self.status
        return (s['scanning'] or s['processing'] or s['pending'] or s['notice'] is not None
                or self._auto_permit is not None)

    def _update(self, **values):
        with self._lock:
            self._status.update(values)
        self.sink.on_scout_changed()

    def request(self):
        with self._lock:
            if self.busy:
                raise ValueError('侦察正在进行')
            self._cancel.clear()
            self._pending.set()
            self._status.update(pending=True, reason='')
        self.sink.on_scout_changed()

    def stop(self):
        self.invalidate_notice('已取消本轮自动侦察', stop=True)

    def invalidate_notice(self, reason, *, stop=False):
        """取消和授权在同一锁内变更；回调必须在锁外，避免与 WebSink 反向加锁。"""
        with self._lock:
            changed = self._notice_state is not None or self._auto_permit is not None
            if changed or stop:
                n = self._notice_state or self._auto_permit
                stage = n['stage'] if n else self._latest_state.stage if self._latest_state else None
                if stage:
                    self._attempted.add(stage)
                self._cancel.set()
                self._notice_state = self._auto_permit = None
                self._pending.clear()
                self._status.update(notice=None, pending=False, reason=reason)
        if (changed or stop) and not self.scanning:
            self.activity.close()
        if changed or stop:
            self.sink.on_scout_changed()
            self._notice(reason)

    def notice_shown(self, notice_id, window):
        if not isinstance(window, str) or not self.sink.scout_hud_visible(window):
            return False
        with self._lock:
            n = self._notice_state
            if not n or n['id'] != notice_id or self._cancel.is_set() or self.stopped():
                return False
            state = self._latest_state
            if (n['generation'] != self.book.started or state is None or state.stage != n['stage']
                    or self.input_problem(state, self.ctx.settings.scout.min_timer)
                    or not self._scheduled(state) or self.ctx.guard.stopped):
                return False
            if self.clock() - n['created'] >= NOTICE_ACK_TIMEOUT and n['ack'] is None:
                return False
            if n['windows'] and window not in n['windows']:
                return False
            try:
                self.activity.check()
            except AbortError:
                return False
            n['windows'].add(window)
            if n['ack'] is None:
                n['ack'] = self.clock()
        return True

    def cancel_notice(self, notice_id):
        with self._lock:
            n = self._notice_state or self._auto_permit
            if not n or n['id'] != notice_id:
                return False
            self._cancel.set()
            self._notice_state = self._auto_permit = None
            self._status.update(notice=None, reason='已取消本轮自动侦察')
        if not self.scanning:
            self.activity.close()
        self.sink.on_scout_changed()
        self._notice('已取消本轮自动侦察')
        return True

    def observe(self, state):
        """原始输入先更新/撤销通知，稳定器不能沿用上一帧的准备阶段。"""
        with self._lock:
            self._latest_state = state
            reason = self.request_problem()
            availability_changed = reason != self._input_reason
            self._input_reason = reason
            n = self._notice_state or self._auto_permit
            invalid = n and (state.stage != n['stage'] or state.phase != Phase.PLANNING
                             or not own_view(state))
        if invalid:
            self.invalidate_notice('画面或回合已变化，跳过本轮自动侦察')
        elif availability_changed:
            self.sink.on_scout_changed()

    def hud_changed(self):
        with self._lock:
            n = self._notice_state or self._auto_permit
            windows = tuple(n['windows']) if n else ()
        if n and windows and not any(self.sink.scout_hud_visible(w) for w in windows):
            self.invalidate_notice('侦察提示已关闭或不可见，跳过本轮自动侦察')

    @staticmethod
    def input_problem(state, min_timer=5):
        """共享可用原因，仅读状态；预告/手动/API 均不能使用非己方或过期玩家栏。"""
        if state is None or state.stage_tuple is None:
            return '等待有效对局画面'
        stage = state.stage_tuple
        if stage[0] < 1 or not 1 <= stage[1] <= 7:
            return '等待有效对局回合'
        if state.phase != Phase.PLANNING:
            return '仅在准备阶段可用'
        if state.timer is None or not isfinite(state.timer) or state.timer < min_timer:
            return f'倒计时至少 {min_timer} 秒才能侦察'
        if state.players_stale:
            return '玩家栏已过期，等待当前画面'
        if not own_view(state):
            return '请先返回自己的棋盘'
        own = [p for p in state.players if p.is_self]
        if len(own) != 1 or type(own[0].slot) is not int or not 0 <= own[0].slot < 8:
            return '尚未识别自己的玩家栏位置'
        return ''

    def request_problem(self, fallback=None):
        with self._lock:
            state = self._latest_state if self._latest_state is not None else fallback
        problem = self.input_problem(state)
        if not problem and (self.stopped() or self.ctx.guard.stopped):
            problem = '侦察已停止或安全守卫急停'
        if not problem and not self.targets(state):
            problem = '没有可侦察的存活对手'
        return problem

    def _scheduled(self, state):
        cfg = self.ctx.settings.scout
        return cfg.enabled and (cfg.schedule == 'every_round' or state.stage in cfg.rounds)

    def _preflight(self, state, *, manual=False):
        ex = self.ctx.executor
        problem = self.input_problem(state)
        problems = ([problem] if problem else []) + list(ex.preflight())
        if self.stopped() or self.ctx.guard.stopped or self._cancel.is_set():
            problems.append('侦察已停止或安全守卫急停')
        if not ex.dry_run and not self.ctx.layout.player_slots_calibrated:
            problems.append('玩家栏尚未标定：请在工具→标定中完成第1/第8个头像标定')
        if self.ctx.settings.vision.backend == 'replay' and not ex.dry_run:
            problems.append('状态文件回放只支持演练侦察；实机请用实时视觉读取游戏内回放')
        if not any(p.is_self and type(p.slot) is int and 0 <= p.slot < 8 for p in state.players):
            problems.append('尚未识别自己的玩家栏位置')
        if not problems and isinstance(ex, MacExecutor):
            win = self.ctx.locator.find(0)
            if not win or self._foreground_pid() != win.pid:
                problems.append('游戏不是前台，请切回游戏后再侦察')
        return problems

    def _automatic_problem(self, state, notice=None):
        cfg = self.ctx.settings.scout
        if not self._scheduled(state):
            return '自动侦察已禁用或本回合未启用'
        problem = self.input_problem(state, cfg.min_timer)
        if problem:
            return problem
        if notice and (notice['generation'] != self.book.started or notice['stage'] != state.stage):
            return '对局或回合已变化'
        if not getattr(self.sink, 'scout_toast_capable', lambda: False)():
            return '侦察提示窗口不可用，跳过本轮自动侦察'
        if notice and notice['windows'] and not any(self.sink.scout_hud_visible(w) for w in notice['windows']):
            return '侦察提示已隐藏，跳过本轮自动侦察'
        if notice:
            try:
                self.activity.check()
            except AbortError as exc:
                return str(exc)
        if notice and notice.get('window') is not None:
            win = self.ctx.locator.find(0)
            if not win or (win.pid, win.window_id) != notice['window']:
                return '游戏窗口已变化，跳过本轮自动侦察'
        problems = self._preflight(state)
        if problems:
            return '；'.join(problems)
        if not self.targets(state):
            return '没有可侦察的存活对手'
        return ''

    def targets(self, state):
        recent = {row.slot: row for row in self.book.latest() if row.slot is not None}
        cfg = self.ctx.settings.scout
        targets = []
        seen = set()
        for p in state.players:
            if p.is_self or p.hp is None or p.hp <= 0 or type(p.slot) is not int or not 0 <= p.slot < 8 or p.slot in seen:
                continue
            seen.add(p.slot)
            row = recent.get(p.slot)
            seen_hp = self.book.last_seen_hp(p.slot)
            if cfg.skip_recent_rounds > 0 and row and row.stale_rounds <= cfg.skip_recent_rounds \
                    and seen_hp is not None and abs(seen_hp - p.hp) < 10:
                continue
            priority = 0 if row is None else 1 if row.stale_rounds >= 2 else 2
            targets.append((priority, -p.hp, p.slot, p))
        return [row[3] for row in sorted(targets)[:cfg.max_targets]]

    def maybe_run(self, state):
        self.observe(state)
        if self._generation != self.book.started:
            self.invalidate_notice('新对局，旧侦察预告已取消')
            self._attempted.clear()
            self._generation = self.book.started
        if self.status['scanning'] or self.status['processing']:
            return False
        manual = self._pending.is_set()
        cfg = self.ctx.settings.scout
        input_problem = self.input_problem(state, 5 if manual else cfg.min_timer)
        eligible = not input_problem
        if manual:
            self._pending.clear()
            if eligible:
                self._attempted.add(state.stage)
                self.run(state, manual=True, _requested=True)
                return True
            self._update(pending=False, reason=input_problem)
            self._notice('无法侦察：' + input_problem)
            return False
        with self._lock:
            n = self._notice_state
        if n:
            reason = self._automatic_problem(state, n)
            if not reason and n['ack'] is None and self.clock() - n['created'] >= NOTICE_ACK_TIMEOUT:
                reason = 'HUD未确认呈现，跳过本轮自动侦察'
            if reason:
                self.invalidate_notice(reason)
                return False
            with self._lock:
                if self._notice_state is not n or self._cancel.is_set():
                    return False
                elapsed = self.clock() - n['ack'] if n['ack'] is not None else 0
                seconds = max(1, ceil(NOTICE_SECONDS - elapsed))
                changed = self._status['notice']['seconds'] != seconds
                self._status['notice']['seconds'] = seconds
                ready = n['ack'] is not None and elapsed >= NOTICE_SECONDS
                if ready:
                    self._notice_state = None
                    self._auto_permit = n
            if changed or ready:
                self.sink.on_scout_changed()
            if ready:
                self.run(state, _permit=n)
            return ready
        if not eligible or not self._scheduled(state) or state.stage in self._attempted:
            return False
        # 新的回合可开始新的授权；同回合取消永不重新清除。
        with self._lock:
            if state.stage in self._attempted:
                return False
            self._cancel.clear()
        reason = self._automatic_problem(state)
        if state.timer < cfg.min_timer + NOTICE_SECONDS:
            reason = '倒计时不足以预告3秒后侦察，跳过本轮'
        if not reason:
            try:
                snapshot = self.activity.start(automatic=True)
            except AbortError as exc:
                # 预告前忙碌仅延后，同回合可在足够时间内重新满足空闲条件。
                self.activity.close()
                self._update(reason=str(exc))
                return False
        if reason:
            self._attempted.add(state.stage)
            self.activity.close()
            self._update(reason=reason)
            self._notice(reason)
            return False
        win = snapshot.window
        n = {'id': uuid4().hex, 'stage': state.stage, 'generation': self.book.started,
             'window': (win.pid, win.window_id) if win else None,
             'created': self.clock(), 'ack': None, 'windows': set()}
        with self._lock:
            if self._cancel.is_set() or self.stopped():
                self.activity.close()
                return False
            self._attempted.add(state.stage)
            self._notice_state = n
            self._status.update(notice={'id': n['id'], 'stage': state.stage, 'seconds': 3}, reason='')
        self.sink.on_scout_changed()
        self._notice('3秒后自动侦察（居中提示呈现确认后计时，用户活动取消本次）')
        return False

    def _make_reader(self):
        from jccbrain.vision.pyeye import build_reader
        # 单独的tracker/cache/阶段记忆，绝不复用主视觉的GameReader。
        # 复用现有图库只读参考；新读取器自己的持有/阶段记忆不回流主视觉。
        return build_reader(self.ctx.db, ocr=self.ctx.settings.vision.ocr)

    @staticmethod
    def _timer_probe(image, initial):
        from jccbrain.vision.eye.digits import read_digits
        from jccbrain.vision.eye.image import crop, load_regions
        cell = crop(image, load_regions()['timer'])
        text = read_digits(cell, image.shape[0]) if cell is not None else ''
        number = int(text) if text.isdigit() else None
        return {'timer': number, 'phase': 'combat' if number is not None and initial.timer is not None
                and number > initial.timer + 3 else None}

    def _foreground_pid(self):
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app else None

    def _check(self):
        if self.stopped():
            raise AbortError('辅助模式已停止')
        if getattr(self, '_run_generation', self.book.started) != self.book.started:
            raise AbortError('对局已变化')
        if self.ctx.guard.stopped:
            raise AbortError(self.ctx.guard.reason or '安全守卫急停')
        if self._cancel.is_set():
            raise AbortError('用户停止')
        if self._phase_abort.is_set():
            raise AbortError('进入战斗或海克斯阶段')
        with self._lock:
            permit, state = self._auto_permit, self._latest_state
        if permit is not None and not self._touched:
            reason = self._automatic_problem(state, permit) if state else '画面已失效'
            if reason:
                raise AbortError(reason)
            if self.clock() - permit['ack'] < NOTICE_SECONDS:
                raise AbortError('HUD预告不足3秒')
        if self._remaining_timer() < 5:
            raise AbortError('倒计时不足5秒')
        if isinstance(self.ctx.executor, MacExecutor) and self._window and self._foreground_pid() != self._window.pid:
            raise AbortError('游戏已离开前台')
        self.activity.check()
        if self._cancel.is_set() or self.stopped():
            raise AbortError('用户停止')

    def _grab(self, *, cleanup=False):
        self._check()
        image, win = self._capture.grab()
        if self._window is not None and (win.pid, win.window_id) != (self._window.pid, self._window.window_id):
            raise AbortError('游戏窗口已变化')
        self._window = win
        self._check()
        if not cleanup:
            probe = self._probe(image, self._initial)
            if probe.get('phase') in ('combat', 'augment'):
                self._phase_abort.set()
            if probe.get('timer') is not None and probe['timer'] < 5:
                raise AbortError('倒计时不足5秒')
            if probe.get('timer') is not None:
                self._timer_value, self._timer_at = probe['timer'], self.clock()
            self._check()
        return image

    def _point(self, player):
        x, y = self.ctx.layout.player_slots[player.slot]
        if isinstance(player.y, (int, float)) and isfinite(player.y) and 0 <= player.y <= 1:
            y = player.y
        return x, y

    def _return_check(self):
        self._check()

    def _execute(self, action):
        runner = self
        class Boundary:
            def check(self): runner._check()
            def begin(self, kind, point, tolerance):
                runner._check()
                runner.activity.begin(kind, point, tolerance)
            def finish(self, check): runner.activity.finish(check)
        def before():
            self._check()  # 必须在 SafetyGuard 限速等待之后检查。
            self._touched = True
            if self.status['notice'] is not None:
                self._update(notice=None)
        self.ctx.executor.execute(action, before_do=before, event_guard=Boundary())

    def _click(self, player, *, cleanup=False):
        point = self._point(player)
        action = ClickPoint(point, f'侦察：玩家栏第{player.slot + 1}位')
        self._execute(action)
        self.sink.on_action(action, not self.ctx.executor.dry_run)

    def _wait_stable(self, before, *, cleanup=False):
        cfg = self.ctx.settings.scout
        previous = before
        cut_at = None
        stable = 0
        started = self.clock()
        deadline = started + cfg.switch_timeout_ms / 1000
        image = None
        while self.clock() < deadline:
            image = self._grab(cleanup=cleanup)
            current = thumbnails(image)
            now = self.clock()
            if cut_at is None and difference(current['board'], before['board']) > cfg.cut_threshold:
                cut_at = now
            # dwell之后才累计稳定帧，给棋子/羁绊栏实际留出加载时间。
            ready = (cut_at is not None and (now-started)*1000 >= cfg.min_dwell_ms
                     and all(difference(current[k], previous[k]) < cfg.stable_threshold for k in SCOUT_REGIONS))
            stable = stable + 1 if ready else 0
            refreshed = difference(current['traits'], before['traits']) > TRAITS_REFRESH_THRESHOLD
            if stable >= cfg.stable_frames and refreshed and now <= deadline:
                return ScoutCapture(image, True, self._timing(started, cut_at, now))
            previous = current
            self.sleep(.01)
        # 无cut/羁绊相同也保留最后一帧；loaded=False禁止棋盘影响卡池/阵容。
        return ScoutCapture(image, False, self._timing(started, cut_at, self.clock()))

    @staticmethod
    def _timing(started, cut_at, now):
        return {'cut_ms': round((cut_at-started)*1000, 1) if cut_at is not None else None,
                'settle_ms': round((now-cut_at)*1000, 1) if cut_at is not None else None,
                'total_ms': round((now-started)*1000, 1)}

    def run(self, state, *, manual=False, _permit=None, _requested=False):
        cfg = self.ctx.settings.scout
        ex = self.ctx.executor
        with self._lock:
            if manual and not _requested:
                if self.busy:
                    return
                self._cancel.clear()
            if not manual and (_permit is None or self._auto_permit is not _permit):
                return
        problems = self._preflight(state, manual=manual)
        if manual and not problems:
            try:
                self.activity.start(automatic=False)
            except AbortError as exc:
                problems.append(str(exc))
        if not manual:
            reason = self._automatic_problem(self._latest_state or state, _permit)
            if reason:
                problems.append(reason)
        self._update(pending=False, reason='')
        if problems:
            self.activity.close()
            with self._lock:
                self._auto_permit = None
            self._update(reason='；'.join(problems))
            self._notice('无法侦察：' + '；'.join(problems))
            return
        targets = self.targets(state)
        if not targets:
            self.activity.close()
            with self._lock:
                self._auto_permit = None
            self._update(reason='没有可侦察的存活对手')
            return
        if not self.ctx.layout.player_slots_calibrated:
            self._notice('玩家栏使用默认坐标；可在工具→标定中标定第1/8个头像')
        me = next(p for p in state.players if p.is_self and type(p.slot) is int and 0 <= p.slot < 8)
        self._awaiting_own = True
        self._phase_abort.clear()
        self._touched = False
        self._initial, self._started = state, self.clock()
        self._timer_value, self._timer_at = state.timer, self._started
        self._window = None
        try:
            origin = ex.mouse_position()
        except Exception as exc:
            self.activity.close()
            with self._lock:
                self._auto_permit = None
                self._awaiting_own = False
            self._update(notice=None, reason=f'读取鼠标位置失败：{exc}')
            return
        self._last_mouse = origin
        generation = self.book.started
        self._run_generation = generation
        self._update(scanning=True, processing=False, captured=0, total=len(targets), processed=0, recognized=0,
                     interruption_s=0, background_s=0, retries=0)
        with self._lock:
            self._retry_targets.clear()
            self._retried_slots.clear()
            self._accept_retries = True
        jobs = queue.Queue(maxsize=2*len(targets))
        scan_done = threading.Event()
        captured_slots = set()
        reason = ''

        def capture_target(target):
            before = thumbnails(self._grab())
            self._click(target)
            frame = self._wait_stable(before)
            if not frame.loaded:
                self._notice(f'侦察玩家栏第{target.slot + 1}位加载超时，保留未加载截图')
            captured_slots.add(target.slot)
            self._update(captured=len(captured_slots))
            jobs.put_nowait((target, ScoutCapture(frame.image.copy(), frame.loaded, frame.timing), time.time()))

        try:
            if ex.dry_run and not self._simulated_frames:
                for target in targets:
                    self._click(target)
                self._notice('[演练] 仅记录侦察点击计划，不采集未切换的真实画面')
                return
            if self._capture is None:
                from jccbrain.vision.eye.capture import WindowCapture
                self._capture = WindowCapture(self.ctx.locator)
            self._update(processing=True)
            self._thread = threading.Thread(target=self._recognize, args=(jobs, scan_done, generation, state.stage),
                                            name='scout-reader', daemon=True)
            self._thread.start()
            for target in targets:
                capture_target(target)
            # 只消费扫描结束前已返回的识别反馈，不等待OCR；每位最多补一次。
            while (target := self._next_retry()) is not None:
                self._check()
                self._update(retries=self.status['retries'] + 1)
                capture_target(target)
        except Exception as exc:
            reason = str(exc)
            self._notice('侦察已中断：' + reason)
        finally:
            with self._lock:
                self._accept_retries = False
            # 所有清理动作仍走execute和guard；急停时不绕过守卫强行点回去。
            try:
                if not self._touched:
                    raise AbortError('尚未开始侦察，不执行返回/鼠标恢复')
                self._check()
                restore = origin
                before = thumbnails(self._grab(cleanup=True)) if self._capture is not None else None
                self._click(me, cleanup=True)
                if before is not None and not self._wait_stable(before, cleanup=True).loaded:
                    self._notice('返回自己加载等待超时；主循环将继续丢弃观战帧')
                self._execute(RestoreMouse(restore))
            except Exception as exc:
                self._notice(f'侦察返回/恢复鼠标未完成：{exc}')
                reason = reason or str(exc)
            if not self.activity.close():
                reason = reason or '输入监听未能停止'
            elapsed = self.clock() - self._started
            with self._lock:
                self._auto_permit = None
                if not self._touched:
                    self._awaiting_own = False
            self._update(scanning=False, notice=None, interruption_s=round(elapsed, 3), reason=reason)
            scan_done.set()
            self._notice(f"侦察 {state.stage}：截到 {self.status['captured']}/{len(targets)}，打断 {elapsed:.2f} s，补截 {self.status['retries']}")

    def _recognize(self, jobs, scan_done, generation, stage):
        started = time.monotonic()
        reader = None
        previous_traits, previous_slot = trait_signature(self._initial.traits), None
        processed_slots, recognized_slots = set(), set()
        self._update(processing=True)
        while not scan_done.is_set() or not jobs.empty():
            try:
                target, frame, captured_at = jobs.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                if self._cancel.is_set() or self.stopped() or generation != self.book.started:
                    continue
                if reader is None:
                    reader = self._reader_factory()
                read = getattr(reader, 'read_scout', None) or reader.read
                result = read(frame.image)
                if isinstance(result, GameState):
                    result = result.raw
                if result.get('stage') and result['stage'] != stage:
                    self._phase_abort.set()
                    continue
                if result.get('phase') in ('combat', 'augment'):
                    self._phase_abort.set()
                # 快读接口可不带phase：后台额外读一个横幅，不能阻塞截图/切换。
                if hasattr(reader, 'ocr') and hasattr(reader, 'regions'):
                    from jccbrain.vision.eye.image import crop
                    cell = crop(frame.image, reader.regions['banner'])
                    text = ''.join(line.text for line in reader.ocr.recognize_batch([reader.ocr.prepare(cell)], ['zh-Hans', 'en-US'], ['banner'])[0]) if cell is not None else ''
                    if any(word in text for word in ('战斗开始', '强化符文', '选择一个')):
                        self._phase_abort.set()
                if generation != self.book.started or self.stopped() or self._cancel.is_set():
                    continue
                shot = ScoutShot.from_dict(result, slot=target.slot, name=target.name, round=stage, t=captured_at,
                                           loaded=frame.loaded, timing=frame.timing,
                                           flags=[] if frame.loaded else ['load_timeout'])
                validate_shot(shot, target, previous_traits, previous_slot)
                timing = shot.timing
                cut = f"{timing['cut_ms']}ms" if timing['cut_ms'] is not None else '未检测'
                settle = f"{timing['settle_ms']}ms" if timing['settle_ms'] is not None else '未切换'
                self._notice(f"slot{target.slot} {target.name or '未知玩家'}：cut {cut} / "
                             f"settle {settle} / total {timing['total_ms']}ms / "
                             f"{'loaded' if shot.loaded else 'loaded=False'} / flags={','.join(shot.flags) or 'none'}")
                if not shot.loaded or shot.flags:
                    self._request_retry(target)
                if 'viewing_mismatch' in shot.flags:
                    self._notice(f'侦察玩家栏第{target.slot + 1}位身份不符，丢弃（viewing={result.get("viewing")}）')
                    continue
                previous_traits, previous_slot = trait_signature(shot.traits), target.slot
                # 原图写盘也在后台，扫描线程只复制入队。
                import cv2
                try:
                    directory = var_dir() / 'scout' / generation
                    directory.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(directory / f'{captured_at:.6f}-slot{target.slot}.png'), frame.image):
                        raise OSError('原图写入失败')
                except OSError:
                    log.exception('侦察原图保存失败，仍保留识别画像')
                if (shot.board or shot.traits or shot.level is not None) and self.book.add(shot):
                    recognized_slots.add(target.slot)
                    self._update(recognized=len(recognized_slots))
            except Exception:
                log.exception('后台侦察识别失败')
                self._notice(f'玩家栏第{target.slot + 1}位后台识别失败，跳过')
            finally:
                processed_slots.add(target.slot)
                self._update(processed=len(processed_slots))
                jobs.task_done()
        self._update(processing=False, background_s=round(time.monotonic() - started, 3))
        s = self.status
        self._notice(f"侦察 {stage}：截到 {s['captured']}/{s['total']}，打断 {s['interruption_s']:.2f} s，后台识别 {s['background_s']:.2f} s，补截 {s['retries']}")
