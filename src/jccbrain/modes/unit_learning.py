"""Assistant-owned, synchronous, cancellable unit inspection transactions.

There is no capture thread and no input sender here: captures belong to PyEye,
and the existing executor/guard owns every action. Off is a zero-work path.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from jccbrain.actuator.actions import ClickPoint, RestoreMouse

log = logging.getLogger(__name__)


def window_key(window):
    names = ('window_id', 'pid', 'x', 'y', 'width', 'height')
    return tuple(getattr(window, k, None) for k in names) if window else None


def supported_window(window):
    if window is None or not getattr(window, 'window_id', 0) or not getattr(window, 'pid', 0):
        return False
    w, h = window.width, window.height
    return (1576 <= w <= 1774 and 1013 <= h <= 1136 and 1.54 <= w/h <= 1.58)


def crop_bounds(unit, image):
    # Mirror the EXISTING feature crop, not a new feature/identity threshold.
    h, w = image.shape[:2]
    b = unit.bar
    return (max(0, math.floor(b.min_x-.05*b.width)), b.max_y+6,
            min(w, math.ceil(b.min_x+1.1*b.width)), min(h, math.ceil(b.max_y+6+1.3*b.width)))


def body_point(unit, image):
    """Blood-bar-relative body centre, checked against actual isolated crop.

    L1's three real targets: width 68..69px, body y=.389551..595537,
    centre lies inside the existing 79..80 x 89..90px crop (1774x1136).
    eval_unit_learning_auto audits ALL detected units, without names/anchors;
    exclusion is geometric, not a learned positive/negative identity threshold.
    """
    if unit.bar is None or unit.crop is None:
        return None
    h, w = image.shape[:2]
    b = unit.bar
    point = b.cx, (b.max_y+.005*h+.65*b.width)/h
    x, y = point
    x0, y0, x1, y1 = crop_bounds(unit, image)
    allowed = (.20 < x < .76 and .18 < y < .615 if unit.zone == 'board'
               else .20 < x < .80 and .615 < y < .77 if unit.zone == 'bench' else False)
    return point if allowed and x0 < x*w < x1 and y0 < y*h < y1 else None


@dataclass(frozen=True)
class Target:
    point: tuple
    x: float
    y: float
    zone: str
    slot: int | None
    distance: float
    known: str | None
    frame_hash: str

    @property
    def key(self):
        return self.zone, self.slot, self.x, self.y


def same_target(a, b):
    # Preserve tracking's strict spatial association without quantization edges.
    return a[:2] == b[:2] and abs(a[2]-b[2]) < .012 and abs(a[3]-b[3]) < .02


def candidates(reader, before, current):
    """Two actual receipt images, pure detection/features and read-only matching."""
    from jccbrain.vision.eye.unit_learning import isolated_crop
    from jccbrain.vision.eye.units import feature_print, fp_distance, TRACK_THRESHOLD, MATCH_THRESHOLD, MATCH_MARGIN
    _, old = reader.units.detect(before.image)
    _, new = reader.units.detect(current.image)
    result = []
    for unit in new:
        point = body_point(unit, current.image)
        if point is None or not isolated_crop(unit, new, current.image):
            continue
        peers = [u for u in old if u.zone == unit.zone and u.slot == unit.slot
                 and abs(u.x-unit.x) < .012 and abs(u.y-unit.y) < .02]
        if len(peers) != 1 or not isolated_crop(peers[0], old, before.image):
            continue
        a, b = feature_print(peers[0].crop), feature_print(unit.crop)
        if a is None or b is None:
            continue
        distance = fp_distance(a, b)
        if distance >= TRACK_THRESHOLD:
            continue
        match = reader.units.gallery.match(b)
        if match and match[1] < MATCH_THRESHOLD-MATCH_MARGIN:
            continue  # a stable, already known identity needs no repeated click
        result.append(Target(point, unit.x, unit.y, unit.zone, unit.slot, distance,
                             match[0] if match else None, current.pixel_hash))
    return sorted(result, key=lambda t: (t.known is not None, t.distance, t.zone, t.x, t.y))


def empty_point(reader, frame):
    """Choose from currently empty own board; no unconditional close coordinate."""
    meta, units = reader.units.detect(frame.image)
    h, w = frame.image.shape[:2]
    occupied = [crop_bounds(u, frame.image) for u in units if u.bar is not None]
    if legend := meta.get('legend'):
        x, y = legend['x']*w, legend['y']*h
        occupied.append((x-.06*w,y-.02*h,x+.06*w,y+.14*h))
    # Also exclude the little legend and all bar pixels, with a scaled margin.
    for u in units:
        if u.bar:
            b = u.bar
            occupied.append((b.min_x-.03*w, b.min_y-.02*h, b.max_x+.03*w, b.max_y+.02*h))
    for y in (.45, .40, .50, .35, .55):
        for x in (.60, .55, .65, .50, .45, .40, .35):
            if all(not (a-.02*w < x*w < c+.02*w and b-.02*h < y*h < d+.02*h)
                   for a,b,c,d in occupied):
                return (x,y)
    return None


class UnitLearningRunner:
    def __init__(self, ctx, sink, stopped=lambda: False, *, scout=None, monitor=None,
                 clock=time.monotonic, wait=None, trace=None):
        self.ctx, self.sink, self.stopped, self.scout = ctx, sink, stopped, scout
        self.clock, self.monitor, self.trace = clock, monitor, trace
        self._cancel = threading.Event()
        self.wait = wait or self._cancel.wait
        self.provider = None
        self.active = False
        self._configured = None
        self._last_attempt = float('-inf')
        self._rounds = OrderedDict()
        self._last_stage = None
        self._blocked_panel = False
        self._counts = dict(attempted=0, verified=0, cancelled=0)
        self._status = dict(mode='off', reason='disabled', **self._counts)
        self._baseline = None
        self._current = None
        self._initial = None
        self._expected_mouse = None
        self._last_notice = None
        self._timing = None

    def _segment(self, name):
        if self._timing is not None:
            now = self.clock()
            key = self._timing_segment
            self._timing['segments_ms'][key] += (now-self._segment_started)*1000
            self._timing_segment, self._segment_started = name, now

    def _process_sample(self, sample, transaction=None):
        start = self.clock()
        try:
            return self.provider.process_unit_learning(transaction, sample=sample, check=self._check)
        finally:
            self._timing['panel_ms'] += (self.clock()-start)*1000

    def _wait_for_sample(self):
        # The receipt is timestamped at actual capture, before scene/panel OCR.
        # Only make up the remaining minimum gap, never stack another .12s on it.
        self._check()
        remaining = .10 - (self.clock()-self._current.frame.monotonic)
        if remaining > 0:
            start = self.clock()
            self.wait(remaining)
            self._timing['wait_ms'] += (self.clock()-start)*1000
        self._check()

    @property
    def status(self):
        return dict(self._status)

    def _emit(self, kind, **data):
        if self.trace:
            self.trace(kind, data)

    def _update(self, reason, **extra):
        mode = getattr(self.ctx.settings.vision, 'unit_learning_mode', 'off')
        self._status = dict(mode=mode, reason=reason, active=self.active, **self._counts, **extra)
        if self.provider:
            self.provider.unit_learning_status(self._status)
        if reason != self._last_notice:
            self.sink.on_log('棋子身份学习：'+reason)
            self._last_notice = reason
        self._emit('status', **self._status)

    def stop(self):
        self._cancel.set()
        if self.provider:
            self.provider.cancel_unit_learning()  # nonblocking, reaches the promotion hook
        if self.monitor:
            stopped=self.monitor.stop()
            if not stopped:self._update('input_monitor_stop_failed')
            return stopped
        return True

    def close(self):
        stopped=self.stop()
        if self.provider:
            try:
                self.provider.end_unit_learning()
                self._update('stopped' if stopped else 'input_monitor_stop_failed')
            except Exception as exc:
                log.exception('停止棋子身份学习失败')
                self._update('cleanup_error', error=f'{type(exc).__name__}: {exc}')
        self._configured = None

    def reset_confirmations(self):
        """Main-loop scene boundary; preserve attempt budgets and panel ownership."""
        if self.active:
            self.stop()  # an unexpected concurrent transition must cancel first
        if self.provider:
            try:
                self.provider.finish_unit_learning_transaction()
            except Exception as exc:
                log.exception('清理场景变更前的棋子确认失败')
                self._update('cleanup_error', error=f'{type(exc).__name__}: {exc}')

    def _configure(self):
        cfg = self.ctx.settings.vision
        cfg.validate_unit_learning()
        mode = cfg.unit_learning_mode
        if mode == 'off':
            if self.provider and self._configured:
                self.provider.end_unit_learning()
                self._configured = None
                self._update('disabled')
            return False
        from jccbrain.vision.pyeye import PyEyeProvider
        if not isinstance(self.ctx.vision, PyEyeProvider) or cfg.backend != 'pyeye':
            self._update('live_pyeye_required')
            return False
        self.provider = self.ctx.vision
        if self._configured != mode:
            from jccbrain.core.paths import var_dir
            self.provider.configure_unit_learning(var_dir()/'unit-learning', mode='record' if mode == 'record' else 'panel',
                                                  automatic_record=False)
            self._configured = mode
        return True

    def _abort(self, reason):
        from jccbrain.vision.eye.unit_learning import LearningCancelled
        raise LearningCancelled(reason)

    def _controls(self, mode='auto'):
        if self._cancel.is_set() or self.stopped():
            self._abort('stopped')
        if self.ctx.guard.stopped:
            self._abort('guard_stopped')
        if self.ctx.settings.vision.unit_learning_mode != mode:
            self._abort('mode_changed')
        if self.scout and (self.scout.busy or self.scout.paused):
            self._abort('scout_priority')

    def _scene(self, sample, *, entrance=False):
        d, frame = sample.state, sample.frame
        if (d.get('phase') != 'planning' or frame.phase != 'planning' or d.get('viewing') is not None
                or frame.viewing is not None or d.get('units_stale') or not d.get('stage') or frame.stage!=d.get('stage')
                or not re.fullmatch(r'[1-9]\d?-[1-9]\d?', str(d.get('stage')))):
            self._abort('fresh_own_planning_required')
        timer = d.get('timer')
        if type(timer) not in (int, float) or not math.isfinite(timer):
            self._abort('timer_unreadable')
        if timer < (self.ctx.settings.vision.unit_learning_min_timer if entrance else 5):
            self._abort('timer_too_short')
        age = self.clock()-frame.monotonic
        if age < 0 or age > 2.:
            self._abort('capture_not_fresh')
        if not supported_window(frame.window):
            self._abort('unsupported_window')
        if frame.size != (int(frame.window.width), int(frame.window.height)):
            self._abort('capture_window_size_mismatch')
        if self._initial:
            if (d.get('stage') != self._initial.state.get('stage')
                    or window_key(frame.window) != window_key(self._initial.frame.window)):
                self._abort('scene_or_window_changed')

    def _input(self, *, entrance=False):
        if self.monitor is None:
            from jccbrain.vision.eye.input_monitor import HIDMonitor
            self.monitor = HIDMonitor(self.ctx.locator)
        if entrance and not self.monitor.start():
            self._emit('input_monitor_failed',snapshot=self.monitor.snapshot().to_json())
            self._abort('input_monitor_unavailable')
        s = self.monitor.snapshot()
        self._emit('input', snapshot=s.to_json())
        if (not s.available or not s.counters or s.mouse is None or not math.isfinite(s.idle_s)
                or not all(math.isfinite(v) for v in s.mouse)):
            self._abort('input_monitor_unavailable')
        if s.held:
            self._abort('held_input')
        if self.ctx.settings.actuator.failsafe_corner and s.mouse[0]<=5 and s.mouse[1]<=5:
            self._abort('failsafe_corner')
        if not supported_window(s.window) or s.foreground_pid != s.window.pid:
            self._abort('game_not_foreground')
        if entrance:
            if s.idle_s < self.ctx.settings.vision.unit_learning_idle_s:
                self._abort('input_not_idle')
            self._baseline = s
            self._expected_mouse = s.mouse
        else:
            from jccbrain.vision.eye.input_monitor import counter_residual, InputMonitorError
            if (s.generation!=self._baseline.generation or not s.tap_healthy
                    or s.foreign_sequence!=self._baseline.foreign_sequence):
                self._abort('hardware_input_changed')
            try:
                if any(counter_residual(self._baseline,s).values()):
                    self._abort('hardware_input_changed')
            except InputMonitorError as exc:
                self._abort(str(exc))
            if window_key(s.window) != window_key(self._baseline.window):
                self._abort('window_changed')
            # Existing executor jitter is bounded; do not reset hardware counters
            # after posting our event, even if a user acts at exactly that time.
            jitter = getattr(self.ctx.settings.actuator, 'jitter_px', 0)
            tolerance = max(.003*s.window.width, 2*jitter)
            if math.dist(s.mouse, self._expected_mouse) > tolerance:
                self._abort('mouse_moved')
        return s

    def _check(self):
        self._controls()
        self._input()
        elapsed = self.clock()-self._started
        if elapsed > 6:
            self._abort('transaction_timeout')
        if self._current:
            self._scene(self._current)
            d = self._current.state
            remaining = min(self._initial.state['timer']-elapsed,
                            d['timer']-(self.clock()-self._current.frame.monotonic))
            if remaining < 5:
                self._abort('timer_too_short')

    def _sample(self):
        # _check is invoked both sides of capture and after isolated scene OCR.
        start = self.clock()
        try:
            sample = self.provider.sample_unit_learning(check=self._check)
        finally:
            self._timing['sample_ms'] += (self.clock()-start)*1000
            self._timing['samples'] += 1
        if sample is None:
            self._abort('temporary_capture_failed')
        if (self._current and (sample.frame.sequence<=self._current.frame.sequence
                              or sample.frame.monotonic<=self._current.frame.monotonic)):
            self._abort('capture_not_fresh')
        self._scene(sample)
        self._current = sample
        for key, value in sample.timings.items():
            self._timing[key] = self._timing.get(key, 0.) + value
        self._check()
        self._emit('sample', sample=sample)
        return sample

    def _action(self, action, *, validate=None, before=None, reuse_closed=False):
        from jccbrain.vision.eye.input_monitor import InputMonitorError, counter_residual
        expected=None
        completed=False
        waiting = self.clock()
        def before_do():
            nonlocal expected
            self._timing['guard_ms'] += (self.clock()-waiting)*1000
            # Only mouse restoration may reuse the just-validated CLOSED frame.
            # This executes AFTER guard waiting, with all current input/scene gates.
            self._check()
            age = self.clock()-self._current.frame.monotonic
            reused = reuse_closed and isinstance(action, RestoreMouse) and 0 <= age <= .25
            if not reused:
                self._sample()
            if validate and not validate(self._current.frame):
                self._abort('action_target_changed')
            self._check()
            if before:
                before(self._current.frame)
                self._check()  # transaction/logging hooks may block or admit user input
            if reuse_closed:
                if reused and self.clock()-self._current.frame.monotonic > .25:
                    self._sample()
                    self._check()
                    reused = False
                from jccbrain.vision.eye.unit_panel import presence
                if presence(self._current.frame.image) >= .9:
                    self._abort('own_panel_still_open')
                self._timing['reused_closed'] += int(reused)
            w=self._baseline.window
            point=(w.x+action.point[0]*w.width,w.y+action.point[1]*w.height) if isinstance(action,ClickPoint) else action.screen_point
            jitter=max(0,float(getattr(self.ctx.settings.actuator,'jitter_px',0))) if isinstance(action,ClickPoint) else 0.
            # Original executor jitter, plus <=.0005W/H for native float-point
            # quantization (~.89/.57 at 1774x1136); PID/order/time are also strict.
            expected=self.monitor.expect_action('click' if isinstance(action,ClickPoint) else 'restore',point,
                                                 (jitter+.0005*w.width,jitter+.0005*w.height))
            # Registration is the FINAL operation before _do. Don't widen that
            # protected boundary with trace/disk I/O; emit the receipt afterwards.
        self._check()
        try:
            self._timing['action_attempts'] += 1
            self.ctx.executor.execute(action, before_do=before_do)
            completed=True
            self._timing['actions'] += 1
            self._emit('input_expectation',action_id=expected,action_kind='click' if isinstance(action,ClickPoint) else 'restore',
                       baseline=self._baseline.to_json())
            # Log the executed action even when reconciliation subsequently
            # cancels. A post-click input failure must never look like zero input.
            self._emit('action', action=action.describe(), point=getattr(action,'point',None),
                       screen_point=getattr(action,'screen_point',None),real=True,action_id=expected)
            self.sink.on_action(action, True)
            matched=self.monitor.finish_action(expected,self._baseline,check=self._controls)
            self._emit('input_reconciled',action_id=expected,snapshot=matched.to_json(),
                       residual=counter_residual(self._baseline,matched))
        except InputMonitorError as exc:
            self._emit('input_reconciliation_failed',action_id=expected,error=str(exc),
                       snapshot=self.monitor.snapshot().to_json())
            self._abort(str(exc))
        except Exception as exc:
            self._emit('action_failed',action_id=expected,error=f'{type(exc).__name__}: {exc}',
                       snapshot=self.monitor.snapshot().to_json())
            raise
        finally:
            # The executor may fail after a partial native send. Do not claim
            # zero input: distinguish completed actions from uncertain attempts.
            if expected is not None and not completed:
                self._timing['uncertain_actions'] += 1
            self.monitor.clear_expected()  # no exemption may leak after a partial/failed execute
        if isinstance(action, ClickPoint):
            w = self._baseline.window
            self._expected_mouse = (w.x+action.point[0]*w.width, w.y+action.point[1]*w.height)
        else:
            self._expected_mouse = action.screen_point
        self._check()  # exact raw HID minus observed/confirmed own events, fixed baseline

    def _round(self, stage):
        key = tuple(map(int, stage.split('-')))
        if self._last_stage is not None and key <= (1,2) and key < self._last_stage:
            self._rounds.clear()
        self._last_stage = key
        value = self._rounds.setdefault(stage, set())
        self._rounds.move_to_end(stage)
        while len(self._rounds) > 64:
            self._rounds.popitem(last=False)
        return value

    def maybe_run(self, raw):
        """After normal processing and scout.maybe_run; True => normal fresh read next."""
        if self._cancel.is_set() or self.stopped():
            return False
        if getattr(self.ctx.settings.vision,'unit_learning_mode','off') == 'off' and self._configured is None:
            return False
        from jccbrain.vision.eye.unit_learning import LearningCancelled, ClickTransaction
        from jccbrain.vision.eye.unit_panel import presence as panel_presence
        try:
            if not self._configure():
                return False
            if self._configured == 'record':
                self._controls('record')
                sample=self.provider.unit_learning_sample()
                if sample is None or raw.raw!=sample.state:
                    self._abort('no_current_capture')
                result=self.provider.process_unit_learning(sample=sample,check=lambda:self._controls('record'))
                self._update(result.get('reason','cancelled') if result.get('status')=='cancelled' else 'record_only',
                             result=result)
                return False
            self._controls()
            initial = self.provider.unit_learning_sample()
            # A stale GameState from a stabilizer/caller cannot substitute for the
            # provider's receipt of this exact ordinary read.
            if initial is None or raw.raw != initial.state:
                self._abort('no_current_capture')
            self._initial = None
            self._scene(initial, entrance=True)
            if self.ctx.executor.dry_run:
                self._abort('dry_run_no_actions_or_promotion')
            self._input(entrance=True)
            if window_key(initial.frame.window) != window_key(self._baseline.window):
                self._abort('window_changed')
            present = panel_presence(initial.frame.image) >= .9
            if self._blocked_panel:
                if present:
                    self._abort('own_panel_still_open')
                self._blocked_panel = False
            attempted = self._round(initial.frame.stage)
            if not present and (len(attempted) >= 2 or self.clock()-self._last_attempt < self.ctx.settings.vision.unit_learning_interval_s):
                self._abort('round_budget_or_interval')
            self._started = self.clock()
            self._timing = dict(samples=0, actions=0, action_attempts=0, uncertain_actions=0, reused_closed=0,
                                sample_ms=0., capture_ms=0., scene_ms=0., receipt_ms=0., panel_ms=0.,
                                guard_ms=0., wait_ms=0., segments_ms=dict(candidate=0.,before_click=0.,
                                    panel_confirmation=0.,close_restore=0.,finish=0.))
            self._timing_segment, self._segment_started = 'candidate', self._started
            self._initial = self._current = initial
            self.active = True
            self._update('passive_panel' if present else 'selecting')
            self._emit('sample', sample=initial)
            if present:
                # User opened this panel: no click, close or mouse restoration.
                self._segment('panel_confirmation')
                result = self._process_sample(initial)
                if result.get('status') == 'pending':
                    self._wait_for_sample()
                    result = self._process_sample(self._sample())
                self._result(result)
                return True
            second = self._sample()
            targets = candidates(self.provider._reader, initial.frame, second.frame)
            self._emit('candidates', targets=[vars(t) for t in targets])
            target = next((t for t in targets if not any(same_target(t.key,k) for k in attempted)), None)
            if target is None:
                self._abort('no_stable_unknown_target')
            attempted.add(target.key)
            self._last_attempt = self.clock()
            self._counts['attempted'] += 1
            transaction = None
            opened = False
            def still_target(frame):
                return any(same_target(t.key, target.key) and math.dist(t.point, target.point) < .012
                           for t in candidates(self.provider._reader, second.frame, frame))
            def clicked(frame):
                nonlocal transaction, opened
                transaction = ClickTransaction(frame, target.point, self.clock(), frame.window_id)
                opened = True
                self._blocked_panel = True
                self._emit('transaction', before=frame.metadata(), target=target.point, clicked_at=transaction.clicked_at)
            # Even if execute raises after sending the click, leave blocked_panel
            # set. No unguarded cleanup after failure/user takeover.
            self._segment('before_click')
            self._action(ClickPoint(target.point, '查看未知棋子身份'), validate=still_target, before=clicked)
            self._segment('panel_confirmation')
            result = {'status':'rejected','reason':'no_two_panel_captures'}
            for _ in range(3):
                self._wait_for_sample()
                result = self._process_sample(self._sample(), transaction)
                if result.get('status') in ('verified','cancelled','error'):
                    break
            self._result(result)
            if result.get('status') in ('cancelled','error'):
                self._abort(result.get('reason','learning_failed'))
            if opened:
                self._segment('close_restore')
                self._close_panel()
            return True
        except LearningCancelled as exc:
            if self.active:
                self._segment('finish')
                self._counts['cancelled'] += 1
            self._update(str(exc))
            return self.active
        except Exception as exc:
            # Optional learning never brings down ordinary Assistant observation.
            log.exception('棋子身份学习失败')
            self._update('learning_error', error=f'{type(exc).__name__}: {exc}')
            return self.active
        finally:
            if self.active:
                try:
                    self.provider.finish_unit_learning_transaction()
                except Exception as exc:
                    log.exception('清理棋子身份学习事务失败')
                    self._update('cleanup_error', error=f'{type(exc).__name__}: {exc}')
                finally:
                    self.active = False
                    self._status['active'] = False
                    self.provider.unit_learning_status(self._status)
            self._initial = self._current = self._baseline = None
            if self.monitor:
                if not self.monitor.stop():  # release on every success/skip/cancel/exception
                    self._update('input_monitor_stop_failed')
            if self._timing is not None:
                self._segment('finish')
                self._timing.update(total_ms=(self.clock()-self._started)*1000, reason=self._status['reason'])
                self._status['timing'] = self._timing
                self.provider.unit_learning_status(self._status)
                self._emit('transaction_timing', **self._timing)
                log.info('棋子学习事务耗时 %s', self._timing)
                self._timing = None

    def _result(self, result):
        if result.get('status') == 'verified':
            self._counts['verified'] += 1
        self._update(result.get('reason',result.get('status','unknown')), result=result)
        self._check()

    def _close_panel(self):
        from jccbrain.vision.eye.unit_panel import presence as panel_presence
        point = empty_point(self.provider._reader, self._current.frame)
        if point is None:
            self._abort('no_safe_empty_close_point')
        def empty(frame):
            return empty_point(self.provider._reader, frame) == point
        self._action(ClickPoint(point,'关闭本次棋子面板'), validate=empty)
        self._wait_for_sample()
        after = self._sample()
        self._blocked_panel = panel_presence(after.frame.image) >= .9
        if self._blocked_panel:
            self._update('own_panel_still_open')
            return
        self._action(RestoreMouse(self._baseline.mouse), reuse_closed=True)
