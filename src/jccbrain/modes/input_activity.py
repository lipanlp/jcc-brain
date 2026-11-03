"""侦察侧只读输入事务。固定基线，复用视觉提供的监听/逐事件归因契约。"""
from __future__ import annotations

import math
import time

from jccbrain.actuator.safety import AbortError
from jccbrain.vision.eye.input_monitor import EVENT_NAMES, HIDMonitor, counter_residual


class InputActivity:
    def __init__(self, locator, *, monitor=None, clock=time.monotonic):
        self.monitor = monitor if monitor is not None else HIDMonitor(locator)
        self.clock = clock
        self.baseline = None
        self.last_at = None
        self.running = False
        self.failed = None
        self.action = None

    def start(self, *, automatic):
        if self.failed == '输入监听未能停止':
            raise AbortError(self.failed)
        if not self.running:
            try:
                self.running = bool(self.monitor.start())
            except Exception as exc:
                self.close()
                raise AbortError('输入监测启动失败') from exc
        if not self.running:
            self.close()
            raise AbortError('输入监测不可用')
        self.failed = None
        self.baseline = self.last_at = None
        s = self._snapshot()
        if s.pending is not None:
            raise AbortError('旧输入期望尚未结束')
        if s.held:
            raise AbortError('用户仍按住按键或鼠标')
        if automatic and s.idle_s < 1.5:
            raise AbortError('等待用户连续空闲1.5秒')
        self.baseline = s
        return s

    def _snapshot(self):
        s = self.monitor.snapshot()
        if (not s.available or not s.tap_healthy or s.error or s.generation <= 0
                or not math.isfinite(s.at) or s.at > self.clock() + .01
                or self.clock() - s.at > .25 or (self.last_at is not None and s.at < self.last_at)
                or not math.isfinite(s.idle_s) or s.idle_s < 0):
            raise AbortError('输入监测异常：' + (s.error or '健康/时钟/代次无效'))
        for counters in (s.counters, s.own_counters):
            if (len(counters) != len(EVENT_NAMES) or set(dict(counters)) != set(EVENT_NAMES)
                    or any(type(v) is not int or v < 0 for _, v in counters)):
                raise AbortError('输入监测计数异常')
        win = s.window
        if not win or s.foreground_pid != win.pid:
            raise AbortError('游戏不是前台或窗口不可用')
        self.last_at = s.at
        return s

    def check(self):
        if self.failed:
            raise AbortError(self.failed)
        try:
            if self.baseline is None:
                raise AbortError('输入事务尚未授权')
            b = self.baseline
            for attempt in range(31):
                s = self._snapshot()
                if s.generation != b.generation:
                    raise AbortError('输入监听代次已变化')
                if s.window != b.window:
                    raise AbortError('游戏窗口已变化')
                if s.foreign_sequence != b.foreign_sequence:
                    raise AbortError('用户活动或未知输入，停止侦察')
                residual = counter_residual(b, s)
                if not any(residual.values()):
                    break
                # HID计数与listen-only tap回调可能异步；等待严格守恒后才允许
                # 下一次发送。固定原基线、不豁免任何计数，陌生事件立即中断。
                pending = s.pending
                if self.action is None or not pending or pending['id'] != self.action or attempt == 30:
                    raise AbortError('用户活动或未对账输入，停止侦察')
                getattr(self.monitor, 'wait', time.sleep)(.005)
            # 唯一允许的 held 是已归因本进程的 down 与其成对 up 之间。
            own_down = s.pending and s.pending['order'] == ('MouseMoved', 'LeftMouseDown', 'LeftMouseUp') and s.pending['consumed'] == 2
            if s.held and not own_down:
                raise AbortError('用户仍按住按键或鼠标')
            return s
        except Exception as exc:
            self.failed = str(exc)
            raise AbortError(self.failed) from exc

    def begin(self, kind, point, tolerance):
        self.check()
        self.action = self.monitor.expect_action(kind, point, tolerance)

    def finish(self, check):
        try:
            self.monitor.finish_action(self.action, self.baseline, check)
            self.check()
        except Exception as exc:
            self.failed = str(exc)
            raise AbortError(self.failed) from exc
        finally:
            self.action = None

    def close(self):
        self.baseline = None
        self.action = None
        try:
            stopped = self.monitor.stop()
        except Exception:
            stopped = False
        self.running = not stopped
        if not stopped:
            self.failed = '输入监听未能停止'
        return stopped
