"""Read-only purchase dependency observation; no input injection or OCR work.

This counter observer owns its own HIDMonitor and never borrows scout's ledger.
MouseMoved is deliberately ignored. Observation races don't imply idle: usable
counter samples still detect relevant deltas, otherwise degrade to raw-first.
"""
from __future__ import annotations

from collections import deque
import math
import threading
import time

from jccbrain.vision.eye.input_monitor import EVENT_NAMES, HIDMonitor

INTERVAL = .05
RELEVANT = frozenset(EVENT_NAMES) - {'MouseMoved'}


class BuyActivityObserver:
    def __init__(self, locator, callback, *, monitor=None, clock=time.monotonic):
        self.monitor = monitor if monitor is not None else HIDMonitor(locator)
        self.callback, self.clock = callback, clock
        self._stop = threading.Event()
        self._thread = None
        self._last = None
        self._condition = None
        self._blocked = True
        self._samples = deque(maxlen=2048)
        self._durations = deque(maxlen=2048)
        self._cpu = 0.
        self._polls = 0
        self._error = None
        self._available = False
        self._generation = 0

    @property
    def status(self):
        return {'available': self._available, 'reason': self._error, 'blocked': self._blocked,
                'polls': self._polls, 'target_interval_ms': INTERVAL * 1000,
                'max_query_ms': max(tuple(self._durations), default=0.) * 1000,
                'max_interval_ms': max(tuple(self._samples), default=0.) * 1000,
                'thread_cpu_s': self._cpu, 'generation': self._generation,
                'running': bool(self._thread and self._thread.is_alive())}

    def start(self):
        if self._thread is not None:
            raise RuntimeError('buy input observer cannot be restarted')
        self._thread = threading.Thread(target=self._run, name='buy-input-observer', daemon=True)
        self._thread.start()

    def _changed(self, condition, blocked, at, *, activity=False):
        changed = condition != self._condition or blocked != self._blocked
        self._condition, self._blocked = condition, blocked
        if changed or activity:
            self.callback(condition, blocked, at)

    def poll(self):
        """One controlled counter sample, also used by explicit injected tests."""
        s = self.monitor.snapshot()
        counters = dict(s.counters)
        usable = (s.tap_healthy and (s.available or s.error == 'input_observation_race')
                  and len(s.counters) == len(EVENT_NAMES) and set(counters) == set(EVENT_NAMES)
                  and all(type(v) is int and v >= 0 for v in counters.values())
                  and type(s.generation) is int and s.generation > 0
                  and math.isfinite(s.at) and 0 <= self.clock() - s.at <= .25)
        if not usable:
            self._available = False
            self._error = s.error or 'input_monitor_unavailable'
            self._last = None
            # Unavailable is NOT idle. It fences the old read once, then permits
            # reliable new raw frames as the documented raw-first fallback.
            self._changed('fallback:' + self._error, False, self.clock())
            return
        previous = self._last
        if previous and (s.at < previous.at or any(counters[k] < dict(previous.counters)[k] for k in counters)):
            self._available = False
            self._error = 'input_counter_or_clock_rewound'
            self._last = None
            self._changed('fallback:' + self._error, False, self.clock())
            return
        self._available, self._error = True, None
        self._generation = s.generation
        win = s.window
        foreground = bool(win and s.foreground_pid == win.pid)
        window = (win.pid, win.window_id, win.x, win.y, win.width, win.height) if win else None
        before = previous.window if previous else None
        old_window = (before.pid, before.window_id, before.x, before.y, before.width, before.height) if before else None
        relevant = bool(previous and any(counters[k] != dict(previous.counters)[k] for k in RELEVANT))
        # A raced raw count query can still expose a just-delivered relevant tap
        # observation before its next HID counter sample. Never inspect keycodes.
        if previous and s.error == 'input_observation_race':
            relevant |= any(e.get('at', -1) > previous.at and e.get('kind') in RELEVANT for e in s.events)
        boundary = bool(previous and (s.generation != previous.generation or window != old_window))
        condition = 'foreground' if foreground else 'focus_lost'
        self._changed(condition, not foreground or s.held, s.at,
                      activity=boundary or foreground and relevant)
        self._last = s

    def _run(self):
        cpu = time.thread_time()
        previous = None
        try:
            try:
                started = self.monitor.start()
            except Exception:
                started = False
            if not started:
                self._available, self._error = False, 'input_monitor_start_failed'
                self._changed('fallback:' + self._error, False, self.clock())
                return
            while not self._stop.is_set():
                start = self.clock()
                if previous is not None:
                    self._samples.append(start - previous)
                previous = start
                try:
                    self.poll()
                except Exception as exc:
                    self._available = False
                    self._error = 'input_monitor_error:' + type(exc).__name__
                    self._last = None
                    self._changed('fallback:' + self._error, False, self.clock())
                self._polls += 1
                elapsed = self.clock() - start
                self._durations.append(elapsed)
                self._cpu = time.thread_time() - cpu
                self._stop.wait(max(0., INTERVAL - elapsed))
        finally:
            try:
                if not self.monitor.stop():
                    self._error = 'input_monitor_stop_failed'
            except Exception:
                self._error = 'input_monitor_stop_failed'
            self._cpu = time.thread_time() - cpu

    def request_stop(self):
        self._stop.set()

    def close(self):
        self.request_stop()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(.8)
        return self._thread is None or not self._thread.is_alive()
