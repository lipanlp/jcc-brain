"""Read-only input monitoring and bounded attribution of our executor events.

Never sends/intercepts/modifies input. Hardware counters and the transaction's
baseline stay fixed; only actual, ordered, source-PID/point/time matched tap
observations can account for our own finite action. No key codes/text are read.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace

EVENT_NAMES = ('LeftMouseDown', 'LeftMouseUp', 'RightMouseDown', 'RightMouseUp',
               'OtherMouseDown', 'OtherMouseUp', 'MouseMoved', 'LeftMouseDragged',
               'RightMouseDragged', 'OtherMouseDragged', 'KeyDown', 'KeyUp',
               'FlagsChanged', 'ScrollWheel')
ACTION_DEADLINE_S = .50  # parent native probe: move/down/up span ~40 ms; never an input-ignore window
RECONCILE_S = .15       # tap callback/HID delivery may be asynchronous, but not unbounded
HEARTBEAT_LIMIT_S = .25


class InputMonitorError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputSnapshot:
    at: float
    available: bool = False
    idle_s: float = 0.
    counters: tuple = ()
    held: bool = False
    mouse: tuple[float, float] | None = None
    foreground_pid: int | None = None
    window: object = None
    error: str | None = None
    generation: int = 0
    own_counters: tuple = ()
    foreign_sequence: int = 0
    tap_healthy: bool = False
    pending: dict | None = None
    events: tuple = ()

    def to_json(self):
        return asdict(self)


def counter_residual(baseline, current):
    """Exact conservation, all categories, relative to the ORIGINAL baseline."""
    a,b = dict(baseline.counters),dict(current.counters)
    x,y = dict(baseline.own_counters),dict(current.own_counters)
    if set(a)!=set(b) or set(x)!=set(a) or set(y)!=set(a):
        raise InputMonitorError('input_counter_schema_changed')
    if any(b[k]<a[k] or y[k]<x[k] for k in a):
        raise InputMonitorError('input_counters_rewound')
    return {k:b[k]-a[k]-(y[k]-x[k]) for k in a}


class EventLedger:
    """Pure callback adapter, independently testable without a native tap."""
    def __init__(self, *, pid=None, clock=time.monotonic):
        self.pid = os.getpid() if pid is None else pid
        self.clock = clock
        self.lock = threading.RLock()
        self.own = Counter({name:0 for name in EVENT_NAMES})
        self.foreign = self.sequence = self.action_id = 0
        self.pending = None
        self.error = None
        self.events = deque(maxlen=64)

    def fail(self, reason):
        with self.lock:
            self.error = reason
            self.pending = None
            self.sequence += 1

    def clear_expected(self):
        with self.lock:
            self.pending = None
            self.sequence += 1

    def expect(self, kind, point, tolerance, source_time_ns):
        with self.lock:
            if self.error or self.pending is not None:
                raise InputMonitorError(self.error or 'unfinished_input_expectation')
            if (kind not in ('click','restore') or len(point)!=2 or len(tolerance)!=2
                    or any(not math.isfinite(v) for v in (*point,*tolerance)) or min(tolerance)<0
                    or not isinstance(source_time_ns,int) or source_time_ns<=0):
                raise InputMonitorError('invalid_input_expectation')
            self.action_id += 1
            self.pending = dict(id=self.action_id, pid=self.pid, point=tuple(point), tolerance=tuple(tolerance),
                order=('MouseMoved','LeftMouseDown','LeftMouseUp') if kind=='click' else ('MouseMoved',),
                consumed=0, started=self.clock(), deadline=self.clock()+ACTION_DEADLINE_S,
                source_time_ns=source_time_ns, previous_time_ns=source_time_ns)
            self.sequence += 1
            return self.action_id

    def observe(self, kind, pid, point=None, *, source_time_ns=0):
        # Called only by listen-only tap; don't retain an event/key object.
        with self.lock:
            now=self.clock();p=self.pending
            match=bool(p and not self.error and p['consumed']<len(p['order']) and pid==p['pid']
                and kind==p['order'][p['consumed']] and point is not None and len(point)==2
                and all(math.isfinite(v) for v in point)
                and all(abs(point[i]-p['point'][i])<=p['tolerance'][i] for i in (0,1))
                and p['started']<=now<=p['deadline']
                and p['source_time_ns']<=source_time_ns<=p['source_time_ns']+int(ACTION_DEADLINE_S*1e9)
                and source_time_ns>=p['previous_time_ns'])
            if match:
                self.own[kind]+=1;p['consumed']+=1;p['previous_time_ns']=source_time_ns
            else:
                self.foreign+=1;self.pending=None
            self.sequence+=1
            self.events.append(dict(at=now,kind=kind,pid=pid,point=point,source_time_ns=source_time_ns,
                                    own=match,action=p['id'] if match else None))

    def state(self):
        with self.lock:
            return dict(own_counters=tuple((n,self.own[n]) for n in EVENT_NAMES),
                foreign_sequence=self.foreign, sequence=self.sequence, error=self.error,
                pending=dict(self.pending) if self.pending else None,events=tuple(dict(e) for e in self.events))


class SessionEventTap:
    """One bounded listen-only runloop. Constructing it does not touch macOS."""
    def __init__(self, ledger, *, clock=time.monotonic, quartz=None):
        self.ledger,self.clock,self._q=ledger,clock,quartz
        self._stop=threading.Event();self._ready=threading.Event()
        self.thread=None;self.tap=self.loop=None;self.heartbeat=0.
        self._resource_lock=threading.Lock();self._invalidated=False

    def start(self):
        self.thread=threading.Thread(target=self._run,name='unit-learning-input',daemon=True)
        self.thread.start()
        if not self._ready.wait(.5):
            self.ledger.fail('tap_start_timeout');self.stop()
        return self.healthy()

    def _run(self):
        source=None
        try:
            if self._q is None:
                import Quartz
                self._q=Quartz
            q=self._q
            if not q.CGPreflightListenEventAccess():
                raise InputMonitorError('input_monitor_permission_unavailable')
            for symbol in ('CGEventGetTimestamp','CGEventGetIntegerValueField',
                           'CGEventGetLocation','CGEventTapIsEnabled'):
                getattr(q,symbol)  # resolve lazy framework symbols before callback/main-thread use
            self._by_type={getattr(q,'kCGEvent'+n):n for n in EVENT_NAMES}
            self.tap=q.CGEventTapCreate(q.kCGSessionEventTap,q.kCGHeadInsertEventTap,
                q.kCGEventTapOptionListenOnly,sum(1<<t for t in self._by_type),self.callback,None)
            if self.tap is None:
                raise InputMonitorError('listen_only_tap_unavailable')
            self.loop=q.CFRunLoopGetCurrent()
            source=q.CFMachPortCreateRunLoopSource(None,self.tap,0)
            q.CFRunLoopAddSource(self.loop,source,q.kCFRunLoopDefaultMode)
            self.heartbeat=self.clock();self._ready.set()
            while not self._stop.is_set():
                result=q.CFRunLoopRunInMode(q.kCFRunLoopDefaultMode,.02,False)
                if self._stop.is_set():break
                now=self.clock()
                if not self._stop.is_set() and result in (q.kCFRunLoopRunFinished,q.kCFRunLoopRunStopped):
                    raise InputMonitorError('tap_runloop_exited')
                if now-self.heartbeat>HEARTBEAT_LIMIT_S:
                    raise InputMonitorError('tap_observation_gap')
                if not q.CGEventTapIsEnabled(self.tap):
                    raise InputMonitorError('tap_disabled')
                self.heartbeat=now
        except Exception as exc:
            self.ledger.fail(f'{type(exc).__name__}: {exc}')
        finally:
            self._ready.set()
            try:
                if source is not None:self._q.CFRunLoopRemoveSource(self.loop,source,self._q.kCFRunLoopDefaultMode)
            except Exception as exc:
                self.ledger.fail(f'tap_remove_failed: {type(exc).__name__}: {exc}')
            finally:
                try:
                    self._invalidate()
                except Exception as exc:
                    self.ledger.fail(f'tap_invalidate_failed: {type(exc).__name__}: {exc}')
                self.tap=self.loop=None
            if not self._stop.is_set() and self.ledger.error is None:
                self.ledger.fail('tap_thread_exited')

    def callback(self, _proxy, kind, event, _info):
        # Always return original event, including failure paths. No interception.
        try:
            if self._stop.is_set():return event
            q=self._q
            if kind in (q.kCGEventTapDisabledByTimeout,q.kCGEventTapDisabledByUserInput):
                self.ledger.fail('tap_disabled_notification');self._stop.set()
            else:
                name=self._by_type.get(kind)
                if name is None:
                    self.ledger.fail('unexpected_tap_type')
                else:
                    pid=int(q.CGEventGetIntegerValueField(event,q.kCGEventSourceUnixProcessID))
                    point=None
                    if name in EVENT_NAMES[:10]:
                        p=q.CGEventGetLocation(event);point=(float(p.x),float(p.y))
                    self.ledger.observe(name,pid,point,source_time_ns=int(q.CGEventGetTimestamp(event)))
        except Exception as exc:
            self.ledger.fail(f'tap_callback_error: {type(exc).__name__}: {exc}')
        return event

    def healthy(self):
        alive=self.thread is not None and self.thread.is_alive()
        if (not alive or self._stop.is_set() or self.tap is None or self.ledger.error
                or self.clock()-self.heartbeat>HEARTBEAT_LIMIT_S):
            return False
        try:
            if not self._q.CGEventTapIsEnabled(self.tap):
                self.ledger.fail('tap_disabled')
                return False
            return True
        except Exception as exc:
            self.ledger.fail(f'tap_health_query_failed: {type(exc).__name__}: {exc}')
            return False

    def event_time_ns(self):
        # On macOS monotonic_ns uses mach_absolute_time, matching native event
        # nanoseconds since boot (conductor event-clock-probe-2). An unsent
        # CGEventCreate(None) has timestamp 0 and cannot supply this clock.
        return time.monotonic_ns()

    def stop(self):
        self._stop.set()
        error=None
        try:
            self._invalidate()  # immediately cease observing, even if the runloop is delayed
        except Exception as exc:
            error=exc
        try:
            if self.loop is not None:self._q.CFRunLoopStop(self.loop)
        except Exception as exc:
            error=error or exc
        if self.thread is not None and self.thread is not threading.current_thread():self.thread.join(.3)
        stopped=self.thread is None or not self.thread.is_alive()
        if not stopped:self.ledger.fail('tap_stop_timeout')
        if error:self.ledger.fail(f'tap_stop_failed: {type(error).__name__}: {error}')
        return stopped and error is None

    def _invalidate(self):
        with self._resource_lock:
            if self.tap is not None and not self._invalidated:
                self._q.CFMachPortInvalidate(self.tap)
                self._invalidated=True


class HIDMonitor:
    def __init__(self, locator, *, tap_factory=SessionEventTap, clock=time.monotonic, wait=time.sleep, raw_query=None):
        self.locator,self.tap_factory,self.clock,self.wait=locator,tap_factory,clock,wait
        self.raw_query=raw_query or self._query
        self.ledger=None;self.tap=None;self.generation=0

    def start(self):
        if self.tap is not None:
            if self.tap.healthy():return True
            if not self.stop():return False
        self.generation+=1;self.ledger=EventLedger(clock=self.clock)
        try:
            self.tap=self.tap_factory(self.ledger)
            return self.tap.start()
        except Exception as exc:
            self.ledger.fail(f'tap_start_failed: {type(exc).__name__}: {exc}')
            return False

    def stop(self):
        if self.ledger:self.ledger.clear_expected()
        if self.tap is None:return True
        try:
            stopped=self.tap.stop()
        except Exception as exc:
            self.ledger.fail(f'tap_stop_failed: {type(exc).__name__}: {exc}')
            return False
        if stopped:self.tap=None
        return stopped

    def clear_expected(self):
        if self.ledger:self.ledger.clear_expected()

    def expect_action(self,kind,point,tolerance):
        if self.tap is None or not self.tap.healthy():raise InputMonitorError('tap_unhealthy')
        return self.ledger.expect(kind,point,tolerance,self.tap.event_time_ns())

    def finish_action(self,action_id,baseline,check=lambda:None):
        deadline=self.clock()+RECONCILE_S
        while True:
            check()
            s=self.snapshot()
            if s.generation!=baseline.generation:
                raise InputMonitorError('input_monitor_restarted')
            if not s.available and s.error!='input_observation_race':
                raise InputMonitorError(s.error or 'tap_unhealthy')
            if s.foreign_sequence!=baseline.foreign_sequence:
                raise InputMonitorError('hardware_input_changed')
            p=s.pending
            done=bool(p and p['id']==action_id and p['consumed']==len(p['order']))
            if s.available and done and not any(counter_residual(baseline,s).values()):
                self.clear_expected()
                return s
            if self.clock()>=deadline:
                raise InputMonitorError('input_counter_gap' if done else 'expected_event_missing')
            self.wait(.005)

    def snapshot(self):
        before=self.ledger.state() if self.ledger else None
        raw=self.raw_query()
        after=self.ledger.state() if self.ledger else None
        healthy=bool(self.tap and self.tap.healthy())
        error=raw.error or (after and after['error']) or (None if healthy else 'tap_unhealthy')
        if before and after and before['sequence']!=after['sequence']:
            error=error or 'input_observation_race'
        return replace(raw,available=raw.available and healthy and error is None,error=error,
            generation=self.generation,own_counters=after['own_counters'] if after else (),
            foreign_sequence=after['foreign_sequence'] if after else 0,tap_healthy=healthy,
            pending=after['pending'] if after else None,events=after['events'] if after else ())

    def _query(self):
        try:
            import Quartz as q
            from AppKit import NSWorkspace
            from jccbrain.actuator.window import mouse_location
            if not q.CGPreflightListenEventAccess():
                raise InputMonitorError('input_monitor_permission_unavailable')
            state=q.kCGEventSourceStateHIDSystemState
            counters=tuple((n,int(q.CGEventSourceCounterForEventType(state,getattr(q,'kCGEvent'+n)))) for n in EVENT_NAMES)
            held=(any(q.CGEventSourceKeyState(state,i) for i in range(128))
                  or any(q.CGEventSourceButtonState(state,i) for i in range(32))
                  or bool(q.CGEventSourceFlagsState(state)&(q.kCGEventFlagMaskShift|q.kCGEventFlagMaskControl
                      |q.kCGEventFlagMaskAlternate|q.kCGEventFlagMaskCommand)))
            app=NSWorkspace.sharedWorkspace().frontmostApplication()
            return InputSnapshot(self.clock(),True,float(q.CGEventSourceSecondsSinceLastEventType(state,q.kCGAnyInputEventType)),
                counters,bool(held),tuple(mouse_location()),int(app.processIdentifier()) if app else None,self.locator.find(0))
        except Exception as exc:
            return InputSnapshot(self.clock(),error=f'{type(exc).__name__}: {exc}')
