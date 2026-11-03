"""No native tap/capture/input. Real callbacks/runloop adapter with fake Quartz."""
import json
import os
import threading
import time
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from jccbrain.vision.eye.input_monitor import (ACTION_DEADLINE_S, EVENT_NAMES, EventLedger,
    HIDMonitor, InputMonitorError, InputSnapshot, SessionEventTap, counter_residual)
from test_unit_learning_auto import auto, run, Clock


class QuartzFake:
    kCGEventSourceUnixProcessID=99
    kCGEventTapDisabledByTimeout=1000
    kCGEventTapDisabledByUserInput=1001
    kCGSessionEventTap=1
    kCGHeadInsertEventTap=2
    kCGEventTapOptionListenOnly=3
    kCFRunLoopDefaultMode=4
    kCFRunLoopRunFinished=5
    kCFRunLoopRunStopped=6
    def __init__(self):
        for i,name in enumerate(EVENT_NAMES):setattr(self,'kCGEvent'+name,i+1)
        self.permission=True;self.enabled=True;self.created=[];self.removed=[];self.invalidated=[]
        self.loop_stop=threading.Event();self.create_none=False;self.run_hook=None;self.queries=[]
    def CGPreflightListenEventAccess(self):return self.permission
    def CGEventTapCreate(self,*args):self.created.append(args);return None if self.create_none else object()
    def CFRunLoopGetCurrent(self):return object()
    def CFMachPortCreateRunLoopSource(self,*args):return object()
    def CFRunLoopAddSource(self,*args):pass
    def CFRunLoopRunInMode(self,*args):
        if self.run_hook:return self.run_hook()
        self.loop_stop.wait(.001);return 7
    def CGEventTapIsEnabled(self,*_):return self.enabled
    def CFRunLoopRemoveSource(self,*args):self.removed.append(args)
    def CFMachPortInvalidate(self,tap):self.invalidated.append(tap)
    def CFRunLoopStop(self,*_):self.loop_stop.set()
    def CGEventGetIntegerValueField(self,event,field):
        assert field==self.kCGEventSourceUnixProcessID  # never query keyboard content/code
        self.queries.append('pid');return event['pid']
    def CGEventGetLocation(self,event):self.queries.append('point');return NS(x=event['point'][0],y=event['point'][1])
    def CGEventGetTimestamp(self,event):self.queries.append('time');return event['ns']
    def CGEventCreate(self,_):return dict(ns=0)  # unsent native events have no generation timestamp


def adapter(ledger):
    q=QuartzFake();tap=SessionEventTap(ledger,clock=ledger.clock,quartz=q)
    tap._by_type={getattr(q,'kCGEvent'+n):n for n in EVENT_NAMES}
    return tap,q


def observe(tap,q,name,pid,point,ns):
    event=dict(pid=pid,point=point,ns=ns,keycode='FORBIDDEN',text='FORBIDDEN')
    assert tap.callback(None,getattr(q,'kCGEvent'+name),event,None) is event


def test_native_clock_uses_integer_monotonic_without_creating_or_timestamping_an_event(monkeypatch):
    ledger=EventLedger();tap,q=adapter(ledger)
    assert q.CGEventGetTimestamp(q.CGEventCreate(None))==0
    create=Mock(side_effect=AssertionError('unsent event is not a clock'))
    get_timestamp=Mock(side_effect=AssertionError('clock must not read an event'))
    q.CGEventCreate=create;q.CGEventGetTimestamp=get_timestamp
    ns=2_506_206_600_103_083
    monkeypatch.setattr(time,'monotonic_ns',lambda:ns)
    assert tap.event_time_ns()==ns and type(tap.event_time_ns()) is int
    create.assert_not_called();get_timestamp.assert_not_called()
    assert not q.created  # no native/listen-only tap was started by a clock read


@pytest.mark.parametrize('case',['valid','zero','old','future','late','foreign'])
def test_native_clock_floor_keeps_observed_event_time_and_source_checks(monkeypatch,case):
    clock=Clock();ledger=EventLedger(pid=33,clock=clock);tap,q=adapter(ledger)
    ns=2_506_206_600_103_083;point=(400.,500.)
    monkeypatch.setattr(time,'monotonic_ns',lambda:ns)
    ledger.expect('click',point,(2.,2.),tap.event_time_ns())
    # Delivery is later than generation. Callback arrival time must not replace
    # the event's native timestamp, otherwise old/future events could pass.
    clock.wait(.4 if case!='late' else ACTION_DEADLINE_S+.001)
    event_ns={'zero':0,'old':ns-1,'future':ns+int(ACTION_DEADLINE_S*1e9)+1}.get(case,ns+20_000_000)
    observe(tap,q,'MouseMoved',0 if case=='foreign' else 33,point,event_ns)
    if case=='valid':
        for offset,name in enumerate(('LeftMouseDown','LeftMouseUp'),start=1):
            observe(tap,q,name,33,point,event_ns+offset*10_000_000)
        assert ledger.foreign==0 and ledger.pending['consumed']==3
        assert sum(ledger.own.values())==3
        assert ledger.events[0]['source_time_ns']==event_ns
    else:
        assert ledger.foreign==1 and not ledger.pending and not sum(ledger.own.values())


def test_callback_matches_only_registered_own_order_and_never_reads_key_content():
    clock=Clock();ledger=EventLedger(pid=33,clock=clock);tap,q=adapter(ledger)
    ledger.expect('click',(400.,500.),(2.,2.),int(clock()*1e9))
    for name in ('MouseMoved','LeftMouseDown','LeftMouseUp'):observe(tap,q,name,33,(401.,501.),int(clock()*1e9))
    s=ledger.state()
    assert s['foreign_sequence']==0 and sum(dict(s['own_counters']).values())==3
    assert s['pending']['consumed']==3
    observe(tap,q,'KeyDown',33,None,int(clock()*1e9))
    assert ledger.foreign==1 and 'point' not in q.queries[-2:]
    assert 'FORBIDDEN' not in json.dumps(ledger.state())


@pytest.mark.parametrize('case',['other_pid','hardware_pid','wrong_type','out_of_order','wrong_point',
    'expired','old_event','future_event','duplicate','extra','no_expectation'])
def test_wrong_or_unobserved_events_are_not_own(case):
    clock=Clock();ledger=EventLedger(pid=33,clock=clock);tap,q=adapter(ledger)
    ns=int(clock()*1e9);point=(400.,500.);name='MouseMoved';pid=33
    if case!='no_expectation':ledger.expect('click',point,(2.,2.),ns)
    if case=='other_pid':pid=34
    if case=='hardware_pid':pid=0
    if case=='wrong_type':name='RightMouseDown'
    if case=='out_of_order':name='LeftMouseDown'
    if case=='wrong_point':point=(403.,500.)
    if case=='expired':clock.wait(ACTION_DEADLINE_S+.001)
    if case=='old_event':ns-=1
    if case=='future_event':ns+=int((ACTION_DEADLINE_S+.01)*1e9)
    if case=='duplicate':observe(tap,q,'MouseMoved',pid,point,ns)
    if case=='extra':
        for n in ('MouseMoved','LeftMouseDown','LeftMouseUp'):observe(tap,q,n,pid,point,ns)
    before=sum(ledger.own.values())
    observe(tap,q,name,pid,point,ns)
    assert ledger.foreign==1 and sum(ledger.own.values())==before and ledger.pending is None


def test_missing_event_and_raw_counter_gap_never_subtracted_without_observation(auto):
    e=auto;m=e.monitor
    assert m.start();baseline=m.snapshot()
    action=m.expect_action('click',e.mouse,(1.,1.))
    e.deliver('MouseMoved',os.getpid(),e.mouse)
    with pytest.raises(InputMonitorError,match='expected_event_missing'):m.finish_action(action,baseline)
    m.clear_expected();m.stop();assert m.start();baseline=m.snapshot()
    action=m.expect_action('restore',e.mouse,(1.,1.))
    m.ledger.observe('MouseMoved',os.getpid(),e.mouse,source_time_ns=int(e.clock()*1e9))
    # Tap observation by itself is insufficient: HID has not counted it.
    with pytest.raises(InputMonitorError,match='input_counter_gap'):m.finish_action(action,baseline)
    m.stop()


def test_bounded_asynchronous_tap_and_hid_delivery_can_reconcile(auto):
    e=auto;m=e.monitor;assert m.start();baseline=m.snapshot()
    action=m.expect_action('restore',e.mouse,(1.,1.));e.raw_counts['MouseMoved']+=1
    polls=[]
    def wait(t):
        e.clock.wait(t);polls.append(t)
        if len(polls)==2:m.ledger.observe('MouseMoved',os.getpid(),e.mouse,source_time_ns=int(e.clock()*1e9))
    m.wait=wait
    result=m.finish_action(action,baseline)
    assert polls and all(n==0 for n in counter_residual(baseline,result).values())
    assert m.ledger.pending is None
    m.stop()


def test_raw_hid_extra_event_even_when_own_tap_matches_fails_conservation(auto):
    e=auto;m=e.monitor;assert m.start();baseline=m.snapshot()
    action=m.expect_action('restore',e.mouse,(1.,1.));e.deliver('MouseMoved',os.getpid(),e.mouse)
    e.raw_counts['MouseMoved']+=1  # dropped external callback cannot be excused
    with pytest.raises(InputMonitorError,match='input_counter_gap'):m.finish_action(action,baseline)
    assert baseline.counters!=m.snapshot().counters  # original baseline was not reset
    m.stop()


@pytest.mark.parametrize('case',['permission','create','disabled','runloop','thread','heartbeat','notification'])
def test_native_adapter_unavailable_disabled_gap_and_thread_failure_fail_closed(case):
    q=QuartzFake();clock=time.monotonic;ledger=EventLedger();tap=SessionEventTap(ledger,quartz=q)
    if case=='permission':q.permission=False
    if case=='create':q.create_none=True
    if case=='disabled':q.enabled=False
    if case=='runloop':q.run_hook=lambda:q.kCFRunLoopRunFinished
    if case=='thread':
        def fail():raise OSError('runloop failed')
        q.run_hook=fail
    if case=='heartbeat':
        fake_clock=Clock();tap.clock=fake_clock
        def gap():fake_clock.wait(.3);return 7
        q.run_hook=gap
    if case=='notification':
        def disabled():
            tap.callback(None,q.kCGEventTapDisabledByTimeout,object(),None);return 7
        q.run_hook=disabled
    tap.start()
    assert tap._ready.wait(.2)
    if tap.thread:tap.thread.join(.3)
    assert not tap.healthy() and ledger.error
    assert tap.stop() and (not q.created if case=='permission' else q.created)
    if case not in ('permission','create'):assert q.removed and q.invalidated


def test_normal_native_adapter_stop_removes_source_and_no_thread_survives():
    q=QuartzFake();ledger=EventLedger();tap=SessionEventTap(ledger,quartz=q)
    assert tap.start() and tap.healthy()
    args=q.created[0]
    assert args[:3]==(q.kCGSessionEventTap,q.kCGHeadInsertEventTap,q.kCGEventTapOptionListenOnly)
    assert tap.stop() and not tap.thread.is_alive()
    assert len(q.removed)==len(q.invalidated)==1
    assert ledger.pending is None


def mac_executor(e,monkeypatch,*,inject=None):
    """Actual MacExecutor._do/click; native _post replaced by callback adapter."""
    from jccbrain.actuator.executor import MacExecutor
    import Quartz
    e.guard.sleep=e.clock.wait
    ex=MacExecutor(None,e.guard,e.ctx.settings.actuator,e.ctx.locator)
    posted=[];actions=[];q=QuartzFake()
    # Actual Quartz event constants differ from the fake's callback IDs.
    names={getattr(Quartz,'kCGEvent'+n):n for n in EVENT_NAMES}
    def post(kind,point):
        name=names[kind];p=(point.x,point.y);posted.append(name);e.mouse=p
        if name=='LeftMouseDown':e.panel=not e.panel
        if not hasattr(e,'callback_adapter') or e.callback_adapter.ledger is not e.monitor.ledger:
            e.callback_adapter,_q=adapter(e.monitor.ledger)
            e.callback_quartz=_q
        e.raw_counts[name]+=1
        observe(e.callback_adapter,e.callback_quartz,name,os.getpid(),p,int(e.clock()*1e9))
        if inject:inject(name,p)
    monkeypatch.setattr(ex,'_post',post)
    monkeypatch.setattr(Quartz,'CGEventPost',Mock(side_effect=AssertionError('native input forbidden')))
    ex.listeners.append(lambda a,real:actions.append(a))
    e.ctx.executor=ex
    return posted,actions


def test_real_mac_executor_matched_own_events_complete_learning_close_restore(auto,monkeypatch):
    e=auto;posts,actions=mac_executor(e,monkeypatch)
    assert run(e)
    assert e.runner.status['verified']==1 and len(e.ctl.store.entries())==1
    assert posts==['MouseMoved','LeftMouseDown','LeftMouseUp']*2+['MouseMoved']
    assert len(actions)==3 and not e.panel
    assert sum(e.raw_counts.values())==7  # retain fact: our events DO affect raw HID
    assert e.provider.read().board[0].name=='甲'
    assert e.monitor.tap is None and e.tap_starts==e.tap_stops
    assert len([d for k,d in e.trace if k=='input_expectation'])==3
    assert not any(d.get('reason')=='hardware_input_changed' for k,d in e.trace if k=='status')


@pytest.mark.parametrize('pid',[0,123456])
def test_user_input_at_identical_click_position_is_not_hidden_by_own_expectation(auto,monkeypatch,pid):
    e=auto
    def inject(name,point):
        if name=='LeftMouseDown':
            e.raw_counts['MouseMoved']+=1
            observe(e.callback_adapter,e.callback_quartz,'MouseMoved',pid,point,int(e.clock()*1e9))
    posts,actions=mac_executor(e,monkeypatch,inject=inject)
    assert run(e)
    assert e.runner.status['reason']=='hardware_input_changed'
    assert not e.ctl.store.entries() and len(actions)==1 and e.panel
    assert len(posts)==3 and not e.monitor.ledger.pending and e.monitor.tap is None
    assert len([d for k,d in e.trace if k=='action'])==1  # executed click is honestly logged


def test_stop_during_pending_expectation_releases_it_and_no_following_action(auto,monkeypatch):
    e=auto
    def inject(name,point):
        if name=='MouseMoved':e.runner.stop()
    posts,actions=mac_executor(e,monkeypatch,inject=inject)
    run(e)
    assert len(actions)==1 and len(posts)==3 and not e.ctl.store.entries()
    assert not e.monitor.ledger.pending and e.monitor.tap is None
    assert e.runner.status['reason']=='stopped'


@pytest.mark.parametrize('mode',['off','record'])
def test_off_record_never_start_input_tap(auto,mode):
    auto.ctx.settings.vision.unit_learning_mode=mode
    run(auto);run(auto);auto.runner.close()
    assert auto.tap_starts==0 and auto.monitor.tap is None


def test_tap_fault_during_panel_ocr_does_not_promote_or_cleanup_click(auto):
    e=auto
    def ocr(_):
        from jccbrain.vision.eye.unit_panel import PanelObservation
        e.monitor.ledger.fail('tap_disabled_notification')
        return PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields')
    e.ctl.panel.read.side_effect=ocr
    run(e)
    assert not e.ctl.store.entries() and len(e.ctx.executor.history)==1
    assert e.runner.status['reason']=='input_monitor_unavailable' and e.monitor.tap is None


def test_startup_timeout_is_bounded_and_pending_listener_cannot_be_replaced():
    q=QuartzFake();release=threading.Event();entered=threading.Event()
    original=q.CGEventTapCreate
    def blocked(*args):entered.set();release.wait(2);return original(*args)
    q.CGEventTapCreate=blocked
    ledger=EventLedger();tap=SessionEventTap(ledger,quartz=q)
    start=time.monotonic()
    assert not tap.start() and entered.is_set()
    assert time.monotonic()-start<1.5 and ledger.error
    release.set();assert tap.stop() and not tap.thread.is_alive()
    assert q.invalidated and q.removed


def test_monitor_backend_start_stop_failures_do_not_escape_or_allow_restart(auto):
    e=auto;monitor=e.monitor
    class Failed:
        def __init__(self,ledger):self.ledger=ledger
        def healthy(self):return False
        def start(self):raise OSError('create failed')
        def stop(self):raise OSError('stop failed')
    monitor.tap_factory=Failed
    assert not monitor.start() and not monitor.stop()
    generation=monitor.generation
    assert not monitor.start() and monitor.generation==generation
    assert monitor.tap is not None and not monitor.snapshot().available


def test_expected_consumption_is_one_shot_and_restart_is_not_a_new_baseline(auto):
    e=auto;m=e.monitor;assert m.start();baseline=m.snapshot()
    token=m.expect_action('restore',e.mouse,(1.,1.));e.deliver('MouseMoved',os.getpid(),e.mouse)
    m.finish_action(token,baseline)
    with pytest.raises(InputMonitorError,match='expected_event_missing'):m.finish_action(token,baseline)
    m.stop();assert m.start()
    with pytest.raises(InputMonitorError,match='input_monitor_restarted'):m.finish_action(token,baseline)
    m.stop()


def test_partial_execute_exception_does_not_leave_exemption_for_next_transaction(auto,monkeypatch):
    e=auto;execute=e.ctx.executor._do
    def partial(_):
        e.mouse=e.monitor.ledger.pending['point']
        e.deliver('MouseMoved',os.getpid(),e.mouse)
        raise OSError('send failed after movement')
    monkeypatch.setattr(e.ctx.executor,'_do',partial)
    run(e)
    assert not e.monitor.ledger.pending and e.monitor.tap is None and not e.ctl.store.entries()
    failed=next(d for k,d in e.trace if k=='action_failed')
    assert sum(dict(failed['snapshot']['own_counters']).values())==1
    monkeypatch.setattr(e.ctx.executor,'_do',execute)
    e.stage='1-5';e.clock.wait(20)
    assert run(e) and e.runner.status['verified']==1
    assert e.monitor.tap is None and e.tap_starts==e.tap_stops


def test_user_input_during_transaction_hook_is_rechecked_before_registering_or_click(auto):
    e=auto;trace=e.runner.trace
    def hook(kind,data):
        trace(kind,data)
        if kind=='transaction':e.deliver('KeyDown',0,None)
    e.runner.trace=hook
    run(e)
    assert not e.ctx.executor.history and not e.ctl.store.entries()
    assert e.runner.status['reason']=='hardware_input_changed'
    assert e.monitor.tap is None and not e.monitor.ledger.pending
