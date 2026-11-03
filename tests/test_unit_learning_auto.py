"""Product scheduler/Provider/Executor flow. Native inputs/actions are simulated."""
import copy
import json
import os
import threading
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jccbrain.actuator.actions import ClickPoint, RestoreMouse
from jccbrain.actuator.executor import Executor
from jccbrain.actuator.safety import AbortError
from jccbrain.actuator.window import WindowInfo
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.modes.base import ModeRunner, ModeSink
from jccbrain.modes.unit_learning import UnitLearningRunner, candidates, body_point, empty_point
from jccbrain.vision.eye.input_monitor import InputSnapshot, HIDMonitor, EVENT_NAMES, SessionEventTap
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.unit_learning import UnitLearningStore, LearningCancelled
from jccbrain.vision.eye.unit_panel import PanelObservation
from jccbrain.vision.eye.units import Bar, TrackedUnit, Gallery, UnitTracker, unit_crop
from jccbrain.vision.pyeye import PyEyeProvider


class Clock:
    t = 100.
    def __call__(self): return self.t
    def wait(self, t): self.t += t


class SimExecutor(Executor):
    dry_run = False
    def __init__(self, env):
        super().__init__(None, env.guard)
        self.env, self.history = env, []
    def _do(self, action):
        self.history.append(action)
        if isinstance(action, ClickPoint):
            self.env.panel = action.label == '查看未知棋子身份'
            self.env.mouse = self.env.window.to_screen(*action.point)
        else:
            self.env.mouse = action.screen_point
        for name in (('MouseMoved','LeftMouseDown','LeftMouseUp') if isinstance(action,ClickPoint) else ('MouseMoved',)):
            self.env.deliver(name,os.getpid(),self.env.mouse)
        if self.env.action_hook: self.env.action_hook(action)


@pytest.fixture
def auto(tmp_path, monkeypatch):
    monkeypatch.setenv('JCC_HOME',str(tmp_path/'home'))
    clock = Clock()
    monkeypatch.setattr('time.monotonic',clock)
    monkeypatch.setattr('time.monotonic_ns',lambda:int(clock()*1e9))
    e = NS(clock=clock, panel=False, action_hook=None, capture_hook=None, input_hook=None,
           mouse=(900.,700.), counters=(('hardware',0),), held=False, available=True, idle=10.,
           window=WindowInfo('game','title',12,20,30,1774,1136,7), front=12,
           phase='planning',timer=30,stage='1-4',viewing=None,stale=False,captures=0)
    settings = Settings(); settings.vision.unit_learning_mode='auto'; settings.actuator.dry_run=False
    guard = NS(stopped=False, check=Mock(), reset=Mock())
    def stop(*_): guard.stopped=True
    guard.stop=stop; e.guard=guard
    gallery=Gallery(tmp_path/'gallery','probe',{'甲'})
    tracker=UnitTracker(gallery)
    bar=Bar(675,742,450,455,.3994,.3983)
    def detect(im):
        return {},[TrackedUnit('board',None,bar.cx,bar.cy,1,bar=bar,crop=unit_crop(im,bar),
                              fp=np.array([1.,0.],np.float32),fp_prepared=True)]
    tracker.detect=detect
    def state(): return dict(phase=e.phase,timer=e.timer,stage=e.stage,viewing=e.viewing,units_stale=e.stale,gold=17)
    reader=NS(units=tracker,dict=Dictionary({'甲':1},champion_traits={'甲':['法师']}),
              ocr=NS(name='fake'),last_ocr_cells=0, main_reads=0)
    def read(im):
        reader.main_reads+=1
        return {**state(), **tracker.observe(im,e.phase,None,prepared=detect(im))}
    reader.read=read
    reader._foreign_context=lambda:NS(_read_learning_scene=lambda im:state())
    def grab():
        e.captures+=1;clock.wait(.13)
        if e.capture_hook:e.capture_hook(e.captures)
        im=np.zeros((1136,1774,3),np.uint8)
        cv2.rectangle(im,(680,470),(736,550),(255,220,0),2)
        if e.panel:im[int(.17*1136):int(.60*1136),int(.772*1774):int(.785*1774)]=(60,120,170)
        return im,e.window
    p=PyEyeProvider(min_interval_s=0);p._reader=reader;p._capture=NS(grab=grab)
    e.provider=p
    ctx=NS(settings=settings,vision=p,guard=guard,locator=NS(find=lambda *_:e.window))
    ctx.executor=SimExecutor(e)
    scout=NS(busy=False,paused=False)
    e.raw_counts=Counter({name:0 for name in EVENT_NAMES});e.last_legacy=0;e.tap_starts=e.tap_stops=0
    class Tap:
        def __init__(self,ledger):self.ledger=ledger;self.alive=False
        def start(self):self.alive=True;e.tap_starts+=1;return True
        def healthy(self):return self.alive and not self.ledger.error
        event_time_ns=SessionEventTap.event_time_ns  # exercise the production clock adapter
        def stop(self):self.alive=False;e.tap_stops+=1;return True
    def query():
        if e.input_hook:e.input_hook()
        legacy=dict(e.counters)['hardware']
        if legacy!=e.last_legacy:
            e.raw_counts['KeyDown']+=legacy-e.last_legacy;e.last_legacy=legacy
            e.monitor.ledger.observe('KeyDown',0,source_time_ns=int(clock()*1e9))
        return InputSnapshot(clock(),e.available,e.idle,tuple(e.raw_counts.items()),e.held,e.mouse,e.front,e.window)
    monitor=HIDMonitor(ctx.locator,tap_factory=Tap,clock=clock,wait=clock.wait,raw_query=query)
    e.monitor=monitor
    def deliver(name,pid,point):
        e.raw_counts[name]+=1
        monitor.ledger.observe(name,pid,point,source_time_ns=int(clock()*1e9))
    e.deliver=deliver
    trace=[];sink=NS(on_log=Mock(),on_action=Mock())
    r=UnitLearningRunner(ctx,sink,scout=scout,monitor=monitor,clock=clock,wait=clock.wait,
                         trace=lambda k,d:trace.append((k,d)))
    # Configure once, then the next REAL Provider.read issues a new receipt.
    r.maybe_run(GameState.from_dict(state()))
    ctl=p._learning
    ctl.panel.read=Mock(return_value=PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields'))
    for mod in ('jccbrain.vision.eye.units','jccbrain.vision.eye.unit_learning'):
        monkeypatch.setattr(mod+'.feature_print',lambda _:np.array([1.,0.],np.float32))
    e.ctx,e.runner,e.scout,e.trace,e.ctl,e.reader=ctx,r,scout,trace,ctl,reader
    return e


def run(e): return e.runner.maybe_run(e.provider.read())


def test_complete_product_flow_promotes_isolated_clean_crop_closes_and_consumes(auto):
    e=auto;raw=e.provider.read();last=e.provider._last;image=e.provider._last_img;timestamp=e.provider._last_t
    prev=copy.deepcopy(e.reader.units.prev);reads=e.reader.main_reads
    assert e.runner.maybe_run(raw)
    assert e.runner.status['verified']==1
    assert not e.runner.status['active']
    assert [type(a) for a in e.ctx.executor.history]==[ClickPoint,ClickPoint,RestoreMouse]
    assert not e.panel and not e.runner._blocked_panel
    assert e.provider._last is last and e.provider._last_img is image and e.provider._last_t==timestamp
    assert e.reader.main_reads==reads and repr(e.reader.units.prev)==repr(prev)
    assert len(e.ctl.store.entries())==1
    assert e.provider.read().board[0].name=='甲'  # ongoing MAIN tracker hot-loads it


@pytest.mark.parametrize('field,value,reason',[
    ('phase','combat','fresh_own_planning_required'),('phase','unknown','fresh_own_planning_required'),
    ('viewing',{'slot':2},'fresh_own_planning_required'),('stale',True,'fresh_own_planning_required'),
    ('stage',None,'fresh_own_planning_required'),('timer',None,'timer_unreadable'),
    ('timer',14,'timer_too_short'),('available',False,'input_monitor_unavailable'),
    ('held',True,'held_input'),('idle',1.,'input_not_idle'),('front',99,'game_not_foreground')])
def test_entrance_fail_closed_without_actions_or_promotions(auto,field,value,reason):
    setattr(auto,field,value);assert not run(auto)
    assert auto.runner.status['reason']==reason
    assert not auto.ctx.executor.history and not auto.ctl.store.entries()


@pytest.mark.parametrize('change,reason',[
    ('window','window_changed'),('front','game_not_foreground'),('input','hardware_input_changed'),
    ('mouse','mouse_moved'),('stop','stopped'),('guard','guard_stopped'),('scout','scout_priority'),
    ('phase','fresh_own_planning_required'),('stage','scene_or_window_changed'),('timer','timer_too_short'),
    ('timeout','transaction_timeout')])
def test_cancel_after_click_never_closes_or_restores_or_promotes(auto,change,reason):
    e=auto
    def hook(_):
        if change=='window':e.window=replace(e.window,x=21)
        if change=='front':e.front=99
        if change=='input':e.counters=(('hardware',1),)
        if change=='mouse':e.mouse=(500.,500.)
        if change=='stop':e.runner.stop()
        if change=='guard':e.guard.stopped=True
        if change=='scout':e.scout.busy=True
        if change=='phase':e.phase='combat'
        if change=='stage':e.stage='1-5'
        if change=='timer':e.timer=3
        if change=='timeout':e.clock.wait(7)
    e.action_hook=hook
    assert run(e)
    assert e.runner.status['reason']==reason
    assert len(e.ctx.executor.history)==1 and e.panel
    assert not e.ctl.store.entries()


def test_user_input_during_second_panel_ocr_cancels_before_store_commit(auto):
    e=auto;calls=[]
    def panel(_):
        calls.append(1)
        if len(calls)==2:e.counters=(('hardware',1),)
        return PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields')
    e.ctl.panel.read.side_effect=panel
    assert run(e)
    assert not e.ctl.store.entries() and len(e.ctx.executor.history)==1
    assert e.runner.status['reason']=='hardware_input_changed'


def test_cancel_immediately_before_atomic_promotion_does_not_expose_vector(auto,monkeypatch):
    e=auto;original=e.ctl.store._commit_dir
    def commit(directory,*a,**kw):
        answer=original(directory,*a,**kw)
        if directory==e.ctl.store.verified:e.runner.stop()
        return answer
    monkeypatch.setattr(e.ctl.store,'_commit_dir',commit)
    run(e)
    assert not e.ctl.store.active_vectors()
    assert len(e.ctx.executor.history)==1


def test_guard_wait_before_do_rechecks_inputs_before_first_real_action(auto):
    auto.guard.check.side_effect=lambda:setattr(auto,'counters',(('hardware',1),))
    run(auto)
    assert not auto.ctx.executor.history and not auto.ctl.store.entries()


def test_cached_capture_cannot_be_second_target_stability_sample(auto,monkeypatch):
    e=auto
    monkeypatch.setattr(e.provider,'sample_unit_learning',lambda **_:e.provider.unit_learning_sample())
    run(e)
    assert e.runner.status['reason']=='capture_not_fresh'
    assert not e.ctx.executor.history and not e.ctl.store.entries()


def test_actual_mac_executor_code_with_simulated_hid_counter_changes_cancels(auto,monkeypatch):
    # Exercise MacExecutor.click's three synthetic posts, replacing _post only.
    # This assumes HID sees them, to prove fail-closed; NOT a native HID finding.
    from jccbrain.actuator.executor import MacExecutor
    import Quartz
    e=auto;e.guard.sleep=e.clock.wait
    ex=MacExecutor(None,e.guard,e.ctx.settings.actuator,e.ctx.locator)
    posts=[]
    def post(kind,point):
        posts.append(kind);e.mouse=(point.x,point.y);e.counters=(('hardware',len(posts)),)
    monkeypatch.setattr(ex,'_post',post)
    monkeypatch.setattr(Quartz,'CGEventPost',Mock(side_effect=AssertionError('must not send native input')))
    e.ctx.executor=ex
    run(e)
    assert len(posts)==3 and e.runner.status['reason']=='hardware_input_changed'
    assert not e.ctl.store.entries() and e.runner._blocked_panel
    Quartz.CGEventPost.assert_not_called()


@pytest.mark.parametrize('change',['phase','window','scout','stop'])
def test_before_do_fresh_scene_rejects_change_before_any_actual_action(auto,change):
    e=auto
    def change_now():
        if change=='phase':e.phase='combat'
        elif change=='window':e.window=replace(e.window,y=31)
        elif change=='scout':e.scout.busy=True
        else:e.runner.stop()
    e.guard.check.side_effect=change_now
    run(e)
    assert not e.ctx.executor.history and not e.ctl.store.entries()


def test_entry_guard_and_failsafe_and_invalid_input_time_fail_closed(auto):
    e=auto;e.guard.stopped=True;run(e)
    assert e.runner.status['reason']=='guard_stopped'
    e.guard.stopped=False;e.mouse=(1.,1.);run(e)
    assert e.runner.status['reason']=='failsafe_corner'
    e.mouse=(900.,700.);e.idle=float('nan');run(e)
    assert e.runner.status['reason']=='input_monitor_unavailable'
    assert not e.ctx.executor.history


def test_same_position_feature_and_isolation_and_body_area_are_required(auto,monkeypatch):
    from jccbrain.vision.eye.unit_learning import FrameSource
    e=auto;e.provider.read();before=e.provider.current_learning_frame()
    im=before.image
    u=e.reader.units.detect(im)[1][0]
    assert body_point(u,im)
    assert body_point(replace(u,zone='board',bar=replace(u.bar,cx=.90)),im) is None
    assert candidates(e.reader,before,before)  # pure geometry helper, not receipt confirmation
    clone=replace(u,bar=replace(u.bar,min_x=u.bar.min_x+10,max_x=u.bar.max_x+10))
    e.reader.units.detect=lambda _:({},[u,clone])
    assert not candidates(e.reader,before,before)
    e.reader.units.detect=lambda _:({},[u])
    sequence=iter([np.array([1.,0.]),np.array([0.,1.])])
    monkeypatch.setattr('jccbrain.vision.eye.units.feature_print',lambda _:next(sequence))
    assert not candidates(e.reader,before,before)


def test_no_blind_close_when_all_empty_points_occupied(auto):
    e=auto;e.provider.read();frame=e.provider.current_learning_frame()
    # Exclusion rectangles of actual detected bars/crops, not a fixed close point.
    units=[]
    for y in (.35,.40,.45,.50,.55):
        for x in (.35,.40,.45,.50,.55,.60,.65):
            b=Bar(int(x*1774-35),int(x*1774+35),int(y*1136),int(y*1136+5),x,y)
            units.append(TrackedUnit('board',None,x,y,1,bar=b,crop=unit_crop(frame.image,b)))
    e.reader.units.detect=lambda _:({},units)
    assert empty_point(e.reader,frame) is None


@pytest.mark.parametrize('busy,paused',[(True,False),(False,True)])
def test_scout_pending_processing_scanning_or_awaiting_own_prevents_learning(auto,busy,paused):
    auto.scout.busy,auto.scout.paused=busy,paused
    assert not run(auto)
    assert auto.runner.status['reason']=='scout_priority'
    assert not auto.ctx.executor.history


def test_already_open_user_panel_confirms_without_any_close_or_mouse_action(auto):
    auto.panel=True
    assert run(auto)
    assert auto.runner.status['verified']==1 and auto.panel
    assert not auto.ctx.executor.history


def test_dryrun_never_clicks_or_confirms_even_visible_panel(auto):
    auto.ctx.executor.dry_run=True;auto.panel=True
    assert not run(auto)
    assert auto.runner.status['reason']=='dry_run_no_actions_or_promotion'
    assert not auto.ctx.executor.history and not auto.ctl.store.entries()


def test_record_records_only_and_mode_off_disconnects_processing(auto):
    e=auto;e.ctx.settings.vision.unit_learning_mode='record'
    run(e);run(e)
    assert e.provider.last_learning['promoted']==0
    assert e.provider._learning.store.entries('review') and not e.provider._learning.store.entries()
    e.ctx.settings.vision.unit_learning_mode='off'
    run(e)
    assert e.provider._learning is None and not e.ctx.executor.history
    assert e.runner.status['mode']=='off' and e.runner.status['reason']=='disabled'


def test_record_waits_until_after_scout_busy_and_never_runs_inside_normal_read(auto):
    e=auto;e.ctx.settings.vision.unit_learning_mode='record'
    run(e)  # reconfigure; next ordinary read has the new receipt
    store=e.provider._learning.store
    e.scout.busy=True
    raw=e.provider.read()
    assert not store.entries('review')
    e.runner.maybe_run(raw)
    assert not store.entries('review') and e.runner.status['reason']=='scout_priority'
    e.scout.busy=False
    run(e)
    assert store.entries('review') and not store.entries()


def test_off_never_touches_vision_executor_or_store():
    settings=Settings()
    ctx=NS(settings=settings)
    r=UnitLearningRunner(ctx,ModeSink())
    assert not r.maybe_run(GameState())
    r.close()
    assert not hasattr(ctx,'vision')


def test_replay_cannot_enter_native_auto_or_record(tmp_path,monkeypatch):
    from jccbrain.vision.pyeye import ImageReplayProvider
    monkeypatch.setenv('JCC_HOME',str(tmp_path/'home'))
    settings=Settings();settings.vision.unit_learning_mode='auto'
    ctx=NS(settings=settings,vision=ImageReplayProvider(tmp_path))
    r=UnitLearningRunner(ctx,NS(on_log=Mock()))
    assert not r.maybe_run(GameState()) and r.status['reason']=='live_pyeye_required'
    assert not (tmp_path/'home').exists()


def test_failed_target_not_retried_same_round_and_new_round_budget_clears(auto):
    e=auto;e.ctl.panel.read.return_value=PanelObservation(False,0.,reason='no_panel')
    run(e);attempts=len(e.ctx.executor.history)
    e.clock.wait(20)
    assert run(e)  # selection gets skipped, no repeated click
    assert len(e.ctx.executor.history)==attempts
    assert e.runner.status['reason']=='no_stable_unknown_target'
    e.stage='1-5';e.clock.wait(20);run(e)
    assert e.runner.status['attempted']==2
    for n in range(70):e.runner._round(f'2-{n+1}')
    assert len(e.runner._rounds)==64
    assert e.runner._round('1-1')==set() and len(e.runner._rounds)==1


def test_interval_and_two_targets_per_round(auto):
    e=auto;raw=e.provider.read()
    e.runner._last_attempt=e.clock()
    assert not e.runner.maybe_run(raw) and e.runner.status['reason']=='round_budget_or_interval'
    e.clock.wait(20);e.runner._round('1-4').update({('a',),('b',)})
    assert not run(e) and e.runner.status['reason']=='round_budget_or_interval'
    assert not e.ctx.executor.history


def test_optional_capture_error_recovers_next_round_without_corrupting_main(auto):
    e=auto;e.reader._foreign_context=lambda:(_ for _ in ()).throw(OSError('temporary'))
    run(e)
    assert e.runner.status['reason']=='temporary_capture_failed' and not e.ctx.executor.history
    e.reader._foreign_context=lambda:NS(_read_learning_scene=lambda im:dict(phase='planning',stage='1-5',timer=30))
    e.stage='1-5';e.clock.wait(20)
    assert run(e) and e.runner.status['verified']==1


def test_transaction_cleanup_failure_does_not_interrupt_assistant_read(auto,monkeypatch):
    e=auto
    monkeypatch.setattr(e.provider,'finish_unit_learning_transaction',Mock(side_effect=OSError('cleanup failed')))
    assert run(e)
    assert e.runner.status['reason']=='cleanup_error' and not e.runner.active
    assert e.provider.read().gold==17


def test_stop_disconnects_even_if_optional_controller_reset_fails(auto,monkeypatch):
    e=auto
    monkeypatch.setattr(e.ctl,'reset',Mock(side_effect=RuntimeError('reset fault')))
    e.runner.close()
    assert e.provider._learning is None and e.provider._learning_source is None
    assert e.runner.status['reason']=='cleanup_error'
    assert e.provider.read().gold==17


def test_missing_stale_wrong_sized_receipts_cannot_start(auto):
    e=auto;raw=e.provider.read()
    e.provider._learning_frame=None
    assert not e.runner.maybe_run(raw) and e.runner.status['reason']=='no_current_capture'
    e.provider.read();e.clock.wait(2.01)
    assert not e.runner.maybe_run(e.provider._last) and e.runner.status['reason']=='capture_not_fresh'
    e.window=replace(e.window,width=1600,height=900)
    assert not run(e) and e.runner.status['reason']=='unsupported_window'


def test_stabilized_timer_does_not_substitute_for_current_pixels(auto):
    e=auto;e.timer=None;raw=e.provider.read()
    raw.raw['timer']=30  # receipt context is copied from last raw, but not frame phase
    # A separate stabilized GameState with old timer is never the actual read.
    old=GameState.from_dict({**raw.raw,'timer':40})
    assert not e.runner.maybe_run(old) and e.runner.status['reason']=='no_current_capture'


def test_readonly_store_survives_disable_and_restart_and_revoke(auto):
    e=auto;run(e);store=e.ctl.store
    e.runner.close()
    assert e.provider._learning is None and len(e.reader.units.gallery.verified_store.active_vectors())==1
    root=store.root.parent
    read_only=UnitLearningStore(root,store.season,store.pool,read_only=True)
    assert len(read_only.active_vectors())==1
    with pytest.raises(PermissionError):read_only.revoke(store.entries()[0]['id'])
    store.revoke(store.entries()[0]['id'])
    assert not read_only.active_vectors() and e.provider.read().board[0].name.startswith('?')


def test_native_monitor_unavailable_is_readonly_and_fail_closed(monkeypatch):
    import Quartz
    monkeypatch.setattr(Quartz,'CGPreflightListenEventAccess',lambda:False)
    locator=NS(find=Mock(side_effect=AssertionError('no window query after denial')))
    snapshot=HIDMonitor(locator).snapshot()
    assert not snapshot.available and snapshot.error
    locator.find.assert_not_called()


def test_mode_runner_join_timeout_retains_old_owner_and_refuses_autoroll(auto):
    e=auto;runner=ModeRunner(e.ctx)
    old=NS(stop_event=threading.Event(),acts=False,learning=e.runner,scout=None)
    thread=NS(is_alive=lambda:True,join=Mock())
    runner.mode,runner._thread=old,thread
    assert runner.stop(timeout=.01) is False
    assert runner.mode is old and runner._thread is thread and e.runner._cancel.is_set()
    new=NS(preflight=Mock(),key='autoroll',acts=True)
    assert runner.stopping and runner.start(new)==['旧模式仍在停止中，请等待线程退出后再启动']
    new.preflight.assert_not_called()
    assert old.stop_event.is_set() and e.guard.stopped


def test_assistant_real_loop_scout_first_then_learning_then_new_normal_read(auto,db,rules):
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.vision.base import VisionError
    e=auto;ctx=Ctx(db,rules,[])
    ctx.settings=e.ctx.settings;ctx.vision=e.provider;ctx.guard=e.guard;ctx.executor=e.ctx.executor
    ctx.locator=e.ctx.locator;ctx.llm=NS(ready=False)
    sink=AdviceSink();mode=AssistantMode(ctx,sink)
    mode.learning=e.runner;e.runner.ctx=ctx
    events=[]
    mode.scout=NS(paused=False,busy=False,mark_own=lambda:None,observe=lambda _:None,stop=lambda:None,
                  maybe_run=lambda st:events.append('scout') or False)
    original=e.runner.maybe_run
    def learning(raw):
        events.append('learning')
        if e.runner.status['verified']:
            assert raw.board[0].name=='甲'
            mode.stop_event.set()
            return False
        result=original(raw)
        return result
    mode.learning.maybe_run=learning
    # Advice was computed on the preceding normal read; no private capture can
    # push an extra advice or enter the stabilizer.
    mode.loop()
    assert events==['scout','learning','scout','learning']
    assert len(sink.advices)==2 and e.reader.main_reads==2
    assert sink.advices[-1].phase.value=='unknown' and sink.advices[-1].headline=='辅助已停止'
    assert mode.stabilizer.acc['gold']==17 and e.provider._learning is None
    assert len(e.ctx.executor.history)==3


def test_real_scout_manual_request_during_ocr_cancels_before_following_action(auto,db):
    from jccbrain.engine.scout import ScoutBook
    from jccbrain.modes.scout import ScoutRunner
    e=auto;e.ctx.advisor=NS(scout=ScoutBook(db))
    sink=ModeSink();scout=ScoutRunner(e.ctx,sink)
    e.runner.scout=scout
    def ocr(_):
        scout.request()
        return PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields')
    e.ctl.panel.read.side_effect=ocr
    run(e)
    assert scout.status['pending'] and e.runner.status['reason']=='scout_priority'
    assert not e.ctl.store.entries() and len(e.ctx.executor.history)==1
    # Synchronous learning has returned; scout owns the next main-loop decision.
    assert not e.runner.active and not scout.scanning


def test_mode_runner_real_threads_wait_for_learning_stop_before_next_mode(auto):
    e=auto;runner=ModeRunner(e.ctx)
    entered=threading.Event();released=threading.Event();next_started=[]
    class Old:
        key='assistant';acts=False;scout=None;learning=e.runner;stop_event=threading.Event();sink=ModeSink()
        def preflight(self):return []
        def loop(self):
            entered.set();released.wait(2)
            assert self.stop_event.is_set() and self.learning._cancel.is_set()
            self.learning.close()
            return 'old exited'
    class New:
        key='autoroll';acts=True;sink=ModeSink();stop_event=threading.Event()
        def preflight(self):return []
        def loop(self):next_started.append(True);return 'done'
    assert not runner.start(Old()) and entered.wait(1)
    assert runner.stop(timeout=.01) is False
    assert runner.running and runner.stopping
    assert runner.start(New()) and e.guard.stopped
    released.set();runner._thread.join(2)
    assert runner.stop(timeout=1) is True
    assert not runner.start(New())
    runner._thread.join(2)
    assert next_started==[True]


def test_factory_only_loads_existing_confirmed_store_without_writing_it(tmp_path,db,monkeypatch):
    from scripts.eval_vision_b8 import manifest
    from jccbrain.vision.pyeye import build_reader
    from jccbrain.vision.eye.items import season_key
    monkeypatch.setenv('JCC_HOME',str(tmp_path/'home'))
    root=tmp_path/'home/unit-learning'
    pool={h.name for h in db.pool_heroes()};name=sorted(pool)[0]
    before_reader=build_reader(db,tmp_path/'gallery')
    assert before_reader.units.gallery.verified_store is None and not root.exists()
    store=UnitLearningStore(root,season_key(db.season_id),pool)
    sid=store.record(dict(source='manual',name=name,confirmation=True),{'unit':np.zeros((12,12,3),np.uint8)})
    store.promote(sid,np.array([1.,0.],np.float32))
    before=manifest(root)
    new=build_reader(db,tmp_path/'gallery')
    assert new.units.gallery.match(np.array([1.,0.],np.float32))==(name,0.)
    assert new.units.gallery.verified_store.read_only and manifest(root)==before


@pytest.mark.parametrize('field,value',[
    ('unit_learning_mode','on'),('unit_learning_idle_s',1.9),('unit_learning_idle_s',61),
    ('unit_learning_idle_s',float('nan')),('unit_learning_interval_s',9),('unit_learning_interval_s',301),
    ('unit_learning_min_timer',9),('unit_learning_min_timer',121),('unit_learning_min_timer',15.5),
    ('unit_learning_min_timer',True)])
def test_settings_invalid_atomic_and_loaded_invalid_never_enables_click(tmp_path,monkeypatch,field,value):
    from jccbrain.webui.settings_schema import apply
    s=Settings();before=copy.deepcopy(s)
    monkeypatch.setenv('JCC_HOME',str(tmp_path))
    with pytest.raises(ValueError):apply(s,{'vision':{'unit_learning_mode':'auto',field:value}})
    assert s==before and not (tmp_path/'settings.json').exists()
    path=tmp_path/'invalid.json';path.write_text(json.dumps({'vision':{'unit_learning_mode':'auto',field:value}}))
    assert Settings.load(path).vision.unit_learning_mode=='off'


def test_settings_default_migration_schema_save_reopen(tmp_path,monkeypatch):
    from jccbrain.webui.settings_schema import apply,schema
    path=tmp_path/'settings.json';path.write_text('{}')
    monkeypatch.setenv('JCC_HOME',str(tmp_path))
    s=Settings.load(path);assert s.vision.unit_learning_mode=='off'
    apply(s,{'vision':{'unit_learning_mode':'auto','unit_learning_idle_s':'3',
                       'unit_learning_interval_s':'20','unit_learning_min_timer':'19'}})
    assert Settings.load(path)==s
    section=next(x for x in schema(s) if x['key']=='vision')
    mode=next(x for x in section['fields'] if x['key']=='unit_learning_mode')
    assert mode['label']=='棋子身份学习' and mode['choices']==['off','record','auto']
    assert '演练' in section['note'] and '辅助大脑' in section['note']
