"""User takeover contract, through actual ScoutRunner with explicit controlled events."""
from dataclasses import replace
import pytest
from test_scout_runner import setup, finish, clicks, WINDOW  # noqa: F401
from test_scout_notice import announce
from jccbrain.actuator.actions import RestoreMouse

@pytest.mark.parametrize('kind', ['MouseMoved','LeftMouseDragged','RightMouseDragged','OtherMouseDragged',
    'LeftMouseDown','LeftMouseUp','RightMouseDown','RightMouseUp','OtherMouseDown','OtherMouseUp',
    'KeyDown','KeyUp','FlagsChanged','ScrollWheel'])
def test_any_event_after_notice_cancels_same_round_without_chasing(setup, kind):
    ctx, runner, cap, state, _, _ = setup
    notice = announce(setup)
    ctx.monitor.event(kind,point=(400.1,300.1))  # even subpixel mouse motion
    runner.clock.sleep(3)
    runner.maybe_run(state)
    assert not ctx.executor.history and cap.calls == 0
    assert runner.status['notice'] is None
    assert not runner.notice_shown(notice,'toast')
    ctx.monitor.held=False
    assert not runner.maybe_run(state) and runner.status['notice'] is None


def test_short_key_press_between_queries_cannot_disappear(setup):
    ctx, runner, _, state, _, _ = setup
    announce(setup)
    ctx.monitor.event('KeyDown'); ctx.monitor.event('KeyUp')
    assert not ctx.monitor.held
    runner.clock.sleep(3)
    runner.maybe_run(state)
    assert not ctx.executor.history

@pytest.mark.parametrize('busy', ['held','idle'])
def test_before_notice_defers_until_idle_in_same_round(setup,busy):
    ctx, runner, _, state, sink, _ = setup
    ctx.settings.scout.enabled=True
    sink.scout_toast_capable=lambda:True
    sink.scout_hud_visible=lambda w=None:w in (None,'toast')
    if busy=='held':ctx.monitor.held=True
    else:ctx.monitor.idle=1.49
    assert not runner.maybe_run(state) and runner.status['notice'] is None
    ctx.monitor.held=False;ctx.monitor.idle=1.5
    assert not runner.maybe_run(state) and runner.status['notice']
    runner.stop()
    assert ctx.monitor.stops and not ctx.executor.history

@pytest.mark.parametrize('fault', ['health','available','race','clock','future','generation','rewind','schema','negative_own','foreign'])
def test_monitor_fault_after_notice_fails_closed(setup,fault):
    ctx,runner,_,state,_,_=setup
    announce(setup)
    if fault=='health':ctx.monitor.healthy=False
    if fault=='available':ctx.monitor.available=False
    if fault=='race':ctx.monitor.change=lambda s:replace(s,error='input_observation_race')
    if fault=='clock':ctx.monitor.change=lambda s:replace(s,at=float('nan'))
    if fault=='future':ctx.monitor.change=lambda s:replace(s,at=s.at+1)
    if fault=='generation':ctx.monitor.generation+=1
    if fault=='rewind':ctx.monitor.counts['KeyDown']=-1
    if fault=='schema':ctx.monitor.counts.pop('KeyUp')
    if fault=='negative_own':ctx.monitor.ledger.own['MouseMoved']+=1
    if fault=='foreign':ctx.monitor.ledger.foreign+=1
    runner.clock.sleep(3);runner.maybe_run(state)
    assert not ctx.executor.history and runner.status['notice'] is None
    assert ctx.monitor.stops

@pytest.mark.parametrize('boundary',['guard','load','retry','return','error','restore','newgame'])
def test_takeover_at_every_execution_boundary_has_no_later_action_or_restore(setup,boundary):
    ctx,runner,cap,state,_,_=setup
    occurred=[]
    def takeover():
        occurred.append(len(ctx.executor.history));ctx.monitor.event('ScrollWheel')
    if boundary=='guard':
        original=ctx.guard.check
        def guard():original();takeover()
        ctx.guard.check=guard
    elif boundary in ('load','error','newgame'):
        def hook(n):
            if n==2:
                if boundary=='newgame':ctx.scout.reset()
                else:takeover()
                if boundary=='error':raise RuntimeError('controlled capture failure')
        cap.hook=hook
    elif boundary=='retry':
        runner._next_retry=lambda:(takeover(),state.players[1])[1]
    else:
        original=runner._click
        def click(player,**kw):
            if boundary=='return' and player.is_self:takeover()
            original(player,**kw)
        runner._click=click
        if boundary=='restore':
            original_execute=ctx.executor.execute
            def execute(action,**kw):
                if isinstance(action,RestoreMouse):takeover()
                return original_execute(action,**kw)
            ctx.executor.execute=execute
    runner.run(state,manual=True);finish(runner)
    assert runner.status['reason']
    assert not any(isinstance(a,RestoreMouse)for a in ctx.executor.history)
    if occurred:assert len(ctx.executor.history)==occurred[0]
    if boundary not in ('guard',):assert runner._awaiting_own
    assert ctx.monitor.stops


def test_own_event_ledger_does_not_exempt_concurrent_user(setup):
    ctx,runner,_,state,_,_=setup
    def hook(kind):
        if kind=='MouseMoved':
            ctx.monitor.hook=lambda _:None
            ctx.monitor.event('KeyDown')
    ctx.monitor.hook=hook
    runner.run(state,manual=True);finish(runner)
    assert not ctx.executor.history and runner.status['reason']
    assert ctx.monitor.ledger.foreign>0


def test_missing_own_event_prevents_later_switches(setup):
    ctx,runner,_,state,_,_=setup
    original=ctx.monitor.event
    ctx.monitor.event=lambda kind,**kw:None if kind=='LeftMouseUp' else original(kind,**kw)
    runner.run(state,manual=True);finish(runner)
    assert not ctx.executor.history and runner.status['reason']
    assert ctx.monitor.stops


def test_foreground_loss_and_geometry_change_during_load_do_not_cleanup(setup):
    ctx,runner,cap,state,_,_=setup
    cap.hook=lambda n:setattr(ctx.monitor,'front',999)if n==2 else None
    runner.run(state,manual=True);finish(runner)
    assert len(clicks(ctx))==1 and runner._awaiting_own
    assert not any(isinstance(a,RestoreMouse)for a in ctx.executor.history)

@pytest.mark.parametrize('takeover', [False,True])
def test_native_boundary_own_sequence_and_paired_release_only(setup,monkeypatch,takeover):
    import sys
    from types import SimpleNamespace
    import jccbrain.actuator.executor as module
    from jccbrain.actuator.executor import MacExecutor
    ctx,runner,cap,state,_,_=setup
    ex=MacExecutor(ctx.layout,ctx.guard,ctx.settings.actuator,ctx.locator)
    ex.preflight=lambda:[]
    ex.mouse_position=lambda:(400.,300.)
    ex._screen=lambda p:WINDOW.to_screen(*p)
    posts=[]
    names={1:'MouseMoved',2:'LeftMouseDown',3:'LeftMouseUp'}
    def post(kind,point):
        posts.append(kind);ctx.monitor.event(names[kind],own=True,point=point)
        if takeover and kind==2:ctx.monitor.event('ScrollWheel')
    ex._post=post
    ex.listeners.append(cap.on_click)
    ctx.executor=ex
    runner._foreground_pid=lambda:WINDOW.pid
    monkeypatch.setitem(sys.modules,'Quartz',SimpleNamespace(kCGEventMouseMoved=1,kCGEventLeftMouseDown=2,
        kCGEventLeftMouseUp=3,CGPointMake=lambda x,y:(x,y)))
    monkeypatch.setattr(module,'time',SimpleNamespace(sleep=lambda _:None))
    runner.run(state,manual=True);finish(runner)
    if takeover:
        assert posts==[1,2,3]  # only matching up after takeover; no new move/target/restore
        assert runner.status['reason'] and runner._awaiting_own
    else:
        assert posts==[1,2,3]*3+[1]
        assert not runner.status['reason']
    assert ctx.monitor.stops


def test_monitor_start_stop_failures_never_authorize_or_revive(setup):
    ctx,runner,_,state,_,_=setup
    ctx.monitor.start=lambda:(_ for _ in ()).throw(RuntimeError('controlled start failure'))
    runner.run(state,manual=True)
    assert not ctx.executor.history and not runner.busy
    assert ctx.monitor.stops
    ctx.monitor.start=lambda:True
    ctx.monitor.stop=lambda:False
    runner.run(state,manual=True);finish(runner)
    count=len(ctx.executor.history)
    runner.run(state,manual=True)
    assert len(ctx.executor.history)==count and '未能停止' in runner.status['reason']


def test_existing_pending_expectation_cannot_authorize_new_transaction(setup):
    ctx,runner,_,state,_,_=setup
    ctx.monitor.expect_action('click',(1.,1.),(0.,0.))
    runner.run(state,manual=True)
    assert not ctx.executor.history and '旧输入期望' in runner.status['reason']
    assert ctx.monitor.stops


def test_stop_during_monitor_start_closes_newly_started_listener(setup):
    ctx,runner,_,state,sink,_=setup
    ctx.settings.scout.enabled=True
    sink.scout_toast_capable=lambda:True
    sink.scout_hud_visible=lambda w=None:w=='toast'
    def start():runner.stop();return True
    ctx.monitor.start=start
    runner.maybe_run(state)
    assert not ctx.executor.history and not runner.activity.running
    assert ctx.monitor.stops>=2 and runner.status['notice'] is None


def test_mouse_origin_failure_closes_listener_and_notice(setup):
    ctx,runner,_,state,_,_=setup
    ctx.executor.mouse_position=lambda:(_ for _ in ()).throw(RuntimeError('controlled origin failure'))
    runner.run(state,manual=True)
    assert not ctx.executor.history and ctx.monitor.stops and not runner.activity.running
    assert not runner.paused and runner.status['notice'] is None
    assert '读取鼠标位置失败' in runner.status['reason']


def test_delayed_own_tap_delivery_is_reconciled_without_exempting_user(setup):
    ctx,runner,_,state,_,_=setup
    pending=[]
    monitor=ctx.monitor
    def event(kind='MouseMoved',*,own=False,point=(1.,1.)):
        monitor.counts[kind]+=1
        monitor.held=kind=='LeftMouseDown'
        pending.append((kind,own,point,int(runner.clock()*1e9)+1))
    def wait(seconds):
        runner.clock.sleep(seconds)
        if pending:
            kind,own,point,stamp=pending.pop(0)
            monitor.ledger.observe(kind,monitor.ledger.pid if own else -1,point,source_time_ns=stamp)
    monitor.event=event;monitor.wait=wait
    runner.run(state,manual=True);finish(runner)
    assert len(clicks(ctx))==3 and isinstance(ctx.executor.history[-1],RestoreMouse)
    assert not runner.status['reason'] and not pending
