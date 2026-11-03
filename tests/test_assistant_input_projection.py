"""Read-only counter observation and input fences; native sender is never used."""
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import pytest
from scout_fakes import ControlledInput
from test_assistant_buy_projection import raw, feed, model, running_api, isolated
from test_webui_api import setup_api
from jccbrain.actuator.window import WindowInfo
from jccbrain.core.buy_source import buy_order
from jccbrain.engine.advisor import BuyProjection
from jccbrain.modes.buy_activity import BuyActivityObserver
from jccbrain.webui.events import EventBus

WINDOW = WindowInfo('game', 'controlled', 101, 0, 0, 1000, 800, 42)

@pytest.fixture
def observation():
    now = [100.]
    monitor = ControlledInput(lambda: now[0], WINDOW)
    records = []
    observer = BuyActivityObserver(None, lambda *a: records.append(a), monitor=monitor, clock=lambda: now[0])
    observer.poll(); records.clear()
    return observer, monitor, records, now

@pytest.mark.parametrize('kind', ['KeyDown', 'KeyUp', 'LeftMouseDown', 'LeftMouseUp', 'RightMouseDown', 'RightMouseUp', 'OtherMouseDown', 'OtherMouseUp', 'LeftMouseDragged', 'RightMouseDragged', 'OtherMouseDragged', 'ScrollWheel', 'FlagsChanged'])
def test_relevant_counter_delta_fences_even_short_press(observation, kind):
    o, m, records, now = observation
    m.event(kind); m.held = False
    now[0] += .05
    o.poll()
    assert records and records[-1][0] == 'foreground'

@pytest.mark.parametrize('raced', [False, True])
def test_only_mouse_motion_never_invalidates(observation, raced):
    o, m, records, now = observation
    m.event('MouseMoved')
    if raced:
        m.change = lambda s: replace(s, available=False, error='input_observation_race')
    now[0] += .05; o.poll()
    assert not records and o.status['available']

@pytest.mark.parametrize('boundary', ['held', 'release', 'focus', 'window', 'generation', 'unhealthy', 'count_rewind', 'clock_rewind', 'race_short', 'invalid_counts'])
def test_monitor_boundaries_and_recovery(observation, boundary):
    o, m, records, now = observation
    if boundary == 'held': m.held = True
    elif boundary == 'release': m.held = True; o.poll(); records.clear(); m.held = False
    elif boundary == 'focus': m.front = -1
    elif boundary == 'window': m.window = replace(WINDOW, window_id=102)
    elif boundary == 'generation': m.generation += 1
    elif boundary == 'unhealthy': m.healthy = False
    elif boundary == 'count_rewind': m.counts['KeyDown'] = 2; o.poll(); records.clear(); m.counts['KeyDown'] = 1
    elif boundary == 'clock_rewind': now[0] -= 1
    elif boundary == 'race_short':
        m.event('KeyDown'); m.event('KeyUp'); m.change = lambda s: replace(s, available=False, error='input_observation_race')
    else: m.change = lambda s: replace(s, counters=())
    now[0] += .05; o.poll(); assert records
    if boundary in ('unhealthy','count_rewind','clock_rewind','invalid_counts'):
        assert not o.status['available'] and not records[-1][1]  # explicit fallback, not idle
    m.healthy = True; m.change = lambda s:s; m.front = WINDOW.pid; m.held = False
    now[0] += .1; o.poll(); assert o.status['available'] and not o.status['blocked']

@pytest.mark.parametrize('failure', ['start_false','start_throw','poll_throw','stop_throw'])
def test_monitor_resources_close_on_failures(failure):
    m = ControlledInput(time.monotonic, WINDOW); records = []
    if failure == 'start_false': m.available = False
    elif failure == 'start_throw': m.start = lambda: (_ for _ in ()).throw(RuntimeError())
    elif failure == 'poll_throw': m.snapshot = lambda: (_ for _ in ()).throw(RuntimeError())
    else: m.stop = lambda: (_ for _ in ()).throw(RuntimeError())
    o = BuyActivityObserver(None, lambda *a: records.append(a), monitor=m)
    o.start()
    deadline = time.monotonic()+1
    while not records and time.monotonic()<deadline: time.sleep(.005)
    assert o.close() and not o.status['running']
    assert records
    if failure.startswith('start'): assert 'fallback:' in records[0][0]
    with pytest.raises(RuntimeError): o.start()

@pytest.mark.parametrize('position', ['before_read','during_read','during_compute','after_compute','continuous'])
def test_epoch_cannot_relabel_same_shop_and_requires_two_new_computes(running_api, position):
    api, ctx, bus, mode = running_api
    old = api._buy_projection
    token = mode.buy_read_token()
    if position == 'during_compute':
        original = mode.compute_advice
        def compute(st, **kw):
            value = original(st, **kw); mode._input_changed('key',False); return value
        mode.compute_advice = compute
        mode._process_frame(raw(),read_token=token); mode.compute_advice = original
    else:
        mode._input_changed('key',False)
        if position == 'continuous': mode._input_changed('key',False)
        mode._process_frame(raw(),read_token=token)
    assert not api._buy_projection.valid
    api._sink.on_buy_projection(old)
    assert not api._buy_projection.valid
    feed(mode,1); assert not api._buy_projection.valid
    feed(mode,1); assert api._buy_projection.valid
    assert api._buy_projection.source.input_epoch == mode.buy_input_epoch

@pytest.mark.parametrize('callback', ['compute','native','state_lock','api_lock','mode_lock'])
def test_input_callback_does_not_wait_for_slow_paths(running_api, callback):
    api, ctx, bus, mode = running_api
    locks = {'state_lock':api._state_lock,'api_lock':api._lock,'mode_lock':mode._lock}
    entered, release = threading.Event(), threading.Event()
    original = mode.compute_advice
    def block():
        entered.set(); assert release.wait(2)
    if callback == 'compute':
        def compute(st,**kw): block(); return original(st,**kw)
        mode.compute_advice = compute
        target=lambda:feed(mode,1)
    elif callback == 'native':
        api._sink._hud_callback=block
        target=api._sink._hud_callback
    else:
        def target():
            with locks[callback]:block()
    t=threading.Thread(target=target);t.start();assert entered.wait(1)
    try:
        start=time.monotonic();mode._input_changed('key',False)
        assert time.monotonic()-start < .1 and not api._buy_projection.valid
    finally:release.set();t.join(2)
    assert not t.is_alive()


def test_stop_fences_before_waiting_for_compute(running_api):
    api, ctx, bus, mode = running_api
    entered, release = threading.Event(), threading.Event()
    original=mode.compute_advice
    def compute(st,**kw): entered.set();assert release.wait(2);return original(st,**kw)
    mode.compute_advice=compute
    worker=threading.Thread(target=lambda:feed(mode,1));worker.start();assert entered.wait(1)
    stop=threading.Thread(target=mode.stop_event.set);stop.start()
    try:
        deadline=time.monotonic()+.1
        while not mode.stopped and time.monotonic()<deadline:time.sleep(.001)
        assert mode.stopped and not api._buy_projection.valid
    finally:release.set();worker.join(2);stop.join(2)
    assert not worker.is_alive() and not stop.is_alive()


def test_old_watcher_cannot_affect_stopped_or_replaced_session(running_api):
    api, ctx, bus, mode=running_api
    token=object();mode._buy_observer_token=token
    mode._input_changed('key',False,token=token);epoch=mode.buy_input_epoch
    mode._buy_observer_token=object();mode._input_changed('late',False,token=token)
    assert mode.buy_input_epoch==epoch
    mode.stop_event.set();epoch=mode.buy_input_epoch;mode._input_changed('late',False)
    assert mode.buy_input_epoch==epoch


def test_explicit_same_frame_clear_wins_and_preview_stays_paired(running_api):
    api, ctx, bus, mode=running_api
    old,preview,adv=api._buy_projection,api._preview_state,api._last_advice
    old=replace(old,source=replace(old.source,frame=api._last_state.buy_source.frame))
    clear=BuyProjection(replace(old.source,eligible=False))
    api._sink.on_buy_projection(clear);api._sink.on_buy_projection(old)
    api._sink.on_state(api._last_state);api._sink.on_advice(adv)
    assert api._buy_projection is clear and api._preview_state is preview


def test_event_bus_fence_rejects_late_events_and_survives_reattach():
    messages=[]; bus=EventBus(messages.append)
    source={'session':4,'input_epoch':2,'frame':10,'eligible':False}
    old={**source,'input_epoch':1,'frame':1000,'eligible':True}
    try:
        bus.emit('buy_projection',{'source':source,'valid':False,'buys':[]})
        for kind in ('state','advice','hud','status'):
            bus.emit(kind,{'buy_source':old})
        bus.emit('buy_projection',{'source':{**source,'eligible':True},'valid':True})
        # Fresh nested clear may still carry a scout notice on cached own state.
        bus.emit('hud',{'buy_source':old,'buy_projection':{'source':source,'valid':False},'scout_notice':{'id':'current'}})
        time.sleep(.15);bus.detach('main');messages.clear();bus.attach(messages.append)
        time.sleep(.15)
        decoded=[json.loads(x[len('window.__jcc_emit('):-1])for x in messages]
        assert decoded and all(x['type']=='buy_projection' for x in decoded)
        assert all(not x['payload']['valid'] for x in decoded)
    finally:bus.close()


def test_live_loop_monitor_start_stop_and_offline_never_starts_os(db, rules, monkeypatch):
    _, sink, offline = model(db, rules)
    monkeypatch.setattr('jccbrain.modes.buy_activity.HIDMonitor', lambda *a:(_ for _ in ()).throw(AssertionError('offline must not observe OS')))
    feed(offline)
    assert offline._buy_observer is None
    for _ in range(2):
        ctx, sink, mode = model(db, rules)
        m=ControlledInput(time.monotonic,WINDOW);mode._buy_monitor=m
        def eof(**kw):raise EOFError('controlled read')
        ctx.vision.read=eof
        with pytest.raises(EOFError):mode.loop()
        assert mode.stopped and m.starts==1 and m.stops==1
        assert not mode._buy_observer.status['running'] and not sink.projections[-1].valid


def test_held_and_fallback_fence_then_fresh_confirmation(running_api):
    api, ctx, bus, mode=running_api
    mode._input_changed('held',True)
    feed(mode,4);assert not api._buy_projection.valid
    mode._input_changed('fallback:unavailable',False)
    feed(mode,1);assert not api._buy_projection.valid
    feed(mode,1);assert api._buy_projection.valid
    mode._input_changed('foreground',False)
    feed(mode,1);assert not api._buy_projection.valid
    feed(mode,1);assert api._buy_projection.valid


def test_input_rejects_late_llm_reply_and_keeps_paired_preview(running_api):
    from jccbrain.engine.llm import LLMReply
    api,ctx,bus,mode=running_api
    old_state,old_advice,preview=api._last_state,api._last_advice,api._preview_state
    entered,release=threading.Event(),threading.Event()
    def answer(*a):entered.set();assert release.wait(2);return LLMReply('must not fill after input',[], '')
    ctx.llm.advise=answer
    t=threading.Thread(target=mode._ask_llm,args=(old_state,old_advice,mode.lifecycle.generation));t.start();assert entered.wait(1)
    mode._input_changed('key',False);release.set();t.join(2)
    assert not t.is_alive() and mode._llm is None and api._preview_state is preview
    assert not api._buy_projection.valid and 'must not fill' not in (api._last_advice.llm_text or '')


def test_provisional_same_frame_clear_cannot_downgrade_confirmation(running_api):
    api,ctx,bus,mode=running_api
    old=api._buy_projection
    api._sink.on_buy_projection(BuyProjection(old.source))
    assert api._buy_projection is old
    messages=[];events=EventBus(messages.append)
    from jccbrain.webui.serialize import buy_projection
    try:
        events.emit('buy_projection',buy_projection(old))
        events.emit('buy_projection',buy_projection(BuyProjection(old.source)))
        time.sleep(.15)
        out=[json.loads(x[len('window.__jcc_emit('):-1])for x in messages]
        assert len(out)==1 and out[0]['payload']['valid']
    finally:events.close()


def test_input_during_full_callback_does_not_confirm_rejected_strategy(running_api):
    api,ctx,bus,mode=running_api
    identity=mode._buy_identity;shown=mode._shown
    compute=mode.compute_advice
    def changed(st,**kw):
        adv=compute(st,**kw);adv.econ.reason='new economic identity';return adv
    mode.compute_advice=changed
    feed(mode,1)  # full strategy's unchanged two-compute gate
    entered,release=threading.Event(),threading.Event()
    original=mode.sink.on_advice
    def callback(adv):entered.set();assert release.wait(2);original(adv)
    mode.sink.on_advice=callback
    t=threading.Thread(target=lambda:feed(mode,1));t.start();assert entered.wait(1)
    mode._input_changed('key',False);release.set();t.join(2)
    assert not t.is_alive() and mode._buy_identity==identity and mode._shown==shown
    assert not api._buy_projection.valid
    mode.sink.on_advice=original
    feed(mode,1);assert not api._buy_projection.valid
    feed(mode,1);assert api._buy_projection.valid and mode._buy_identity!=identity


def test_cached_llm_is_tagged_to_its_input_epoch(running_api):
    from jccbrain.engine.llm import LLMReply
    api,ctx,bus,mode=running_api
    mode._llm=(api._last_state.stage,LLMReply('old epoch reply',[],''))
    mode._llm_input_epoch=mode.buy_input_epoch
    mode._input_changed('key',False)
    feed(mode,2)
    assert 'old epoch reply' not in (mode._latest.llm_text or '')


def test_raw_players_rank_and_derived_stable_rank_share_buy_semantics(running_api):
    api,ctx,bus,mode=running_api
    players=[{'name':'self','hp':75,'self':True,'slot':0},{'name':'other','hp':80,'slot':1}]
    for _ in range(5):mode._process_frame(raw(hp=75,players=players))
    assert api._last_state.hp_rank == (2,2)
    assert api._buy_projection.valid and api._buy_projection.buys
    source=api._buy_projection
    for i in range(4):mode._process_frame(raw(hp=75,players=players,timer=30-i))
    assert api._buy_projection is source
    mode._input_changed('key',False)
    mode._process_frame(raw(hp=75,players=players));assert not api._buy_projection.valid
    mode._process_frame(raw(hp=75,players=players));assert api._buy_projection.valid


def test_new_game_republication_orders_after_lifecycle_clear(running_api):
    api,ctx,bus,mode=running_api
    feed(mode,1,phase='ended')
    assert api._last_advice.phase.value=='ended'
    feed(mode,4,stage='1-2',phase='planning',level=3,gold=10,xp=[0,6])
    assert api._last_state.stage=='1-2' and api._last_advice.stage=='1-2'
    assert api._last_advice.phase.value=='planning' and api._buy_projection.valid


def test_confirmed_post_input_advice_is_freshly_paired_for_reattach(running_api):
    api,ctx,bus,mode=running_api
    old=api._last_advice;preview=api._preview_state
    mode._input_changed('key',False)
    feed(mode,1)
    assert api._last_advice is old and api._preview_state is preview
    feed(mode,1)
    assert api._buy_projection.valid
    assert api._last_advice.buy_source.input_epoch==mode.buy_input_epoch
    assert api._last_advice.buy_source.frame==api._preview_state.buy_source.frame
    assert api._last_advice.buy_source.input_epoch==api._preview_state.buy_source.input_epoch
