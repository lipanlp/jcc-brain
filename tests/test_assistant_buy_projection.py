"""Production raw pipeline and API projection safety, independent of strategy gate."""
from copy import deepcopy
from dataclasses import replace
import socket
from unittest.mock import Mock

import pytest

from conftest import make_state
from test_assistant import Ctx, AdviceSink
from test_webui_api import setup_api, FakeHud
from jccbrain.core.models import GameState, Phase
from jccbrain.engine.advisor import Advice, BuyAdvice, HeuristicAdvisor
from jccbrain.modes.assistant import AssistantMode
from jccbrain.webui import serialize
from jccbrain.actuator.layout import Layout
from jccbrain.core.settings import HudSettings


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('offline only')))


class Sink(AdviceSink):
    def __init__(self):
        super().__init__()
        self.projections = []
    def on_buy_projection(self, projection):
        self.projections.append(projection)


def raw(**changes):
    changes.setdefault('shop', {'open': True, 'slots': [{'name': '绯红树怪', 'cost': 1}, {'name': '奥恩', 'cost': 1}]})
    return GameState.from_dict(make_state(**changes))


def model(db, rules):
    ctx = Ctx(db, rules, [])
    ctx.game_lock_id = '104'
    sink = Sink()
    return ctx, sink, AssistantMode(ctx, sink)


def feed(mode, n=4, **changes):
    for _ in range(n):
        result = mode._process_frame(raw(**changes))
    return result


def test_equipment_churn_does_not_starve_confirmed_buys(db, rules):
    _, sink, mode = model(db, rules)
    feed(mode)
    full = len(sink.advices)
    shop = {'open': True, 'slots': [None, None, None, {'name': '绯红树怪', 'cost': 1}]}
    for i in range(12):
        mode._process_frame(raw(shop=shop, item_bench=[{'name': '无尽之刃'}] if i % 2 else []))
        if i == 1:
            assert not sink.projections or not sink.projections[-1].valid
        if i >= 2:
            assert sink.projections and sink.projections[-1].valid
            assert [(b.slot, b.name) for b in sink.projections[-1].buys] == [(3, '绯红树怪')]
    assert len(sink.advices) == full  # entire strategy never became stable


def test_serializer_cannot_apply_old_slot_reason_to_new_card(db, rules):
    st = raw()
    adv = HeuristicAdvisor(db, rules).advise(st, '104')
    adv.buys = [BuyAdvice(0, '绯红树怪', 1, '旧目标卡', 4)]
    st.shop[0].name = '奥恩'
    result = serialize.hud(st, adv, db, Layout(), HudSettings())
    assert not any(s['key'] == 0 and s['kind'] == 'buy' for s in result['shop'])


def test_single_computation_noise_never_publishes(db, rules):
    _, sink, mode = model(db, rules)
    feed(mode)
    original = mode.compute_advice
    def noise(st, **kwargs):
        adv = original(st, **kwargs)
        adv.buys = [replace(b, reason='one-frame-noise', priority=b.priority + .1) for b in adv.buys]
        return adv
    mode.compute_advice = noise
    feed(mode, 1)
    mode.compute_advice = original
    feed(mode, 2)
    assert not any(b.reason == 'one-frame-noise' for p in sink.projections for b in p.buys)


def test_single_raw_shop_noise_revokes_until_reconfirmed(db, rules):
    _, sink, mode = model(db, rules)
    feed(mode)
    before = sink.projections[-1]
    result = feed(mode, 1, shop={'open': True, 'slots': [{'name': '远古巨龙', 'cost': 1}, {'name': '奥恩', 'cost': 1}]})
    assert result[0].shop[0].name == '绯红树怪'
    assert before.valid and not sink.projections[-1].valid
    feed(mode, 1)
    assert not sink.projections[-1].valid
    feed(mode, 1)
    assert sink.projections[-1].valid


def test_timer_coordinates_and_items_do_not_change_buy_fingerprint(db, rules):
    from jccbrain.core.buy_source import buy_fingerprint
    a = raw(board=[{'name': '奥恩', 'star': 2, 'x': .2, 'y': .4}])
    b = raw(timer=3, board=[{'name': '奥恩', 'star': 2, 'x': .7, 'y': .6, 'items': [{'name': '无尽之刃'}]}],
            item_bench=[{'name': '暴风大剑'}])
    assert buy_fingerprint(a) == buy_fingerprint(b)


@pytest.mark.parametrize('changes', [
    {'gold': 0}, {'level': 7}, {'stage': '4-1'}, {'phase': 'combat'},
    {'hp_rank': [4, 4]}, {'odds': [10, 20, 30, 30, 10]}, {'players_stale': True},
    {'shop': {'open': False}},
    {'shop': {'open': True, 'stale': True, 'slots': [{'name': '绯红树怪', 'cost': 1}]}},
    {'units_stale': True}, {'shop': {'open': True, 'slots': [None]}},
    {'shop': {'open': True, 'slots': [{'name': '绯红树怪', 'cost': 2}]}},
    {'shop': {'open': True, 'slots': [{'name': '绯红树怪', 'cost': 1, 'upgrade': 2}]}},
    {'shop': {'open': True, 'slots': [{'name': '绯红树怪', 'cost': 1, 'star': 2}]}},
    {'board': [{'name': '绯红树怪', 'star': 3}]},
])
def test_source_rejects_changed_dependencies_even_before_new_advice(db, rules, changes):
    from jccbrain.core.buy_source import source_matches_state
    _, sink, mode = model(db, rules)
    st, adv, _ = feed(mode)
    projection = sink.projections[-1]
    assert projection.valid and source_matches_state(projection.source, st)
    changed = raw(**changes)
    changed.buy_source = st.buy_source
    assert not source_matches_state(projection.source, changed)
    assert serialize.hud(changed, adv, db, Layout(), HudSettings(), projection=projection)['shop'] == []


@pytest.fixture
def running_api(setup_api):
    api, ctx, bus = setup_api
    api._icons.close(); api._icons.prefetch = lambda *a: None
    hud = FakeHud(); hud.status = {'overlay': True, 'panel': True}
    api._hud = hud
    ctx.game_lock_id = '104'
    mode = AssistantMode(ctx, api._sink)
    api._runner.mode = mode
    feed(mode)
    yield api, ctx, bus, mode
    api._runner.mode = None


@pytest.mark.parametrize('change', [
    {'phase': 'ended'}, {'stage': None, 'gold': None, 'level': None},
    {'phase': 'spectate', 'viewing': {'self': False, 'slot': 2}},
    {'shop': {'open': False}}, {'units_stale': True},
])
def test_invalid_raw_clears_before_scout_callback_hud(running_api, change):
    api, ctx, bus, mode = running_api
    original = mode.scout.observe
    def refresh(st):
        api._emit_hud()
        return original(st)
    mode.scout.observe = refresh
    bus.events.clear()
    mode._process_frame(raw(**change))
    assert not any(p['shop'] for k, p in bus.events if k == 'hud')
    if change.get('phase') == 'spectate':
        assert api._last_state.phase != Phase.SPECTATE
    old = api._last_advice
    api._refresh_hud()
    assert not [p for k, p in bus.events if k == 'hud'][-1]['shop']


def test_on_state_clears_old_highlights_and_late_projection_cannot_revive(running_api):
    api, ctx, bus, mode = running_api
    old_projection = api._buy_projection
    old_advice = api._last_advice
    old_state = api._last_state
    bus.events.clear()
    feed(mode, 2, shop={'open': True, 'slots': [None, {'name': '绯红树怪', 'cost': 1}]})
    assert not api._buy_projection.valid
    assert not [p for k, p in bus.events if k == 'hud'][-1]['shop']
    api._sink.on_buy_projection(old_projection)
    api._sink.on_state(old_state)
    api._sink.on_advice(old_advice)  # cannot replace projection even if full advice is eligible
    api._refresh_hud()
    assert not api._buy_projection.valid
    feed(mode, 1, shop={'open': True, 'slots': [None, {'name': '绯红树怪', 'cost': 1}]})
    assert api._buy_projection.valid and api._buy_projection.buys[0].slot == 1
    newest = api._buy_projection
    api._sink.on_buy_projection(old_projection)
    api._sink.on_advice(old_advice)
    api._refresh_hud()
    assert api._buy_projection is newest
    assert all(p['key'] == 1 for p in [p for k, p in bus.events if k == 'hud'][-1]['shop'])


def test_spectate_return_preserves_strategy_gate_and_recovers_buys(running_api):
    api, ctx, bus, mode = running_api
    full = api._last_advice
    feed(mode, 1, phase='spectate', viewing={'self': False})
    assert not api._buy_projection.valid
    feed(mode, 2)
    assert api._buy_projection.valid and api._last_advice is full


def test_selection_stop_and_new_mode_reject_old_generation(running_api):
    api, ctx, bus, mode = running_api
    old_projection, old_advice = api._buy_projection, api._last_advice
    ctx.game_lock_id = ''
    ctx.game_watch_ids = ['104']
    mode.refresh_selection()
    assert not api._buy_projection.valid
    api._sink.on_buy_projection(old_projection); api._sink.on_advice(old_advice)
    assert not api._buy_projection.valid
    feed(mode, 3)
    before = api._buy_projection
    mode.stop_event.set()
    api._sink.on_buy_projection(before); api._sink.on_advice(api._last_advice)
    assert not api._buy_projection or not api._buy_projection.valid
    next_mode = AssistantMode(ctx, api._sink); api._runner.mode = next_mode
    feed(next_mode, 3)
    newest = api._buy_projection
    api._sink.on_buy_projection(before)
    assert api._buy_projection is newest


def test_dynamic_buys_do_not_repair_full_preview_with_unpaired_state(running_api):
    api, ctx, bus, mode = running_api
    preview_state, full = deepcopy(api._preview_state), api._last_advice
    shop = {'open': True, 'slots': [None, None, None, {'name': '绯红树怪', 'cost': 1}]}
    for i in range(3):
        feed(mode, 1, shop=shop, item_bench=[{'name': '无尽之刃'}] if i % 2 else [])
    assert api._buy_projection.valid
    assert api._last_advice is full and api._preview_state == preview_state


def test_changed_economic_identity_waits_for_full_confirmation(db, rules):
    _, sink, mode = model(db, rules)
    feed(mode)
    original = mode.compute_advice
    def changed(st, **kwargs):
        adv = original(st, **kwargs)
        adv.econ.reason = 'new strategy identity'
        return adv
    mode.compute_advice = changed
    full = len(sink.advices)
    for i in range(8):
        feed(mode, 1, item_bench=[{'name': '无尽之刃'}] if i % 2 else [])
        assert not sink.projections[-1].valid
    assert len(sink.advices) == full
    feed(mode, 2, item_bench=[])
    assert len(sink.advices) > full and sink.projections[-1].valid


def test_late_llm_fills_text_without_replacing_dynamic_buys(running_api):
    import threading
    from jccbrain.engine.llm import LLMReply
    api, ctx, bus, mode = running_api
    old_state, old_advice = api._last_state, api._last_advice
    entered, release = threading.Event(), threading.Event()
    def answer(*args):
        entered.set()
        assert release.wait(2)
        return LLMReply('controlled reply', [], '')
    ctx.llm.advise = answer
    thread = threading.Thread(target=mode._ask_llm, args=(old_state, old_advice, mode.lifecycle.generation))
    thread.start(); assert entered.wait(2)
    shop = {'open': True, 'slots': [None, None, {'name': '绯红树怪', 'cost': 1}]}
    feed(mode, 3, shop=shop)
    before = api._buy_projection
    release.set(); thread.join(2); assert not thread.is_alive()
    api._refresh_hud()
    assert api._buy_projection is before and before.valid and before.buys[0].slot == 2
    assert all(p['key'] == 2 for p in [p for k, p in bus.events if k == 'hud'][-1]['shop'])


def test_wrong_season_and_generation_are_rejected_even_with_matching_shop(running_api):
    api, ctx, bus, mode = running_api
    before = api._buy_projection
    for source in [replace(before.source, season='other'),
                   replace(before.source, generation=before.source.generation + 1),
                   replace(before.source, selection='outdated')]:
        api._sink.on_buy_projection(replace(before, source=source))
        assert api._buy_projection is before


def test_real_eventbus_coalesces_projection_without_losing_state_barrier(running_api):
    import json
    import threading
    from jccbrain.webui.events import EventBus, WebSink
    api, ctx, _, mode = running_api
    delivered, ready = [], threading.Event()
    def callback(js):
        delivered.append(json.loads(js[len('window.__jcc_emit('):-1]))
        if any(e['type'] == 'hud' for e in delivered): ready.set()
    bus = EventBus(callback)
    try:
        api._bus = bus
        api._sink = WebSink(bus, api._db, on_state=api._on_state, on_advice=api._on_advice,
                            on_buy_projection=api._on_buy_projection, hud_callback=api._emit_hud)
        mode.sink = api._sink
        old = api._buy_projection
        feed(mode, 3, shop={'open': True, 'slots': [None, None, {'name': '绯红树怪', 'cost': 1}]})
        api._sink.on_buy_projection(old)
        assert ready.wait(2)
        projection = [e['payload'] for e in delivered if e['type'] == 'buy_projection'][-1]
        hud = [e['payload'] for e in delivered if e['type'] == 'hud'][-1]
        assert projection['valid'] and [b['slot'] for b in projection['buys']] == [2]
        assert [s['key'] for s in hud['shop']] == [2]
    finally:
        bus.close()


def test_old_own_state_cannot_lower_spectate_clear_barrier(running_api):
    api, ctx, bus, mode = running_api
    old_state, old_projection = api._last_state, api._buy_projection
    feed(mode, 1, phase='spectate', viewing={'self': False})
    barrier = api._buy_projection
    api._sink.on_state(old_state)
    api._sink.on_buy_projection(old_projection)
    api._refresh_hud()
    assert api._buy_projection is barrier and not barrier.valid
    assert not [p for k, p in bus.events if k == 'hud'][-1]['shop']
