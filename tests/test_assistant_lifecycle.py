"""终局/无HUD守卫、原有观战隔离及真实线程发布边界。"""
from copy import deepcopy
import threading
from unittest.mock import Mock

import pytest

from conftest import make_state
from test_assistant import Ctx, AdviceSink
from jccbrain.core.models import GameState, Phase
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.llm import LLMReply
from jccbrain.modes.assistant import AssistantMode
from jccbrain.webui import serialize
from jccbrain.webui.events import WebSink
from jccbrain.actuator.layout import Layout
from jccbrain.core.settings import HudSettings


def frame(**kw):
    return GameState.from_dict(make_state(**kw))


def no_hud():
    # phase、单位、装备、商店都可能由reader保留，不能据此唤回对局。
    return frame(stage=None, level=None, gold=None, phase='combat',
                 board=[{'name': '奥恩'}], item_bench=[{'name': '无尽之刃'}],
                 shop={'open': True, 'slots': [{'name': '奥恩', 'cost': 1}]})


def assert_empty(advice):
    payload = serialize.advice(advice)
    for key in ('comps', 'buys', 'sells', 'augments', 'targets', 'items', 'item_plan',
                'augments_taken', 'opponents', 'notes'):
        assert payload[key] == []
    for key in ('target', 'board', 'econ', 'llm_text', 'ai_augment', 'comp_status', 'switch_reason'):
        assert payload[key] is None
    assert payload['held_items'] == {'components': {}, 'completed': [], 'emblems': []}
    assert payload['traits_view'] == {'current': [], 'final': []}
    return payload


def mode_for(db, rules):
    ctx = Ctx(db, rules, [])
    sink = AdviceSink()
    return ctx, sink, AssistantMode(ctx, sink)


@pytest.mark.parametrize('phase', ['ended', 'combat'])
def test_first_terminal_or_missing_hud_clears_without_updating_models(db, rules, phase):
    ctx, sink, mode = mode_for(db, rules)
    mode._process_frame(frame())
    mode.tracker.select = Mock(wraps=mode.tracker.select)
    mode.shop_stats.update = Mock(wraps=mode.shop_stats.update)
    mode._observe_opponent = Mock()
    raw = frame(phase='ended', placement=1) if phase == 'ended' else no_hud()
    before = deepcopy(raw.raw)
    assert mode._process_frame(raw) is None
    assert_empty(sink.advices[-1])
    assert sink.advices[-1].headline.startswith('对局结束' if phase == 'ended' else '等待有效对局画面')
    for _ in range(3):
        mode._process_frame(raw)
    mode.tracker.select.assert_not_called()
    mode.shop_stats.update.assert_not_called()
    mode._observe_opponent.assert_not_called()
    assert not mode.stabilizer.acc and raw.raw == before


@pytest.mark.parametrize('change', [
    {'gold': None}, {'level': None}, {'stage': None},
    {'phase': 'combat', 'gold': None, 'shop': {'open': False}},
    {'phase': 'augment', 'stage': None, 'gold': None, 'level': None,
     'augments': [{'name': '', 'desc': '获得一个装备锻造器'}]},
])
def test_partial_hud_closed_shop_and_real_augment_are_not_paused(db, rules, change):
    _, _, mode = mode_for(db, rules)
    assert mode._process_frame(frame(**change)) is not None
    assert mode.lifecycle.active


def test_empty_augment_placeholders_do_not_prove_hud(db, rules):
    _, sink, mode = mode_for(db, rules)
    mode._process_frame(frame(phase='augment', stage=None, gold=None, level=None,
                              augments=[{'name': ''}, {'name': ''}, {'name': ''}]))
    assert_empty(sink.advices[-1])
    assert not mode.lifecycle.active


def test_pause_recovers_same_stage_without_carrying_old_hud_or_llm(db, rules):
    _, sink, mode = mode_for(db, rules)
    mode._process_frame(frame(stage='4-1', gold=80))
    old_generation = mode.lifecycle.generation
    mode._llm = ('4-1', LLMReply('旧请求', [], ''))
    mode._process_frame(no_hud())
    assert mode.lifecycle.generation > old_generation
    result = mode._process_frame(frame(stage='4-1', gold=3))
    assert result and result[0].gold == 3
    assert result[1].llm_text is None and mode.lifecycle.active
    assert sink.advices[-1].stage == '4-1'


def test_ended_latches_until_three_credible_new_game_frames_and_resets_context(db, rules):
    ctx, sink, mode = mode_for(db, rules)
    ctx.settings.assistant.default_lock_comp_id = '104'
    mode._process_frame(frame(stage='6-3'))
    ctx.game_lock_id = ''
    ctx.game_watch_ids = ['104']
    ctx.game_augments = ['飞升']
    mode._process_frame(frame(stage='6-3', phase='ended', placement=1))
    for raw in [frame(stage='6-3'), frame(stage='1-2', level=9), no_hud()]:
        assert mode._process_frame(raw) is None
        assert_empty(sink.advices[-1])
    early = frame(stage='1-2', level=2, gold=3)
    assert mode._process_frame(early) is None
    assert mode._process_frame(early) is None
    # 无HUD打断连续确认，不能累计跨缺帧的三次信号。
    assert mode._process_frame(no_hud()) is None
    assert mode._process_frame(early) is None
    assert mode._process_frame(early) is None
    assert mode._process_frame(early) is not None
    assert ctx.game_watch_ids == [] and ctx.game_augments == [] and ctx.game_lock_id == '104'
    assert mode.lifecycle.active
    generation = mode.lifecycle.generation
    mode._process_frame(early)
    assert mode.lifecycle.generation == generation  # 不能连续重复新局重置。


def test_advisor_ended_returns_before_inventory_scout_or_matcher(db, rules, monkeypatch):
    advisor = HeuristicAdvisor(db, rules, scout=Mock())
    advisor.matcher.match = Mock(side_effect=AssertionError('不应匹配'))
    monkeypatch.setattr('jccbrain.engine.advisor.items.held', Mock(side_effect=AssertionError('不应读装备')))
    adv = advisor.advise(frame(phase='ended', placement=2, board=[{'name': '奥恩'}]))
    assert_empty(adv)
    advisor.scout.update.assert_not_called()
    assert adv.headline == '对局结束，第 2 名'


def test_foreign_ended_or_invalid_frames_never_enter_own_stabilizer(db, rules):
    _, sink, mode = mode_for(db, rules)
    mode._process_frame(frame(board=[{'name': '奥恩'}]))
    before = deepcopy(mode.stabilizer.acc)
    for raw in [frame(phase='ended', viewing={'slot': 1}),
                frame(phase='spectate', board=[{'name': '苍蓝哨戒'}])]:
        assert mode._process_frame(raw) is None
    assert mode.stabilizer.acc == before and mode.lifecycle.active
    assert not any(a.phase == Phase.ENDED for a in sink.advices)


@pytest.mark.parametrize('transition', ['pause', 'ended', 'same_stage_resume', 'new_game_same_stage', 'stop'])
def test_real_thread_late_llm_reply_cannot_publish_across_generation(db, rules, transition):
    ctx, sink, mode = mode_for(db, rules)
    state, adv, generation = mode._process_frame(frame(stage='1-2' if transition == 'new_game_same_stage' else '4-1'))
    entered, release = threading.Event(), threading.Event()
    def blocked(*_):
        entered.set()
        assert release.wait(2)
        return LLMReply('迟到请求', ['错误回填'], '')
    ctx.llm.advise = blocked
    worker = threading.Thread(target=mode._ask_llm, args=(state, adv, generation))
    worker.start()
    try:
        assert entered.wait(2)
        done = threading.Event()
        def transition_now():
            if transition == 'stop':
                mode.stop_event.set()
            else:
                mode._process_frame(frame(phase='ended') if transition in ('ended', 'new_game_same_stage') else no_hud())
                if transition == 'same_stage_resume':
                    mode._process_frame(frame(stage='4-1'))
                if transition == 'new_game_same_stage':
                    for _ in range(3):mode._process_frame(frame(stage='1-2', level=2, gold=3))
            done.set()
        guard = threading.Thread(target=transition_now)
        guard.start()
        assert done.wait(.5)  # 网络仍未放行，暂停/停止必须已返回。
        guard.join(1)
        count = len(sink.advices)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and len(sink.advices) == count
    assert not any(a.llm_text and '迟到' in a.llm_text for a in sink.advices)
    assert mode._llm is None


@pytest.mark.parametrize('transition', ['pause', 'stop'])
def test_publication_and_transition_are_serialized_in_real_threads(db, rules, transition):
    ctx, sink, mode = mode_for(db, rules)
    state, adv, generation = mode._process_frame(frame())
    entered, release, transition_done = threading.Event(), threading.Event(), threading.Event()
    original = sink.on_advice
    order = []
    def publish(value):
        if value.llm_text:
            entered.set()
            assert release.wait(2)
            order.append('reply')
        original(value)
    sink.on_advice = publish
    ctx.llm.advise = lambda *_: LLMReply('有效返回', [], '')
    worker = threading.Thread(target=mode._ask_llm, args=(state, adv, generation))
    worker.start()
    assert entered.wait(2)
    def transition_now():
        if transition == 'stop':mode.stop_event.set()
        else:mode._process_frame(no_hud())
        order.append('transition')
        transition_done.set()
    guard = threading.Thread(target=transition_now)
    guard.start()
    try:
        assert not transition_done.wait(.03)  # 已进入发布时，清空/停止不能越过它。
    finally:
        release.set()
        worker.join(2);guard.join(2)
    assert not worker.is_alive() and not guard.is_alive()
    assert order == ['reply', 'transition']
    if transition == 'pause':assert_empty(sink.advices[-1])


def test_websink_state_advice_and_hud_projection_clear_every_action(db, rules):
    ctx = Ctx(db, rules, [])
    events = []
    latest = {'state': None, 'advice': None}
    class Bus:
        def emit(self, kind, payload):events.append((kind, payload))
    bus = Bus()
    def hud():
        bus.emit('hud', serialize.hud(latest['state'], latest['advice'], db, Layout(), HudSettings()))
    sink = WebSink(bus, lambda:db, on_state=lambda value:latest.update(state=value),
                   on_advice=lambda value:latest.update(advice=value), hud_callback=hud)
    mode = AssistantMode(ctx, sink)
    mode._process_frame(frame(board=[{'name': '奥恩', 'x': .4, 'y': .4}]))
    mode._process_frame(no_hud())
    assert_empty(latest['advice'])
    for key in ('units', 'shop', 'augments', 'positions', 'arrows'):
        assert [p for kind,p in events if kind=='hud'][-1][key] == []
    assert latest['state'].gold is None and not latest['state'].board


def test_own_ended_clears_even_while_scout_is_awaiting_return(db, rules):
    _, sink, mode = mode_for(db, rules)
    mode._process_frame(frame())
    mode.scout = Mock(paused=True)
    mode._process_frame(frame(phase='ended', placement=1))
    assert sink.advices[-1].phase == Phase.ENDED
    assert_empty(sink.advices[-1])


@pytest.mark.parametrize('phase', ['unknown', 'combat'])
def test_no_hud_clears_while_scout_awaits_return_without_marking_own(db, rules, phase):
    _, sink, mode = mode_for(db, rules)
    mode._process_frame(frame())
    mode.scout = Mock(paused=True)
    raw = no_hud()
    raw.phase = Phase.parse(phase)
    raw.raw['phase'] = phase
    mode._process_frame(raw)
    assert_empty(sink.advices[-1])
    assert not mode.lifecycle.active
    mode.scout.mark_own.assert_not_called()
