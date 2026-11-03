"""视觉已选海克斯的消费链路和按运营曲线推导的利息豁免；不依赖实时识别。"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain.context import AppContext
from jccbrain.core.models import GameState, Phase
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor, Advice
from jccbrain.engine.economy import EconAction, EconDecision
from jccbrain.engine.llm import build_context
from jccbrain.engine.stabilizer import StateStabilizer
from jccbrain.modes.assistant import AssistantMode
from jccbrain.webui.api import Api
from jccbrain.webui import serialize


def test_from_dict_preserves_selected_augment_order_and_copies_list():
    raw = {'augments_taken': ['存心失利', '飞升']}
    state = GameState.from_dict(raw)
    assert state.augments_taken == ['存心失利', '飞升']
    state.augments_taken.append('新的')
    assert raw['augments_taken'] == ['存心失利', '飞升']


@pytest.mark.parametrize('raw', [{}, {'augments_taken': None}, {'augments_taken': []}, {'augments_taken': '飞升'}])
def test_missing_or_invalid_selected_augment_field_is_empty(raw):
    assert GameState.from_dict(raw).augments_taken == []
    assert GameState().augments_taken == []


def test_stabilizer_accumulates_without_reordering_or_mutating_input():
    stabilizer = StateStabilizer()
    source = make_state(stage='2-1', augments_taken=['飞升'])
    original = deepcopy(source)
    first = stabilizer.update(source)
    for raw in [make_state(stage='2-2'), make_state(stage=None, phase='unknown'),
                make_state(stage='3-2', augments_taken=['存心失利', '飞升', '存心失利', None, ''])]:
        out = stabilizer.update(raw)
    assert out['augments_taken'] == ['飞升', '存心失利']
    assert first['augments_taken'] == ['飞升'] and source == original
    assert stabilizer.update(make_state(stage='3-3', augments_taken=['飞升']))['augments_taken'] == ['飞升', '存心失利']


@pytest.mark.parametrize('gap', [False, True])
def test_stabilizer_new_game_resets_selected_augments(gap):
    stabilizer = StateStabilizer()
    stabilizer.update(make_state(stage='4-2', augments_taken=['飞升']))
    if gap:
        stabilizer.update(make_state(stage=None, phase='unknown'))
    assert stabilizer.update(make_state(stage='1-2', augments_taken=['存心失利']))['augments_taken'] == ['存心失利']
    assert stabilizer.update(make_state(stage='1-3'))['augments_taken'] == ['存心失利']
    stabilizer.reset()
    assert stabilizer.update(make_state(stage='1-3'))['augments_taken'] == []


def test_ended_before_stage_two_uses_existing_three_frame_new_game_rule():
    stabilizer = StateStabilizer()
    stabilizer.update(make_state(stage='1-3', augments_taken=['飞升']))
    stabilizer.update(make_state(stage=None, phase='ended'))
    for _ in range(2):
        assert stabilizer.update(make_state(stage='1-1'))['augments_taken'] == ['飞升']
    assert stabilizer.update(make_state(stage='1-1'))['augments_taken'] == []


def test_advisor_normalizes_vision_names_and_keeps_bonus_and_llm(db, rules):
    advisor = HeuristicAdvisor(db, rules)
    state = GameState(stage='2-3', augments_taken=[' 飞 升 ', '飞升3', '存心失利额外文字'])
    base = advisor.advise(replace(state, augments_taken=[]))
    advice = advisor.advise(state)
    assert advice.augments_taken == ['飞升', '存心失利']
    assert advice.target.score - base.target.score == pytest.approx(.03)
    assert '海克斯 飞升 适配（均名 -0.30）' in advice.target.why
    assert build_context(state, advice)['augments_taken'] == ['飞升', '存心失利']
    assert serialize.advice(advice)['augments_taken'] == ['飞升', '存心失利']
    assert serialize.state(state, db)['augments_taken'] == state.augments_taken
    json.dumps(serialize.state(state, db))


@pytest.fixture
def automatic_mode(db, rules, tmp_path, monkeypatch):
    monkeypatch.setenv('JCC_HOME', str(tmp_path))
    ctx = AppContext(Settings())
    ctx.db, ctx.rules = db, rules
    ctx.settings.save = Mock()
    bus = SimpleNamespace(emit=Mock())
    api = Api(ctx, bus)
    yield AssistantMode(ctx, api._sink), ctx, api, bus
    api._shutdown()


def test_mode_vision_to_readonly_status_and_new_game(automatic_mode):
    mode, ctx, api, bus = automatic_mode
    for names in (['飞 升'], ['飞升3'], [], ['存心失利']):
        state = GameState(stage='2-5', augments_taken=names)
        advice = mode.compute_advice(state)
    assert advice.augments_taken == ctx.game_augments == ['飞升', '存心失利']
    assert api.status()['data']['augments_taken'] == ['飞升', '存心失利']
    status_events = [call.args[1]['augments_taken'] for call in bus.emit.call_args_list if call.args[0]=='status']
    assert status_events == [[], ['飞升'], ['飞升', '存心失利']]
    assert not hasattr(api, 'mark_augment') and not hasattr(api, 'unmark_augment')
    for _ in range(3): advice = mode.compute_advice(GameState(stage='1-2'))
    assert ctx.game_augments == advice.augments_taken == []
    assert api.status()['data']['augments_taken'] == []
    ctx.settings.save.assert_not_called()


def test_stabilizer_and_mode_do_not_reintroduce_previous_game(automatic_mode):
    mode, ctx, api, _ = automatic_mode
    for raw in [make_state(stage='5-2', augments_taken=['飞升'])] + [make_state(stage='1-2')]*4:
        state = GameState.from_dict(mode.stabilizer.update(raw))
        advice = mode.compute_advice(state)
    assert ctx.game_augments == advice.augments_taken == []
    assert mode.stabilizer.acc['augments_taken'] == []
    assert api.status()['data']['augments_taken'] == []


@pytest.mark.parametrize('style,stage,expected', [
    ('fast8', '3-5', EconAction.LEVEL), ('reroll', '3-5', EconAction.SAVE),
    ('standard', '3-5', EconAction.SAVE), ('fast9', '3-5', EconAction.LEVEL),
    ('fast8', '2-5', EconAction.LEVEL), ('reroll', '2-5', EconAction.LEVEL),
    ('fast8', '4-2', EconAction.SAVE), ('standard', '4-2', EconAction.LEVEL),
    ('fast8', '5-1', EconAction.LEVEL), ('standard', '5-2', EconAction.LEVEL),
])
def test_critical_rounds_follow_current_style_curve(db, rules, style, stage, expected):
    advisor = HeuristicAdvisor(db, rules)
    state = GameState(stage=stage, level=6, hp=80, gold=52, xp=(29,36))
    advice = Advice(stage, Phase.PLANNING, econ=EconDecision(EconAction.LEVEL, '升级', xp_buys=2))
    advisor._finish_econ(state, advice, style)
    assert advice.econ.action == expected
    if expected == EconAction.SAVE:
        assert '非关键回合' in advice.econ.reason and advice.econ.xp_buys == 0


@pytest.mark.parametrize('style,stage,expected', [
    ('fast8', '3-6', EconAction.LEVEL), ('standard', '3-4', EconAction.LEVEL),
    ('fast8', '2-1', EconAction.SAVE), ('fast9', '5-1', EconAction.LEVEL),
])
def test_critical_rounds_use_loaded_rule_keys_instead_of_hardcoded_list(db, rules, style, stage, expected):
    changed = replace(rules, fast8_levels={'3-6':7}, standard_levels={'3-4':6})
    advisor = HeuristicAdvisor(db, changed)
    state = GameState(stage=stage, level=5, hp=80, gold=52, xp=(13,20))
    advice = Advice(stage, Phase.PLANNING, econ=EconDecision(EconAction.LEVEL, '升级', xp_buys=2))
    advisor._finish_econ(state, advice, style)
    assert advice.econ.action == expected
