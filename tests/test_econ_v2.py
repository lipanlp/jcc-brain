"""成型运营与所有升级来源的利息保护，使用合成阵容。"""
from dataclasses import replace

import pytest

from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.data.models import Guide
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher
from jccbrain.engine.economy import EconAction, EconDecision


def advisor_for(db, rules, style):
    base = db.comp('104')
    if style != 'reroll':
        base = replace(base, guide=Guide('g', '95' if style == 'fast9' else '速8', None, None, d_time='上8'))
    advisor = HeuristicAdvisor(db, rules)
    advisor.matcher = CompMatcher([base], db)
    return advisor, base


@pytest.mark.parametrize('stage,style,level,gold,action,reason', [
    ('4-1', 'fast8', 8, 80, EconAction.LEVEL, '存钱上 9 找 5 费'),
    ('4-2', 'fast8', 8, 80, EconAction.LEVEL, '存钱上 9 找 5 费'),
    ('4-2', 'fast8', 8, 2, EconAction.SAVE, '存钱上 9 找 5 费'),
    ('4-2', 'reroll', 8, 80, EconAction.SAVE, '存钱升人口'),
    ('4-2', 'fast9', 9, 80, EconAction.SAVE, '保持经济'),
])
def test_completed_comp_economy(db, rules, stage, style, level, gold, action, reason):
    advisor, comp = advisor_for(db, rules, style)
    state = GameState(stage=stage, phase=Phase.PLANNING, level=level, gold=gold, hp=80, xp=(78, 80),
                      board=[Unit(h.name, star=3) for h in comp.heroes])
    advice = advisor.advise(state)
    assert all(t.need == 0 for t in advice.targets)
    assert advice.econ.action == action and advice.econ.reason.startswith('阵容已成型')
    assert reason in advice.econ.reason
    if action == EconAction.LEVEL:
        assert advice.econ.xp_buys == 1


def test_completed_comp_does_not_override_emergency_all_in(db, rules):
    advisor, comp = advisor_for(db, rules, 'fast8')
    adv = advisor.advise(GameState(stage='4-2', level=8, gold=30, hp=30,
                        board=[Unit(h.name, star=3) for h in comp.heroes]))
    assert adv.econ.action == EconAction.ALL_IN and adv.econ.roll_budget == 30
    assert '阵容已成型' not in adv.econ.reason


@pytest.mark.parametrize('stage,hp,gold,style,expected', [
    ('2-3', 80, 52, 'fast8', EconAction.SAVE),
    ('2-1', 80, 52, 'fast8', EconAction.LEVEL),
    ('3-2', 80, 52, 'fast8', EconAction.LEVEL),
    ('4-1', 80, 52, 'fast8', EconAction.LEVEL),
    ('4-2', 80, 52, 'fast8', EconAction.SAVE),
    ('2-3', 49, 52, 'fast8', EconAction.LEVEL),
    ('2-3', 80, 59, 'fast8', EconAction.LEVEL),
    ('5-1', 80, 52, 'fast9', EconAction.LEVEL),
    ('5-1', 80, 52, 'fast8', EconAction.LEVEL),
])
def test_interest_guard_and_critical_rounds(db, rules, monkeypatch, stage, hp, gold, style, expected):
    advisor, _ = advisor_for(db, rules, style)
    monkeypatch.setattr('jccbrain.engine.economy.decide', lambda *args: EconDecision(EconAction.LEVEL, '原升级', xp_buys=2))
    adv = advisor.advise(GameState(stage=stage, level=6, gold=gold, hp=hp, xp=(29, 36)))
    assert adv.econ.action == expected
    if expected == EconAction.SAVE:
        assert adv.econ.reason == '差 7 经验，非关键回合，不为升级掉利息'
        assert adv.econ.xp_buys == adv.econ.roll_budget == 0


def test_incomplete_targets_continue_rolling(db, rules):
    advisor, _ = advisor_for(db, rules, 'fast8')
    adv = advisor.advise(GameState(stage='4-2', level=8, gold=50, hp=80))
    assert any(t.need > 0 for t in adv.targets) and adv.econ.action == EconAction.ROLL


def test_completed_level_is_still_subject_to_interest_guard(db, rules):
    advisor, comp = advisor_for(db, rules, 'fast8')
    adv = advisor.advise(GameState(stage='4-3', level=8, gold=52, hp=80, xp=(73, 80),
                         board=[Unit(h.name, star=3) for h in comp.heroes]))
    assert adv.econ.action == EconAction.SAVE
    assert adv.econ.reason.startswith('阵容已成型：差 7 经验')


def test_probability_generated_level_also_protects_interest(db, rules, monkeypatch):
    from test_rules_v2 import expensive_advisor
    advisor = expensive_advisor(db, rules)
    monkeypatch.setattr('jccbrain.engine.economy.decide', lambda *args: EconDecision(EconAction.ROLL, 'D', roll_budget=20))
    adv = advisor.advise(GameState(stage='4-3', level=6, gold=52, hp=80, xp=(31, 36)))
    assert adv.econ.action == EconAction.SAVE and '不为升级掉利息' in adv.econ.reason


@pytest.mark.parametrize('cap,gold,buys,expected', [
    (5, 68, 4, EconAction.LEVEL),  # 68→52，实际利息仍为5
    (5, 80, 1, EconAction.LEVEL),
    (5, 62, 4, EconAction.SAVE),   # 62→46，实际利息5→4
    (5, 52, 2, EconAction.SAVE),
    (3, 48, 3, EconAction.LEVEL),  # 自定义封顶，48→36仍为3
    (3, 38, 3, EconAction.SAVE),
    (7, 88, 4, EconAction.LEVEL),  # 不把5硬编码为利息上限
    (7, 68, 4, EconAction.SAVE),
])
def test_finish_econ_uses_rule_interest_cap(db, rules, cap, gold, buys, expected):
    from jccbrain.engine.advisor import Advice
    advisor = HeuristicAdvisor(db, replace(rules, max_interest=cap))
    state = GameState(stage='3-3', level=7, gold=gold, hp=80, xp=(2, 56))
    adv = Advice(stage=state.stage, phase=state.phase,
                 econ=EconDecision(EconAction.LEVEL, '升级', xp_buys=buys))
    advisor._finish_econ(state, adv, 'fast8')
    assert adv.econ.action == expected
    assert adv.econ.xp_buys == (buys if expected == EconAction.LEVEL else 0)


def test_production_economy_above_interest_cap_is_not_blocked(db, rules):
    from jccbrain.engine.advisor import Advice
    from jccbrain.engine.economy import decide
    state = GameState(stage='3-3', level=7, gold=68, hp=80, xp=(2, 56))
    econ = decide(state, rules, 'fast8', 4)
    assert econ.action == EconAction.LEVEL and econ.xp_buys == 4
    assert rules.interest(68) == rules.interest(68 - econ.xp_buys * rules.xp_cost)
    adv = Advice(stage=state.stage, phase=state.phase, econ=econ)
    HeuristicAdvisor(db, rules)._finish_econ(state, adv, 'fast8')
    assert adv.econ is econ


@pytest.mark.parametrize('stage,hp,action', [
    ('4-1', 80, EconAction.LEVEL),
    ('4-3', 49, EconAction.LEVEL),
    ('4-3', 30, EconAction.ALL_IN),
])
def test_interest_fix_preserves_critical_low_hp_and_emergency(db, rules, stage, hp, action):
    from jccbrain.engine.advisor import Advice
    state = GameState(stage=stage, level=7, gold=52, hp=hp, xp=(2, 56))
    econ = EconDecision(action, '原决策', xp_buys=2 if action == EconAction.LEVEL else 0,
                        roll_budget=52 if action == EconAction.ALL_IN else 0)
    adv = Advice(stage=stage, phase=state.phase, econ=econ)
    HeuristicAdvisor(db, rules)._finish_econ(state, adv, 'fast8')
    assert adv.econ is econ
