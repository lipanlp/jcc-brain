"""概率计算窗口缺省值不等于刷不到；合成边界走生产后处理。"""
from dataclasses import replace

import pytest

from jccbrain.core.models import GameState, Phase
from jccbrain.data.models import CompHero
from jccbrain.engine.advisor import Advice, HeuristicAdvisor, TargetOdds
from jccbrain.engine.comps import CompMatch
from jccbrain.engine.economy import EconAction, EconDecision
from jccbrain.engine.odds import PoolState, hit_odds, prob_with_gold


def case(db, rules, *, name='远古巨龙', level=6, gold=50, hp=80,
         xp=(31, 36), live=None, taken=0, need=3, action=EconAction.ROLL):
    hero = db.hero(name)
    comp = replace(db.comp('104'), heroes=[CompHero('target', name, hero.cost, is_carry=True)])
    state = GameState(stage='4-1', phase=Phase.PLANNING, level=level, gold=gold, hp=hp, xp=xp)
    pool = PoolState.from_db(db, rules)
    pool.taken[name] = taken
    odds = hit_odds(pool, name, level, need, live)
    adv = Advice(stage=state.stage, phase=state.phase, comps=[CompMatch(comp, .5, .5)],
                 econ=EconDecision(action, '基础运营', roll_budget=30),
                 targets=[TargetOdds(name, 0, need, 3 if need == 9 else 2, odds)])
    return HeuristicAdvisor(db, rules), state, adv, pool, live


@pytest.mark.parametrize('action', [EconAction.ROLL, EconAction.ALL_IN])
def test_positive_probability_outside_window_keeps_base_action(db, rules, action):
    advisor, state, adv, pool, live = case(db, rules, name='奥恩', level=8,
        need=9, live=[.01, .2, .3, .3, .19], action=action)
    row = adv.targets[0]
    assert row.odds.gold_for_50 is None and row.odds.slot_prob > 0
    assert row.odds.remaining >= row.need
    original = adv.econ
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ is original and adv.econ.action == action
    assert '按 30 金预算' in adv.econ.reason and '计算范围' in adv.econ.reason
    assert '刷不到' not in adv.econ.reason
    assert '当前等级基本刷不到' not in '\n'.join(adv.lines())
    chance = prob_with_gold(pool, row.name, row.need, state.level, 30, live)
    assert f'约 {int(chance * 10 + .5)} 成' in adv.econ.reason


@pytest.mark.parametrize('taken', [8, 9])
@pytest.mark.parametrize('action', [EconAction.ROLL, EconAction.ALL_IN])
def test_pool_shortage_does_not_level(db, rules, taken, action):
    advisor, state, adv, pool, live = case(db, rules, taken=taken, action=action)
    original = adv.econ
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ is original and adv.econ.action == action
    assert '按当前已知卡池' in adv.econ.reason
    assert '卡池不足' in '\n'.join(adv.lines())


def test_true_zero_can_level_with_affordable_rule_improvement(db, rules):
    advisor, state, adv, pool, live = case(db, rules)
    assert adv.targets[0].odds.slot_prob == 0
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.LEVEL and adv.econ.xp_buys == 2
    assert adv.econ.keep_gold == 42
    assert '下一等级按规则估计' in adv.econ.reason
    assert '当前出现概率为 0' in '\n'.join(adv.lines())


def test_true_zero_unaffordable_saves_cost_instead_of_negative_balance(db, rules):
    advisor, state, adv, pool, live = case(db, rules, gold=7)
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.SAVE and adv.econ.xp_buys == 0
    assert '8 金' in adv.econ.reason and '先攒' in adv.econ.reason


@pytest.mark.parametrize('supported', [False, True])
def test_zero_next_level_unsupported_or_not_improved_does_not_level(db, rules, supported):
    table = {6: [50, 30, 20, 0, 0]}
    if supported:
        table[7] = [50, 30, 20, 0, 0]
    rules = replace(rules, level_odds=table)
    advisor, state, adv, pool, live = case(db, rules)
    original = adv.econ
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ is original and adv.econ.action == EconAction.ROLL
    assert '当前出现概率为 0' in adv.econ.reason


@pytest.mark.parametrize('gold,xp,rules_missing', [(None, (31, 36), False), (50, None, True)])
def test_zero_unknown_funding_or_xp_does_not_immediately_level(db, rules, gold, xp, rules_missing):
    if rules_missing:
        rules = replace(rules, xp_to_next={})
    advisor, state, adv, pool, live = case(db, rules, gold=gold, xp=xp)
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action != EconAction.LEVEL


def test_zero_all_in_is_not_overridden(db, rules):
    advisor, state, adv, pool, live = case(db, rules, action=EconAction.ALL_IN)
    original = adv.econ
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ is original and adv.econ.action == EconAction.ALL_IN
    assert '当前出现概率为 0' in adv.econ.reason


def test_live_positive_overrides_zero_rule_probability(db, rules):
    advisor, state, adv, pool, live = case(db, rules, live=[0, 0, 0, 0, 1])
    assert rules.odds(6)[4] == 0 and adv.targets[0].odds.slot_prob > 0
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.ROLL


def test_live_zero_uses_next_rule_estimate_not_next_live_claim(db, rules):
    advisor, state, adv, pool, live = case(db, rules, level=8, live=[1, 0, 0, 0, 0])
    assert rules.odds(8)[4] > 0 and adv.targets[0].odds.slot_prob == 0
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.LEVEL
    assert '下一等级按规则估计' in adv.econ.reason


def test_zero_level_still_passes_interest_guard(db, rules):
    advisor, state, adv, pool, live = case(db, rules, gold=52)
    state.stage = '4-3'
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.LEVEL
    advisor._finish_econ(state, adv, 'fast8')
    assert adv.econ.action == EconAction.SAVE and '掉利息' in adv.econ.reason


def test_finite_budget_keeps_existing_budget_probability(db, rules):
    advisor, state, adv, pool, live = case(db, rules, level=8)
    adv.econ.roll_budget = 7
    assert adv.targets[0].odds.gold_for_50 is not None
    advisor._roll_odds(state, adv, pool, live)
    assert adv.econ.action == EconAction.ROLL
    assert '按 7 金预算约 0 成把握凑齐' in adv.econ.reason
