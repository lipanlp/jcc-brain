"""概率预算、卡池与卖卡保护；全部使用合成数据。"""
from dataclasses import replace

import pytest

from jccbrain.core.models import GameState, Phase, Unit, TraitStatus
from jccbrain.data.models import CompHero
from jccbrain.engine.advisor import Advice, BuyAdvice, HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher, CompMatch
from jccbrain.engine.economy import EconAction, EconDecision
from jccbrain.engine.odds import PoolState, hit_curve, hit_odds, prob_with_gold
from jccbrain.engine.tracker import CompTracker


def test_gold_probability_monotonic_and_same_spending_as_hit_odds(db, rules):
    pool = PoolState.from_db(db, rules)
    values = [prob_with_gold(pool, '奥恩', 3, 5, gold) for gold in range(50)]
    assert all(0 <= a <= b <= 1 for a, b in zip(values, values[1:]))
    assert values[0] == 0
    assert prob_with_gold(pool, '奥恩', 3, 5, 23) == hit_curve(pool, '奥恩', 3, 5, max_rolls=10)[10]
    odds = hit_odds(pool, '奥恩', 5, 3)
    assert prob_with_gold(pool, '奥恩', 3, 5, odds.gold_for_50) >= .5
    assert prob_with_gold(pool, '奥恩', 3, 5, odds.gold_for_50 - 1) < .5


@pytest.mark.parametrize('name,need,level,gold,expected', [
    ('奥恩', 0, 5, 0, 1), ('奥恩', 1, 5, -1, 0), ('奥恩', 1, 5, 2, 0),
    ('远古巨龙', 1, 6, 100, 0), ('未知', 1, 5, 100, 0),
])
def test_probability_boundaries(db, rules, name, need, level, gold, expected):
    assert prob_with_gold(PoolState.from_db(db, rules), name, need, level, gold) == expected


def test_probability_pool_exhaustion_and_live_odds(db, rules):
    pool = PoolState.from_db(db, rules)
    pool.taken['奥恩'] = 29
    assert prob_with_gold(pool, '奥恩', 2, 5, 200) == 0
    assert prob_with_gold(pool, '奥恩', 1, 5, 100, [0, 0, 0, 0, 1]) == 0


def expensive_advisor(db, rules, subcarry=False):
    comp = replace(db.comp('104'), heroes=[CompHero('dragon', '远古巨龙', 5,
                   is_carry=not subcarry, is_subcarry=subcarry, star_target=2)])
    advisor = HeuristicAdvisor(db, rules)
    advisor.matcher = CompMatcher([comp])
    return advisor


@pytest.mark.parametrize('subcarry', [False, True])
def test_unavailable_carry_levels_up(db, rules, monkeypatch, subcarry):
    advisor = expensive_advisor(db, rules, subcarry)
    monkeypatch.setattr('jccbrain.engine.economy.decide', lambda *args: EconDecision(EconAction.ROLL, 'D', roll_budget=30))
    adv = advisor.advise(GameState(stage='4-1', phase=Phase.PLANNING, level=6, xp=(31, 36), gold=50))
    assert adv.econ.action == EconAction.LEVEL and adv.econ.xp_buys == 2
    assert '当前 6 级刷不到 远古巨龙（5 费）' in adv.econ.reason


def test_unavailable_carry_does_not_override_all_in(db, rules):
    adv = expensive_advisor(db, rules).advise(GameState(stage='4-1', level=6, hp=20, gold=30))
    assert adv.econ.action == EconAction.ALL_IN and adv.econ.roll_budget == 30
    assert '远古巨龙 当前出现概率为 0' in adv.econ.reason


def test_short_budget_explains_probability(db, rules, monkeypatch):
    monkeypatch.setattr('jccbrain.engine.economy.decide', lambda *args: EconDecision(EconAction.ROLL, 'D', roll_budget=7))
    adv = expensive_advisor(db, rules).advise(GameState(stage='4-2', level=8, gold=30))
    assert adv.econ.action == EconAction.ROLL
    assert '按 7 金预算约 0 成把握凑齐 远古巨龙' in adv.econ.reason


@pytest.mark.parametrize('star,remaining,penalty', [(3, 5, 0), (3, 4, .1), (2, 3, 0), (2, 2, .1), (3, 0, .2)])
def test_carry_pool_thresholds(db, rules, star, remaining, penalty):
    comp = replace(db.comp('104'), heroes=[replace(db.comp('104').heroes[0], star_target=star)])
    matcher = CompMatcher([comp])
    state = GameState()
    before = matcher.match(state)[0]
    pool = PoolState.from_db(db, rules)
    pool.taken['绯红树怪'] = rules.pool_size[1] - remaining
    after = matcher.match(state, pool=pool)[0]
    assert before.score - after.score == pytest.approx(penalty)
    assert any(f'剩 {remaining} 张（按自己持有与手动登记估算）' in why for why in after.why) == bool(penalty)


def test_advisor_pool_counts_self_and_manual_taken_and_refreshes_candidates(db, rules):
    advisor, tracker = HeuristicAdvisor(db, rules), CompTracker()
    state = GameState(stage='2-1', board=[Unit('绯红树怪')])
    before = advisor.advise(state, tracker=tracker).target
    after = advisor.advise(state, extra_taken={'绯红树怪': 25}, tracker=tracker).target
    assert before.score - after.score == pytest.approx(.1)
    assert any('剩 4 张' in why for why in after.why)


def unrelated_state(db, size=7):
    db.heroes_by_name['孤卡'] = replace(db.hero('奥恩'), name='孤卡', traits=('无关羁绊',))
    return GameState(bench=[Unit('孤卡', zone='bench', slot=0)] +
                           [Unit(f'?{i}', zone='bench', slot=i) for i in range(1, size)])


@pytest.mark.parametrize('size,count', [(6, 0), (7, 1), (9, 1)])
def test_sells_only_when_two_or_fewer_slots_left(db, rules, size, count):
    advisor = HeuristicAdvisor(db, rules)
    st = unrelated_state(db, size)
    target = CompMatch(db.comp('104'), .5, 0)
    result = advisor._sells(st, target, [BuyAdvice(0, '绯红树怪', 1, '必须买', 3)])
    assert len(result) == count
    if result:
        assert result[0].reason == '与阵容和羁绊无关，腾备战席'


@pytest.mark.parametrize('protection', ['pair', 'two_star', 'board', 'buy', 'watch', 'active_trait', 'main_trait'])
def test_sell_protections(db, rules, protection):
    advisor = HeuristicAdvisor(db, rules)
    st = unrelated_state(db)
    target = CompMatch(db.comp('104'), .5, 0)
    buys, watches = [], []
    if protection == 'pair': st.bench[1] = Unit('孤卡', zone='bench', slot=1)
    if protection == 'two_star': st.bench[0].star = 2
    if protection == 'board': st.board = [Unit('孤卡')]
    if protection == 'buy': buys = [BuyAdvice(0, '孤卡', 1, '本帧买', 3)]
    if protection == 'watch':
        target = CompMatch(replace(target.comp, heroes=[]), .5, 0)
        db.comp('104').heroes.append(CompHero('lone', '孤卡', 1))
        watches = ['104']
    if protection == 'active_trait': st.traits = [TraitStatus('无关羁绊', 2, True)]
    if protection == 'main_trait': target = CompMatch(replace(target.comp, traits=['无关羁绊']), .5, 0)
    assert advisor._sells(st, target, buys, watches) == []


def test_sell_target_board_unknown_and_max_two(db, rules):
    advisor = HeuristicAdvisor(db, rules)
    st = unrelated_state(db)
    for i in (1, 2):
        name = f'孤卡{i}'
        db.heroes_by_name[name] = replace(db.hero('孤卡'), name=name)
        st.bench[i] = Unit(name, zone='bench', slot=i)
    st.bench[3] = Unit('绯红树怪', zone='bench', slot=3)
    st.board = [Unit('奥恩')]
    result = advisor._sells(st, CompMatch(db.comp('104'), .5, 0))
    assert len(result) == 2 and {s.name for s in result} <= {'孤卡', '孤卡1', '孤卡2'}
