from collections import Counter

import pytest

from jccbrain.engine.odds import (PoolState, any_hit_per_roll, copies_needed, hit_curve, hit_odds,
                                  simulate_rolldown)


@pytest.fixture
def pool(db, rules):
    return PoolState.from_db(db, rules)


def test_remaining_and_tier_total(pool):
    assert pool.remaining("奥恩") == 30
    assert pool.tier_total(1) == 90                     # 3 个 1 费 × 30
    pool.taken["奥恩"] = 5
    assert pool.remaining("奥恩") == 25 and pool.tier_total(1) == 85


def test_slot_prob_formula(pool):
    # 7 级 1 费概率 19%，奥恩占 1 费池的 1/3
    assert pool.slot_prob("奥恩", 7) == pytest.approx(0.19 / 3)
    assert pool.slot_prob("远古巨龙", 3) == 0
    assert pool.slot_prob("不存在", 7) == 0


def test_hit_curve_monotonic_and_bounds(pool):
    c = hit_curve(pool, "奥恩", 3, 5, max_rolls=80)
    assert c[0] == 0
    assert all(b >= a - 1e-12 for a, b in zip(c, c[1:]))
    assert 0.5 < c[-1] <= 1.0


def test_hit_curve_impossible_when_pool_exhausted(pool):
    pool.taken["远古巨龙"] = 8                           # 只剩 1 张
    assert hit_curve(pool, "远古巨龙", 2, 9)[-1] == 0


def test_single_roll_matches_closed_form(pool):
    p = pool.slot_prob("奥恩", 6)
    c = hit_curve(pool, "奥恩", 1, 6, max_rolls=1)
    assert c[1] == pytest.approx(1 - (1 - p) ** 5)


def test_hit_odds_fields(pool):
    o = hit_odds(pool, "奥恩", 5, need=3)
    assert o.cost == 1 and o.remaining == 30
    assert o.rolls_for_50 <= o.rolls_for_80
    assert o.gold_for_80 == o.rolls_for_80 * 2 + 3


def test_any_hit(pool):
    one = any_hit_per_roll(pool, ["奥恩"], 5)
    two = any_hit_per_roll(pool, ["奥恩", "绯红树怪"], 5)
    assert 0 < one < two < 1


def test_simulation_agrees_with_dp(pool):
    # 只追 1 张、预算 10 次刷新：模拟与精确值相差应在几个百分点内
    gold = 10 * 2 + 1
    exact = hit_curve(pool, "奥恩", 1, 5, max_rolls=10)[10]
    sim = simulate_rolldown(pool, {"奥恩": 1}, gold, 5, keep_gold=0, sims=4000, seed=7)
    assert sim.hit_rate["奥恩"] == pytest.approx(exact, abs=0.04)


def test_simulation_respects_keep_gold(pool):
    sim = simulate_rolldown(pool, {"远古巨龙": 3}, 30, 3, keep_gold=20, sims=200, seed=1)
    assert sim.avg_rolls == 5 and sim.all_hit_rate == 0


def test_copies_needed():
    assert copies_needed(4, 2) == 0
    assert copies_needed(4, 3) == 5
    assert copies_needed(0, 1) == 1


def test_taken_counter_copy_isolated(pool):
    p2 = pool.copy()
    p2.taken["奥恩"] += 3
    assert pool.taken == Counter()
