"""D牌共同预算、冻结任务与最后商店；仅假画面/演练执行器。"""
from types import SimpleNamespace

import pytest

from jccbrain.actuator.actions import BuyShop, Reroll
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.modes.base import ModeSink
from test_autoroll import FakeCtx, ScriptedVision, TARGET, SECOND, make_mode, snapshot


@pytest.mark.parametrize('keep', [0, 20, 30, 50, 37])
@pytest.mark.parametrize('surplus,expected', [(0, []), (1, [BuyShop(0)]), (2, [BuyShop(0), BuyShop(1)])])
def test_each_reserve_applies_to_cumulative_buys(make_mode, keep, surplus, expected):
    st = snapshot([TARGET] * 3, gold=keep + surplus)
    mode, _ = make_mode([st])
    mode.keep_gold = keep
    assert mode._buys(st, {TARGET: 3}) == expected


@pytest.mark.parametrize('gold', [None, -1, True, 1.5, float('nan'), float('inf'), '50'])
def test_unknown_or_invalid_gold_never_buys_or_refreshes(make_mode, gold):
    mode, _ = make_mode([snapshot([TARGET], gold=gold)])
    assert '金币' in mode.loop()
    assert not mode.ctx.executor.history


@pytest.mark.parametrize('cost', [0, -1, True, 1.5, float('nan'), float('inf'), '1'])
def test_illegal_observed_cost_does_not_fall_back_to_db(make_mode, cost):
    st = snapshot([TARGET])
    st.shop[0].cost = cost
    mode, _ = make_mode([st])
    assert mode._buys(st, {TARGET: 3}) == []
    assert '成本未知或无效' in mode.loop()
    assert not mode.ctx.executor.history


def test_missing_shop_cost_uses_known_db_but_never_invents_one_gold(make_mode, monkeypatch):
    st = snapshot([TARGET])
    st.shop[0].cost = None
    mode, _ = make_mode([st])
    assert mode._buys(st, {TARGET: 3}) == [BuyShop(0)]
    monkeypatch.setattr(mode.ctx.db, 'hero', lambda name: None)
    assert '成本未知或无效' in mode.loop()
    assert not mode.ctx.executor.history


@pytest.mark.parametrize('cost', [-1, True, float('nan'), '2'])
def test_invalid_refresh_cost_stops(make_mode, cost):
    st = snapshot()
    st.refresh_cost = cost
    mode, _ = make_mode([st])
    assert '刷新成本' in mode.loop()
    assert not mode.ctx.executor.history


def test_last_refresh_can_buy_without_exceeding_limit(make_mode):
    mode, _ = make_mode([snapshot(), snapshot([TARGET], gold=18),
                         snapshot(gold=17, bench=[(TARGET, 1)])],
                        targets=[RollTarget(TARGET, 1)], max_rolls=1)
    assert '目标全部凑齐' in mode.loop()
    assert mode.ctx.executor.history == [Reroll(), BuyShop(0)]
    assert mode.rolls == 1 and mode.bought == 1


def test_target_finished_even_at_reserve_and_no_action(make_mode):
    mode, _ = make_mode([snapshot(gold=30, bench=[(TARGET, 2)])])
    mode.keep_gold = 30
    assert '目标全部凑齐' in mode.loop()
    assert not mode.ctx.executor.history


def test_constructor_deep_copies_targets_and_technical_settings(db, rules):
    ctx = FakeCtx(db, rules, ScriptedVision([snapshot()]))
    ctx.settings.autoroll.buy_pairs_only = True
    ctx.settings.autoroll.read_interval_ms = 127
    targets = [RollTarget(TARGET, 2)]
    mode = AutoRollMode(ctx, ModeSink(), targets, 30, 9)
    targets[0].name = SECOND
    targets[0].star = 3
    targets.clear()
    ctx.settings.autoroll.keep_gold = 0
    ctx.settings.autoroll.max_rolls = 200
    ctx.settings.autoroll.buy_pairs_only = False
    ctx.settings.autoroll.read_interval_ms = 900
    assert mode.resolve_targets(snapshot()) == [RollTarget(TARGET, 2)]
    assert mode._buys(snapshot([TARGET], gold=40), {TARGET: 3}) == []
    assert mode._cfg.read_interval_ms == 127
    assert mode.status['keep'] == 30 and mode.status['max_rolls'] == 9
    exported = mode.status
    exported['targets'][0]['star'] = 1
    assert mode.status['targets'][0]['star'] == 2


def test_automatic_target_and_star_are_resolved_only_once(db, rules, monkeypatch):
    ctx = FakeCtx(db, rules, ScriptedVision([snapshot()]))
    ctx.settings.autoroll.target_star = 3
    mode = AutoRollMode(ctx, ModeSink())
    hero = SimpleNamespace(name=TARGET, star_target=None, is_carry=True, is_subcarry=False)
    adv = SimpleNamespace(target=SimpleNamespace(comp=SimpleNamespace(heroes=[hero])))
    monkeypatch.setattr(mode, 'advise', lambda st: adv)
    ctx.settings.autoroll.target_star = 1
    assert mode.resolve_targets(snapshot()) == [RollTarget(TARGET, 3)]
    hero.name = SECOND
    hero.star_target = 1
    assert mode.resolve_targets(snapshot()) == [RollTarget(TARGET, 3)]


def test_invalid_gold_also_safe_with_actual_probability_report(db, rules):
    ctx = FakeCtx(db, rules, ScriptedVision([snapshot(gold=float('nan'))]))
    mode = AutoRollMode(ctx, ModeSink(), [RollTarget(TARGET, 2)])
    assert mode.loop() == '金币读数无效，已停止'
    assert not ctx.executor.history


@pytest.mark.parametrize('keep', [0, 20, 30, 50, 37])
def test_at_each_reserve_loop_stops_without_buy_or_refresh(make_mode, keep):
    mode, _ = make_mode([snapshot([TARGET], gold=keep)])
    mode.keep_gold = keep
    assert '保留线' in mode.loop()
    assert not mode.ctx.executor.history


def test_mixed_costs_use_remaining_budget_in_same_batch(make_mode):
    st = snapshot([TARGET, SECOND, '苍蓝哨戒'], gold=36)
    for slot, cost in zip(st.shop, [5, 2, 1]):
        if slot: slot.cost = cost
    mode, _ = make_mode([st])
    mode.keep_gold = 30
    assert mode._buys(st, {TARGET: 3, SECOND: 3, '苍蓝哨戒': 3}) == [BuyShop(0), BuyShop(2)]
