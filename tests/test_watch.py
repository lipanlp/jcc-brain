"""本局关注：真实来牌确认、统计展示、候选与锁定、API 和推送边界。"""
from dataclasses import replace
import json
import logging
from unittest.mock import Mock

import pytest

from ai_fakes import authorize_ai

from conftest import make_state
from jccbrain.actuator.layout import Layout
from jccbrain.context import AppContext
from jccbrain.core.models import GameState
from jccbrain.core.settings import HudSettings, Settings
from jccbrain.data.models import CompHero
from jccbrain.engine.advisor import Advice, HeuristicAdvisor
from jccbrain.engine.comps import CompMatch, CompMatcher
from jccbrain.engine.llm import LLMReply, build_context
from jccbrain.engine.shopstats import ShopStats
from jccbrain.engine.tracker import CompTracker
from jccbrain.modes.assistant import AssistantMode, advice_signature
from jccbrain.modes.base import ModeSink
from jccbrain.webui import serialize
from jccbrain.webui.api import Api
from jccbrain.webui.events import WebSink


def state(*names, open=True, **kw):
    return GameState.from_dict(make_state(shop={'open': open, 'slots': [
        {'name': n, 'cost': 1} if n else None for n in names]}, **kw))


def confirm(stats, st):
    stats.update(st)
    stats.update(st)


def test_same_shop_counts_once_including_duplicate_copies():
    stats = ShopStats()
    st = state('绯红树怪', '绯红树怪', '奥恩')
    for _ in range(20):
        stats.update(st)
    assert stats.counts == {'绯红树怪': 2, '奥恩': 1}


def test_single_frame_noise_never_counts_or_recounts_original():
    stats = ShopStats()
    a, b = state('奥恩'), state('绯红树怪')
    confirm(stats, a)
    stats.update(b)
    confirm(stats, a)
    assert stats.counts == {'奥恩': 1}


def test_confirmed_change_then_return_counts_new_shop():
    stats = ShopStats()
    for st in (state('奥恩'), state('绯红树怪'), state('奥恩')):
        confirm(stats, st)
    assert stats.counts == {'奥恩': 2, '绯红树怪': 1}


def test_closed_shop_neither_counts_nor_connects_confirmation():
    stats = ShopStats()
    a = state('奥恩')
    stats.update(a)
    confirm(stats, state('奥恩', open=False))
    stats.update(a)
    assert not stats.counts
    stats.update(a)
    confirm(stats, state('奥恩', open=False))
    confirm(stats, a)
    assert stats.counts == {'奥恩': 1}


def test_slot_order_is_part_of_snapshot_and_empty_slots_do_not_count():
    stats = ShopStats()
    confirm(stats, state('奥恩', None, '绯红树怪'))
    confirm(stats, state(None, '奥恩', '绯红树怪'))
    assert stats.counts == {'奥恩': 2, '绯红树怪': 2}


def test_reset_clears_counts_and_confirmation():
    stats = ShopStats()
    st = state('奥恩')
    confirm(stats, st)
    stats.reset()
    stats.update(st)
    assert not stats.counts
    stats.update(st)
    assert stats.counts == {'奥恩': 1}


def test_buying_cards_does_not_recount_survivors():
    stats = ShopStats()
    for names in [('A', 'B', 'C', 'D', 'E'),
                  (None, 'B', 'C', 'D', 'E'),
                  (None, None, 'C', 'D', 'E')]:
        confirm(stats, state(*names))
    assert stats.counts == dict.fromkeys('ABCDE', 1)


def test_refresh_with_new_hero_after_buying_counts_the_new_shop():
    stats = ShopStats()
    confirm(stats, state('A', 'B', 'C', 'D', 'E'))
    confirm(stats, state(None, 'B', 'C', 'D', 'E'))
    for _ in range(5):
        stats.update(state('F', 'B', 'C', 'D', 'E'))
    assert stats.counts == {'A': 1, 'B': 2, 'C': 2, 'D': 2, 'E': 2, 'F': 1}


def test_restocking_same_hero_in_bought_slot_is_a_new_shop():
    stats = ShopStats()
    confirm(stats, state('A', 'B', 'C', 'D', 'E'))
    confirm(stats, state(None, 'B', 'C', 'D', 'E'))
    confirm(stats, state('A', 'B', 'C', 'D', 'E'))
    assert stats.counts == dict.fromkeys('ABCDE', 2)


def test_locked_shop_across_rounds_and_selling_do_not_recount():
    stats = ShopStats()
    for stage in ('2-3', '2-4', '2-5'):
        confirm(stats, state('A', 'B', 'C', 'D', 'E', stage=stage,
                             bench=[{'slot': 0, 'name': '奥恩'}] if stage == '2-3' else []))
    assert stats.counts == dict.fromkeys('ABCDE', 1)


def test_single_frame_new_hero_after_buying_is_not_counted():
    stats = ShopStats()
    confirm(stats, state('A', 'B', 'C', 'D', 'E'))
    confirm(stats, state(None, 'B', 'C', 'D', 'E'))
    stats.update(state('F', 'B', 'C', 'D', 'E'))
    confirm(stats, state(None, 'B', 'C', 'D', 'E'))
    assert stats.counts == dict.fromkeys('ABCDE', 1)


def test_buying_every_slot_empty_then_refresh_counts_normally():
    stats = ShopStats()
    names = list('ABCDE')
    confirm(stats, state(*names))
    for i in range(5):
        names[i] = None
        confirm(stats, state(*names))
    assert stats.counts == dict.fromkeys('ABCDE', 1)
    confirm(stats, state('F', 'G', 'H', 'I', 'J'))
    assert stats.counts == dict.fromkeys('ABCDEFGHIJ', 1)


@pytest.fixture
def watch_db(db):
    base = db.comp('104')
    def hero(name, **kw):
        h = db.hero(name)
        return CompHero(h.id, h.name, h.cost, **kw)
    db.comps = {
        'A': replace(base, id='A', name='红蓝', heroes=[hero('绯红树怪', is_carry=True), hero('苍蓝哨戒', is_core=True)]),
        'B': replace(base, id='B', name='奥蓝', heroes=[hero('奥恩', is_carry=True), hero('苍蓝哨戒', is_subcarry=True)]),
        'C': replace(base, id='C', name='螃蟹', heroes=[hero('峡谷迅捷蟹', is_carry=True)]),
        'D': replace(base, id='D', name='石甲虫', heroes=[hero('远古石甲虫', is_carry=True)]),
        'E': replace(base, id='E', name='巨龙', heroes=[hero('远古巨龙', is_carry=True)]),
    }
    return db


def test_match_stats_and_owned_copies_are_correct_without_changing_score(watch_db):
    st = state('绯红树怪', bench=[{'slot': 0, 'name': '苍蓝哨戒', 'star': 2}],
               board=[{'name': '绯红树怪', 'star': 1}])
    stats = ShopStats()
    stats.counts.update({'绯红树怪': 4, '苍蓝哨戒': 3, '奥恩': 2, '不相关': 9})
    matcher = CompMatcher(watch_db.comps_ranked())
    before = matcher.match_ids(st, ['A', 'B'])
    after = matcher.match_ids(st, ['A', 'B'], shop_stats=stats)
    assert [(m.seen, m.seen_carry, m.owned_copies) for m in after] == [(7, 4, 4), (5, 2, 3)]
    assert [m.score for m in before] == [m.score for m in after]
    assert [m.fit for m in before] == [m.fit for m in after]
    assert matcher.match(st, shop_stats=stats)[0].seen > 0


@pytest.fixture
def ranking(watch_db):
    scores = {cid: CompMatch(c, .9-i*.1, .1) for i, (cid, c) in enumerate(watch_db.comps.items())}
    matcher = Mock(spec=CompMatcher)
    fresh = lambda cid: replace(scores[cid], why=[])
    matcher.match.side_effect = lambda st, top=None: sorted([fresh(cid) for cid in scores], key=lambda m: -m.score)[:top]
    matcher.match_ids.side_effect = lambda st, ids: [fresh(cid) for cid in ids if cid in scores]
    return matcher, scores


def test_watch_has_all_four_sorted_candidates_and_bypasses_settlement(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    st = state(stage='4-3', bench=[{'slot': 0, 'name': '绯红树怪', 'star': 2}])
    matches = tracker.select(st, matcher, None, ['D', 'C', 'B', 'A'])
    assert [m.comp.id for m in matches] == list('ABCD')
    assert tracker.status == '关注' and not tracker.settled and tracker.target_id == 'A'


def test_watch_keeps_primary_within_round_and_refreshes_comparison_scores(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    st = state(stage='2-2')
    tracker.select(st, matcher, None, ['A', 'B'])
    scores['B'].score = 1.1  # 即使瞬时明显领先，同回合仍只比较。
    matches = tracker.select(st, matcher, None, ['A', 'B'])
    assert [m.comp.id for m in matches] == ['A', 'B'] and matches[1].score == 1.1
    assert tracker.target_id == 'A' and not tracker.lead
    matcher.match.assert_called_once()


def test_watch_change_immediately_reevaluates_and_logs_names(ranking, caplog):
    matcher, _ = ranking
    tracker = CompTracker()
    st = state(stage='2-2')
    tracker.select(st, matcher, None)
    with caplog.at_level(logging.INFO, logger='jccbrain.rounds'):
        matches = tracker.select(st, matcher, None, ['B', 'C'])
    assert [m.comp.id for m in matches] == ['B', 'C'] and matcher.match.call_count == 2
    assert '状态 关注' in caplog.text and '关注 奥蓝、螃蟹' in caplog.text


def test_lock_pauses_but_unlock_restores_watches(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    st = state(stage='2-2')
    tracker.select(st, matcher, None, ['A', 'B', 'C'])
    locked = tracker.select(st, matcher, 'B', ['A', 'B', 'C'])
    assert len(locked) == 1 and locked[0].comp.id == 'B' and tracker.status == '锁定'
    restored = tracker.select(st, matcher, None, ['A', 'B', 'C'])
    assert [m.comp.id for m in restored] == ['B', 'A', 'C'] and tracker.status == '关注'


@pytest.mark.parametrize('settled', [True, False])
def test_cancel_watch_restores_automatic_rules_and_preserves_settled(ranking, settled):
    matcher, _ = ranking
    tracker = CompTracker()
    st = state(stage='2-2')
    tracker.select(st, matcher, None)
    tracker.settled = settled
    tracker.select(state(stage='4-3'), matcher, None, ['B', 'C'])
    assert tracker.settled == settled
    tracker.select(st, matcher, None, [])
    assert tracker.settled == settled and tracker.status == ('定型' if settled else '过渡')
    assert len(tracker.candidate_ids) == (1 if settled else 2)


def test_watch_buys_only_primary_and_shared_role_uses_primary_weight(watch_db, rules):
    advisor = HeuristicAdvisor(watch_db, rules)
    st = state('奥恩', '绯红树怪', '苍蓝哨戒', stage='3-1', gold=2, board_count=[5, 5])
    advice = advisor.advise(st, tracker=CompTracker(), watch_ids=['A', 'B'])
    assert advice.target.comp.id == 'A'
    buys = {b.name: b for b in advice.buys}
    assert set(buys) == {'绯红树怪', '苍蓝哨戒'}
    assert sum(b.cost for b in buys.values()) <= 2
    assert buys['苍蓝哨戒'].priority == 3 + watch_db.comp('A').heroes[1].weight
    assert buys['苍蓝哨戒'].reason == '红蓝 核心'
    assert buys['绯红树怪'].reason == '红蓝 主C'
    assert [m.comp.id for m in advice.comps] == ['A', 'B']


def test_locked_buys_ignore_other_watch_targets(watch_db, rules):
    st = state('绯红树怪', '奥恩', stage='3-1')
    advice = HeuristicAdvisor(watch_db, rules).advise(st, 'A', tracker=CompTracker(), watch_ids=['A', 'B'])
    assert advice.locked and [b.name for b in advice.buys] == ['绯红树怪']


def test_watch_hud_marks_only_primary_shop_and_board_targets(watch_db):
    matches = [CompMatch(watch_db.comp(cid), .5, .2) for cid in ('A', 'B')]
    advice = Advice('3-2', state().phase, comps=matches, comp_status='关注')
    st = state('奥恩', '绯红树怪', '苍蓝哨戒', '远古巨龙', board=[
        {'name': '奥恩', 'x': .3, 'y': .4}, {'name': '绯红树怪', 'x': .4, 'y': .4}])
    payload = serialize.hud(st, advice, watch_db, Layout(), HudSettings())
    assert [(s['key'], s['kind']) for s in payload['shop']] == [(1, 'target'), (2, 'target')]
    assert [u['target'] for u in payload['units']] == [False, True]
    advice.comp_status = '锁定'
    assert serialize.hud(st, advice, watch_db, Layout(), HudSettings()) == payload


def test_watched_llm_context_all_four_and_reply_guard(watch_db):
    matches = [CompMatch(watch_db.comp(cid), .5, .2, seen=i, seen_carry=1, owned_copies=3)
               for i, cid in enumerate('ABCD')]
    advice = Advice('2-2', state().phase, comps=matches, comp_status='关注')
    context = build_context(state(), advice)
    assert len(context['watched_comps']) == 4
    assert context['main_comp'] == '红蓝'
    assert context['watched_comps'][3] == {'name': '石甲虫', 'seen': 3, 'seen_carry': 1, 'owned_copies': 3, 'fit': .2}
    mode = object.__new__(AssistantMode)
    mode._llm = ('2-2', LLMReply('比较', [], '', '范围外'))
    authorize_ai(mode, advice)
    mode._attach_llm(advice)
    assert not advice.notes
    mode._llm = ('2-2', LLMReply('比较', [], '', '奥蓝'))
    mode._attach_llm(advice)
    assert advice.notes == ['AI 建议转：奥蓝']
    locked = replace(advice, comp_status='锁定', locked=True)
    assert 'watched_comps' not in build_context(state(), locked)


def test_fourth_candidate_statistics_trigger_signature_and_serialize(watch_db):
    matches = [CompMatch(watch_db.comp(cid), .5, .2) for cid in 'ABCD']
    advice = Advice('2-2', state().phase, comps=matches, comp_status='关注')
    for field in ('seen', 'seen_carry', 'owned_copies'):
        modified = replace(advice, comps=matches[:3]+[replace(matches[3], **{field: 9})])
        assert advice_signature(advice) != advice_signature(modified)
        assert serialize.advice(modified)['comps'][3][field] == 9
        json.dumps(serialize.advice(modified))


@pytest.fixture
def api_ctx(watch_db, monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path/'runtime'))
    ctx, bus = AppContext(Settings()), Mock()
    ctx.db = watch_db
    api = Api(ctx, bus)
    yield api, ctx, bus
    api._shutdown()
    ctx.close()


def test_watch_api_deduplicates_emits_and_keeps_manual_lock_without_persistence(api_ctx):
    api, ctx, bus = api_ctx
    ctx.game_lock_id = 'A'
    ctx.settings.assistant.default_lock_comp_id = 'A'
    ctx.settings.save = Mock()
    response = api.set_watch_comps(['A', 'B', 'A'])
    assert response['ok'] and ctx.game_watch_ids == ['A', 'B']
    expected = [{'id': 'A', 'name': '红蓝', 'canonical_id': 'A', 'available': True},
                {'id': 'B', 'name': '奥蓝', 'canonical_id': 'B', 'available': True}]
    assert response['data']['watch'] == expected
    assert response['data']['lock']['id'] == 'A' and ctx.game_lock_id == 'A'
    assert response['data']['default_lock']['id'] == 'A'  # 长期默认保留给下一局。
    bus.emit.assert_any_call('watch_changed', {'watch': expected})
    ctx.settings.save.assert_not_called()
    assert api.set_watch_comps([])['data']['watch'] == [] and not ctx.game_watch_ids


@pytest.mark.parametrize('bad', ['A', None, [1], [{}], ['A', 'missing'], list('ABCDE')])
def test_watch_api_rejects_invalid_atomically(api_ctx, bad):
    api, ctx, bus = api_ctx
    ctx.game_watch_ids = ['A']
    response = api.set_watch_comps(bad)
    assert not response['ok'] and ctx.game_watch_ids == ['A']
    bus.emit.assert_not_called()


def test_new_game_clears_watches_and_shop_history(api_ctx):
    api, ctx, bus = api_ctx
    ctx.game_watch_ids = ['A', 'B']
    ctx.game_lock_id = ''
    sink = Mock(spec=ModeSink)
    mode = AssistantMode(ctx, sink)
    confirm(mode.shop_stats, state('奥恩'))
    mode.compute_advice(state('奥恩', stage='2-5'))
    for _ in range(2):
        mode.compute_advice(state('绯红树怪', stage='1-2'))
        assert ctx.game_watch_ids == ['A', 'B']
    mode.compute_advice(state('绯红树怪', stage='1-2'))
    assert ctx.game_watch_ids == [] and not mode.shop_stats.counts
    assert mode.tracker.status == '探索'
    sink.on_watch_changed.assert_called_once_with()
    mode.compute_advice(state('绯红树怪', stage='1-2'))
    assert mode.shop_stats.counts == {'绯红树怪': 1}


def test_web_sink_pushes_watch_reset_status():
    bus = Mock()
    status = {'watch': []}
    sink = WebSink(bus, lambda: None, status_getter=lambda: status)
    sink.on_watch_changed()
    bus.emit.assert_any_call('watch_changed', {'watch': []})
    bus.emit.assert_any_call('status', status)


def test_mode_initialization_keeps_preselected_watch_list_and_updates_stats(api_ctx):
    _, ctx, _ = api_ctx
    ctx.game_watch_ids = ['A', 'B']
    mode = AssistantMode(ctx, ModeSink())
    first = mode.compute_advice(state('奥恩', stage='1-3'))
    second = mode.compute_advice(state('奥恩', stage='1-3'))
    assert ctx.game_watch_ids == ['A', 'B'] and first.comp_status == second.comp_status == '关注'
    assert next(m for m in first.comps if m.comp.id == 'B').seen == 0
    assert next(m for m in second.comps if m.comp.id == 'B').seen == 1


def test_watch_requires_two_round_boundaries_and_never_settles(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-2'), matcher, None, ['A', 'B', 'C'])
    scores['B'].score = 1.1
    tracker.select(state(stage='2-3'), matcher, None, ['A', 'B', 'C'])
    assert tracker.target_id == 'A' and tracker.lead == {'B': 1}
    for stage in ['2-3', None, '2-3']:
        tracker.select(state(stage=stage), matcher, None, ['A', 'B', 'C'])
        assert tracker.target_id == 'A' and tracker.lead == {'B': 1}
    matches = tracker.select(state(stage='2-4'), matcher, None, ['A', 'B', 'C'])
    assert [m.comp.id for m in matches] == ['B', 'A', 'C']
    assert tracker.switch_reason == '奥蓝 连续 2 回合领先 0.20' and not tracker.lead
    tracker.select(state(stage='5-1'), matcher, None, ['A', 'B', 'C'])
    assert tracker.status == '关注' and not tracker.settled and len(tracker.candidate_ids) == 3


def test_watch_broken_or_changed_challenger_lead_restarts(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-1'), matcher, None, ['A', 'B', 'C'])
    for stage, b, c, lead in [('2-2', 1.1, .7, {'B': 1}), ('2-3', .95, .7, {}),
                              ('2-4', 1.1, .7, {'B': 1}), ('2-5', 1.1, 1.2, {'C': 1})]:
        scores['B'].score, scores['C'].score = b, c
        tracker.select(state(stage=stage), matcher, None, ['A', 'B', 'C'])
        assert tracker.target_id == 'A' and tracker.lead == lead
    assert tracker.select(state(stage='2-6'), matcher, None, ['A', 'B', 'C'])[0].comp.id == 'C'


def test_enter_watch_retains_existing_valid_primary_even_if_lower_scored(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-2'), matcher, None)
    scores['B'].score = 1.1
    tracker.select(state(stage='2-2'), matcher, None, ['A', 'B'])
    assert tracker.target_id == 'A' and tracker.status == '关注' and not tracker.lead


def test_append_remove_watch_keeps_primary_without_counting_a_new_round(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-1'), matcher, None, ['A', 'B'])
    scores['B'].score = 1.1
    tracker.select(state(stage='2-2'), matcher, None, ['A', 'B'])
    scores['C'].score = 1.2
    tracker.select(state(stage='2-3'), matcher, None, ['A', 'B', 'C'])
    assert tracker.target_id == 'A' and not tracker.lead
    tracker.select(state(stage='2-3'), matcher, None, ['A', 'C'])
    assert tracker.target_id == 'A' and not tracker.lead
    tracker.select(state(stage='2-4'), matcher, None, ['A', 'C'])
    assert tracker.target_id == 'A' and tracker.lead == {'C': 1}
    result = tracker.select(state(stage='2-4'), matcher, None, ['B', 'C'])
    assert [m.comp.id for m in result] == ['C', 'B'] and not tracker.lead


def test_watch_reload_clears_lead_and_replaces_removed_primary(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-1'), matcher, None, ['A', 'B'])
    scores['B'].score = 1.1
    tracker.select(state(stage='2-2'), matcher, None, ['A', 'B'])
    replacement = Mock(spec=CompMatcher)
    replacement.match.side_effect = matcher.match.side_effect
    replacement.match_ids.side_effect = matcher.match_ids.side_effect
    tracker.select(state(stage='2-2'), replacement, None, ['A', 'B'])
    assert tracker.target_id == 'A' and not tracker.lead
    tracker.select(state(stage='2-3'), replacement, None, ['A', 'B'])
    assert tracker.target_id == 'A' and tracker.lead == {'B': 1}
    del scores['A']
    tracker.select(state(stage='2-3'), replacement, None, ['A', 'B'])
    assert tracker.target_id == 'B' and not tracker.lead


def test_watch_reset_and_invalid_ids_keep_no_obsolete_lead(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(stage='2-1'), matcher, None, ['A', 'B'])
    scores['B'].score = 1.1
    tracker.select(state(stage='2-2'), matcher, None, ['A', 'B'])
    tracker.reset()
    assert tracker.lead == {} and tracker.target_id is None
    assert tracker.select(state(stage='1-1'), matcher, None, ['missing'])[0].comp.id == 'B'
    assert tracker.status == '探索'
    scores.clear()
    assert tracker.select(state(stage='1-1'), matcher, None, ['missing']) == [] and not tracker.lead


def test_secondary_exclusive_direct_upgrade_is_still_explained(watch_db, rules):
    from jccbrain.engine.economy import EconAction, EconDecision
    advisor = HeuristicAdvisor(watch_db, rules)
    st = state('奥恩', stage='3-1', gold=10, board_count=[5, 5], bench=[
        {'slot': 0, 'name': '奥恩'}, {'slot': 1, 'name': '奥恩'}])
    target = CompMatch(watch_db.comp('A'), .5, .2)
    buys = advisor._buys(st, target, EconDecision(EconAction.ROLL, '测试', keep_gold=0))
    assert [(b.name, b.reason) for b in buys] == [('奥恩', '买了直接升星')]


def test_secondary_exclusive_filler_requires_actual_board_space(watch_db, rules):
    from jccbrain.engine.economy import EconAction, EconDecision
    advisor = HeuristicAdvisor(watch_db, rules)
    target = CompMatch(watch_db.comp('A'), .5, .2)
    econ = EconDecision(EconAction.ROLL, '测试', keep_gold=0)
    st = state('奥恩', stage='2-2', gold=10, board_count=[4, 5])
    buy, = advisor._buys(st, target, econ)
    assert buy.name == '奥恩' and buy.reason.startswith('补场上空位')
    assert not advisor._buys(replace(st, board_count=(5, 5)), target, econ)


def test_watch_bench_reservation_and_total_spend_remain_bounded(watch_db, rules):
    from jccbrain.engine.economy import EconAction, EconDecision
    advisor = HeuristicAdvisor(watch_db, rules)
    target = CompMatch(watch_db.comp('A'), .5, .2)
    st = state('绯红树怪', '苍蓝哨戒', '奥恩', stage='2-2', gold=2, board_count=[4, 5],
               bench=[{'slot': i, 'name': '远古巨龙'} for i in range(8)])
    buys = advisor._buys(st, target, EconDecision(EconAction.ROLL, '测试', keep_gold=0))
    assert len(buys) == 1 and buys[0].name == '绯红树怪' and sum(b.cost for b in buys) <= 2
