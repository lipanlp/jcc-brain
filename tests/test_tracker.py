"""阵容按回合收敛：时机、迟滞、定型、锁定与展示链路。"""
from __future__ import annotations

from ai_fakes import authorize_ai

import logging
from dataclasses import replace
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain.core.models import GameState
from jccbrain.data.models import CompHero
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatch, CompMatcher
from jccbrain.engine.llm import LLMReply, build_context
from jccbrain.engine.tracker import CompTracker
from jccbrain.modes.assistant import AssistantMode, advice_signature
from jccbrain.modes.base import ModeSink
from jccbrain.webui.serialize import advice as serialize_advice


def state(stage='2-1', **kw):
    return GameState.from_dict(make_state(stage=stage, **kw))


@pytest.fixture
def ranking(db):
    base = db.comps_ranked()[0]
    scores = {
        cid: CompMatch(replace(base, id=cid, name=cid), score, 0.1)
        for cid, score in zip('ABCDEF', (0.6, 0.5, 0.4, 0.3, 0.2, 0.1))
    }
    matcher = Mock(spec=CompMatcher)

    def fresh(cid):
        return replace(scores[cid], why=[], owned=[], missing=[], in_shop=[])

    matcher.match.side_effect = lambda st, top=None, **kwargs: sorted(
        [fresh(cid) for cid in scores], key=lambda m: -m.score)[:top]
    matcher.match_ids.side_effect = lambda st, ids, **kwargs: [fresh(cid) for cid in ids if cid in scores]
    return matcher, scores


def test_round_frames_refresh_only_selected_candidates_and_keep_order(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    assert [m.comp.id for m in tracker.select(state('1-3'), matcher, None)] == ['A', 'B', 'C']
    scores['B'].score = 0.99
    for st in (state('1-3', gold=40), state(None), state('1-3', phase='combat')):
        result = tracker.select(st, matcher, None)
        assert [m.comp.id for m in result] == ['A', 'B', 'C']
        assert result[1].score == 0.99
    matcher.match.assert_called_once()
    assert matcher.match.call_args.kwargs == {'top': None}
    assert matcher.match_ids.call_count == 3
    tracker.select(state('1-4'), matcher, None)
    assert matcher.match.call_count == 2 and tracker.target_id == 'B'


def test_real_matcher_refreshes_shop_and_owned_without_reordering(db):
    base = db.comps_ranked()[0]
    alternative = replace(base, id='other', name='另一套', heroes=[CompHero('o', '奥恩', 1, is_carry=True)])
    matcher = CompMatcher([base, alternative])
    tracker = CompTracker()
    original = tracker.select(state('1-3'), matcher, None)
    changed = state('1-3', bench=[{'slot': 0, 'name': '奥恩', 'star': 1}],
                    shop={'open': True, 'slots': [{'name': '奥恩', 'cost': 1}]})
    refreshed = tracker.select(changed, matcher, None)
    assert [m.comp.id for m in refreshed] == [m.comp.id for m in original]
    match = next(m for m in refreshed if m.comp.id == 'other')
    assert [h.name for h in match.owned] == ['奥恩'] and not match.missing
    assert match.in_shop == ['奥恩'] and '商店有 奥恩' in match.why


def test_candidates_narrow_at_stage_two_and_settlement(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    assert len(tracker.select(state('1-3'), matcher, None)) == 3
    assert len(tracker.select(state('2-2'), matcher, None)) == 2
    assert tracker.status == '过渡'
    assert len(tracker.select(state('3-2'), matcher, None)) == 1
    assert tracker.status == '定型' and tracker.settled


def test_switch_requires_two_rounds_and_reason_lasts_only_current_round(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(), matcher, None)
    scores['B'].score = 0.7
    tracker.select(state('2-2'), matcher, None)
    assert tracker.target_id == 'A' and tracker.lead == {'B': 1} and tracker.switch_reason is None
    tracker.select(state('2-2'), matcher, None)
    assert tracker.lead == {'B': 1}
    tracker.select(state('2-3'), matcher, None)
    assert tracker.target_id == 'B' and tracker.lead == {}
    assert tracker.switch_reason == 'B 连续 2 回合领先 0.10'
    assert tracker.candidate_ids == ['B', 'A']
    tracker.select(state(None), matcher, None)
    assert tracker.switch_reason is not None
    tracker.select(state('2-4'), matcher, None)
    assert tracker.switch_reason is None


def test_interrupted_lead_restarts_count(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(), matcher, None)
    for stage, score in [('2-2', 0.7), ('2-3', 0.65), ('2-4', 0.7)]:
        scores['B'].score = score
        tracker.select(state(stage), matcher, None)
        assert tracker.target_id == 'A'
    assert tracker.lead == {'B': 1}
    tracker.select(state('2-5'), matcher, None)
    assert tracker.target_id == 'B'


def test_different_challenger_must_build_own_consecutive_lead(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(), matcher, None)
    scores['B'].score = 0.7
    tracker.select(state('2-2'), matcher, None)
    scores['C'].score = 0.8
    tracker.select(state('2-3'), matcher, None)
    assert tracker.lead == {'C': 1} and tracker.target_id == 'A'


def test_small_lead_never_switches(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state(), matcher, None)
    scores['B'].score = 0.65
    for stage in ('2-2', '2-3', '2-4', '2-5'):
        tracker.select(state(stage), matcher, None)
        assert tracker.target_id == 'A' and tracker.lead == {}


@pytest.mark.parametrize('trigger', ['stage', 'fit', 'carry'])
def test_three_settlement_triggers(ranking, trigger):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state('1-3'), matcher, None)
    kw = {}
    if trigger == 'fit':
        scores['A'].fit = 0.5
    elif trigger == 'carry':
        carry = next(h for h in scores['A'].comp.heroes if h.is_carry)
        kw['bench'] = [{'slot': 0, 'name': carry.name, 'star': 2}]
    result = tracker.select(state('3-2' if trigger == 'stage' else '2-2', **kw), matcher, None)
    assert tracker.settled and tracker.status == '定型'
    assert [m.comp.id for m in result] == ['A']


def test_settled_challenger_only_produces_hint(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state('3-2'), matcher, None)
    scores['B'].score = 0.7
    tracker.select(state('3-3'), matcher, None)
    assert tracker.hint is None
    result = tracker.select(state('3-4'), matcher, None)
    assert [m.comp.id for m in result] == ['A'] and tracker.switch_reason is None
    assert tracker.hint == '可考虑转 B（连续 2 回合领先 0.10）'
    tracker.select(state('3-4'), matcher, None)
    assert tracker.hint is not None
    scores['B'].score = 0.5
    tracker.select(state('3-5'), matcher, None)
    assert tracker.hint is None and tracker.lead == {}


def test_lock_and_unlock_immediately_reevaluate_in_same_round(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    tracker.select(state('1-3'), matcher, None)
    result = tracker.select(state('1-3'), matcher, 'B')
    assert tracker.target_id == 'B' and tracker.status == '锁定' and not tracker.settled
    assert len(result) == 1 and '本局锁定' in result[0].why
    result = tracker.select(state('1-3'), matcher, 'B')
    assert '本局锁定' in result[0].why and matcher.match.call_count == 2
    result = tracker.select(state('1-3'), matcher, None)
    assert tracker.target_id == 'A' and tracker.status == '探索' and len(result) == 3
    assert matcher.match.call_count == 3


def test_lock_preserves_existing_settlement(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    tracker.select(state('3-2'), matcher, None)
    tracker.select(state('3-2'), matcher, 'B')
    assert tracker.settled and tracker.target_id == 'B' and tracker.status == '锁定'
    tracker.select(state('3-2'), matcher, None)
    assert tracker.settled and tracker.target_id == 'B' and tracker.status == '定型'


def test_invalid_lock_does_not_claim_locked(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    tracker.select(state('1-3'), matcher, 'missing')
    assert tracker.status == '探索' and tracker.target_id == 'A'


def test_reload_removing_target_restarts_even_within_round(ranking):
    matcher, scores = ranking
    tracker = CompTracker()
    tracker.select(state('3-2'), matcher, None)
    del scores['A']
    result = tracker.select(state('3-2'), matcher, None)
    assert tracker.target_id == 'B' and [m.comp.id for m in result] == ['B']
    assert matcher.match.call_count == 2 and tracker.lead == {}


def test_reset_clears_all_game_state_and_returns_to_exploration(ranking):
    matcher, _ = ranking
    tracker = CompTracker()
    tracker.select(state('3-2'), matcher, 'B')
    tracker.lead = {'A': 1}
    tracker.hint = '旧提示'
    tracker.switch_reason = '旧原因'
    tracker.reset()
    assert (tracker.round, tracker.target_id, tracker.status, tracker.hint, tracker.switch_reason) == (None,) * 5
    assert not tracker.settled and not tracker.candidate_ids and not tracker.lead
    tracker.select(state('1-3'), matcher, None)
    assert tracker.status == '探索' and tracker.target_id == 'A'


@pytest.mark.parametrize('role,k', [('carry', 0.5), ('subcarry', 0.3), ('other', 0.1)])
def test_two_star_weight_depends_on_role(db, role, k):
    base = db.comps_ranked()[0]
    hero = CompHero('test', '奥恩', 1, is_carry=role == 'carry', is_subcarry=role == 'subcarry')
    comp = replace(base, heroes=[hero, CompHero('missing', '缺失', 1)])
    matcher = CompMatcher([comp])
    def match(star):
        return matcher.match(state(bench=[{'slot': 0, 'name': '奥恩', 'star': star}]))[0]
    one, two, three = match(1), match(2), match(3)
    assert two.score > one.score
    expected_fit = min(hero.weight * (1 + k) / (hero.weight + 1), 1)
    assert two.fit == pytest.approx(expected_fit)
    assert three.fit == pytest.approx(min(hero.weight * (1 + 2 * k) / (hero.weight + 1), 1))


def test_round_log_contains_summary_once_per_round(ranking, caplog):
    matcher, _ = ranking
    tracker = CompTracker()
    with caplog.at_level(logging.INFO, logger='jccbrain.rounds'):
        for stage in ('1-3', '1-3', None, '2-1', '2-1'):
            tracker.select(state(stage), matcher, None)
    records = [r for r in caplog.records if r.name == 'jccbrain.rounds']
    assert len(records) == 2
    assert records[0].message == '回合 1-3｜状态 探索｜目标 A｜候选 A、B、C｜切换：无｜提示：无'
    assert '回合 2-1｜状态 过渡｜目标 A' in records[1].message


def test_advisor_without_tracker_keeps_top_five_cli_path(db, rules):
    advisor = HeuristicAdvisor(db, rules)
    base = db.comps_ranked()[0]
    advisor.matcher = CompMatcher([replace(base, id=str(i)) for i in range(6)])
    st = state('2-2')
    expected = advisor.matcher.match(st, top=5)
    advice = advisor.advise(st)
    assert advice.comps == expected and len(advice.comps) == 5
    assert advice.comp_status is None and advice.switch_reason is None


@pytest.mark.parametrize('stage,locked,status,text', [
    ('1-3', None, '探索', '前期按强度过渡（参考 '),
    ('2-2', None, '过渡', '倾向 '),
    ('3-2', None, '定型', '走 '),
    ('1-3', '104', '锁定', '走 '),
])
def test_advisor_status_headline_serialization_and_llm_context(db, rules, stage, locked, status, text):
    advice = HeuristicAdvisor(db, rules).advise(state(stage), locked, tracker=CompTracker())
    assert advice.comp_status == status and text + advice.target.name in advice.headline
    assert serialize_advice(advice)['comp_status'] == status
    assert build_context(state(stage), advice)['comp_status'] == status


def test_settled_hint_is_in_notes_and_ai_switch_is_reference_only(db, rules, ranking):
    matcher, scores = ranking
    advisor = HeuristicAdvisor(db, rules)
    advisor.matcher = matcher
    tracker = CompTracker()
    advisor.advise(state('3-2'), tracker=tracker)
    scores['B'].score = 0.7
    advisor.advise(state('3-3'), tracker=tracker)
    advice = advisor.advise(state('3-4'), tracker=tracker)
    assert tracker.hint in advice.notes
    mode = AssistantMode.__new__(AssistantMode)
    mode._llm = (advice.stage, LLMReply('继续运营', [], '', switch_comp='B'))
    authorize_ai(mode, advice)
    mode._attach_llm(advice)
    assert 'AI 建议转：B（已定型，仅供参考）' in advice.notes
    assert advice.target.comp.id == 'A'


def test_switch_reason_serializes_and_changes_push_signature(db, rules, ranking):
    matcher, scores = ranking
    advisor = HeuristicAdvisor(db, rules)
    advisor.matcher = matcher
    tracker = CompTracker()
    advisor.advise(state(), tracker=tracker)
    scores['B'].score = 0.7
    advisor.advise(state('2-2'), tracker=tracker)
    advice = advisor.advise(state('2-3'), tracker=tracker)
    assert serialize_advice(advice)['switch_reason'] == 'B 连续 2 回合领先 0.10'
    assert advice_signature(advice) != advice_signature(replace(advice, switch_reason=None))


def test_mode_resets_tracker_when_season_changes(db, rules):
    from test_assistant import Ctx
    ctx = Ctx(db, rules, [])
    mode = AssistantMode(ctx, ModeSink())
    mode.compute_advice(state('3-2'))
    mode.tracker.reset = Mock(wraps=mode.tracker.reset)
    mode._season_id = 'previous-season'
    mode.compute_advice(state('1-3'))
    mode.tracker.reset.assert_called_once_with()
    assert mode.tracker.status == '探索'
