"""隔离的侦察画像、卡池及撞阵回归；不读取运行目录。"""
from dataclasses import replace
import json

import pytest

from jccbrain.core.models import GameState, Phase, Player, TraitStatus, Unit
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher
from jccbrain.engine.llm import SYSTEM_PROMPT, build_context
from jccbrain.engine.scout import ScoutBook, ScoutShot
from jccbrain.modes.assistant import advice_signature
from jccbrain.webui.serialize import advice as serialize


@pytest.fixture
def book(db, tmp_path):
    out = ScoutBook(db, directory=tmp_path / 'scout')
    out.update(GameState(stage='4-1', players=[Player('自己', 80, True, 0), Player('甲', 60, False, 1), Player('乙', 40, False, 2)]))
    return out


def shot(**kw):
    return ScoutShot(**{'slot': 1, 'round': '4-1', **kw})


def test_trait_counts_closer_wins(db, book):
    base = db.comp('104')
    smaller = replace(base, id='small', name='两野怪', heroes=base.heroes[:2])
    book.comps_getter = lambda: [base, smaller]
    row = book.add(shot(traits=[TraitStatus('峡谷野怪', 2, True)]))
    assert row.comp_guess == {'id': 'small', 'name': '两野怪', 'fit': 1, 'by': 'traits'}


def test_no_trait_match_is_unknown(book):
    assert book.add(shot(traits=[TraitStatus('不存在', 7, True)])).comp_guess is None


def test_units_refine_when_better(db, book):
    row = book.add(shot(traits=[TraitStatus('峡谷野怪', 1)], board=[Unit(h.name, 2) for h in db.comp('104').heroes]))
    assert row.comp_guess['by'] == 'units' and row.comp_guess['fit'] == 1
    assert len(row.carries) == 5


def test_good_traits_not_overwritten_by_weak_board(book):
    row = book.add(shot(traits=[TraitStatus('峡谷野怪', 5)], board=[Unit('奥恩')]))
    assert row.comp_guess['by'] == 'traits' and row.comp_guess['fit'] == 1


def test_slot_name_self_and_players_parser(book):
    st = GameState.from_dict({'players': [{'name': '丙', 'hp': 70, 'self': False, 'slot': 3, 'y': .3}]})
    assert st.players[0].slot == 3 and st.players[0].y == .3
    assert book.add(shot()).name == '甲'
    assert book.add(shot(slot=0, name='自己')) is None
    assert book.add(ScoutShot()) is None
    assert len(book.latest()) == 1


def test_players_and_history_are_independent_and_copied(book):
    for i in range(5):
        book.add(shot(t=i, level=i + 1))
    book.add(shot(slot=2))
    assert len(book.latest()) == 2
    assert [r.level for r in book.history['slot:1']] == [3, 4, 5]
    rows = book.latest()
    rows[0].name = '篡改'
    assert book.latest()[0].name == '甲'


def test_old_background_result_does_not_replace_newer_round(book):
    book.add(shot(round='4-3', t=100, level=8))
    assert book.add(shot(round='4-2', t=200, level=5)) is None
    assert book.latest()[0].level == 8
    assert book.add(shot(round='4-3', t=90)) is None


def test_staleness_and_death_returns_pool(book):
    book.add(shot(board=[Unit('绯红树怪', 2)]))
    book.update(GameState(stage='4-3', players=[Player('甲', 0, False, 1)]))
    assert book.latest()[0].stale_rounds == 2
    assert not book.taken()


def test_reset_clears_each_game(book):
    book.add(shot())
    started = book.started
    book.reset()
    assert book.latest() == [] and book.history == {} and book.players == []
    assert book.started != started and book.path is None


def test_pool_stars_and_manual_max(db, rules, book):
    book.add(shot(board=[Unit('绯红树怪', 2), Unit('苍蓝哨戒', 3)], bench=[Unit('绯红树怪')]))
    book.add(shot(slot=2, board=[Unit('绯红树怪')]))
    advisor = HeuristicAdvisor(db, rules, scout=book)
    pool = advisor.pool(GameState(board=[Unit('绯红树怪')]), {'绯红树怪': 3, '苍蓝哨戒': 12})
    assert pool.taken['绯红树怪'] == 6  # 自己1 + max(侦察5, 手动3)
    assert pool.taken['苍蓝哨戒'] == 12
    assert '侦察（4-1 回合）' in pool.basis
    assert pool.copy().opponents == pool.opponents


def test_traits_only_does_not_reduce_pool(db, rules, book):
    book.add(shot(traits=[TraitStatus('峡谷野怪', 5)]))
    assert HeuristicAdvisor(db, rules, scout=book).pool(GameState()).taken == {}


@pytest.mark.parametrize('opponents, expected', [(1, .08), (2, .16), (3, .20), (7, .20)])
def test_carry_penalty_and_cap(db, rules, book, opponents, expected):
    comp = db.comp('104')
    base = CompMatcher([comp]).match(GameState())[0].score
    for slot in range(1, opponents + 1):
        book.add(shot(slot=slot, board=[Unit('绯红树怪', 2)]))
    # 单独验证主C竞争，移除猜测以免同时施加撞阵分。
    pool = HeuristicAdvisor(db, rules, scout=book).pool(GameState())
    for row in pool.opponents:
        row.comp_guess = None
    match = CompMatcher([comp]).match(GameState(), pool=pool)[0]
    assert base - match.score == pytest.approx(expected)
    assert any(f'被 {opponents} 家拿着' in why for why in match.why)


def test_two_singles_count_as_contested_and_one_does_not(db, rules, book):
    book.add(shot(board=[Unit('绯红树怪')]))
    advisor = HeuristicAdvisor(db, rules, scout=book)
    first = advisor.matcher.match(GameState(), pool=advisor.pool(GameState()))[0]
    assert not any('被' in why for why in first.why)
    book.add(shot(t=10**12, board=[Unit('绯红树怪'), Unit('绯红树怪')]))
    second = advisor.matcher.match(GameState(), pool=advisor.pool(GameState()))[0]
    assert any('被 1 家' in why for why in second.why)


@pytest.mark.parametrize('n,expected', [(1, .05), (2, .10), (4, .15)])
def test_comp_contest_penalty_and_cap(db, rules, book, n, expected):
    advisor = HeuristicAdvisor(db, rules, scout=book)
    base = advisor.matcher.match(GameState())[0].score
    for slot in range(1, n + 1):
        book.add(shot(slot=slot, traits=[TraitStatus('峡谷野怪', 5)]))
    match = advisor.matcher.match(GameState(), pool=advisor.pool(GameState()))[0]
    assert base - match.score == pytest.approx(expected)
    assert f'{n} 家在玩 野怪小红' in match.why


def test_persistence_each_shot_and_failure_nonfatal(book, tmp_path, caplog):
    book.add(shot(board=[Unit('绯红树怪')]))
    book.add(shot(t=10**12, level=7))
    data = [json.loads(s) for s in book.path.read_text().splitlines()]
    assert len(data) == 2 and data[0]['board'][0]['name'] == '绯红树怪'
    bad = tmp_path / 'file'
    bad.write_text('blocked')
    book.directory = bad
    assert book.add(shot(t=10**12 + 1, level=8)).level == 8
    assert '侦察快照写入失败' in caplog.text


def test_llm_serialize_and_signature(db, rules, book):
    advisor = HeuristicAdvisor(db, rules, scout=book)
    before = advisor.advise(GameState(stage='4-1', phase=Phase.PLANNING))
    book.add(shot(traits=[TraitStatus('峡谷野怪', 5)], board=[Unit('绯红树怪', 2)], level=7))
    book.add(shot(slot=2, traits=[TraitStatus('峡谷野怪', 5)]))
    adv = advisor.advise(GameState(stage='4-3', phase=Phase.PLANNING))
    data = json.loads(json.dumps(serialize(adv), ensure_ascii=False))
    assert data['opponents'][0]['name'] == '甲' and data['opponents'][0]['contested']
    assert data['opponents'][0]['stale_rounds'] == 2
    ctx = build_context(GameState(), adv)
    assert [r['hp'] for r in ctx['opponents']] == [60, 40]
    assert set(ctx['opponents'][0]) == {'name', 'hp', 'round', 'level', 'comp', 'by', 'carries', 'top_traits'}
    assert '结合对手阵容' in SYSTEM_PROMPT
    assert advice_signature(before) != advice_signature(adv)


def test_legacy_without_fields_keeps_results(db, rules, book):
    st = GameState.from_dict({'stage': '3-2', 'phase': 'planning', 'gold': 50, 'level': 6})
    plain = HeuristicAdvisor(db, rules).advise(st)
    scoped = HeuristicAdvisor(db, rules, scout=book).advise(st)
    assert serialize(plain) == serialize(scoped)


def test_mode_new_game_resets_book(db, rules, book):
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    ctx = Ctx(db, rules, [])
    ctx.advisor.scout = book
    mode = AssistantMode(ctx, AdviceSink())
    mode.compute_advice(GameState(stage='4-1'))
    book.add(shot())
    for _ in range(3):
        mode.compute_advice(GameState(stage='1-1'))
    assert book.latest() == []


def test_shot_from_dict_missing_fields():
    assert ScoutShot.from_dict({}).board == []
    s = ScoutShot.from_dict({'viewing': {'slot': 2}, 'traits': [{'name': '猎人', 'count': 2}], 'board': [{'name': '?1'}]}, round='4-1')
    assert s.slot == 2 and s.round == '4-1' and not s.board[0].known
