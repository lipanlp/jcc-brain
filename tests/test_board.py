"""站位解析与人口方案；合成数据库不依赖 var。"""
from dataclasses import replace
import json

import pytest

from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.data.db import _pos
from jccbrain.data.models import GuideHero
from jccbrain.data.positions import parse_pos
from jccbrain.engine.advisor import Advice, HeuristicAdvisor
from jccbrain.engine.board import BoardCell, BoardPlan, board_for
from jccbrain.modes.assistant import AssistantMode, advice_signature
from jccbrain.vision.base import VisionError
from jccbrain.webui.serialize import advice as serialize_advice


@pytest.mark.parametrize('value,expected', [
    ('1,7', (1, 7)), (' 4 , 2 ', (4, 2)), ((2, 3), (2, 3)), ([3, 4], (3, 4)),
    (None, None), ('bad', None), ('1,2,3', None), ('1.5,2', None),
    ('0,1', None), ('5,1', None), ((1, 8), None), ([True, 2], None),
])
def test_parse_positions(value, expected):
    assert parse_pos(value) == expected
    if value is None or isinstance(value, str):
        assert _pos(value) == expected


@pytest.mark.parametrize('population,source', [(6, '攻略 L6'), (7, '攻略 L6'), (8, '攻略 L8'), (9, '攻略 L8')])
def test_exact_or_nearest_lower_population(board_comp, db, population, source):
    plan = board_for(board_comp, population, GameState(level=population), db)
    assert plan.source == source and plan.population == population
    assert plan.cells[0].name == '奥恩'


@pytest.mark.parametrize('population,key', [(4, 'early'), (6, 'mid'), (7, 'mid'), (8, 'final')])
def test_phase_fallback_without_population_keys(board_comp, db, population, key):
    board_comp.guide.boards = {k: v for k, v in board_comp.guide.boards.items() if not k.startswith('L')}
    plan = board_for(board_comp, population, GameState(level=population), db)
    assert plan.source == f'攻略 {key}'


def test_population_prefers_board_capacity_then_level(board_comp, db):
    assert board_for(board_comp, None, GameState(level=8, board_count=(2, 6)), db).population == 6
    assert board_for(board_comp, None, GameState(level=8), db).population == 8
    assert board_for(board_comp, None, GameState(), db) is None


def test_no_guide_sorts_carries_and_core_before_cost(db):
    comp = db.comp('104')
    plan = board_for(comp, 3, GameState(level=3), db)
    assert plan.source == '成型站位'
    assert [c.name for c in plan.cells] == ['绯红树怪', '远古巨龙', '远古石甲虫']
    assert [(c.row, c.col) for c in plan.cells] == [(4, 7), (4, 3), (1, 5)]


def test_summons_duplicates_conflicts_and_invalid_positions_are_filtered(board_comp, db):
    guide = board_comp.guide
    base = guide.boards['L8'][0]
    guide.boards['L8'] = [base,
        replace(base, name='召唤物', unit_type='summon', position='2,2'),
        replace(base, position='3,3'),
        replace(base, name='冲突棋子', position='1,1'),
        replace(base, name='越界棋子', position='5,1'),
        replace(base, name='缺坐标', position=None)]
    plan = board_for(board_comp, 8, GameState(level=8), db)
    assert [(c.name, c.row, c.col) for c in plan.cells] == [('奥恩', 1, 1)]


def test_owned_uses_highest_star_across_board_and_bench(board_comp, db):
    st = GameState(level=8, board=[Unit('奥恩', 2)],
                   bench=[Unit('奥恩', 1, zone='bench'), Unit('绯红树怪', 3, zone='bench')])
    cells = {c.name: c for c in board_for(board_comp, 8, st, db).cells}
    assert cells['奥恩'].owned_star == 2 and cells['奥恩'].on_board
    assert cells['绯红树怪'].owned_star == 3 and not cells['绯红树怪'].on_board
    assert cells['苍蓝哨戒'].owned_star is None and not cells['苍蓝哨戒'].on_board


@pytest.mark.parametrize('guide_item,comp_item,expected', [
    ('攻略装备', '阵容装备', '攻略装备'),
    (None, '阵容装备', '阵容装备'),
    (None, None, '默认主C装备'),
])
def test_first_item_source_priority(board_comp, db, guide_item, comp_item, expected):
    guide = board_comp.guide
    guide.carry_items = ['默认主C装备']
    carry = next(h for h in guide.boards['L8'] if h.is_carry)
    guide.boards['L8'] = [replace(carry, items=(guide_item,) if guide_item else ())]
    board_comp.heroes = [replace(h, items=(comp_item,) if comp_item else ()) for h in board_comp.heroes]
    cell = board_for(board_comp, 8, GameState(level=8), db).cells[0]
    assert cell.item == expected and cell.is_carry


def test_noncarry_does_not_inherit_default_carry_item(board_comp, db):
    cell = board_for(board_comp, 8, GameState(level=8), db).cells[0]
    assert cell.name == '奥恩' and cell.item is None


def test_next_hint_has_new_hero_positions_and_ownership(board_comp, db):
    st = GameState(level=7, bench=[Unit('远古巨龙', 1, zone='bench')])
    plan = board_for(board_comp, 7, st, db)
    assert plan.next_hint == '升 8 级：上 远古巨龙（第4行第3列，已有）；上 厄斐琉斯（第4行第5列，还缺）'
    assert board_for(board_comp, 8, GameState(level=8), db).next_hint is None
    assert board_for(board_comp, 10, GameState(level=10), db).next_hint is None


def test_missing_positions_produce_no_plan(db):
    comp = db.comp('104')
    comp.heroes = [replace(h, position=None) for h in comp.heroes]
    assert board_for(comp, 8, GameState(level=8), db) is None


def test_advisor_computes_board_each_frame_and_serializes(board_comp, db, rules):
    advisor = HeuristicAdvisor(db, rules)
    first = advisor.advise(GameState(level=7))
    second = advisor.advise(GameState(level=8, board=[Unit('奥恩', 2)]))
    assert first.board.source == '攻略 L6' and second.board.source == '攻略 L8'
    data = serialize_advice(second)
    json.dumps(data)
    assert data['board']['population'] == 8
    assert data['board']['cells'][0]['owned_star'] == 2
    assert data['board']['cells'][0]['on_board']
    assert serialize_advice(advisor.advise(GameState()))['board'] is None


def _board_advice():
    return Advice('2-3', Phase.PLANNING, headline='存钱', board=BoardPlan(
        '攻略 L6', 7, [BoardCell(1, 1, '奥恩', 1, False, None, 2, 1, False)]))


@pytest.mark.parametrize('field,value', [
    ('row', 2), ('col', 3), ('name', '绯红树怪'), ('owned_star', 2),
    ('on_board', True), ('item', '无尽之刃'), ('is_carry', True),
])
def test_visible_cell_changes_update_signature(field, value):
    original = _board_advice()
    cell = replace(original.board.cells[0], **{field: value})
    changed = replace(original, board=replace(original.board, cells=[cell]))
    assert advice_signature(original) != advice_signature(changed)


@pytest.mark.parametrize('field,value', [('source', '攻略 L8'), ('population', 8), ('next_hint', '升 8 级：上 厄斐琉斯')])
def test_plan_changes_update_signature(field, value):
    original = _board_advice()
    changed = replace(original, board=replace(original.board, **{field: value}))
    assert advice_signature(original) != advice_signature(changed)


def test_equal_board_contents_and_absent_board_are_stable():
    original = _board_advice()
    same = replace(original, board=replace(original.board, cells=[replace(c) for c in original.board.cells]))
    assert advice_signature(original) == advice_signature(same)
    assert advice_signature(original) != advice_signature(replace(original, board=None))
    assert advice_signature(replace(original, board=None)) == advice_signature(replace(same, board=None))


@pytest.mark.parametrize('change', ['population', 'owned_star', 'on_board'])
def test_board_only_change_is_pushed_after_two_frames_and_not_repeated(db, rules, change):
    from unittest.mock import Mock
    from conftest import make_state
    from test_assistant import AdviceSink, Ctx

    original = _board_advice()
    if change == 'population':
        changed = replace(original, board=replace(original.board, population=8))
    else:
        cell = replace(original.board.cells[0], **{change: 2 if change == 'owned_star' else True})
        changed = replace(original, board=replace(original.board, cells=[cell]))
    # 第 3 帧只变一次不推；第 5/6 帧连续一致才推，第 7 帧不重复。
    sequence = [original, original, changed, original, changed, changed, changed]
    ctx = Ctx(db, rules, [make_state(stage='2-3')] * len(sequence))
    ctx.llm.ready = False
    sink = AdviceSink()
    mode = AssistantMode(ctx, sink)
    mode.compute_advice = Mock(side_effect=sequence)
    frames = []
    sink.on_advice = lambda advice: (frames.append(mode.compute_advice.call_count), sink.advices.append(advice))
    with pytest.raises(VisionError, match='回放结束'):
        mode.loop()
    assert frames[:-1] == [1, 6] and sink.advices[:-1] == [original, changed]
    assert frames[-1] == 7 and sink.advices[-1].headline == '辅助已停止'
    assert sink.advices[-1].board is None and not sink.advices[-1].targets


@pytest.mark.parametrize('field,value', [('items', ['无尽之刃']), ('role', 'tank')])
def test_items_and_role_change_signature(field, value):
    original = _board_advice()
    cell = replace(original.board.cells[0], **{field: value})
    assert advice_signature(original) != advice_signature(replace(original, board=replace(original.board, cells=[cell])))


@pytest.mark.parametrize('case,expected', [('tie', '峡谷迅捷蟹'), ('more_items', '奥恩'), ('no_items', None), ('back_row', None)])
def test_tank_role_prefers_front_row_equipment_then_cost(board_comp, db, case, expected):
    guide = board_comp.guide
    heroes = [GuideHero('11500', '奥恩', position='1,1', cost=1, items=('护甲',)),
              GuideHero('12510', '峡谷迅捷蟹', position='2,1', cost=2, items=('护甲',)),
              GuideHero('11508', '绯红树怪', position='4,7', cost=1, is_carry=True, items=('无尽之刃',))]
    if case == 'more_items':
        heroes[0] = replace(heroes[0], items=('护甲', '回复'))
    elif case == 'no_items':
        heroes = [replace(h, items=()) if not h.is_carry else h for h in heroes]
    elif case == 'back_row':
        heroes = [replace(h, position='3,1' if i==0 else '4,1') if not h.is_carry else h for i,h in enumerate(heroes)]
    guide.boards['L8'] = heroes
    plan = board_for(board_comp, 8, GameState(level=8), db)
    assert [c.name for c in plan.cells if c.role=='tank'] == ([expected] if expected else [])
    assert next(c for c in plan.cells if c.name=='绯红树怪').role == 'carry'


@pytest.mark.parametrize('count,ready', [(0, False), (1, False), (2, True)])
def test_recipe_components_count_each_owned_copy_once(board_comp, db, count, ready):
    plan = board_for(board_comp, 8, GameState(level=8, item_bench=[{'name':'暴风大剑'}]*count), db)
    cell = next(c for c in plan.cells if c.name=='绯红树怪')
    detail = cell.item_details[0]
    assert detail['components'] == ['暴风大剑', '暴风大剑']
    assert detail['held'] == ['暴风大剑']*count and detail['ready'] is ready
    assert cell.item == cell.items[0] and cell.traits == ['峡谷野怪', '猎人']


def test_completed_item_ready_and_unknown_recipe(board_comp, db):
    carry = next(h for h in board_comp.guide.boards['L8'] if h.is_carry)
    board_comp.guide.boards['L8'] = [replace(carry, items=('无尽之刃', '无配方装备', '第三件', '第四件'))]
    plan = board_for(board_comp, 8, GameState(level=8, item_bench=[{'name':'无尽之刃','kind':'成型装备'}]), db)
    cell = plan.cells[0]
    assert cell.items == ['无尽之刃', '无配方装备', '第三件']
    assert cell.item_details[0]['ready'] and cell.item_details[0]['held'] == ['无尽之刃']
    assert cell.item_details[1] == {'name':'无配方装备','components':[],'held':[],'ready':False}


def test_extended_board_serializes_details(board_comp, db, rules):
    advice = HeuristicAdvisor(db, rules).advise(GameState(level=8))
    data = serialize_advice(advice)
    json.dumps(data)
    cell = next(c for c in data['board']['cells'] if c['is_carry'])
    assert cell['items'] == ['无尽之刃'] and cell['role'] == 'carry'
    assert cell['traits'] and cell['item_details'][0]['components'] == ['暴风大剑', '暴风大剑']


def test_base_component_itself_is_not_marked_as_completed_item(board_comp, db):
    carry = next(h for h in board_comp.guide.boards['L8'] if h.is_carry)
    board_comp.guide.boards['L8'] = [replace(carry, items=('暴风大剑',))]
    plan = board_for(board_comp, 8, GameState(level=8, item_bench=[{'name':'暴风大剑','kind':'基础装备'}]), db)
    assert plan.cells[0].item_details == [{'name':'暴风大剑','components':[],'held':[],'ready':False}]
