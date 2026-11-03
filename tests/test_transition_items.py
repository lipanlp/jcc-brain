"""动态装备安排的资产账本、适配依据和当前单位边界；仅用合成库。"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.data.models import CompHero, Item, SkillValue
from jccbrain.engine import items
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.llm import build_context
from jccbrain.modes.assistant import advice_signature
from jccbrain.webui.serialize import advice as project


def gear(*names):
    return [{'name': name} for name in names]


@pytest.fixture
def equipment_db(db):
    db.items.update({item.id: item for item in [
        replace(db.items['2010'], description='+35物理加成 +35%暴击率'),
        Item('rod', '无用大棒', '基础装备'), Item('glove', '拳套', '基础装备'),
        Item('tear', '女神之泪', '基础装备'), Item('belt', '巨人腰带', '基础装备'),
        Item('jg', '珠光护手', '成型装备', '+35法术加成 +35%暴击率', ('rod', 'glove')),
        Item('warmog', '狂徒铠甲', '成型装备', '+500生命上限', ('belt', 'belt')),
        Item('spear', '朔极之矛', '成型装备', '+15物理加成 +15法术加成', ('1001', 'tear')),
    ]})
    for name, role in [('绯红树怪', 'AD'), ('厄斐琉斯', 'AD'), ('远古巨龙', 'AP'), ('奥恩', 'AP'), ('苍蓝哨戒', '坦克'), ('峡谷迅捷蟹', '坦克')]:
        hero = replace(db.hero(name), role_tag=role)
        db.heroes[hero.id] = db.heroes_by_name[name] = hero
    return db


def target(db, *entries):
    base = db.comp('104')
    heroes = []
    for index, (name, names) in enumerate(entries or [('绯红树怪', ('无尽之刃',))]):
        hero = next((h for h in base.heroes if h.name == name), None)
        if hero is None:
            data = db.hero(name)
            hero = CompHero(data.id, data.name, data.cost)
        heroes.append(replace(hero, is_carry=index == 0, is_subcarry=index == 1, items=tuple(names)))
    return replace(base, heroes=heroes, guide=None)


def arrange(db, comp=None, state=None, evidence=None):
    comp = comp or target(db)
    state = state or GameState(phase=Phase.PLANNING, stage='3-2', level=6)
    before = deepcopy((comp, state, items.held(state, db)))
    result = items.plan(comp, items.held(state, db), state, db, evidence_comps=evidence)
    assert before == (comp, state, items.held(state, db))
    allocated = [(a['source'], a['index']) for row in result for a in row['assets']]
    assert len(allocated) == len(set(allocated))
    by_unit = Counter(row['current_unit']['key'] for row in result if row['ready'])
    for key, n in by_unit.items():
        index = int(key.split(':')[1])
        assert len(state.board[index].items) + n <= 3
    return result


@pytest.mark.parametrize('assets,action', [(['无尽之刃'], '佩戴'), (['暴风大剑', '暴风大剑'], '合成')])
def test_ready_assets_are_given_to_actual_final_hero(equipment_db, assets, action):
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear(*assets))
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == action and row['ready']
    assert row['current_unit']['name'] == row['final_to'] == '绯红树怪'
    assert len(row['assets']) == len(assets)


def test_completed_item_in_bench_is_not_skipped(equipment_db):
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear('无尽之刃'))
    assert items.held(st, equipment_db).completed == ['无尽之刃']
    assert arrange(equipment_db, state=st)[0]['action'] == '佩戴'


def test_equipped_components_cannot_be_borrowed_even_from_target(equipment_db):
    st = GameState(board=[Unit('绯红树怪', items=gear('暴风大剑')), Unit('奥恩', items=gear('暴风大剑'))],
                   item_bench=gear('暴风大剑'))
    assert items.held(st, equipment_db).components == Counter({'暴风大剑': 3})
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == '备料' and row['missing'] == ['暴风大剑']
    assert row['assets'] == [{'source': 'item_bench', 'index': 0, 'name': '暴风大剑'}]


@pytest.mark.parametrize('count,action', [(1, '备料'), (2, '合成')])
def test_duplicate_recipe_component_needs_two_independent_copies(equipment_db, count, action):
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear(*(['暴风大剑'] * count)))
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == action and len(row['assets']) == count


def test_full_recipe_reservation_precedes_partial_recipe(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃', '朔极之矛')))
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear('暴风大剑', '女神之泪'))
    result = arrange(equipment_db, comp, st)
    assert len(result) == 1 and result[0]['item'] == '朔极之矛' and result[0]['action'] == '合成'


def test_shared_recipe_is_not_allocated_twice(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃', '朔极之矛')))
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear('暴风大剑', '暴风大剑', '女神之泪'))
    result = arrange(equipment_db, comp, st)
    assert [r['action'] for r in result] == ['合成', '备料']
    assert result[1]['missing'] == ['暴风大剑']


def test_same_item_for_two_final_heroes_keeps_two_demands(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃',)), ('远古巨龙', ('无尽之刃',)))
    st = GameState(board=[Unit('绯红树怪'), Unit('远古巨龙')], item_bench=gear('无尽之刃', '无尽之刃'))
    result = arrange(equipment_db, comp, st)
    assert [r['final_to'] for r in result] == ['绯红树怪', '远古巨龙']
    assert [r['current_unit']['name'] for r in result] == ['绯红树怪', '远古巨龙']
    assert all(r['action'] == '佩戴' for r in result)


def test_final_owners_worn_copy_reserved_before_other_demand(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃',)), ('远古巨龙', ('无尽之刃',)))
    st = GameState(board=[Unit('绯红树怪'), Unit('远古巨龙', items=gear('无尽之刃'))], item_bench=gear('无尽之刃'))
    result = arrange(equipment_db, comp, st)
    assert len(result) == 1 and result[0]['final_to'] == '绯红树怪' and result[0]['action'] == '佩戴'


def test_worn_completed_item_is_not_free_inventory_or_duplicate_craft(equipment_db):
    st = GameState(board=[Unit('厄斐琉斯', star=2, items=gear('无尽之刃')), Unit('绯红树怪')],
                   item_bench=gear('暴风大剑', '暴风大剑'))
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == '暂留' and not row['ready'] and not row['assets']
    assert row['current_unit']['name'] == '厄斐琉斯' and '不是装备栏空闲' in row['reason']
    assert '拆卸器转交' in row['reason']


def test_worn_item_on_bench_does_not_become_current_board_recipient(equipment_db):
    st = GameState(board=[Unit('厄斐琉斯')], bench=[Unit('绯红树怪', zone='bench', items=gear('无尽之刃'))])
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == '暂留' and row['current_unit'] is None
    assert '备战席绯红树怪' in row['reason'] and not row['assets']
    assert row['existing_item']['holder'] == '备战席绯红树怪'


def test_same_named_units_consume_their_own_slots(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃', '无尽之刃', '无尽之刃')))
    st = GameState(board=[Unit('厄斐琉斯', star=2, items=[{}, {}]), Unit('厄斐琉斯', items=[{}, {}])],
                   item_bench=gear('无尽之刃', '无尽之刃', '无尽之刃'))
    result = arrange(equipment_db, comp, st)
    assert [r['action'] for r in result] == ['佩戴', '佩戴', '暂留']
    assert [r['current_unit']['key'] for r in result[:2]] == ['board:0', 'board:1']
    assert all('棋盘第' in r['current_unit']['label'] for r in result[:2])


@pytest.mark.parametrize('entries', [[{}, {}, {}], gear('?未知', '?未知', '?未知'), gear('暴风大剑', '女神之泪', '拳套')])
def test_all_equipment_slots_including_unknown_occupy_capacity(equipment_db, entries):
    st = GameState(board=[Unit('绯红树怪', items=entries)], item_bench=gear('无尽之刃'))
    assert arrange(equipment_db, state=st)[0]['action'] == '暂留'


@pytest.mark.parametrize('wanted,final,good,bad', [('无尽之刃', '绯红树怪', '厄斐琉斯', '奥恩'),
                                                ('珠光护手', '远古巨龙', '奥恩', '厄斐琉斯'),
                                                ('狂徒铠甲', '苍蓝哨戒', '峡谷迅捷蟹', '奥恩')])
def test_roles_do_not_mix_even_if_wrong_role_higher_star(equipment_db, wanted, final, good, bad):
    comp = target(equipment_db, (final, (wanted,)))
    st = GameState(board=[Unit(bad, star=3), Unit(good)], item_bench=gear(wanted))
    row = arrange(equipment_db, comp, st)[0]
    assert row['give_to'] == good and row['ready']


def test_two_star_transition_not_limited_to_current_guide(equipment_db):
    comp = target(equipment_db)
    assert '厄斐琉斯' not in {h.name for h in comp.heroes}
    st = GameState(board=[Unit('奥恩', star=3), Unit('厄斐琉斯', star=2)], item_bench=gear('无尽之刃'))
    row = arrange(equipment_db, comp, st)[0]
    assert row['give_to'] == '厄斐琉斯' and row['final_to'] == '绯红树怪'
    assert '二星厄斐琉斯' in row['reason'] and '拆卸转交' in row['reason']


def test_actual_final_hero_has_priority_over_stronger_transition(equipment_db):
    st = GameState(board=[Unit('厄斐琉斯', star=3), Unit('绯红树怪')], item_bench=gear('无尽之刃'))
    assert arrange(equipment_db, state=st)[0]['give_to'] == '绯红树怪'


def test_missing_transition_disappears_and_final_arrival_recomputes(equipment_db):
    st = GameState(board=[Unit('厄斐琉斯', star=2)], item_bench=gear('无尽之刃'))
    assert arrange(equipment_db, state=st)[0]['give_to'] == '厄斐琉斯'
    st.board = [Unit('绯红树怪')]
    assert arrange(equipment_db, state=st)[0]['give_to'] == '绯红树怪'
    st.board[0].items = gear('无尽之刃'); st.item_bench = []
    result = arrange(equipment_db, state=st)
    assert all(not row['ready'] for row in result) and all('厄斐琉斯' not in row['reason'] for row in result)
    assert '无需重复' in result[0]['reason']


@pytest.mark.parametrize('board,stale', [([], False), ([Unit('?1', star=3)], False),
                                      ([Unit('远古石甲虫', star=3)], False), ([Unit('厄斐琉斯', star=2)], True)])
def test_unknown_missing_evidence_and_stale_keep_assets_in_bench(equipment_db, board, stale):
    st = GameState(board=board, units_stale=stale, item_bench=gear('无尽之刃'))
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == '暂留' and row['current_unit'] is None and not row['ready']
    assert '先确认' in row['reason']


def test_skill_damage_scaling_fallback_without_role_tags(equipment_db):
    hero = replace(equipment_db.hero('厄斐琉斯'), role_tag=None, skill_values=(SkillValue('伤害', '物理加成'),))
    equipment_db.heroes[hero.id] = equipment_db.heroes_by_name[hero.name] = hero
    st = GameState(board=[Unit('奥恩', star=3), Unit('厄斐琉斯', star=2)], item_bench=gear('无尽之刃'))
    row = arrange(equipment_db, state=st, evidence=[])[0]
    assert row['give_to'] == '厄斐琉斯' and '技能伤害明确含物理加成' in row['reason']


def test_other_real_comp_equipment_preference_is_explicit_fallback(equipment_db):
    other = target(equipment_db, ('远古石甲虫', ('无尽之刃',)))
    other = replace(other, id='other', name='合成适配证据')
    st = GameState(board=[Unit('远古石甲虫', star=2)], item_bench=gear('无尽之刃'))
    row = arrange(equipment_db, state=st, evidence=[other])[0]
    assert row['give_to'] == '远古石甲虫' and '真实阵容「合成适配证据」' in row['reason']


def test_unknown_form_with_conflicting_ad_ap_roles_is_not_guessed(equipment_db):
    base = equipment_db.hero('厄斐琉斯')
    variant = replace(base, id='ap-variant', role_tag='AP', variant_of=base.id, in_pool=False)
    equipment_db.heroes[variant.id] = variant
    st = GameState(board=[Unit('厄斐琉斯', star=3)], item_bench=gear('无尽之刃'))
    assert arrange(equipment_db, state=st)[0]['current_unit'] is None


def test_core_transition_never_gets_sell_instruction(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃',)), ('厄斐琉斯', ()))
    st = GameState(board=[Unit('厄斐琉斯', star=2)], item_bench=gear('无尽之刃'))
    row = arrange(equipment_db, comp, st)[0]
    assert row['give_to'] == '厄斐琉斯' and '出售' not in row['reason']


def test_no_demands_or_no_known_assets_have_clear_fallback(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ()))
    assert '缺少明确装备需求' in arrange(equipment_db, comp)[0]['reason']
    st = GameState(board=[Unit('厄斐琉斯')], item_bench=gear('?未知'))
    assert arrange(equipment_db, state=st)[0]['action'] == '暂留'


def test_production_projection_and_llm_keep_actual_plan_and_signature(equipment_db, rules):
    st = GameState(stage='3-2', phase=Phase.PLANNING, gold=50, level=6,
                   board=[Unit('厄斐琉斯', star=2)], item_bench=gear('无尽之刃'))
    advisor = HeuristicAdvisor(equipment_db, rules)
    adv = advisor.advise(st, locked_comp='104')
    assert project(adv)['item_plan'] == build_context(st, adv)['item_plan'] == adv.item_plan
    assert any('二星厄斐琉斯' in line for line in adv.lines())
    different = deepcopy(adv); different.item_plan[0]['current_unit']['key'] = 'board:9'
    assert advice_signature(adv) != advice_signature(different)


def test_one_worn_copy_cannot_satisfy_two_heroes(equipment_db):
    comp = target(equipment_db, ('绯红树怪', ('无尽之刃',)), ('远古巨龙', ('无尽之刃',)))
    st = GameState(board=[Unit('厄斐琉斯', items=gear('无尽之刃')), Unit('远古巨龙')], item_bench=gear('无尽之刃'))
    result = arrange(equipment_db, comp, st)
    assert [row['action'] for row in result] == ['暂留', '佩戴']
    assert result[0]['final_to'] == '绯红树怪' and result[1]['final_to'] == '远古巨龙'


def test_unknown_wearer_equipment_never_becomes_free(equipment_db):
    st = GameState(board=[Unit('?1', items=gear('无尽之刃')), Unit('厄斐琉斯')])
    row = arrange(equipment_db, state=st)[0]
    assert row['action'] == '暂留' and row['current_unit'] is None
    assert '未识别单位' in row['reason'] and not row['assets']


def test_fulfilled_target_does_not_consume_spare_materials(equipment_db):
    st = GameState(board=[Unit('绯红树怪', items=gear('无尽之刃'))], item_bench=gear('暴风大剑', '暴风大剑'))
    row = arrange(equipment_db, state=st)[0]
    assert not row['assets'] and not row['ready'] and '无需重复' in row['reason']


def test_missing_recipe_part_is_not_invented(equipment_db):
    equipment_db.items['bad'] = Item('bad', '资料缺件装备', '成型装备', recipe=('1001', 'missing-id'))
    comp = target(equipment_db, ('绯红树怪', ('资料缺件装备',)))
    st = GameState(board=[Unit('绯红树怪')], item_bench=gear('暴风大剑', '暴风大剑'))
    result = arrange(equipment_db, comp, st)
    assert all(row['action'] == '暂留' and not row['assets'] for row in result)
    assert 'missing-id' not in str(result)


def test_ap_shield_without_explicit_output_role_or_equipment_is_conservative(equipment_db):
    hero = replace(equipment_db.hero('奥恩'), role_tag=None,
                   skill_values=(SkillValue('伤害', '法术加成'), SkillValue('护盾', '法术加成')))
    equipment_db.heroes[hero.id] = equipment_db.heroes_by_name[hero.name] = hero
    comp = target(equipment_db, ('远古巨龙', ('珠光护手',)))
    st = GameState(board=[Unit('奥恩', star=3)], item_bench=gear('珠光护手'))
    assert arrange(equipment_db, comp, st, evidence=[])[0]['current_unit'] is None


def test_early_warning_does_not_count_worn_components_as_usable(equipment_db):
    st = GameState(stage='2-3', hp=40, board=[Unit('绯红树怪', items=gear('暴风大剑', '暴风大剑'))])
    assert all('2 阶段别囤' not in row['reason'] for row in arrange(equipment_db, state=st))


def test_llm_distinguishes_global_held_from_available_and_equipped(equipment_db, rules):
    st = GameState(stage='3-2', phase=Phase.PLANNING, level=6,
                   board=[Unit('绯红树怪', items=gear('暴风大剑', '暴风大剑'))],
                   item_bench=gear('暴风大剑'))
    adv = HeuristicAdvisor(equipment_db, rules).advise(st, locked_comp='104')
    context = build_context(st, adv)
    assert context['items']['components'] == {'暴风大剑': 3}
    assert context['available_item_bench'] == gear('暴风大剑')
    assert context['equipped_items'][0]['occupied_slots'] == 2
    assert context['equipped_items'][0]['key'] == 'board:0'
    assert not context['units_stale']
