from dataclasses import replace
import json

from jccbrain.core.models import GameState, TraitStatus
from jccbrain.data.models import Item, Trait
from jccbrain.engine.traits import traits_view
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.webui import serialize


def test_final_uses_guide_not_population_board_and_deduplicates(board_comp, db):
    board_comp.guide.boards['final'] += [board_comp.guide.heroes[0]]
    view = traits_view(board_comp, GameState(level=2), db)
    final = {t['name']: t for t in view['final']}
    assert final['峡谷野怪']['count'] == 4  # guide final, not comp.heroes (5) or early (1)
    assert final['峡谷野怪']['tier_index'] == 0
    assert final['峡谷野怪']['next_breakpoint'] == 5
    assert '永恒之森' not in final  # 2人，未达第一档


def test_fallback_comp_counts_tiers_and_current_link(db):
    state = GameState(traits=[TraitStatus('峡谷野怪', 3, True), TraitStatus('永恒之森', 2, False),
                              TraitStatus('未知', None, False)])
    view = traits_view(db.comp('104'), state, db)
    final = {t['name']: t for t in view['final']}
    assert final['峡谷野怪']['count'] == 5 and final['峡谷野怪']['tier_index'] == 1
    assert final['峡谷野怪']['picture_key'] == 'trait:峡谷野怪'
    a, b, c = view['current']
    assert a['active'] and a['in_comp'] and a['tier_index'] == 0
    assert not b['active'] and not b['in_comp'] and b['tier_index'] == -1
    assert c['emblem'] is None and c['count'] is None and c['next_breakpoint'] is None


def test_emblem_by_id_recipe_and_bench_quantity(db):
    t = db.trait('峡谷野怪')
    db.items['emblem'] = Item('emblem', '测试纹章', '转职纹章', recipe=('1001', '1001'), trait_id=t.id)
    state = GameState(item_bench=[{'name': '暴风大剑'}, {'name': '测试纹章'}])
    row = next(t for t in traits_view(db.comp('104'), state, db)['final'] if t['name']=='峡谷野怪')
    assert row['emblem'] == {'name':'测试纹章', 'components':['暴风大剑','暴风大剑'], 'held':['暴风大剑'], 'owned':True}


def test_emblem_name_link_and_missing_recipe(db):
    db.items['emblem'] = Item('emblem', '测试纹章', '转职纹章', trait_name='峡谷野怪')
    row = traits_view(db.comp('104'), GameState(), db)['final'][0]
    assert row['emblem']['components'] == row['emblem']['held'] == []
    assert row['emblem']['owned'] is False


def test_special_trait_not_assumed_active_and_max_tier(db):
    db.traits['峡谷野怪'] = replace(db.trait('峡谷野怪'), breakpoints=(0,), activation='special')
    view = traits_view(db.comp('104'), GameState(traits=[TraitStatus('猎人', 5, True)]), db)
    assert '峡谷野怪' not in [t['name'] for t in view['final']]
    assert view['current'][0]['next_breakpoint'] is None and view['current'][0]['tier_index'] == 3


def test_split_level_descriptions_are_grouped_by_threshold(db):
    db.traits['神谕'] = Trait('t', '神谕', 'class', (2,3), level_desc=('(2)团队1', '神谕3', '(3)团队2', '神谕4'))
    view = traits_view(None, GameState(traits=[TraitStatus('神谕', 3, True)]), db)
    assert view['current'][0]['level_desc'] == ['团队1|神谕3', '团队2|神谕4']


def test_advisor_and_serialization_carry_view(db, rules):
    adv = HeuristicAdvisor(db, rules).advise(GameState(stage='2-1'))
    payload = serialize.advice(adv)
    assert payload['traits_view']['final']
    assert json.loads(json.dumps(payload))['traits_view'] == adv.traits_view


def test_trait_picture_loaded_from_database(db):
    with db.transaction() as c:
        c.execute('UPDATE traits SET picture=? WHERE name=?', ('https://example.com/trait.png','峡谷野怪'))
    db.reload()
    assert db.trait('峡谷野怪').picture == 'https://example.com/trait.png'


def test_trait_changes_trigger_advice_push_without_changing_target(db, rules):
    from jccbrain.modes.assistant import advice_signature
    advisor = HeuristicAdvisor(db, rules)
    a = advisor.advise(GameState(stage='2-1', traits=[TraitStatus('峡谷野怪', 3, True)]))
    b = advisor.advise(GameState(stage='2-1', traits=[TraitStatus('峡谷野怪', 4, True)]))
    assert a.target.comp.id == b.target.comp.id
    assert advice_signature(a) != advice_signature(b)
