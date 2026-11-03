"""静态详情的来源边界、原始信息和异常位置，不构造实战持有状态。"""
from dataclasses import replace
import json

from jccbrain.data.models import CompHero, Guide, GuideHero, Trait, TraitLevel
from jccbrain.webui.serialize import comp_detail


def attach(db, guide):
    comp = db.comps['104']
    comp.guide, comp.guides, comp.guide_id = guide, [guide], guide.id
    guide.comp = comp
    db.guides[guide.id] = guide
    return comp


def test_dataj_default_stays_own_formation_with_linked_stages(db):
    guide = Guide('linked', '另一份原始攻略', 'S', None,
                  boards={'final': [GuideHero('11500', '奥恩', position='1,1')],
                          'early': [GuideHero('11511', '苍蓝哨戒', position='1,2')]})
    comp = attach(db, guide)
    detail = comp_detail(comp, db)
    stages = detail['visualization']['stages']
    assert stages[0]['origin'] == 'comp' and stages[0]['source_name'] == comp.name
    assert [u['name'] for u in stages[0]['units']] == [h.name for h in comp.heroes]
    assert stages[0]['units'][0]['position'] == list(comp.heroes[0].position)
    assert {s['id'] for s in stages[1:]} == {'guide:final', 'guide:early'}
    assert all(s['origin'] == 'guide' and s['source_name'] == guide.name for s in stages[1:])
    assert detail['id'] == '104' and detail['avg_placement'] == comp.avg_placement
    json.dumps(detail)


def test_raw_official_stages_preserve_other_units_and_population_labels(db):
    guide = Guide('original', '原始攻略', 'S', 'v1', three_star=['奥恩'], channels=('ingame',),
                  early_info='完整过渡', d_time='完整D牌', location_info='完整站位', enemy_info='完整克制',
                  hex_info='完整海克斯', equipment_info='完整装备', adventure_info='完整仙灵',
                  boards={'final': [GuideHero('11500', '奥恩', position='1,1'),
                                    GuideHero('pet', '召唤单位', unit_type='pet', position='1,2')],
                          'early': [GuideHero('11511', '苍蓝哨戒', position='2,1')],
                          'L8': [GuideHero('11500', '奥恩', star=2, position='3,2'),
                                 GuideHero('pet', '召唤单位', unit_type='pet', position='3,3')]})
    raw = db._guide_to_comp(guide)
    raw.guide, raw.guides = guide, [guide]
    detail = comp_detail(raw, db)
    stages = {s['id']: s for s in detail['visualization']['stages']}
    assert detail['id'] == 'official:original' and detail['channels'] == ['ingame']
    assert detail['guide']['text'] == guide.text() and detail['name'] == guide.name
    assert stages['final']['population'] is None
    assert stages['final']['hero_count'] == stages['final']['other_count'] == 1
    assert stages['L8']['population'] == 8 and stages['L8']['hero_count'] == 1
    assert stages['final']['units'][0]['target_star'] == 3
    assert stages['L8']['units'][0]['target_star'] == 3
    assert stages['L8']['units'][0]['source_star'] == 2  # 追三目标与来源棋子星级分别保留。
    assert stages['early']['units'][0]['target_star'] is None
    assert stages['final']['units'][1]['unit_type'] == 'pet'
    sections = detail['visualization']['guide']['sections']
    assert [s['text'] for s in sections] == ['完整过渡', '完整D牌', '完整站位', '完整克制', '完整海克斯', '完整装备', '完整仙灵']
    assert guide.boards['final'][0].position == '1,1'


def test_conflicting_invalid_and_unknown_positions_are_not_moved_or_lost(db):
    guide = Guide('positions', '异常站位', None, None,
                  boards={'final': [GuideHero(str(i), name, position=pos) for i, (name, pos) in enumerate([
                      ('奥恩', '1,1'), ('苍蓝哨戒', '1,1'), ('远古巨龙', '5,1'),
                      ('绯红树怪', None), ('峡谷迅捷蟹', 'bad'), ('远古石甲虫', '4,7')])]})
    comp = db._guide_to_comp(guide)
    comp.guide = guide
    units = comp_detail(comp, db)['visualization']['stages'][0]['units']
    assert len(units) == 6
    assert [u['position'] for u in units] == [None, None, None, None, None, [4, 7]]
    assert all('冲突' in u['position_note'] for u in units[:2])
    assert units[2]['position_raw'] == '5,1' and '越界' in units[2]['position_note']
    assert units[3]['position_note'] == '未提供站位'
    assert units[4]['position_raw'] == 'bad'
    assert units[5]['position_note'] == ''


def test_traits_deduplicate_names_and_do_not_count_other_units_or_activate_special(db):
    hero = db.hero('奥恩')
    db.heroes_by_name['奥恩'] = replace(hero, traits=('特殊', '永恒之森'))
    db.traits['特殊'] = Trait('special', '特殊', 'origin', (0,), levels=(TraitLevel(None, '必须满足特殊条件'),), activation='special')
    guide = Guide('traits', '羁绊依据', None, None,
                  boards={'final': [GuideHero('11500', '奥恩'), GuideHero('11500', '奥恩'),
                                    GuideHero('pet', '苍蓝哨戒', unit_type='pet')]})
    comp = db._guide_to_comp(guide)
    comp.guide = guide
    traits = {t['name']: t for t in comp_detail(comp, db)['visualization']['stages'][0]['traits']}
    assert set(traits) == {'特殊', '永恒之森'}
    assert traits['特殊']['count'] == traits['永恒之森']['count'] == 1
    assert traits['特殊']['tier'] is None and traits['特殊']['activation'] == 'special'
    assert traits['永恒之森']['tier'] == 0 and traits['永恒之森']['breakpoints'] == [3, 5, 7]


def test_item_recipes_keep_duplicate_parts_and_missing_entities(db):
    comp = db.comps['104']
    comp.heroes.append(CompHero('missing', '未收录英雄', 0, items=('未收录装备',)))
    visual = comp_detail(comp, db)['visualization']
    recipe = visual['items']['无尽之刃']['components']
    assert [p['name'] for p in recipe] == ['暴风大剑', '暴风大剑']
    assert len(recipe) == 2 and all(p['known'] for p in recipe)
    assert visual['items']['未收录装备']['known'] is False
    assert visual['items']['未收录装备']['components'] == []
    assert visual['heroes']['未收录英雄']['known'] is False
    assert visual['guide'] is None and len(visual['stages']) == 1


def test_disabled_guide_cannot_leak_stages_text_entities_or_items(db):
    guide = Guide('secret', '关闭渠道的原文', 'S', None, channels=('ingame',),
                  early_info='关闭渠道思路', carry_items=['攻略独有装备'], recommended_augments=['攻略独有海克斯'],
                  boards={'early': [GuideHero('11500', '奥恩')]})
    attach(db, guide)
    filtered = db.comp('104', enabled_sources={'dataj'})
    detail = comp_detail(filtered, db)
    assert detail['guide'] is None and detail['visualization']['guide'] is None
    assert len(detail['visualization']['stages']) == 1
    assert '关闭渠道' not in json.dumps(detail, ensure_ascii=False)
    assert '攻略独有装备' not in detail['visualization']['items']
    assert '攻略独有海克斯' not in detail['visualization']['augments']
    assert db.comps['104'].guide is guide


def test_guide_recommendations_keep_source_stage_and_descriptions(db):
    guide = Guide('recommend', '额外资料', None, None, recommended_augments=['飞升'],
                  alternative_augments=['未收录海克斯'], carry_items=['无尽之刃'], carry_items_backup=['暴风大剑'],
                  adventures={'2': ['未收录仙灵']})
    visual = comp_detail(attach(db, guide), db)['visualization']
    assert visual['guide']['adventures'] == [{'stage': '2', 'names': ['未收录仙灵']}]
    assert visual['augments']['飞升']['description'] == db.augment('飞升').description
    assert visual['augments']['未收录海克斯']['known'] is False
    assert visual['mechanics']['未收录仙灵']['known'] is False
    assert visual['guide']['carry_items'] == ['无尽之刃']
