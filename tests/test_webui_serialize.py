"""序列化契约使用真实模型和合成数据库，不依赖 GUI、网络或运行目录。"""
from dataclasses import replace
from copy import deepcopy
import json

import pytest

from conftest import make_state
from jccbrain.core.models import GameState, Unit, Phase, ShopSlot
from jccbrain.actuator.layout import Layout
from jccbrain.core.settings import HudSettings
from jccbrain.data.models import Encounter, Guide, GuideHero, Mechanic
from jccbrain.engine.advisor import Advice, AugmentAdvice, BuyAdvice, HeuristicAdvisor, TargetOdds
from jccbrain.engine.comps import CompMatch
from jccbrain.webui import serialize


def test_state_preserves_values_and_unknown_units(db):
    model = GameState.from_dict(make_state(timer=23, interest=5, streak={'kind': 'win', 'n': 3},
                                         bench=[{'name': '奥恩', 'star': 2, 'slot': 0}],
                                         board=[{'name': '?1', 'x': .321234, 'y': .5}], units_stale=True))
    data = serialize.state(model, db)
    json.dumps(data)
    assert data['phase'] == 'planning'
    assert data['xp'] == [0, 36] and data['streak'] == {'kind': 'win', 'n': 3}
    assert data['bench'][0]['cost'] == 1 and data['bench'][0]['traits']
    assert data['board'][0]['cost'] is None and data['board'][0]['traits'] == []
    assert data['board'][0]['x'] == .321234 and not data['board'][0]['known']
    assert data['summary'] == model.summary() and data['units_stale']


def test_empty_state_is_json(db):
    data = serialize.state(GameState(), db)
    json.dumps(data)
    assert data['xp'] is None and data['streak'] is None
    assert data['shop'] == [None] * 5
    assert serialize.unit(Unit('?'), db)['cost'] is None


def test_advisor_payload_preserves_odds_and_fit(db, rules):
    model = GameState.from_dict(make_state(bench=[{'name': '绯红树怪', 'slot': 0}]))
    original = HeuristicAdvisor(db, rules).advise(model)
    data = serialize.advice(original)
    json.dumps(data)
    assert data['phase'] == 'planning'
    assert data['target']['id'] == original.target.comp.id
    assert data['comps'][0]['fit'] == original.comps[0].fit
    assert all('均名' not in line for row in data['comps'] for line in row['why'])
    assert data['econ']['action'] == original.econ.action.value
    assert isinstance(data['targets'][0]['per_roll'], float)
    assert data['targets'][0]['gold_for_50'] == original.targets[0].odds.gold_for_50
    assert data['targets'][0]['have'] == original.targets[0].have


def test_augments_choose_exactly_one_best_on_tie():
    original = Advice(None, GameState().phase, augments=[AugmentAdvice(0, 'A', .5, 'a'),
                                                        AugmentAdvice(1, 'B', .5, 'b')])
    data = serialize.advice(original)
    json.dumps(data)
    assert [row['best'] for row in data['augments']] == [True, False]


def test_absent_target_odds_stays_null():
    original = Advice(None, GameState().phase, targets=[TargetOdds('奥恩', 1, 2, 2, None)])
    data = serialize.advice(original)
    json.dumps(data)
    assert data['target'] is None and data['econ'] is None
    row = data['targets'][0]
    assert row['name'] == '奥恩' and row['have'] == 1 and row['need'] == 2
    assert all(row[key] is None for key in ('cost', 'remaining', 'per_roll', 'gold_for_50'))


def test_missing_level_preserves_known_costs_without_fabricating_odds(db, rules):
    original = HeuristicAdvisor(db, rules).advise(GameState.from_dict(make_state(level=None)))
    assert original.target and original.targets and all(row.odds is None for row in original.targets)
    snapshot = deepcopy(original)
    data = serialize.advice(original)
    expected = {hero.name: hero.cost for hero in original.target.comp.heroes}
    assert [row['name'] for row in data['targets']] == [row.name for row in original.targets]
    assert [row['cost'] for row in data['targets']] == [expected[row.name] for row in original.targets]
    assert all(row[key] is None for row in data['targets'] for key in ('remaining', 'per_roll', 'gold_for_50'))
    assert original == snapshot


@pytest.mark.parametrize('cost', [1, 2, 3, 4, 5])
def test_no_odds_cost_uses_current_target_hero(db, cost):
    comp = deepcopy(db.comps_ranked()[0])
    comp.heroes = [replace(comp.heroes[0], cost=cost)]
    original = Advice(None, Phase.PLANNING, comps=[CompMatch(comp, 1, 1)],
                      targets=[TargetOdds(comp.heroes[0].name, 1, 2, 2, None)])
    assert serialize.advice(original)['targets'][0] == {
        'name': comp.heroes[0].name, 'have': 1, 'need': 2, 'target_star': 2, 'cost': cost,
        'remaining': None, 'per_roll': None, 'gold_for_50': None,
    }


@pytest.mark.parametrize('cost', [None, 0, -1, 6, 1.5, float('nan'), float('inf'), '5', True])
def test_no_odds_rejects_unknown_or_invalid_comp_cost(db, cost):
    comp = deepcopy(db.comps_ranked()[0])
    comp.heroes = [replace(comp.heroes[0], cost=cost)]
    original = Advice(None, Phase.PLANNING, comps=[CompMatch(comp, 1, 1)],
                      targets=[TargetOdds(comp.heroes[0].name, 0, 3, 2, None)])
    row = serialize.advice(original)['targets'][0]
    assert all(row[key] is None for key in ('cost', 'remaining', 'per_roll', 'gold_for_50'))


def test_no_odds_unknown_name_does_not_borrow_another_hero_cost(db):
    comp = db.comps_ranked()[0]
    original = Advice(None, Phase.PLANNING, comps=[CompMatch(comp, 1, 1)],
                      targets=[TargetOdds('未确认英雄', 0, 3, 2, None)])
    assert serialize.advice(original)['targets'][0]['cost'] is None


def test_existing_odds_remain_authoritative_and_target_order_is_unchanged(db, rules):
    original = HeuristicAdvisor(db, rules).advise(GameState.from_dict(make_state()))
    assert original.target and all(row.odds for row in original.targets)
    original.target.comp = replace(original.target.comp, heroes=[replace(h, cost=4) for h in original.target.comp.heroes])
    snapshot = deepcopy(original)
    rows = serialize.advice(original)['targets']
    assert [row['name'] for row in rows] == [row.name for row in original.targets]
    for row, source in zip(rows, original.targets):
        assert all(row[key] == getattr(source.odds, key) for key in ('cost', 'remaining', 'per_roll', 'gold_for_50'))
    assert original == snapshot


def test_comp_detail_and_statistics(db):
    comp = db.comps_ranked()[0]
    row = serialize.comp_row(comp)
    detail = serialize.comp_detail(comp, db)
    json.dumps(detail)
    assert row['top4_rate'] == 68.9  # dataj 百分数原样保留
    assert row['source_label'] == 'dataj' and row['style_label'] == '追三星'
    assert any(hero['carry'] for hero in row['core'])
    assert detail['heroes'][0]['position'] == [4, 7]
    assert detail['heroes'][0]['role'] == '主C'
    assert detail['guide'] is None
    assert detail['augments'][0]['placement_delta'] == comp.augments[0].placement_delta


@pytest.fixture
def rich_db(db):
    db.mechanics['test'] = Mechanic('test', '测试仙灵', '测试分类', 3, '仙灵效果', (('2-1', '3-1'),), ('升级效果',))
    db.encounters['test'] = Encounter('test', '测试奇遇', '奇遇效果')
    db.guides['test'] = Guide('test', '测试攻略', 'S', None, early_info='过渡描述', author='作者',
                              recommended_augments=['飞升'], equipment_order=['暴风大剑'], three_star=['奥恩'],
                              boards={'final': [GuideHero('11500', '奥恩', cost=1)]})
    return db


def test_official_comp_guide_has_no_model_cycles(rich_db):
    comp = rich_db.comps_ranked()[0]
    comp.source, comp.category, comp.guide_id = 'official', '新手推荐', 'test'
    comp.guide = rich_db.guides['test']
    comp.guide.comp = comp
    detail = serialize.comp_detail(comp, rich_db)
    json.dumps(detail)
    assert detail['source_label'] == '官网·新手推荐'
    assert detail['guide']['text'] == '【前期过渡】过渡描述'
    assert detail['guide']['author'] == '作者'
    assert detail['guide']['recommended_augments'] == ['飞升']


@pytest.mark.parametrize('kind,query', [('heroes', '奥恩'), ('traits', '永恒之森'), ('items', '暴风大剑'),
                                        ('augments', '飞升'), ('mechanics', '仙灵效果'),
                                        ('encounters', '奇遇效果'), ('guides', '过渡描述')])
def test_database_categories_and_filter(rich_db, kind, query):
    rows = serialize.db_rows(rich_db, kind, query)
    assert rows
    json.dumps(rows)
    assert serialize.db_rows(rich_db, kind, '绝不匹配的名字') == []


def test_hero_description_not_truncated(db):
    hero = db.hero('奥恩')
    db.heroes_by_name[hero.name] = replace(hero, skill_desc='完整技能' * 100)
    rows = serialize.db_rows(db, 'heroes', '奥恩')
    assert rows[0]['skill_desc'] == '完整技能' * 100


def test_trait_levels_are_structured(db):
    rows = serialize.db_rows(db, 'traits')
    assert rows and all('description' in row and isinstance(row['levels'], list) for row in rows)
    assert any(row['levels'] for row in rows)
    assert set(next(row['levels'][0] for row in rows if row['levels'])) == {'count', 'text', 'values'}


def test_unknown_category_is_error(db):
    with pytest.raises(ValueError, match='未知的数据类别：unknown'):
        serialize.db_rows(db, 'unknown')


def test_unit_and_item_bench_preserve_equipment_dictionaries(db):
    item = {'name': '无尽之刃', 'kind': '成装', 'x': .35, 'y': .55, 'slot': 1}
    component = {'name': '暴风大剑', 'kind': '散件', 'x': .1, 'y': .8}
    st = GameState(board=[Unit('奥恩', items=[item])], item_bench=[component])
    result = serialize.state(st, db)
    json.dumps(result)
    assert result['board'][0]['items'] == [item]
    assert result['item_bench'] == [component]
    assert serialize.unit(Unit('奥恩'), db)['items'] == []


def test_hud_idle_without_state(db):
    cfg = HudSettings(opacity=.75, font_size=15)
    assert serialize.hud(None, None, db, Layout(), cfg) == {
        'mode': 'idle', 'opacity': .75, 'font_size': 15, 'scout_notice': None, 'units': [], 'shop': [], 'augments': [],
        'positions': [], 'arrows': [],
    }


def test_hud_board_coordinates_titles_and_flags(db, rules):
    layout = Layout(bench_x0=.2, bench_step=.07, bench_y=.73)
    st = GameState(phase=Phase.PLANNING, board=[Unit('绯红树怪', 2, x=.42, y=.51),
                                               Unit('?1', x=.3, y=.4), Unit('奥恩')],
                   bench=[Unit('奥恩', 3, 'bench', 4), Unit('?2', zone='bench')])
    adv = HeuristicAdvisor(db, rules).advise(st)
    result = serialize.hud(st, adv, db, layout, HudSettings())
    json.dumps(result)
    assert result['mode'] == 'board' and len(result['units']) == 3
    board, unknown, bench = result['units']
    assert board['x'] == pytest.approx(.42, abs=1e-6)
    assert board['y_bottom'] == pytest.approx(.51 - .022, abs=1e-6)
    bx, by = layout.bench_slot(4)
    assert bench['x'] == pytest.approx(bx, abs=1e-6)
    assert bench['y_bottom'] == pytest.approx(by - .075, abs=1e-6)
    assert board['title'] == '★★ 绯红树怪' and bench['title'] == '★★★ 奥恩'
    assert board['target'] and board['cost'] == db.hero('绯红树怪').cost and board['traits']
    assert unknown['unknown'] and unknown['title'] == '★ ?1 未识别' and unknown['cost'] == 0
    assert [u['key'] for u in result['units']] == ['board:0', 'board:1', 'bench:4']


@pytest.mark.parametrize('show_names,show_traits', [(False, True), (True, False)])
def test_hud_name_and_trait_settings(db, show_names, show_traits):
    st = GameState(board=[Unit('奥恩', x=.4, y=.5)])
    result = serialize.hud(st, None, db, Layout(), HudSettings(show_names=show_names, show_traits=show_traits))
    assert not result['units'] if not show_names else result['units'][0]['traits'] == ''


def test_hud_shop_buys_take_precedence_and_use_qt_rectangles(db, rules):
    st = GameState(phase=Phase.PLANNING, shop_open=True,
                   shop=[ShopSlot('绯红树怪'), ShopSlot('绯红树怪'), ShopSlot('卡兹克'), None, None])
    adv = HeuristicAdvisor(db, rules).advise(st)
    adv.buys = [BuyAdvice(0, '绯红树怪', 3, '追目标卡', 1)]
    layout = Layout()
    result = serialize.hud(st, adv, db, layout, HudSettings())
    assert [s['kind'] for s in result['shop']] == ['buy', 'target']
    assert [s['text'] for s in result['shop']] == ['买 · 追目标卡', '目标卡']
    for slot in result['shop']:
        assert slot['x'] == pytest.approx(layout.shop_slots[slot['key']][0] - .118 / 2, abs=1e-6)
        assert (slot['y'], slot['w'], slot['h']) == (.838, .118, .15)
    st.shop_open = False
    assert serialize.hud(st, adv, db, layout, HudSettings())['shop'] == []
    assert serialize.hud(st, None, db, layout, HudSettings())['shop'] == []


def test_hud_augments_only_and_single_best(db):
    st = GameState(phase=Phase.AUGMENT, board=[Unit('奥恩', x=.4, y=.5)], shop_open=True)
    adv = Advice(None, Phase.AUGMENT, augments=[AugmentAdvice(0, 'A', .8, ''),
                                              AugmentAdvice(1, 'B', .8, ''), AugmentAdvice(5, '无效', .1, '')])
    layout = Layout()
    result = serialize.hud(st, adv, db, layout, HudSettings())
    assert result['mode'] == 'augments' and not result['units'] and not result['shop']
    assert [a['best'] for a in result['augments']] == [True, False]
    assert result['augments'][0]['text'] == '规则推荐 · 评分 0.800'
    for a in result['augments']:
        x, y = layout.augment_options[a['key']]
        assert a['x'] == pytest.approx(x, abs=1e-6)
        assert a['y'] == pytest.approx(y - .17, abs=1e-6)
    adv.augments = []
    assert serialize.hud(st, adv, db, layout, HudSettings())['mode'] == 'board'


@pytest.mark.parametrize('headline,summary', [
    ('存钱；买 任意英雄甲、任意英雄乙；走 任意阵容', '存钱；走 任意阵容'),
    ('买 任意英雄', '观望'), ('前期按强度过渡（参考 买路财）', '前期按强度过渡（参考 买路财）'),
])
def test_purchase_display_metadata_preserves_original_advice(db, rules, headline, summary):
    original = HeuristicAdvisor(db, rules).advise(GameState.from_dict(make_state()))
    original.headline = headline
    original.comps[0].why = ['已有 2/8', '商店有 任意英雄甲、任意英雄乙', '主C剩余 12', '均名 4.2']
    snapshot = deepcopy(original)
    data = serialize.advice(original)
    assert data['headline'] == headline and data['headline_summary'] == summary
    assert data['comps'][0]['why'] == ['已有 2/8', '商店有 任意英雄甲、任意英雄乙', '主C剩余 12']
    assert data['comps'][0]['why_parts'] == [
        {'kind': 'strategy', 'text': '已有 2/8'},
        {'kind': 'shop', 'text': '商店有 任意英雄甲、任意英雄乙'},
        {'kind': 'strategy', 'text': '主C剩余 12'},
    ]
    assert original == snapshot
