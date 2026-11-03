"""装备/海克斯的数量分配、规则与运行期生命周期；只用合成数据。"""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jccbrain.context import AppContext
from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.core.settings import Settings
from jccbrain.data.models import Item, Guide, GuideHero
from jccbrain.engine import items
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher
from jccbrain.engine.llm import build_context, SYSTEM_PROMPT
from jccbrain.modes.assistant import AssistantMode, advice_signature
from jccbrain.modes.base import ModeSink
from jccbrain.webui.api import Api
from jccbrain.webui.serialize import advice as serialize_advice


@pytest.fixture
def item_db(db):
    db.items.update({i.id: i for i in [
        Item('tear', '女神之泪', '基础装备'), Item('glove', '拳套', '基础装备'),
        Item('spear', '朔极之矛', '成型装备', recipe=('1001', 'tear')),
        Item('emblem', '野怪纹章', '转职纹章', trait_name='峡谷野怪'),
        Item('bright', '光明装备', '光明'), Item('artifact', '神器装备', '神器'),
    ]})
    return db


def equipped(*names):
    return [{'name': name} for name in names]


def only_wishes(db, *wishes):
    base = db.comp('104')
    return replace(base, heroes=[replace(base.heroes[0], items=tuple(wishes))])


def test_held_classifies_short_kinds_and_equipped_items(item_db):
    st = GameState(item_bench=[{'name': '暴风大剑', 'kind': '基础'}, {'name': '野怪纹章', 'kind': '纹章'},
                              {'name': '光明装备', 'kind': '光明'}, {'name': '?未知', 'kind': '成型'},
                              {'name': '未识别纹章', 'kind': '纹章'}],
                   board=[Unit('绯红树怪', items=equipped('神器装备', '无尽之刃'))],
                   bench=[Unit('奥恩', zone='bench', items=equipped('女神之泪'))])
    inv = items.held(st, item_db)
    assert inv.components == Counter({'暴风大剑': 1, '女神之泪': 1})
    assert inv.completed == ['光明装备', '神器装备', '无尽之刃']
    assert inv.emblems == [('野怪纹章', '峡谷野怪')]


def test_wishlist_source_order_dedup_and_limit(item_db):
    comp = only_wishes(item_db, '无尽之刃', '朔极之矛')
    comp.guide = Guide('g', '测试', None, None, carry_items=['朔极之矛', '攻略一', '攻略二'],
                       boards={'final': [GuideHero('h', '绯红树怪', is_carry=True, items=('攻略二', '攻略三', '攻略四', '超额'))]})
    assert items.wishlist(comp) == ['无尽之刃', '朔极之矛', '攻略一', '攻略二', '攻略三', '攻略四']


def test_guide_only_wishlist_and_recipient(item_db):
    comp = only_wishes(item_db)
    comp.guide = Guide('g', '测试', None, None, boards={'final': [GuideHero('h', '绯红树怪', is_carry=True, items=('朔极之矛',))]})
    assert items.wishlist(comp) == ['朔极之矛']
    assert items.recipient(comp, '朔极之矛') == '绯红树怪'


def test_fit_completed_first_preserves_components_for_second_item(item_db):
    comp = only_wishes(item_db, '无尽之刃', '朔极之矛')
    inv = items.HeldItems(Counter({'暴风大剑': 1, '女神之泪': 1}), ['无尽之刃'])
    assert items.item_fit(comp, inv, item_db) == (1, ['无尽之刃', '朔极之矛'])
    assert inv.components == Counter({'暴风大剑': 1, '女神之泪': 1})


def test_fit_does_not_reuse_components_or_double_count_same_recipe(item_db):
    comp = only_wishes(item_db, '无尽之刃', '朔极之矛')
    assert items.item_fit(comp, items.HeldItems(Counter({'暴风大剑': 2, '女神之泪': 1})), item_db) == (.5, ['无尽之刃'])
    assert items.item_fit(comp, items.HeldItems(Counter({'暴风大剑': 1})), item_db) == (0, [])
    assert items.item_fit(only_wishes(item_db), items.HeldItems(), item_db) == (0, [])


@pytest.mark.parametrize('stage,bonus', [('2-5', .12), ('3-1', .06)])
def test_equipment_score_stage_weight(item_db, stage, bonus):
    matcher = CompMatcher([only_wishes(item_db, '无尽之刃')], item_db)
    base = matcher.match(GameState(stage=stage))[0]
    result = matcher.match(GameState(stage=stage, item_bench=equipped('暴风大剑', '暴风大剑')))[0]
    assert result.score - base.score == pytest.approx(bonus)
    assert result.fit == base.fit and '装备契合：无尽之刃' in result.why


def test_emblem_bonus_is_capped_and_requires_main_trait(item_db):
    matcher = CompMatcher([item_db.comp('104')], item_db)
    base = matcher.match(GameState())[0]
    st = GameState(item_bench=equipped('野怪纹章', '野怪纹章', '野怪纹章'))
    result = matcher.match(st)[0]
    assert result.score - base.score == pytest.approx(.08) and '纹章 野怪纹章' in result.why
    other = replace(item_db.comp('104'), traits=['其他'])
    assert CompMatcher([other], item_db).match(st)[0].score == CompMatcher([other], item_db).match(GameState())[0].score


@pytest.mark.parametrize('delta,bonus', [(-.3, .03), (-1, .05), (.9, 0), (None, 0)])
def test_taken_augment_bonus_and_dedup(item_db, delta, bonus):
    comp = item_db.comp('104')
    comp = replace(comp, augments=[replace(comp.augments[0], placement_delta=delta)])
    matcher = CompMatcher([comp], item_db)
    before = matcher.match(GameState())[0]
    after = matcher.match(GameState(), augments_taken=['飞升', '飞升'])[0]
    assert after.score - before.score == pytest.approx(bonus)
    assert any('海克斯 飞升 适配' in why for why in after.why) == bool(bonus)


@pytest.mark.parametrize('owned,parts,ready,fragment', [
    (True, ['暴风大剑', '暴风大剑'], True, '合成 无尽之刃'),
    (True, ['暴风大剑'], False, '还差 暴风大剑'),
    (False, ['暴风大剑', '暴风大剑'], False, '当前没有已知上场单位'),
])
def test_item_plan_ready_partial_and_absent_carry(item_db, owned, parts, ready, fragment):
    st = GameState(board=[Unit('绯红树怪')] if owned else [], item_bench=equipped(*parts))
    plan = items.plan(only_wishes(item_db, '无尽之刃'), items.held(st, item_db), st, item_db)
    assert len(plan) == 1 and plan[0]['ready'] == ready
    assert plan[0]['final_to'] == '绯红树怪' and fragment in plan[0]['reason']
    assert plan[0]['give_to'] == ('绯红树怪' if ready else '')


def test_plan_assigns_available_completed_and_uses_each_component_once(item_db):
    comp = only_wishes(item_db, '无尽之刃', '朔极之矛')
    st = GameState(board=[Unit('绯红树怪')], item_bench=equipped('暴风大剑', '暴风大剑', '女神之泪'))
    assert [p['ready'] for p in items.plan(comp, items.held(st, item_db), st, item_db)] == [True, False]
    st.item_bench = equipped('无尽之刃', '暴风大剑', '女神之泪')
    result = items.plan(comp, items.held(st, item_db), st, item_db)
    assert [p['item'] for p in result] == ['无尽之刃', '朔极之矛']
    assert [p['action'] for p in result] == ['佩戴', '合成']


def test_partial_wishlist_cannot_starve_another_ready_item(item_db):
    comp = only_wishes(item_db, '无尽之刃', '朔极之矛')
    st = GameState(board=[Unit('绯红树怪')], item_bench=equipped('暴风大剑', '女神之泪'))
    plan = items.plan(comp, items.held(st, item_db), st, item_db)
    assert len(plan) == 1 and plan[0]['item'] == '朔极之矛' and plan[0]['ready']


@pytest.mark.parametrize('stage,hp,streak,warning', [('2-3', 69, None, True), ('2-3', 90, {'kind': 'loss', 'n': 2}, True),
                                                 ('2-3', 70, None, False), ('3-1', 60, None, False)])
def test_early_hoarding_warning(item_db, stage, hp, streak, warning):
    st = GameState(stage=stage, hp=hp, streak=streak, item_bench=equipped('暴风大剑', '暴风大剑'))
    plan = items.plan(item_db.comp('104'), items.held(st, item_db), st, item_db)
    assert any('2 阶段别囤装备' in p['reason'] for p in plan) == warning


def test_carousel_suggests_missing_component_instead_of_completed_item(item_db):
    comp = only_wishes(item_db, '朔极之矛')
    hint = items.carousel_hint(comp, items.HeldItems(Counter({'暴风大剑': 1})), item_db)
    assert hint == '选秀优先拿 女神之泪（补出 朔极之矛）'
    assert items.carousel_hint(comp, items.HeldItems(completed=['朔极之矛']), item_db) == '选秀优先拿主C或主C所需散件'


def test_advisor_serialization_signature_and_llm_inventory(item_db, rules):
    advisor = HeuristicAdvisor(item_db, rules)
    st = GameState(stage='2-4', phase=Phase.PLANNING, level=5, gold=30,
                   item_bench=equipped('暴风大剑', '暴风大剑'), board=[Unit('绯红树怪')])
    adv = advisor.advise(st, augments_taken=['飞升'])
    result = serialize_advice(adv)
    context = build_context(st, adv)
    json.dumps(result)
    assert result['item_plan'][0]['ready'] and context['item_plan'] == result['item_plan']
    assert context['items']['components'] == {'暴风大剑': 2} and context['augments_taken'] == ['飞升']
    assert any('选秀优先拿' in n for n in adv.notes)
    assert '结合玩家当前装备与已选海克斯' in SYSTEM_PROMPT
    assert advice_signature(adv) != advice_signature(replace(adv, item_plan=[]))
    assert advice_signature(adv) != advice_signature(replace(adv, augments_taken=[]))


@pytest.fixture
def item_api(item_db, rules, tmp_path, monkeypatch):
    monkeypatch.setenv('JCC_HOME', str(tmp_path))
    ctx = AppContext(Settings())
    ctx.db, ctx.rules = item_db, rules
    events = []
    api = Api(ctx, SimpleNamespace(emit=lambda key, value: events.append((key, value))))
    yield api, ctx, events
    api._icons.close()


def test_automatic_augments_no_save_and_status(item_api):
    api, ctx, events = item_api
    ctx.settings.save = Mock()
    mode = AssistantMode(ctx, api._sink)
    assert not hasattr(api, 'mark_augment') and not hasattr(api, 'unmark_augment')
    for names in (['飞 升'], ['飞升', '飞升'], []):
        mode.compute_advice(GameState(stage='2-5', augments_taken=names))
        assert api.status()['data']['augments_taken'] == ['飞升']
    ctx.settings.save.assert_not_called()
    assert events[-1][0] == 'status' and events[-1][1]['augments_taken'] == ['飞升']


def test_mode_new_game_clears_taken_augments_and_callback(item_api):
    _, ctx, _ = item_api
    sink = Mock(spec=ModeSink)
    mode = AssistantMode(ctx, sink)
    adv = mode.compute_advice(GameState(stage='2-5', augments_taken=['飞升']))
    sink.on_augments_changed.reset_mock()
    assert adv.augments_taken == ['飞升']
    for _ in range(3): adv = mode.compute_advice(GameState(stage='1-2'))
    assert ctx.game_augments == [] and adv.augments_taken == []
    sink.on_augments_changed.assert_called_once_with()
