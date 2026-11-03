"""情境分析、两种权重及AI选择的端到端契约；全部使用离线合成库。"""
from dataclasses import replace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ai_fakes import authorize_ai

from jccbrain.core.models import GameState, Phase, AugmentOption, Unit
from jccbrain.core.settings import HudSettings
from jccbrain.actuator.layout import Layout
from jccbrain.data.models import Augment, Item
from jccbrain.engine import augments
from jccbrain.engine.advisor import HeuristicAdvisor, Advice
from jccbrain.engine.comps import CompMatch
from jccbrain.engine.llm import build_context, parse_reply, SYSTEM_PROMPT
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.base import ModeSink
from jccbrain.webui import serialize
from jccbrain.context import AppContext
from jccbrain.core.settings import Settings


@pytest.fixture
def context(db, rules):
    db.items['i2'] = Item('i2', '巨人杀手', '成型装备')
    db.items['i3'] = Item('i3', '鬼索的狂暴之刃', '成型装备')
    db.items['e'] = Item('e','峡谷野怪纹章','转职纹章', trait_name='峡谷野怪')
    comp = db.comp('104')
    comp.heroes[0] = replace(comp.heroes[0], items=('无尽之刃','巨人杀手','鬼索的狂暴之刃'), star_target=2)
    for a in [Augment('p','潘多拉的装备',2,'获得随机装备，每回合重铸装备。'),
              Augment('ticket','棱彩门票',3,'每当你的商店刷新时，有45%几率获得1次免费刷新。'),
              Augment('j','珠光莲花 I',2,'己方获得15%暴击几率和技能暴击。'),
              Augment('wild','野怪之力',2,'峡谷野怪获得攻击力。', fetter_id=db.trait('峡谷野怪').id),
              Augment('forest','森林之力',2,'永恒之森获得生命值。'),
              Augment('carry','树怪强化',2,'获得一个绯红树怪并提升其星级。', hero_enhancement_type='1')]:
        db.augments[a.id] = a
        db.augments_by_name[a.name] = a
    ctx = AppContext(Settings())
    ctx.db, ctx.rules = db, rules
    return ctx


def state(**kwargs):
    return GameState(stage='3-2', phase=Phase.AUGMENT, gold=50, hp=80, **kwargs)


def profile(ctx, st=None):
    return augments.situation(st or state(), Advice('3-2',Phase.AUGMENT, comps=[CompMatch(ctx.db.comp('104'),1,1)]), ctx.db)


def info(*category, traits=(), heroes=()):
    return {'category':list(category),'traits':list(traits),'heroes':list(heroes),'tier':2,'description':''}


@pytest.mark.parametrize('name,category', [('潘多拉的装备','item'), ('棱彩门票','econ'),
    ('珠光莲花 I','combat'), ('野怪之力','trait'), ('树怪强化','hero')])
def test_classification_real_style_names_and_metadata(context, name, category):
    assert category in augments.classify(name, context.db)['category']


def test_classification_multi_category_and_missing_db(context):
    result=augments.classify('未知选项',context.db,'获得装备和金币，强化峡谷野怪与绯红树怪的攻击。')
    assert result['category'] == ['item','econ','trait','hero','combat']
    assert result['traits'] == ['峡谷野怪'] and result['heroes'] == ['绯红树怪']
    assert augments.classify('潘多拉的装备 II',context.db)['category'] == ['item']
    assert augments.classify('未知',context.db)['category'] == []


def test_nonzero_metadata_only_establishes_category(context):
    a=Augment('meta','元数据项',3,'',fetter_type='1',hero_enhancement_type='1')
    context.db.augments_by_name[a.name]=a
    result=augments.classify(a.name,context.db)
    assert result['category']==['trait','hero'] and result['traits']==result['heroes']==[]


def test_situation_counts_equipment_components_and_held_emblems(context):
    st=state(item_bench=[{'name':'暴风大剑'},{'name':'暴风大剑'},{'name':'峡谷野怪纹章'}],
             board=[Unit('绯红树怪',items=[{'name':'无尽之刃'}])], streak={'kind':'loss','n':3})
    p=profile(context,st)
    assert p['missing_item_count']==2 and p['component_count']==2
    assert p['missing_items']==['巨人杀手','鬼索的狂暴之刃']
    assert p['held_emblems']==[{'name':'峡谷野怪纹章','trait':'峡谷野怪'}]
    assert p['streak']=={'kind':'loss','n':3} and p['target_traits']==['峡谷野怪']
    assert p['style']=='reroll' and p['stage_major']==3


def test_unknown_economy_health_and_target_are_not_zero(context):
    p=augments.situation(GameState(),Advice(None,Phase.UNKNOWN),context.db)
    assert p['low_gold'] is p['danger'] is p['stage_major'] is None
    assert p['missing_item_count']==0 and p['style'] is None
    assert augments.situation_score(info('econ','combat'),p)[0] == .5


def test_item_shortage_thresholds(context):
    p=profile(context)
    score,reasons=augments.situation_score(info('item'),p)
    assert score==pytest.approx(.8) and '缺 3 件装备' in reasons[0]
    assert augments.situation_score(info('item'),p|{'component_count':3})[0] == .5
    assert augments.situation_score(info('item'),p|{'missing_item_count':1})[0] == .5


@pytest.mark.parametrize('gold,stage,style,expected', [(29,3,'fast8',.7),(30,3,'fast8',.5),
    (29,4,'fast8',.5),(60,5,'reroll',.7),(None,3,'fast8',.5)])
def test_econ_tempo_boundaries(context,gold,stage,style,expected):
    p=profile(context)|{'gold':gold,'low_gold':gold<30 if gold is not None else None,'stage_major':stage,'style':style}
    assert augments.situation_score(info('econ'),p)[0] == pytest.approx(expected)


def test_low_health_penalizes_econ_and_rewards_combat(context):
    p=profile(context)|{'danger':True,'hp':20,'low_gold':True}
    assert augments.situation_score(info('econ'),p)[0] == pytest.approx(.4)
    assert augments.situation_score(info('combat'),p)[0] == pytest.approx(.75)
    assert augments.situation_score(info('combat'),p|{'danger':False,'hp':40})[0] == .5


def test_trait_match_mismatch_and_emblem(context):
    p=profile(context)
    assert augments.situation_score(info('trait',traits=['峡谷野怪']),p)[0] == .75
    assert augments.situation_score(info('trait',traits=['永恒之森']),p)[0] == .35
    assert augments.situation_score(info('trait',traits=['永恒之森']),p|{'held_emblems':[{'trait':'永恒之森'}]})[0]==.75
    assert augments.situation_score(info('trait'),p)[0]==.5  # 关联未知不乱扣分


def test_hero_match_and_score_clamp(context):
    p=profile(context)
    assert augments.situation_score(info('hero',heroes=['绯红树怪']),p)[0] == .7
    assert augments.situation_score(info('hero',heroes=['不在阵容']),p)[0] == .5
    assert augments.situation_score(info('item','trait','hero',traits=['峡谷野怪'],heroes=['绯红树怪']),p)[0] == 1


def test_statistical_and_missing_statistics_weights_and_reasons(context):
    p=profile(context)
    stats=context.db.comp('104').augments[0]
    row=augments.rate('潘多拉的装备',replace(stats,placement_delta=-.3),info('item'),p)
    assert row['base_score']==.65 and row['situation_score']==.8 and row['stats_weight']==.6
    assert row['score']==round(.6*.65+.4*.8,3)
    assert '阵容数据：均名 -0.30' in row['reason'] and '局势：缺 3 件装备' in row['reason']
    generic=augments.rate('潘多拉的装备',None,info('item'),p)
    assert generic['score']==round(.5*.4+.5*.8,3) and generic['stats_weight']==.5
    assert '无该阵容统计' in generic['reason']
    avg=augments.rate('x',replace(stats,placement_delta=None,avg_placement=3.5),info('combat'),p)
    assert avg['stats_weight']==.6 and avg['base_score']==pytest.approx(4.5/7)


def test_advisor_scores_situation_and_llm_context(context):
    st=state(augments=[AugmentOption('潘多拉的装备'),AugmentOption('棱彩门票'),AugmentOption('珠光莲花 I')])
    a=context.advisor.advise(st)
    assert max(a.augments,key=lambda r:r.score).name=='潘多拉的装备'
    assert '潘多拉的装备' in a.headline and a.augment_situation['missing_item_count']==3
    c=build_context(st,a)
    assert c['augment_situation']==a.augment_situation
    assert set(c['augment_options'][0])=={'name','category','desc','rule_score'}
    assert c['augment_options'][0]['category']==['item']
    a.augments[0].desc='长'*100
    assert len(build_context(st,a)['augment_options'][0]['desc'])==80
    assert '不要只看阵容推荐' in SYSTEM_PROMPT and 'headline必须' in SYSTEM_PROMPT
    json.dumps(c,allow_nan=False)


@pytest.mark.parametrize('text,expected', [('选飞升','飞升'),('不选飞升',None),('飞升不推荐',None),
    ('选潘多拉的装备 II','潘多拉的装备 II'),('飞升或潘多拉的装备',None),('选不在选项里的',None),
    ('建议 潘多拉 的装备','潘多拉的装备')])
def test_ai_choice_parses_only_current_unique_name(text,expected):
    assert augments.recommended_augment(text,['飞升','潘多拉的装备','潘多拉的装备 II'])==expected


def test_ai_disagreement_serialization_and_hud_badges(context):
    st=state(augments=[AugmentOption('潘多拉的装备'),AugmentOption('棱彩门票'),AugmentOption('珠光莲花 I')])
    a=context.advisor.advise(st)
    mode=AssistantMode(context,ModeSink())
    mode._llm=(st.stage,parse_reply('{"headline":"选择珠光莲花 I","reason":"血量告急需要即时战力"}'))
    mode._llm_options=tuple(o.name for o in a.augments)
    authorize_ai(mode, a)
    mode._attach_llm(a)
    assert a.ai_augment=='珠光莲花 I'
    rows=serialize.advice(a)['augments']
    assert [r['name'] for r in rows if r['best']]==['潘多拉的装备']
    assert [r['name'] for r in rows if r['ai_best']]==['珠光莲花 I']
    hud=serialize.hud(st,a,context.db,Layout(),HudSettings())
    assert hud['augments'][0]['best'] and not hud['augments'][0]['ai_best']
    assert '规则推荐' in hud['augments'][0]['text'] and 'AI 推荐' in hud['augments'][2]['text']
    json.dumps(hud,allow_nan=False)
    same=replace(a,ai_augment='潘多拉的装备')
    both=serialize.hud(st,same,context.db,Layout(),HudSettings())['augments'][0]
    assert both['best'] and both['ai_best']


def test_refreshed_options_and_late_reply_cannot_mark_new_choice(context):
    st=state(augments=[AugmentOption('飞升')])
    old=context.advisor.advise(st)
    mode=AssistantMode(context,ModeSink())
    new=context.advisor.advise(replace(st,augments=[AugmentOption('棱彩门票')]))
    mode._latest=new
    mode._llm=(st.stage,parse_reply('{"headline":"飞升"}'))
    mode._llm_options=('飞升',)
    mode._attach_llm(new)
    assert new.ai_augment is new.llm_text is None
    context.llm=SimpleNamespace(advise=lambda *_:parse_reply('{"headline":"飞升"}'))
    mode._ask_llm(st,old)
    assert new.ai_augment is new.llm_text is None


def test_named_equipment_and_delayed_economy_are_not_combat(context):
    result=augments.classify('测试',context.db,'经过8场战斗获得一个无尽之刃和金币。')
    assert result['category']==['item','econ']


@pytest.mark.parametrize('hp,gold,danger,low', [(39,29,True,True),(40,30,False,False)])
def test_profile_health_and_gold_boundaries(context,hp,gold,danger,low):
    p=profile(context,GameState(stage='4-1',hp=hp,gold=gold))
    assert p['danger'] is danger and p['low_gold'] is low
