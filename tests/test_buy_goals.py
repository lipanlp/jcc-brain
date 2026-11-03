"""有效目标份数与一次建议内的持有/合成规划，不操作游戏。"""
from dataclasses import replace
import itertools

import pytest

from jccbrain.core.models import GameState, Phase, ShopSlot, Unit
from jccbrain.data.models import CompHero, Guide, GuideHero
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatch
from jccbrain.engine.economy import EconAction, EconDecision

NAMES = {1:'奥恩', 2:'峡谷迅捷蟹', 3:'远古石甲虫', 4:'厄斐琉斯', 5:'远古巨龙'}
ROLL = EconDecision(EconAction.ROLL, '受控测试', keep_gold=0)


def held(name, copies, zone='board'):
    units = []
    for star, amount in ((3,9), (2,3), (1,1)):
        for _ in range(copies // amount): units.append(Unit(name, star, zone))
        copies %= amount
    return units


def scenario(db, rules, cost=1, star=2, copies=0, *, stage='5-1', level=8, **roles):
    name = NAMES[cost]
    h = CompHero(db.hero(name).id, name, cost, star_target=star, **roles)
    comp = replace(db.comp('104'), heroes=[h], guide=None)
    advisor = HeuristicAdvisor(db, rules, comps_getter=lambda:[comp])
    state = GameState(stage=stage, level=level, gold=100, hp=80, phase=Phase.PLANNING,
        board=held(name,copies), board_count=(level or 1,level or 1), shop_open=True, shop=[ShopSlot(name,cost)])
    return advisor, comp, state


@pytest.mark.parametrize('cost,star,early,done', list(itertools.product(range(1,6), [None,1,2,3], [False,True], [False,True])))
def test_all_costs_stop_at_default_or_explicit_goal_in_every_stage(db, rules, cost, star, early, done):
    goal = star or 2; total={1:1,2:3,3:9}[goal]; have=total if done else total-1
    advisor, comp, st = scenario(db,rules,cost,star,have,stage='2-2' if early else '5-1',level=5 if early else 8,is_carry=True)
    adv=advisor.advise(st,locked_comp=comp.id)
    row,=adv.targets
    assert (row.target_star,row.need)==(goal,total-have)
    assert [b.name for b in adv.buys]==([] if done else [NAMES[cost]])


@pytest.mark.parametrize('cost,have', list(itertools.product(range(1,4), [0,2,3,5,6,8,9])))
def test_late_noncarry_core_only_preserves_explicit_three_star_at_six_copies(db,rules,cost,have):
    advisor, comp, st=scenario(db,rules,cost,3,have,is_core=True)
    adv=advisor.advise(st,locked_comp=comp.id);row,=adv.targets
    goal=2 if have<6 else 3
    assert row.target_star==goal and row.need==max({2:3,3:9}[goal]-have,0)
    assert bool(adv.buys)==(row.need>0)
    assert comp.heroes[0].star_target==3


@pytest.mark.parametrize('stage,level,goal', [('4-7',7,3),('5-1',7,2),('4-2',8,2),
    (None,8,2),('5-1',None,2),(None,None,3),('bad',None,3),('4-7',None,3),(None,7,3)])
def test_late_evidence_requires_known_stage_or_known_level(db,rules,stage,level,goal):
    advisor,comp,st=scenario(db,rules,1,3,3,stage=stage,level=level)
    adv=advisor.advise(st,locked_comp=comp.id)
    assert adv.targets[0].target_star==goal
    assert bool(adv.buys)==(goal==3)


@pytest.mark.parametrize('protection', ['carry','subcarry','recommended','equipped_board','equipped_bench','guide_items','guide_level_items'])
def test_explicit_reroll_role_or_equipment_keeps_three_star(db,rules,protection):
    roles={'is_carry':protection=='carry','is_subcarry':protection=='subcarry',
           'items':('无尽之刃',) if protection=='recommended' else ()}
    advisor,comp,st=scenario(db,rules,1,3,3,**roles)
    if protection.startswith('equipped'):
        st.board[0].items=[{'name':'暴风大剑'}]
        if protection=='equipped_bench':st.bench,st.board=st.board,[];st.bench[0].zone='bench'
    if protection.startswith('guide_'):
        comp.guide=Guide('test','攻略','S',None,boards={'L8' if protection=='guide_level_items' else 'final':[GuideHero('1',NAMES[1],items=('无尽之刃',))]})
    adv=advisor.advise(st,locked_comp=comp.id)
    assert adv.targets[0].target_star==3 and adv.targets[0].need==6
    assert len(adv.buys)==1


@pytest.mark.parametrize('equipment', [[],[{'name':None}],[{'name':''}],[{'name':'?1'}]])
def test_unrecognized_equipment_does_not_invent_a_protection(db,rules,equipment):
    advisor,comp,st=scenario(db,rules,1,3,3);st.board[0].items=equipment
    assert advisor.advise(st,locked_comp=comp.id).targets[0].target_star==2


@pytest.mark.parametrize('cost', [4,5])
def test_high_cost_explicit_three_star_is_not_downgraded(db,rules,cost):
    advisor,comp,st=scenario(db,rules,cost,3,3)
    assert advisor.advise(st,locked_comp=comp.id).targets[0].target_star==3


@pytest.mark.parametrize('have,star,count', [(2,2,1),(8,3,1),(0,1,1),(1,2,2),(6,3,3),(3,2,0),(9,3,0)])
def test_identical_shop_slots_share_remaining_goal(db,rules,have,star,count):
    advisor,comp,st=scenario(db,rules,1,star,have,is_carry=True)
    st.shop=[ShopSlot(NAMES[1],1) for _ in range(5)]
    buys=advisor._buys(st,CompMatch(comp,.8,.8),ROLL)
    assert len(buys)==count and [b.slot for b in buys]==list(range(count))
    assert st.owned_copies().get(NAMES[1],0)==have  # 规划不原地改状态


def test_unaffordable_same_name_slot_does_not_consume_goal(db,rules):
    advisor,comp,st=scenario(db,rules,1,2,2)
    st.gold=1;st.shop=[ShopSlot(NAMES[1],2),ShopSlot(NAMES[1],1)]
    assert [b.slot for b in advisor._buys(st,CompMatch(comp,.8,.8),ROLL)]==[1]


def test_merge_frees_actual_bench_space_for_following_target(db,rules):
    advisor,comp,st=scenario(db,rules,1,2)
    comp.heroes.append(CompHero('x',NAMES[2],2,star_target=2))
    st.bench=held(NAMES[1],2,'bench')+[Unit(f'?占位{i}',1,'bench') for i in range(7)]
    st.shop=[ShopSlot(NAMES[1],1),ShopSlot(NAMES[2],2)]
    assert [b.name for b in advisor._buys(st,CompMatch(comp,.8,.8),ROLL)]==[NAMES[1],NAMES[2]]


def test_planned_five_copy_merge_allows_next_buy_but_not_overfull_bench(db,rules):
    advisor,comp,st=scenario(db,rules,1,3,3,is_carry=True)
    st.bench=held(NAMES[1],2,'bench')+[Unit(f'?占位{i}',1,'bench') for i in range(7)]
    st.shop=[ShopSlot(NAMES[1],1) for _ in range(3)]
    assert len(advisor._buys(st,CompMatch(comp,.8,.8),ROLL))==2


def test_board_merge_does_not_invent_extra_bench_space(db,rules):
    advisor,comp,st=scenario(db,rules,1,2,2,is_carry=True)
    comp.heroes.append(CompHero('x',NAMES[2],2,star_target=2))
    st.bench=[Unit(f'?占位{i}',1,'bench') for i in range(9)]
    st.shop=[ShopSlot(NAMES[1],1),ShopSlot(NAMES[2],2)]
    assert [b.name for b in advisor._buys(st,CompMatch(comp,.8,.8),ROLL)]==[NAMES[1]]


def test_non_target_direct_upgrade_does_not_repeat_after_planned_merge(db,rules):
    advisor,comp,st=scenario(db,rules,2,2)
    st.board=held(NAMES[1],2);st.shop=[ShopSlot(NAMES[1],1),ShopSlot(NAMES[1],1)]
    assert len(advisor._buys(st,CompMatch(comp,.8,.8),ROLL))==1


def test_projected_board_goal_changes_without_mutating_comp_or_guide(db,rules):
    advisor,comp,st=scenario(db,rules,1,3,3)
    hero=GuideHero('1',NAMES[1],position='1,1',star=3)
    comp.guide=Guide('test','测试攻略','S',None,boards={'final':[hero]})
    adv=advisor.advise(st,locked_comp=comp.id)
    assert adv.board.cells[0].star_target==adv.targets[0].target_star==2
    assert adv.targets[0].need==0 and not adv.buys
    assert comp.heroes[0].star_target==3 and comp.guide.boards['final'][0].star==3
    assert any('两星' in note for note in adv.notes)


def test_goals_change_without_persisting_previous_call_counts(db,rules):
    advisor,comp,st=scenario(db,rules,1,3,3)
    assert not advisor.advise(st,locked_comp=comp.id).buys
    st.stage='2-2';st.level=5
    assert advisor.advise(st,locked_comp=comp.id).targets[0].need==6
    st.board=[]
    assert advisor.advise(st,locked_comp=comp.id).targets[0].need==9


def test_keep_line_for_non_target_pair_and_new_target_has_no_cached_goal(db,rules):
    advisor,comp,st=scenario(db,rules,2,2,stage='2-2',level=5)
    st.gold=10;st.board=held(NAMES[1],1);st.traits=[]
    from jccbrain.core.models import TraitStatus
    st.traits=[TraitStatus('护卫',1)]
    st.shop=[ShopSlot(NAMES[1],1)]
    assert not advisor._buys(st,CompMatch(comp,.8,.8),EconDecision(EconAction.SAVE,'保留',keep_gold=10))
    assert len(advisor._buys(st,CompMatch(comp,.8,.8),ROLL))==1
    # 同一 advisor 另一主目标以不同星级要求重新计算，不保留前次候选计数。
    changed=replace(comp,id='another',heroes=[CompHero('1',NAMES[1],1,star_target=1)])
    st.board=[];st.shop=[ShopSlot(NAMES[1],1) for _ in range(5)]
    assert len(advisor._buys(st,CompMatch(changed,.8,.8),ROLL))==1
    changed=replace(changed,heroes=[replace(changed.heroes[0],star_target=3,is_carry=True)])
    assert len(advisor._buys(st,CompMatch(changed,.8,.8),ROLL))==5


@pytest.mark.parametrize('guide_star', [1,2])
def test_current_population_guide_lower_star_is_not_raised_by_final_goal_downgrade(db,rules,guide_star):
    advisor,comp,st=scenario(db,rules,1,3,3)
    comp.guide=Guide('test','当前人口攻略','S',None,boards={'L8':[GuideHero('1',NAMES[1],position='1,1',star=guide_star)]})
    adv=advisor.advise(st,locked_comp=comp.id)
    assert adv.targets[0].target_star==2
    assert adv.board.cells[0].star_target==guide_star
    assert comp.guide.boards['L8'][0].star==guide_star
