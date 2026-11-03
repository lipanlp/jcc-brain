"""运营偏好穿过实际后处理与模式共享设置；合成对照，不发送原生输入。"""

import pytest

from conftest import make_state
from jccbrain.actuator.executor import DryRunExecutor
from jccbrain.context import AppContext
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine import economy
from jccbrain.modes.assistant import AssistantMode, advice_signature
from jccbrain.modes.autopilot import AutopilotMode
from jccbrain.webui.api import Api
from test_econ_v2 import advisor_for
from test_webui_api import Bus, Runner, data

PREFERENCES = ('adaptive', 'health', 'economy')


def snapshot(**edits):
    return GameState.from_dict(make_state(**edits))


@pytest.mark.parametrize('style,cost,edits,expected', [
    ('reroll', 1, dict(stage='3-1', level=5, gold=65), [('roll',50,15),('roll',40,25),('roll',60,5)]),
    ('reroll', 2, dict(stage='3-2', level=6, gold=55), [('roll',50,5),('roll',40,15),('save',60,0)]),
    ('reroll', 3, dict(stage='4-1', level=7, gold=65), [('roll',50,15),('roll',40,25),('roll',60,5)]),
    ('fast8', 4, dict(stage='4-2', level=8, gold=50), [('roll',10,40),('roll',0,50),('roll',20,30)]),
    ('fast9', 5, dict(stage='5-1', level=9, gold=50), [('roll',10,40),('roll',0,50),('roll',20,30)]),
    ('fast8', 4, dict(stage='3-5', level=7, gold=40, hp=60), [('save',40,0),('roll',30,10),('save',40,0)]),
    ('standard', 4, dict(stage='3-3', level=7, gold=56), [('level',50,0),('level',50,0),('save',50,0)]),
])
def test_raw_preferences_follow_same_route_and_change_only_optional_budget(rules, style, cost, edits, expected):
    decisions = [economy.decide(snapshot(**edits), rules, style, cost, p) for p in PREFERENCES]
    assert [(d.action.value,d.keep_gold,d.roll_budget) for d in decisions] == expected
    assert decisions[0] == economy.decide(snapshot(**edits), rules, style, cost)
    assert any('保血优先' in d.reason or '经济优先' in d.reason for d in decisions)


@pytest.mark.parametrize('style,edits,expected', [
    ('reroll', dict(stage='3-1',level=5,gold=65), [('roll',50,15),('roll',40,25),('roll',60,5)]),
    ('fast8', dict(stage='4-2',level=8,gold=50), [('roll',10,40),('roll',0,50),('roll',20,30)]),
    ('fast9', dict(stage='5-1',level=9,gold=50), [('roll',10,40),('roll',0,50),('roll',20,30)]),
    ('fast8', dict(stage='3-5',level=7,gold=40,hp=60), [('save',40,0),('roll',30,10),('save',40,0)]),
    ('fast8', dict(stage='3-3',level=7,gold=56), [('level',50,0),('level',50,0),('save',50,0)]),
])
def test_final_advisor_preferences_survive_probability_and_finish_processing(db, rules, style, edits, expected):
    advisor, _ = advisor_for(db,rules,style)
    state = snapshot(**edits)
    answers = [advisor.advise(state, economy_preference=p) for p in PREFERENCES]
    assert [(a.econ.action.value,a.econ.keep_gold,a.econ.roll_budget) for a in answers] == expected
    assert all(a.target.comp.style == style for a in answers)
    assert answers[0].econ == advisor.advise(state).econ
    assert len({a.target.comp.id for a in answers}) == 1


@pytest.mark.parametrize('style,edits', [
    ('reroll',dict(stage='3-1',level=4,gold=40,xp=[0,10])),  # 不在合法慢D等级。
    ('fast8',dict(stage='4-1',level=6,gold=60,xp=[0,36])),  # 关键升级。
    ('fast9',dict(stage='3-2',level=5,gold=40,xp=[16,20])),
    ('fast8',dict(stage='4-2',level=7,gold=80,xp=[0,48],hp=60)),
    ('fast8',dict(stage='4-1',level=7,gold=40,xp=[0,48],hp=60)),
    ('fast8',dict(stage='4-3',level=7,gold=40,hp=45)),     # 必要稳血。
    ('reroll',dict(stage='3-2',level=6,gold=40,streak={'kind':'loss','n':4})),
])
def test_preferences_keep_critical_route_and_recovery_choices(db,rules,style,edits):
    advisor,_ = advisor_for(db,rules,style)
    state=snapshot(**edits)
    assert [advisor.advise(state,economy_preference=p).econ for p in PREFERENCES] == [advisor.advise(state).econ]*3


@pytest.mark.parametrize('style', ['reroll','fast8','fast9'])
@pytest.mark.parametrize('hp', [30,20])
def test_all_preferences_preserve_emergency_all_in_even_formed(db,rules,style,hp):
    advisor,comp=advisor_for(db,rules,style)
    state=snapshot(stage='5-1',level=9,gold=50,hp=hp,board=[{'name':h.name,'star':3} for h in comp.heroes])
    for pref in PREFERENCES:
        adv=advisor.advise(state,economy_preference=pref)
        assert adv.econ.action == economy.EconAction.ALL_IN
        assert adv.econ.keep_gold == 0 and adv.econ.roll_budget == 50


@pytest.mark.parametrize('style', ['reroll','fast8','fast9'])
def test_low_rank_needed_stabilization_is_not_delayed(db,rules,style):
    advisor,_=advisor_for(db,rules,style)
    state=snapshot(stage='4-1',level=7 if style!='reroll' else 5,gold=30 if style!='reroll' else 60,hp=70,xp=[0,48],
        hp_rank=[6,8],players=[{'name':str(i),'hp':90-i,'self':False} for i in range(8)])
    baseline=advisor.advise(state).econ
    assert '血量第' in baseline.reason
    assert all(advisor.advise(state,economy_preference=p).econ == baseline for p in PREFERENCES)


@pytest.mark.parametrize('style,level',[('reroll',8),('fast8',8),('fast9',9)])
def test_formed_comp_stops_purposeless_rolling_for_every_preference(db,rules,style,level):
    advisor,comp=advisor_for(db,rules,style)
    state=snapshot(stage='4-3',level=level,gold=52,hp=80,xp=[73,80],
                   board=[{'name':h.name,'star':3} for h in comp.heroes])
    for preference in PREFERENCES:
        final=advisor.advise(state,economy_preference=preference)
        assert final.econ.action not in (economy.EconAction.ROLL,economy.EconAction.ALL_IN)
        assert '阵容已成型' in final.econ.reason


@pytest.mark.parametrize('edits', [dict(gold=None),dict(level=None),dict(stage=None),dict(hp=None),
    dict(phase='unknown'),dict(phase='spectate'),dict(players_stale=True),
    dict(viewing={'slot':1},players=[{'name':'自己','self':True,'slot':0,'hp':80}])])
def test_uncertain_or_foreign_input_never_gains_new_spending_rules(rules,edits):
    state=snapshot(stage='3-1',level=5,gold=65,**edits) if not set(edits)&{'stage','level','gold'} else snapshot(**{'stage':'3-1','level':5,'gold':65,**edits})
    baseline=economy.decide(state,rules,'reroll',1)
    assert all(economy.decide(state,rules,'reroll',1,p) == baseline for p in PREFERENCES)


def test_positive_and_zero_probability_postprocessing_still_takes_priority(db,rules):
    from test_rules_v2 import expensive_advisor
    advisor=expensive_advisor(db,rules)
    state=snapshot(stage='3-5',level=7,gold=40,hp=60,xp=[47,48],odds=[100,0,0,0,0])
    final=advisor.advise(state,economy_preference='health')
    assert final.econ.action == economy.EconAction.LEVEL
    assert '刷不到' in final.econ.reason and final.econ.xp_buys == 1


def test_saved_preference_reaches_both_live_modes_next_decision_without_automatic_buy(db,tmp_path,monkeypatch):
    monkeypatch.setenv('JCC_HOME',str(tmp_path/'runtime'))
    ctx=AppContext(Settings());ctx.db=db
    api=Api(ctx,Bus());api._runner=Runner()
    try:
        mode=AssistantMode(ctx,api._sink);api._runner.start(mode)
        ctx.executor=DryRunExecutor(ctx.layout,ctx.guard,0)
        adviser=ctx.advisor
        state=snapshot(stage='3-1',level=5,gold=65,
                       shop={'open':True,'slots':[{'name':'绯红树怪','cost':1}]+[None]*4})
        adaptive=mode._process_frame(state)[1]
        data(api.save_feature_settings({'assistant':{'economy_preference':'health'}}))
        health=mode._process_frame(state)[1]
        assert adaptive.econ.roll_budget == 15 and health.econ.roll_budget == 25
        assert advice_signature(adaptive) != advice_signature(health)
        assert api._runner.mode is mode and ctx.advisor is adviser and health.buys
        assert ctx.executor.history == [] and not mode.acts
        data(api.save_feature_settings({'assistant':{'economy_preference':'economy'}}))
        pilot=AutopilotMode(ctx,api._sink)
        assert pilot.compute_advice(state).econ.roll_budget == 5
        assert mode.compute_advice(state).econ.roll_budget == 5
        assert data(api.status())['economy_preference']=='economy'
        assert Settings.load().assistant.economy_preference=='economy'
    finally:
        api._shutdown();ctx.close()


def test_health_does_not_claim_a_budget_change_when_optional_level_is_unchanged(rules):
    state=snapshot(stage='3-3',level=7,gold=56,hp=80)
    assert economy.decide(state,rules,'fast8',4,'health') == economy.decide(state,rules,'fast8',4,'adaptive')
