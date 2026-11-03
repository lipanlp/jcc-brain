"""预选持久化、有效范围、异步隔离和去评级偏置的生产契约。"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain.context import AppContext
from jccbrain.core import comp_presets
from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.core.settings import Settings
from jccbrain.data.models import Guide
from jccbrain.engine.comps import CompMatcher, _strength
from jccbrain.engine.llm import LLMReply, build_context
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.autopilot import AutopilotMode
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.modes.base import ModeSink
from jccbrain.webui.api import Api
from test_comp_sources import Bus, data


@pytest.fixture
def presets(db, monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    ctx = AppContext(Settings()); ctx.db = db
    base = db.comp('104')
    a = replace(base, id='A', name='预选红', heroes=[replace(base.heroes[0], name='绯红树怪', is_carry=True)], guide=None, guides=[])
    b = replace(base, id='B', name='预选蓝', heroes=[replace(base.heroes[0], name='远古巨龙', is_carry=True)], guide=None, guides=[])
    db.comps.clear(); db.comps.update(A=a, B=b)
    api = Api(ctx, Bus()); api._icons.close()
    yield ctx, api
    api._shutdown(); ctx.close()


def create(api, members, name=' 自定义 '):
    return data(api.save_comp_preset(api._ctx.db.season_id, None, name, members))['saved_id']


def enable(api, cid):
    return data(api.select_comp_preset(api._ctx.db.season_id, cid))


def st(**kw):
    return GameState.from_dict(make_state(stage='3-2', phase='planning', board=[{'name':'绯红树怪'}], **kw))


def test_old_settings_default_all_and_dynamic_members(presets):
    ctx, api = presets
    initial = data(api.comp_presets()); assert initial['active_id'] == 'all'
    assert initial['scope']['effective_count'] == 2
    ctx.db.comps['C'] = replace(ctx.db.comps['A'], id='C')
    assert data(api.comp_presets())['scope']['candidate_ids'] == ['A', 'B', 'C']
    assert Settings.load().assistant.comp_presets == {}


def test_crud_persistence_restart_and_delete_active_only(presets):
    ctx, api = presets
    cid = create(api, ['A', 'A']); second = create(api, ['B'], '第二个')
    first = next(r for r in data(api.comp_presets())['configs'] if r['id'] == cid)
    assert first['name'] == '自定义' and first['members'] == ['A']
    enable(api, cid)
    data(api.save_comp_preset(ctx.db.season_id, cid, ' 改名 ', ['A','B']))
    restored = AppContext(Settings.load()); restored.db = ctx.db
    assert restored.assistant_selection()['preset_id'] == cid and restored.assistant_selection()['name'] == '改名'
    assert restored.assistant_selection()['candidate_ids'] == ['A','B']
    data(api.delete_comp_preset(ctx.db.season_id, second)); assert ctx.assistant_selection()['preset_id'] == cid
    data(api.delete_comp_preset(ctx.db.season_id, cid)); assert ctx.assistant_selection()['preset_id'] == 'all'


@pytest.mark.parametrize('method,args', [('save_comp_preset',('all','改名',['A'])),('delete_comp_preset',('all',))])
def test_builtin_immutable(presets, method, args):
    ctx, api = presets
    assert not getattr(api, method)(ctx.db.season_id,*args)['ok']
    assert ctx.assistant_selection()['preset_id'] == 'all'


@pytest.mark.parametrize('name,members', [('', ['A']),(' '*4,['A']),('x'*41,['A']),(None,['A']),('x','A'),('x',[1]),('x',['unknown']),('x',['A\n'])])
def test_invalid_create_does_not_persist(presets, name, members):
    ctx, api = presets
    before = deepcopy(ctx.settings.assistant.comp_presets)
    assert not api.save_comp_preset(ctx.db.season_id,None,name,members)['ok']
    assert ctx.settings.assistant.comp_presets == before


@pytest.mark.parametrize('raw', [None, [], 'bad', {'18_S19':None}, {'18_S19':{'selected':'missing','configs':[]}}, {'18_S19':{'selected':'all','configs':'bad'}}])
def test_corrupt_configs_pause_without_expanding(presets, raw):
    ctx, api = presets; ctx.settings.assistant.comp_presets = raw
    out = data(api.comp_presets()); assert out['error'] and out['scope']['candidate_ids'] == []
    adv = AssistantMode(ctx,ModeSink()).compute_advice(st())
    assert adv.target is None and not adv.buys and not adv.sells and adv.econ is None
    assert not api.select_comp_preset(ctx.db.season_id,'all')['ok']


def test_failed_atomic_save_keeps_memory_and_file(presets, monkeypatch):
    ctx, api = presets; cid = create(api,['A']); before = asdict(ctx.settings)
    from jccbrain.core import paths
    path = paths.settings_path(); saved = path.read_bytes()
    def fail(*args): raise OSError('受控磁盘保存失败')
    monkeypatch.setattr('jccbrain.core.settings.os.replace',fail)
    result = api.save_comp_preset(ctx.db.season_id,cid,'不能生效',['B'])
    assert not result['ok'] and '保存失败' in result['error']
    assert asdict(ctx.settings) == before and path.read_bytes() == saved
    assert [p for p in path.parent.iterdir() if p.name.startswith('tmp')] == []


def test_season_isolation_and_old_request_rejected(presets):
    ctx, api = presets; sid = ctx.db.season_id; cid = create(api,['A']); enable(api,cid)
    ctx.db.use_season('16_S19')
    assert ctx.assistant_selection()['preset_id'] == 'all'
    assert not api.select_comp_preset(sid,cid)['ok']
    assert not api.delete_comp_preset('16_S19',cid)['ok']
    assert comp_presets.read(ctx.settings.assistant.comp_presets,sid)[0]['selected'] == cid


def test_source_closed_and_missing_saved_members_preserved(presets):
    ctx, api = presets; cid = create(api,['A']); enable(api,cid)
    data(api.set_source('dataj',False)); assert ctx.assistant_selection()['candidate_ids'] == []
    row = next(r for r in data(api.comp_presets())['configs'] if r['id']==cid)
    assert row['members']==['A'] and row['unavailable'][0]['id']=='A'
    data(api.set_source('dataj',True)); assert ctx.assistant_selection()['candidate_ids']==['A']
    del ctx.db.comps['A']; ctx.advisor.reload()
    assert data(api.save_comp_preset(ctx.db.season_id,cid,'保留消失记录',['A']))['saved_id']==cid
    assert not api.save_comp_preset(ctx.db.season_id,cid,'不能新增未知',['A','Z'])['ok']


@pytest.mark.parametrize('unavailable', ['source_closed', 'deleted'])
def test_existing_unavailable_watch_can_be_removed_individually(presets, unavailable):
    ctx, api = presets
    cid = create(api, ['A']); enable(api, cid)
    data(api.set_watch_comps(['A', 'B']))
    if unavailable == 'source_closed':
        data(api.set_source('dataj', False))
    else:
        del ctx.db.comps['A']; del ctx.db.comps['B']; ctx.advisor.reload()
    assert ctx.game_watch_ids == ['A', 'B']
    assert data(api.set_watch_comps(['B', 'B']))['watch'][0]['id'] == 'B'
    assert ctx.game_watch_ids == ['B']
    assert ctx.assistant_selection()['candidate_ids'] == []
    # 已删除的旧项现在属于新增，不可借“曾经关注”绕过来源限制。
    assert not api.set_watch_comps(['A', 'B'])['ok']
    assert ctx.game_watch_ids == ['B']
    if unavailable == 'source_closed':
        data(api.set_source('dataj', True))
        assert ctx.assistant_selection()['candidate_ids'] == []  # B不在仅A预选内，不兜底。
    assert data(api.set_watch_comps([]))['watch'] == []


@pytest.mark.parametrize('bad', [None, 'A', [1], [{}], ['A', 'new'], ['A', 'C'], ['A', 'B', 'C', 'D', 'E']])
def test_unavailable_watch_retention_rejects_new_invalid_ids_atomically(presets, bad):
    ctx, api = presets
    ctx.db.comps['C'] = replace(ctx.db.comps['A'], id='C')
    data(api.set_watch_comps(['A', 'B']))
    data(api.set_source('dataj', False))
    revision = ctx.assistant_selection()['revision']
    events = list(api._bus.events)
    assert not api.set_watch_comps(bad)['ok']
    assert ctx.game_watch_ids == ['A', 'B']
    assert ctx.assistant_selection()['revision'] == revision
    assert api._bus.events == events


def test_catalog_aliases_resolve_once_without_changing_browser_or_scout(presets):
    ctx, api = presets; a=ctx.db.comps['A']; guide=Guide('linked','关联原始攻略','D',None); guide.comp=a
    alias=replace(a,id='official:linked',source='official',guide=guide,guide_id='linked');ctx.db.guides['linked']=guide;ctx.db.official_comps[alias.id]=alias
    cid=create(api,['A','official:linked']);enable(api,cid)
    scope=ctx.assistant_selection();assert scope['catalog_count']==2 and scope['effective_count']==1 and scope['candidate_ids']==['A']
    assert len(data(api.comps())['rows'])==3 and len(ctx.enabled_comps())==2
    assert {c.id for c in ctx.scout.comps_getter()}=={'A','B'}


def test_empty_custom_and_watch_disjoint_never_fallback(presets):
    ctx, api = presets;mode=AssistantMode(ctx,ModeSink());cid=create(api,[]);enable(api,cid)
    assert mode.compute_advice(st()).target is None
    data(api.save_comp_preset(ctx.db.season_id,cid,'仅A',['A']))
    data(api.set_watch_comps(['B']));scope=ctx.assistant_selection();assert scope['candidate_ids']==[] and scope['watch_outside'][0]['id']=='B'
    assert ctx.game_watch_ids==['B'] and mode.compute_advice(st()).target is None
    data(api.set_watch_comps(['A','B']));assert mode.compute_advice(st()).target.comp.id=='A' and ctx.game_watch_ids==['A','B']


@pytest.mark.parametrize('source',['game','default'])
def test_manual_lock_outside_preset_has_priority_and_unlock_restores(presets, source):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid)
    data(api.set_watch_comps(['A']))
    data(api.lock_comp_for_game('B') if source=='game' else api.set_default_lock_comp('B'))
    scope=ctx.assistant_selection();assert scope['lock_outside'] and '配置外' in scope['notice']
    mode=AssistantMode(ctx,ModeSink());assert mode.compute_advice(st()).target.comp.id=='B'
    assert ctx.game_watch_ids==['A']
    data(api.lock_comp_for_game(''));assert mode.compute_advice(st()).target.comp.id=='A'
    data(api.set_source('dataj',False));assert mode.compute_advice(st()).target is None


def test_same_round_change_resets_tracker_and_llm_but_rename_does_not(presets):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid);mode=AssistantMode(ctx,api._sink);api._runner.mode=mode
    adv=mode.compute_advice(st());assert adv.target.comp.id=='A';mode._latest=adv;mode._llm=('3-2',LLMReply('旧AI',[],'旧'))
    mode.tracker.lead={'A':1};generation=mode.lifecycle.generation;mode.tracker.reset=Mock(wraps=mode.tracker.reset)
    data(api.save_comp_preset(ctx.db.season_id,cid,'仅改名',['A']))
    assert mode.tracker.lead=={'A':1} and mode.lifecycle.generation==generation and mode._llm is not None
    mode.tracker.reset.assert_not_called()
    data(api.save_comp_preset(ctx.db.season_id,cid,'仅改名',['A']))
    mode.tracker.reset.assert_not_called()
    data(api.save_comp_preset(ctx.db.season_id,cid,'仅改名',['B']))
    assert mode.lifecycle.generation>generation and mode._llm is None and not mode.tracker.lead
    assert api._last_advice.target is None
    assert mode.compute_advice(st()).target.comp.id=='B'


def test_late_llm_cannot_publish_after_configuration_change(presets):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid);mode=AssistantMode(ctx,api._sink);api._runner.mode=mode
    from ai_fakes import authorize_ai
    adv=mode.compute_advice(st());authorize_ai(mode,adv);mode._latest=adv;entered=threading.Event();release=threading.Event()
    def reply(*args):entered.set();assert release.wait(3);return LLMReply('旧集合结果',[],'迟到')
    ctx.llm=SimpleNamespace(advise=reply)
    worker=threading.Thread(target=mode._ask_llm,args=(st(),adv));worker.start();assert entered.wait(2)
    data(api.save_comp_preset(ctx.db.season_id,cid,'改为B',['B']));release.set();worker.join(3);assert not worker.is_alive()
    assert mode._llm is None and all(not p.get('llm_text')for k,p in api._bus.events if k=='advice')
    assert mode.compute_advice(st()).target.comp.id=='B'


def test_api_does_not_hold_api_lock_while_waiting_for_mode(presets, monkeypatch):
    ctx,api=presets;cid=create(api,['A']);mode=AssistantMode(ctx,api._sink);api._runner.mode=mode;mode.compute_advice(st())
    waiting=threading.Event();original=mode.refresh_selection
    def refresh():waiting.set();return original()
    monkeypatch.setattr(mode,'refresh_selection',refresh)
    results=[]
    with mode._lock:
        worker=threading.Thread(target=lambda:results.append(api.select_comp_preset(ctx.db.season_id,cid)));worker.start();assert waiting.wait(2)
        assert api._lock.acquire(timeout=1)
        api._lock.release()
    worker.join(3);assert not worker.is_alive() and results[0]['ok']


def test_other_modes_manual_roll_and_complete_data_scope_are_unchanged(presets):
    ctx,api=presets;full={c.id for c in ctx.enabled_comps()};auto=AutopilotMode(ctx,ModeSink())
    before=[m.comp.id for m in auto.compute_advice(st()).comps]
    cid=create(api,[]);enable(api,cid)
    assert [m.comp.id for m in auto.compute_advice(st()).comps]==before
    roll=AutoRollMode(ctx,ModeSink(),targets=[RollTarget('远古巨龙',2)])
    assert roll.resolve_targets(st())[0].name=='远古巨龙'
    assert {c.id for c in ctx.enabled_comps()}==full and len(data(api.comps())['rows'])==2
    assert AssistantMode(ctx,ModeSink()).compute_advice(st()).target is None


@pytest.mark.parametrize('avg,expected', [(None,.5),(3.0,1),(3.5,1),(4.5,.5),(5.5,0),(6.,0)])
def test_strength_uses_only_stats_and_neutral_missing(presets, avg, expected):
    ctx,_=presets
    for tier in ['S','A','B','C','D',None]:assert _strength(replace(ctx.db.comps['A'],tier=tier,avg_placement=avg))==expected


def test_tier_and_input_order_never_affect_rank_or_llm_reason(presets):
    ctx,_=presets; a=replace(ctx.db.comps['A'],heroes=ctx.db.comps['B'].heroes,avg_placement=None,tier='S');b=replace(a,id='B',name='B',tier='D')
    for seq in [[a,b],[b,a],[replace(b,tier='S'),replace(a,tier='D')]]:
        ranked=CompMatcher(seq,ctx.db).match(st(),top=None);assert [m.comp.id for m in ranked]==['A','B'] and ranked[0].score==ranked[1].score
        assert all('级，'not in why for m in ranked for why in m.why)
    low,high=replace(a,avg_placement=5.5),replace(b,avg_placement=3.5)
    assert CompMatcher([low,high]).match(GameState(),top=None)[0].comp.id=='B'
    adv=AssistantMode(ctx,ModeSink()).compute_advice(st());context=build_context(st(),adv)
    assert all('tier'not in row for row in context['candidate_comps'])
    assert all('级，'not in line for line in context['rule_advice'])
    assert context['comparison_scope']['allowed_comps']


@pytest.mark.parametrize('contents',['{bad','[]','{"assistant":null}'])
def test_corrupt_settings_file_never_defaults_to_all(presets, tmp_path, contents):
    ctx,_=presets;path=tmp_path/'broken.json';path.write_text(contents)
    ctx.settings=Settings.load(path)
    assert ctx.assistant_selection()['candidate_ids']==[] and ctx.assistant_selection()['error']


@pytest.mark.parametrize('bad',[{},'p-bad',False,123])
def test_bad_config_id_and_old_season_update_are_rejected(presets,bad):
    ctx,api=presets
    assert not api.select_comp_preset(ctx.db.season_id,bad)['ok']
    assert not api.save_comp_preset(ctx.db.season_id,bad,'x',['A'])['ok']
    assert not api.save_comp_preset('old-season',None,'x',['A'])['ok']


def test_llm_outside_effective_scope_cannot_add_switch_note(presets):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid);mode=AssistantMode(ctx,ModeSink());adv=mode.compute_advice(st())
    mode._llm=('3-2',LLMReply('转预选蓝',['错误范围'],'理由',switch_comp='预选蓝'))
    mode._attach_llm(adv)
    assert adv.llm_text is None and not any('AI 建议转' in n for n in adv.notes)


def test_equipment_evidence_remains_complete_when_selection_is_empty(presets,monkeypatch):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid)
    captured=[]
    from jccbrain.engine import items
    original=items.plan
    def capture(comp,inventory,state,db,evidence_comps=None):captured.append({c.id for c in evidence_comps});return original(comp,inventory,state,db,evidence_comps)
    monkeypatch.setattr(items,'plan',capture)
    AssistantMode(ctx,ModeSink()).compute_advice(st())
    assert captured==[{'A','B'}]


def test_candidate_buys_are_within_selected_union_and_empty_has_no_actions(presets):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid);mode=AssistantMode(ctx,ModeSink())
    raw=make_state(stage='2-2',gold=60,board=[{'name':'远古巨龙','star':1},{'name':'远古巨龙','star':1}],shop={'open':True,'slots':[{'name':'绯红树怪','cost':1},{'name':'远古巨龙','cost':1}]})
    adv=mode.compute_advice(GameState.from_dict(raw));assert adv.target.comp.id=='A';assert adv.buys and {b.name for b in adv.buys}=={'绯红树怪'}
    data(api.save_comp_preset(ctx.db.season_id,cid,'空',[]));adv=mode.compute_advice(GameState.from_dict(raw));assert adv.target is None and not adv.buys and adv.econ is None


def test_added_alias_or_outside_watch_does_not_reset_effective_selection(presets):
    ctx,api=presets;cid=create(api,['A']);enable(api,cid);data(api.set_watch_comps(['A']))
    mode=AssistantMode(ctx,api._sink);api._runner.mode=mode;mode.compute_advice(st());mode.tracker.reset=Mock(wraps=mode.tracker.reset)
    data(api.set_watch_comps(['A','B']));mode.tracker.reset.assert_not_called()
    guide=Guide('alias','重复条目','S',None);guide.comp=ctx.db.comps['A'];ctx.db.guides['alias']=guide
    ctx.db.official_comps['official:alias']=replace(ctx.db.comps['A'],id='official:alias',source='official',guide=guide,guide_id='alias')
    data(api.save_comp_preset(ctx.db.season_id,cid,'加同义条目',['A','official:alias']))
    mode.tracker.reset.assert_not_called();assert ctx.assistant_selection()['catalog_count']==2 and ctx.assistant_selection()['effective_count']==1


def test_wrong_storage_layer_is_not_silent_all(presets):
    ctx,_=presets;ctx.settings.assistant.comp_presets={'selected':'all','configs':[]}
    assert ctx.assistant_selection()['error'] and ctx.assistant_selection()['candidate_ids']==[]


def test_config_change_keeps_ended_lifecycle_neutral(presets):
    ctx,api=presets;mode=AssistantMode(ctx,api._sink);api._runner.mode=mode;mode.compute_advice(st())
    mode.lifecycle.observe(GameState(phase=Phase.ENDED,placement=1))
    cid=create(api,['A']);enable(api,cid)
    assert api._last_advice.phase==Phase.ENDED and '对局结束' in api._last_advice.headline
    assert api._last_advice.target is None and not api._last_advice.buys


def test_population_preview_pairs_same_target_with_selection_revision(presets):
    ctx,api=presets;first=create(api,['A']);enable(api,first);mode=AssistantMode(ctx,api._sink);api._runner.mode=mode
    state=st();api._sink.on_state(state);api._sink.on_advice(mode.compute_advice(state));old=ctx.assistant_selection()['revision']
    second=create(api,['A'],'同成员新配置');enable(api,second);api._sink.on_state(state);api._sink.on_advice(mode.compute_advice(state))
    now=ctx.assistant_selection()['revision'];assert old!=now
    assert not api.population_preview(8,'A',ctx.db.season_id,old)['ok']
    assert data(api.population_preview(8,'A',ctx.db.season_id,now))['selection_revision']==now


def test_directory_without_roster_is_not_counted_as_decidable(presets):
    ctx,api=presets;ctx.db.comps['C']=replace(ctx.db.comps['A'],id='C',heroes=[])
    out=data(api.comp_presets());assert len(out['catalog'])==3 and out['scope']['effective_count']==2
    cid=create(api,['C']);enable(api,cid);out=data(api.comp_presets());row=next(r for r in out['configs']if r['id']==cid)
    assert row['selected_count']==1 and row['effective_count']==0 and row['unavailable'][0]['id']=='C'
    assert AssistantMode(ctx,ModeSink()).compute_advice(st()).target is None
