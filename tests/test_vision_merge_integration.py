"""M1: real lifecycle/scout/learning/API contracts, isolated data and fake input."""
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain.actuator.layout import Layout
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.engine.scout import ScoutBook
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.scout import ScoutRunner
from jccbrain.modes.unit_learning import UnitLearningRunner
from jccbrain.vision.base import VisionError
from jccbrain.vision.eye.unit_panel import PanelObservation
from jccbrain.webui import settings_schema
from test_assistant import Ctx, AdviceSink
from test_unit_learning_auto import auto, run
from test_webui_api import setup_api, isolation, data


@pytest.mark.parametrize('edits', [
    {'phase':'ended'}, {'phase':'spectate'}, {'viewing':{'slot':7}},
    {'phase':'combat','stage':None,'gold':None,'level':None},
])
def test_unaccepted_lifecycle_frames_never_reach_learning_or_llm(db,rules,monkeypatch,edits):
    ctx=Ctx(db,rules,[make_state(**edits)]);mode=AssistantMode(ctx,AdviceSink())
    learn=Mock(return_value=False)
    monkeypatch.setattr(UnitLearningRunner,'maybe_run',learn)
    ctx.llm.should_call=Mock()
    with pytest.raises(VisionError,match='回放结束'):mode.loop()
    learn.assert_not_called();ctx.llm.should_call.assert_not_called()
    assert mode.stopped and mode.learning._cancel.is_set()


@pytest.mark.parametrize('transition',['waiting','ended','new_game'])
def test_lifecycle_boundary_discards_real_pending_chain_and_old_receipt(auto,db,rules,transition):
    e=auto;e.panel=True;e.provider.read()
    sample=e.provider.unit_learning_sample();ctl=e.provider._learning
    assert e.provider.process_unit_learning(sample=sample)['status']=='pending'
    ctx=Ctx(db,rules,[]);ctx.vision=e.provider
    mode=AssistantMode(ctx,AdviceSink());old=e.runner;mode.learning=old
    mode._process_frame(GameState.from_dict(make_state(stage='4-1')))
    if transition=='new_game':mode._on_new_game()
    else:
        raw=make_state(phase='ended') if transition=='ended' else make_state(phase='combat',stage=None,level=None,gold=None)
        assert mode._process_frame(GameState.from_dict(raw)) is None
    assert mode.learning is old and not old._cancel.is_set()
    assert ctl.previous is None and not ctl.source.validates(sample.frame)
    assert e.provider._learning is ctl and not ctl.store.entries()
    assert e.provider.unit_learning_sample() is None


def test_waiting_resume_clears_confirmation_without_renewing_round_budget_or_panel_ownership(auto,db,rules):
    e=auto;ctx=Ctx(db,rules,[]);ctx.vision=e.provider
    mode=AssistantMode(ctx,AdviceSink());mode.learning=e.runner
    attempted=e.runner._round('4-1');attempted.update({(.3,.4,'board'),(.5,.4,'board')})
    e.runner._last_attempt=99.;e.runner._blocked_panel=True
    valid=GameState.from_dict(make_state(stage='4-1'))
    mode._process_frame(valid)
    mode._process_frame(GameState.from_dict(make_state(phase='combat',stage=None,level=None,gold=None)))
    mode._process_frame(valid)
    assert e.runner._round('4-1') is attempted and len(attempted)==2
    assert e.runner._last_attempt==99. and e.runner._blocked_panel


def test_optional_scene_cleanup_fault_does_not_block_main_lifecycle(auto,db,rules,monkeypatch):
    e=auto;ctx=Ctx(db,rules,[]);ctx.vision=e.provider
    mode=AssistantMode(ctx,AdviceSink());mode.learning=e.runner
    monkeypatch.setattr(e.provider,'finish_unit_learning_transaction',Mock(side_effect=OSError('cleanup probe')))
    assert mode._process_frame(GameState.from_dict(make_state(phase='combat',stage=None,level=None,gold=None))) is None
    assert not mode.lifecycle.active and e.runner.status['reason']=='cleanup_error'


def scout_for(e,db):
    e.ctx.advisor=NS(scout=ScoutBook(db));e.ctx.layout=Layout()
    e.ctx.layout.player_slots_calibrated=True
    e.ctx.settings.scout.enabled=True;e.ctx.settings.scout.schedule='every_round'
    sink=e.runner.sink;sink.on_scout_changed=Mock();sink.scout_toast_capable=lambda:True;sink.scout_hud_visible=lambda w=None:w=='toast'
    from scout_fakes import ControlledInput
    monitor=ControlledInput(e.clock,e.ctx.locator.find(0))
    scout=ScoutRunner(e.ctx,sink,clock=e.clock,sleep=e.clock.wait,input_monitor=monitor)
    state=GameState.from_dict(make_state(stage=e.stage,timer=30,players=[
        {'name':'自己','self':True,'slot':0,'hp':80}, {'name':'对手','slot':1,'hp':70}]))
    scout.book.update(state)
    e.runner.scout=scout
    return scout,state


def announce(scout,state,clock,permit):
    assert not scout.maybe_run(state)
    notice=scout.status['notice'];assert notice and scout.busy
    if permit:
        # Keep the actual permit handoff outstanding; scanning transport is not
        # part of this test and must never capture or click a game.
        scout.run=Mock()
        assert scout.notice_shown(notice['id'],'toast')
        clock.wait(3)
        assert scout.maybe_run(state) and scout._auto_permit is not None
        assert scout.status['notice']['id']==notice['id'] and scout.busy  # 首动作前保留独立toast可见性授权


@pytest.mark.parametrize('permit',[False,True])
def test_real_scout_notice_and_outstanding_permit_preempt_learning_before_capture(auto,db,permit):
    e=auto;scout,state=scout_for(e,db)
    announce(scout,state,e.clock,permit)
    run(e)
    assert e.runner.status['reason']=='scout_priority'
    assert not e.ctx.executor.history and not e.ctl.store.entries() and e.tap_starts==0


@pytest.mark.parametrize('permit',[False,True])
def test_notice_or_permit_arriving_during_learning_ocr_cancels_promotion_and_cleanup(auto,db,permit):
    e=auto;scout,state=scout_for(e,db)
    def ocr(_):
        announce(scout,state,e.clock,permit)
        return PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields')
    e.ctl.panel.read.side_effect=ocr
    run(e)
    assert scout.busy and e.runner.status['reason']=='scout_priority'
    assert not e.ctl.store.entries() and len(e.ctx.executor.history)==1
    assert e.monitor.tap is None and not e.runner.active


def test_current_api_keeps_learning_in_common_vision_and_one_global_dry_run(setup_api,tmp_path):
    api,ctx,bus=setup_api
    common=data(api.settings_schema());feature_before=data(api.feature_settings())
    vision=next(s for s in common if s['key']=='vision')
    learning={f['key']:f for f in vision['fields'] if f['key'].startswith('unit_learning_')}
    assert set(learning)=={'unit_learning_mode','unit_learning_idle_s','unit_learning_interval_s','unit_learning_min_timer'}
    assert learning['unit_learning_mode']['value']=='off'
    assert '顶部全局演练' in vision['note'] and '托管页' in vision['note']
    assert 'vision' not in settings_schema.FEATURE_FIELDS
    assert {s['key'] for s in common}=={'vision','data','llm','actuator'}
    assert 'dry_run' not in {f['key'] for s in common for f in s['fields']}
    assert {s:set(v) for s,v in feature_before['values'].items()}==settings_schema.FEATURE_FIELDS
    valid={'vision':{'unit_learning_mode':'auto','unit_learning_idle_s':3,
                     'unit_learning_interval_s':20,'unit_learning_min_timer':19}}
    saved=data(api.save_settings(valid))
    reopened=Settings.load()
    assert asdict(reopened.vision)==asdict(ctx.settings.vision)
    assert data(api.feature_settings())==feature_before
    for value in (False,True):
        status=data(api.set_dry_run(value))
        assert status['dry_run'] is value and data(api.autoroll_defaults())['dry_run'] is value
        assert Settings.load().actuator.dry_run is value
    previous=asdict(ctx.settings);path=Path(os.environ['JCC_HOME'])/'settings.json';before=path.read_bytes()
    bad=api.save_settings({'vision':{'unit_learning_idle_s':1},'autoroll':{'keep_gold':37},
                           'assistant':{'economy_preference':'health'}})
    assert not bad['ok'] and asdict(ctx.settings)==previous and path.read_bytes()==before
    other_bad=api.save_settings({'vision':{'unit_learning_mode':'record'},'hud':{'font_size':40}})
    assert not other_bad['ok'] and path.read_bytes()==before  # preserve main full validation
    full=settings_schema.schema(ctx.settings,ctx.db,full=True)
    assert any(f['key']=='dry_run' for s in full for f in s['fields'])
    proof=dict(home=os.environ['JCC_HOME'],schema=common,feature_settings=feature_before,
               saved_status=saved,reopened=asdict(reopened),reopened_schema=data(api.settings_schema()),
               rejected_learning=bad,rejected_main=other_bad,unchanged_after_invalid=True,
               final_status=data(api.status()),final_settings=asdict(Settings.load()),events=bus.events)
    (tmp_path/'api-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
