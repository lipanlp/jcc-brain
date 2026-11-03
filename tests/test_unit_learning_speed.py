"""Private scene reading and timing/freshness at the real executor boundary."""
from unittest.mock import Mock

import numpy as np
import pytest

from test_unit_learning_auto import auto, run
from test_vision_spectate_isolation import make, scene, state
from jccbrain.actuator.actions import ClickPoint, RestoreMouse
from jccbrain.vision.eye import reader as module
from jccbrain.vision.eye.image import Region
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader


def test_fast_scene_uses_current_digits_and_private_memory_without_identity_work(tmp_path, monkeypatch, scene):
    own, foreign = scene
    r = make(tmp_path, 'scene')
    r.regions['timer'] = Region(.02, 0, .01, .01, digits='timer')
    r.regions['shop'] = Region(.05, 0, .01, .01)
    monkeypatch.setattr(module, 'read_digits', lambda im, _h, kind: '30' if kind == 'timer' else
                        ('1-1' if im[0, 0, 0] == 2 else '3-1'))
    frozen = state(r)
    ctx = r._foreign_context()
    for method in ('prepare', 'observe', 'read_current', 'detect'):
        setattr(ctx.units, method, Mock(side_effect=AssertionError('no unit work in scene reader')))
    ctx.augments_taken.read = Mock(side_effect=AssertionError('no augment work'))
    prepare = Mock(wraps=ctx.ocr.prepare)
    ctx.ocr.prepare = prepare
    out = ctx._read_learning_scene(own)
    assert out['stage'] == '3-1' and out['timer'] == 30 and out['phase'] == 'planning'
    assert set(out) <= {'stage','timer','phase','viewing','players','players_stale','right_panel',
                        'hp','banner','placement','next_round','round_type'}
    assert prepare.call_count == 1  # banner only, never shop
    assert ctx._read_learning_scene(foreign)['phase'] == 'spectate'
    assert ctx.last_stage == '1-1' and state(r) == frozen
    # Missing current HUD must not be filled from the last known stage/timer.
    monkeypatch.setattr(module, 'read_digits', lambda *_: None)
    out = ctx._read_learning_scene(own)
    assert 'stage' not in out and 'timer' not in out
    with pytest.raises(ValueError, match='isolated'):
        r._read_learning_scene(own)


@pytest.mark.parametrize('banner,players,result,scene_kind,expected', [
    ('准备阶段','','',None,'planning'), ('战斗开始','','',None,'combat'),
    ('','伤害','',None,'combat'), ('强化符文','','',None,'augment'),
    ('','','第二名',None,'ended'), ('准备阶段','','','carousel','carousel'),
    ('准备阶段','','','dialog','dialog'), ('准备阶段','','','spectate','spectate')])
def test_fast_scene_shares_original_priority_and_timer_transition(tmp_path, monkeypatch, banner, players, result, scene_kind, expected):
    r = make(tmp_path, 'phase')
    image = np.zeros((1136,1774,3),np.uint8)
    monkeypatch.setattr(module, 'detect_scene', lambda *_: scene_kind)
    monkeypatch.setattr(module, 'detect_reconnect', lambda *_: False)
    def begin(self, im, only=None, before_ocr=None, **_):
        assert only == {'stage','timer','banner','result','players'}
        fields = dict(stage='3-1', timer='30', banner=banner, players=players, result=result)
        cells = {k:[[Line(v,1.,0,0,1,1)]] for k,v in fields.items() if v}
        return lambda: cells, {}
    monkeypatch.setattr(GameReader, '_begin_cells', begin)
    ctx = r._foreign_context()
    assert ctx._read_learning_scene(image)['phase'] == expected


def test_cleanup_uses_six_temporary_captures_and_fresh_close_frame(auto):
    assert run(auto)
    timing = auto.runner.status['timing']
    assert auto.captures == 7 and timing['samples'] == 6
    assert timing['actions'] == timing['action_attempts'] == 3 and timing['uncertain_actions'] == 0
    assert timing['reused_closed'] == 1
    assert [type(a) for a in auto.ctx.executor.history] == [ClickPoint,ClickPoint,RestoreMouse]
    assert sum(timing['segments_ms'].values()) == pytest.approx(timing['total_ms'])
    assert timing['capture_ms'] == pytest.approx(6*130)


def test_slow_panel_ocr_already_supplies_interval_without_extra_wait(auto):
    e = auto
    original = e.ctl.panel.read.return_value
    e.ctl.panel.read.side_effect = lambda _: (e.clock.wait(.15),original)[1]
    run(e)
    samples = [d['sample'] for k,d in e.trace if k == 'sample']
    panel = [s for s in samples if s.frame.sequence in (4,5)]
    assert len(panel) == 2 and panel[1].frame.monotonic-panel[0].frame.monotonic >= .15
    # First panel / close loading still waits, second panel does not stack .12s.
    assert e.runner.status['timing']['wait_ms'] == pytest.approx(200)
    assert e.runner.status['verified'] == 1


def test_fast_ocr_waits_only_for_remaining_gap(auto):
    run(auto)
    assert auto.runner.status['verified'] == 1
    assert auto.runner.status['timing']['wait_ms'] == pytest.approx(300)
    records = auto.ctl.store.entries()
    evidence = __import__('json').loads((auto.ctl.store.verified/records[0]['id']/'evidence.json').read_text())
    assert evidence['frame']['monotonic']-evidence['previous']['monotonic'] >= .10


def test_partial_ocr_wait_is_only_unspent_fraction_of_interval(auto):
    answer=auto.ctl.panel.read.return_value
    auto.ctl.panel.read.side_effect=lambda _: (auto.clock.wait(.04),answer)[1]
    run(auto)
    assert auto.runner.status['verified']==1
    assert auto.runner.status['timing']['wait_ms']==pytest.approx(260)


def test_restore_guard_wait_expiring_closed_frame_requires_new_capture(auto):
    e = auto
    def guard():
        if len(e.ctx.executor.history) == 2:
            e.clock.wait(.30)
    e.guard.check.side_effect = guard
    run(e)
    timing = e.runner.status['timing']
    assert timing['reused_closed'] == 0 and timing['samples'] == 7
    assert timing['guard_ms'] == pytest.approx(300)
    assert len(e.ctx.executor.history) == 3


def test_slow_new_restore_scene_does_not_trigger_a_second_redundant_capture(auto):
    e=auto
    original=e.reader._foreign_context
    def context():
        ctx=original();read=ctx._read_learning_scene
        ctx._read_learning_scene=lambda image:(e.clock.wait(.30),read(image))[1]
        return ctx
    e.reader._foreign_context=context
    run(e)
    t=e.runner.status['timing']
    assert t['actions']==3 and t['reused_closed']==0 and t['samples']==7


@pytest.mark.parametrize('action_index',[0,1,2])
def test_guard_wait_user_intervention_never_admits_next_action_or_cleanup(auto,action_index):
    e = auto
    def guard():
        if len(e.ctx.executor.history) == action_index:
            e.clock.wait(.30)
            e.counters = (('hardware',1),)
    e.guard.check.side_effect = guard
    run(e)
    assert len(e.ctx.executor.history) == action_index
    assert e.runner.status['timing']['actions'] == action_index
    # A .30s probe gap plus an intervening raw event also invalidates the tap;
    # either path must cancel before the next send.
    assert e.runner.status['reason'] in ('hardware_input_changed','input_monitor_unavailable')


def test_close_click_revalidates_empty_pixels_after_guard_wait(auto,monkeypatch):
    e = auto
    from jccbrain.modes import unit_learning
    def guard():
        if len(e.ctx.executor.history) == 1:
            e.clock.wait(.30)
            monkeypatch.setattr(unit_learning, 'empty_point', lambda *_: None)
    e.guard.check.side_effect = guard
    run(e)
    assert len(e.ctx.executor.history) == 1 and e.panel
    assert e.runner.status['reason'] == 'action_target_changed'
    assert e.runner.status['timing']['actions'] == 1


def test_partial_executor_exception_is_reported_as_uncertain_input(auto):
    e = auto
    e.action_hook = lambda _: (_ for _ in ()).throw(OSError('failed after send'))
    run(e)
    t = e.runner.status['timing']
    assert len(e.ctx.executor.history) == 1 and t['actions'] == 0 and t['uncertain_actions'] == 1
    assert t['action_attempts'] == 1 and e.runner.status['reason'] == 'learning_error'
