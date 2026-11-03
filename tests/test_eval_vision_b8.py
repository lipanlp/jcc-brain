"""B8 的差异豁免只接受任务批准的结构位置和时间范围。"""
import copy

import pytest

from scripts.eval_vision_b8 import compare_frames


def frame(phase='planning', name='?i1', **fields):
    return dict(_file='0001.png', rep=0, prediction=dict(phase=phase,
                board=[dict(name='本人', x=.4, y=.3, items=[dict(name=name, slot=0, x=.5)])],
                bench=[], item_bench=[dict(name=name, slot=1)], units_stale=False, **fields))


def violations(old, new):
    return [change for change in compare_frames(old, new) if change['rule'] is None]


def test_identical_frames_have_no_changes():
    assert compare_frames([frame()], [frame()]) == []


def test_spectate_units_and_removal_of_foreign_auto_label_are_explicitly_reported():
    old = frame('spectate', auto_labeled=['外来'])
    new = copy.deepcopy(old)
    new['prediction'].update(board=[dict(name='?1', x=.7, y=.4)], bench=[dict(slot=2, name='?2')])
    del new['prediction']['auto_labeled']
    changes = compare_frames([old], [new])
    assert not violations([old], [new])
    assert {r['rule'] for r in changes} == {'spectate_current_units', 'remove_foreign_auto_labeled'}
    assert {r['field'] for r in changes} >= {'board.0.x', 'bench', 'auto_labeled'}


@pytest.mark.parametrize('change', ['phase', 'legend', 'gold', 'learning', 'stale'])
def test_spectate_does_not_exempt_non_unit_fields_or_stale_output(change):
    old = frame('spectate', legend={'x': .2}, gold=1)
    new = copy.deepcopy(old)
    pred = new['prediction']
    if change == 'phase':
        pred['phase'] = 'combat'
    elif change == 'legend':
        pred['legend']['x'] = .3
    elif change == 'gold':
        pred['gold'] = 2
    elif change == 'learning':
        pred['auto_labeled'] = ['外来']
    else:
        pred['units_stale'] = True
    assert violations([old], [new])


def test_only_immediate_non_own_scene_can_clear_foreign_units():
    old = [frame('spectate'), frame('carousel')]
    new = copy.deepcopy(old)
    new[1]['prediction'].update(board=[], bench=[])
    assert not violations(old, new)
    for phase in ('planning', 'combat'):
        old[1]['prediction']['phase'] = new[1]['prediction']['phase'] = phase
        assert violations(old, new)
    old[1]['prediction']['phase'] = new[1]['prediction']['phase'] = 'carousel'
    old.insert(1, frame('planning'))
    new.insert(1, frame('planning'))
    assert violations(old, new)


@pytest.mark.parametrize('name', ['装备', '?i2x', '?2', '?i'])
def test_unknown_item_cannot_turn_known_or_match_incomplete_regex(name):
    old = [frame('spectate'), frame()]
    new = copy.deepcopy(old)
    new[1]['prediction']['item_bench'][0]['name'] = name
    assert violations(old, new)


def test_unknown_item_number_only_changes_after_spectate_in_same_repeat():
    old = [frame(), frame('spectate'), frame()]
    new = copy.deepcopy(old)
    new[2]['prediction']['item_bench'][0]['name'] = '?i42'
    assert not violations(old, new)
    new[0]['prediction']['item_bench'][0]['name'] = '?i42'
    assert violations(old, new)
    new = copy.deepcopy(old)
    old[2]['rep'] = new[2]['rep'] = 1
    new[2]['prediction']['item_bench'][0]['name'] = '?i42'
    assert violations(old, new)


@pytest.mark.parametrize('change', ['slot', 'x', 'badge', 'length', 'order', 'own_name', 'own_x'])
def test_unknown_item_rule_preserves_other_attributes_lists_and_own_units(change):
    old = [frame('spectate'), frame()]
    old[1]['prediction']['item_bench'].append(dict(name='已知', slot=2))
    new = copy.deepcopy(old)
    pred = new[1]['prediction']
    pred['board'][0]['items'][0]['name'] = '?i9'  # legitimate change alongside the forbidden one
    if change == 'length':
        pred['item_bench'].append(dict(name='?i3'))
    elif change == 'order':
        pred['item_bench'].reverse()
    elif change.startswith('own_'):
        pred['board'][0]['name' if change == 'own_name' else 'x'] = '外来' if change == 'own_name' else .9
    else:
        pred['item_bench'][0][change] = 7
    assert violations(old, new)


@pytest.mark.parametrize('change', ['missing', 'order', 'file', 'rep'])
def test_frame_count_identity_order_and_repeat_are_fixed(change):
    old = [frame(), frame()]
    old[1]['_file'] = '0002.png'
    new = copy.deepcopy(old)
    if change == 'missing':
        new.pop()
    elif change == 'order':
        new.reverse()
    elif change == 'file':
        new[1]['_file'] = '0003.png'
    else:
        new[1]['rep'] = 1
    assert violations(old, new)


def test_scalar_json_types_are_not_coerced_by_python_equality():
    old, new = frame(gold=1), frame(gold=True)
    assert violations([old], [new])


def test_process_classification_excludes_own_ancestors_but_detects_background_vision():
    from scripts.eval_vision_b8 import classify_processes
    raw = ('100 1 0.0 zsh -lc eval_vision_b8.py suite\n'
           '101 100 0.0 python scripts/eval_vision_b8.py suite\n'
           '906 894 9.9 .venv/bin/python -u scripts/record_assistant.py --minutes 60\n'
           '907 1 2.0 /Applications/金铲铲大脑.app/Contents/MacOS/金铲铲大脑\n')
    review = classify_processes(raw, own_pid=101, own_parent=100)
    assert review['timing_status'] == '未测（环境不净）'
    assert [line.split()[0] for line in review['suspects']] == ['906', '907']
    assert classify_processes(raw.split('906')[0], own_pid=101, own_parent=100)['timing_status'] == 'measured'


def opponent_value():
    return dict(name=None, board=[dict(name='?1',x=.3,y=.2)], mirrored=True)


def pixel_audit(**changes):
    return dict(frame_count=53, removed_count=0, reviewed_count=24, added_count=24,
                unreviewed_count=0, false_new_count=0, uses_product_enemy_bars=True, **changes)


def test_s3b_enabling_requires_quantity_ownership_and_complete_pixel_review():
    from scripts.eval_vision_b8 import opponent_update_gate
    good = dict(enabled=True,counts=dict(board_count=[33,36],friendly_as_enemy=[0,99]),
                frames=[dict(_file='0033.jpg',rep=0,prediction=opponent_value(),production=opponent_value())])
    assert opponent_update_gate(good,pixel_audit())['errors'] == []
    for path,value in [('board_count',[29,36]),('friendly_as_enemy',[1,99])]:
        bad = copy.deepcopy(good);bad['counts'][path] = value
        assert opponent_update_gate(bad,pixel_audit())['errors']
    for key,value in [('unreviewed_count',1),('false_new_count',1),('removed_count',1),('reviewed_count',23)]:
        audit=pixel_audit();audit[key]=value
        assert opponent_update_gate(good,audit)['errors']
    bad=copy.deepcopy(good);bad['frames'][0]['production']=None
    assert opponent_update_gate(bad,pixel_audit())['errors']


def test_s3b_flag_allows_only_candidate_and_passed_combat_opponent():
    from scripts.eval_vision_b8 import compare_opponent_frames
    old=frame('combat');old['prediction']['opponent']=None
    new=copy.deepcopy(old);new['prediction']['opponent']=opponent_value()
    changes=compare_opponent_frames([old],[new],group='standard-items-phase',enabled=True,gate_passed=True)
    assert {c['rule'] for c in changes} == {'enabled_combat_opponent_after_gate'}
    for enabled,passed in [(False,True),(True,False),(False,False)]:
        assert any(c['rule'] is None for c in compare_opponent_frames(
            [old],[new],group='standard-items-phase',enabled=enabled,gate_passed=passed))
    for phase in ('planning','spectate','carousel'):
        a,b=copy.deepcopy(old),copy.deepcopy(new)
        a['prediction']['phase']=b['prediction']['phase']=phase
        assert any(c['rule'] is None for c in compare_opponent_frames(
            [a],[b],group='digits',enabled=True,gate_passed=True))


@pytest.mark.parametrize('change',['board','bench','phase','gold','item_bench'])
def test_s3b_flag_never_exempts_own_fields_even_when_enabled(change):
    from scripts.eval_vision_b8 import compare_opponent_frames
    old=frame('combat',gold=2);old['prediction']['opponent']=None
    new=copy.deepcopy(old);new['prediction']['opponent']=opponent_value()
    new['prediction'][change] = 'changed'
    assert any(c['rule'] is None for c in compare_opponent_frames(
        [old],[new],group='units-standard',enabled=True,gate_passed=True))


def test_s3b_disabled_mode_is_strict_even_on_spectate_and_unknown_equipment():
    from scripts.eval_vision_b8 import compare_opponent_frames
    old=[frame('spectate'),frame()];new=copy.deepcopy(old)
    new[0]['prediction']['board']=[]
    new[1]['prediction']['item_bench'][0]['name']='?i44'
    changes=compare_opponent_frames(old,new,group='digits',enabled=False,gate_passed=False)
    assert len(changes) == 2 and all(c['rule'] is None for c in changes)


def test_s3b_candidate_changes_do_not_exempt_truth_or_frame_identity():
    from scripts.eval_vision_b8 import compare_opponent_frames
    old=dict(rep=0,_file='0033.jpg',prediction=opponent_value(),production=None,false_friendly=[],truth_count=1)
    new=copy.deepcopy(old);new['prediction']['board'].append(dict(name='?2',x=.4,y=.3))
    changes=compare_opponent_frames([old],[new],group='opponent',enabled=False,gate_passed=False)
    assert all(c['rule'] == 'explicit_opponent_candidate_update' for c in changes)
    for key,value in [('truth_count',2),('_file','0036.jpg'),('rep',1)]:
        bad=copy.deepcopy(new);bad[key]=value
        assert any(c['rule'] is None for c in compare_opponent_frames(
            [old],[bad],group='opponent',enabled=False,gate_passed=False))


def test_s3b_raw_report_derivation_uses_original_truth_and_repeat3():
    from scripts.eval_vision_b8 import validate_opponent_report
    wanted={f'{i:04d}.jpg':dict(board_count=1,enemy_positions=[dict(x=.3,y=.2)],friendly_positions=[])
            for i in range(12)}
    rows=[dict(rep=rep,_file=file,prediction=opponent_value(),production=None,false_friendly=[],truth_count=1)
          for rep in range(3) for file in wanted]
    report=dict(frames=rows,counts=dict(board_count=[36,36],friendly_as_enemy=[0,36]))
    assert validate_opponent_report(report,wanted)==[]
    for change in ('counts','frame','false'):
        bad=copy.deepcopy(report)
        if change=='counts':bad['counts']['board_count'][0]=35
        elif change=='frame':bad['frames'].pop()
        else:bad['frames'][0]['false_friendly']=[dict(name='false')]
        assert validate_opponent_report(bad,wanted)


def test_process_classification_detects_playcover_game_and_launcher():
    from scripts.eval_vision_b8 import classify_processes
    commands = ['/Users/user/Library/Containers/io.playcover.PlayCover/Applications/com.tencent.jkchess.app/jkchess',
                '/Applications/PlayCover.app/Contents/MacOS/PlayCover']
    for command in commands:
        snapshot=classify_processes('100 1 36.3 '+command+'\n',own_pid=101,own_parent=1)
        assert snapshot['timing_status'] == '未测（环境不净）'
        assert len(snapshot['suspects']) == 1
