"""Session audit risks: alignment, provenance, abstention and one-to-one gates."""
import json
from pathlib import Path

import pytest

from scripts.eval_vision_session import (align, frame_files, label_value, matching, name_coverage,
                                         score, sha, time_series, validate_truth)


def known(value):
    return dict(status='known', value=value, evidence='manual pixels')


def human_row(name='00001.jpg', count=0):
    return dict(_file=name, sha256='abc', phase=known('combat'), stage=known('2-3'),
                gold=known(7), level=known(5), opponent_audit=True,
                enemy=dict(**known(count), positions=[[.5, .3]]*count,
                           friendly_positions=[], uncertain_positions=[], exclusions='green and thick bars excluded'))


def truth(row):
    return dict(schema='vision-session-human-v1', provenance='human_pixel_review', reviewer='manual', frames=[row])


def entry():
    return dict(file='00001.jpg', sha256='abc', reasons=['opponent_early_2-3'])


def prediction(name='00001.jpg', points=()):
    return dict(_file=name, prediction=dict(phase='combat', stage='2-3', gold=7, level=5,
                opponent=dict(board=[dict(x=x, y=y) for x, y in points])))


@pytest.mark.parametrize('names', [['00001.jpg', '00003.jpg'], ['00001.jpg', '00001.png'], ['x.jpg']])
def test_frame_numbers_reject_missing_duplicate_numeric_ids_and_non_numeric(tmp_path, names):
    folder = tmp_path/'frames'; folder.mkdir()
    for name in names:
        (folder/name).write_bytes(b'x')
    with pytest.raises(ValueError, match='numbering'):
        frame_files(tmp_path)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'order', 'file'])
def test_alignment_rejects_changed_sequence(change):
    files = [Path('00001.jpg'), Path('00002.jpg')]
    rows = [dict(_file=p.name) for p in files]
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(rows[0])
    elif change == 'order': rows.reverse()
    else: rows[1]['_file'] = '00003.jpg'
    with pytest.raises(ValueError, match='alignment'):
        align(files, rows)


@pytest.mark.parametrize('change', ['sha256', 'index', 'error', 'schema_error', 'prediction'])
def test_predictions_must_bind_current_pixels_and_successful_schema(tmp_path, change):
    p = tmp_path/'00001.jpg'; p.write_bytes(b'image bytes')
    row = dict(_file=p.name, index=1, sha256=sha(p), prediction={}, error=None, schema_error=None)
    align([p], [row], predictions=True)
    row[change] = None if change == 'prediction' else 'bad'
    with pytest.raises(ValueError):
        align([p], [row], predictions=True)


@pytest.mark.parametrize('machine', [[], {'phase': 'combat', '_file': '00001.jpg'},
                                      {'schema': 'vision-session-human-v1', 'provenance': 'machine_log'}])
def test_machine_states_are_never_human_truth(machine):
    with pytest.raises(ValueError, match='machine states'):
        validate_truth(machine, [entry()])


@pytest.mark.parametrize('change', ['missing', 'hash', 'opponent_selection', 'count', 'coordinate', 'reason', 'field_type'])
def test_fixed_human_selection_positions_and_evidence_are_required(change):
    row = human_row(count=1); data = truth(row)
    validate_truth(data, [entry()])
    if change == 'missing': data['frames'] = []
    elif change == 'hash': row['sha256'] = 'other'
    elif change == 'opponent_selection': row['opponent_audit'] = False
    elif change == 'count': row['enemy']['value'] = 2
    elif change == 'coordinate': row['enemy']['positions'][0][0] = float('nan')
    elif change == 'reason': row['phase']['evidence'] = ''
    else: row['gold']['value'] = True
    with pytest.raises(ValueError):
        validate_truth(data, [entry()])


def test_unknown_is_not_wrong_or_correct_and_never_helps_quantity_gate():
    row = human_row()
    row['phase'] = dict(status='unknown', value=None, evidence='celebration transition')
    row['enemy'].update(status='unknown', value=None, evidence='occluded')
    metrics, _ = score([row], {row['_file']: prediction()})
    assert metrics['phase'] == dict(correct=0, decidable=0, wrong=0, unknown=1, total=1)
    assert metrics['enemy_count']['unknown'] == 1
    assert not metrics['quantity_ownership_gate']
    with pytest.raises(ValueError):
        label_value(dict(status='unknown', value=2, evidence='not readable'))


def test_victory_is_phase_failure_even_when_machine_says_combat_consistently():
    row = human_row(); row['phase'] = known('ended')
    metrics, details = score([row], {row['_file']: prediction()})
    assert metrics['phase']['wrong'] == 1
    assert not metrics['phase_all_correct']
    assert details[0]['fields']['phase']['truth'] == 'ended'


def test_one_truth_cannot_be_reused_by_duplicate_predictions():
    assert len(matching([[.5, .3], [.501, .301]], [[.5, .3]])) == 1
    row = human_row(count=1)
    metrics, _ = score([row], {row['_file']: prediction(points=[(.5, .3), (.501, .301)])})
    assert metrics['enemy_recalled'] == 1
    assert metrics['unmatched_predictions'] == 1
    assert metrics['enemy_count']['wrong'] == 1


def test_matching_finds_augmenting_path_instead_of_greedy_undercount():
    # First prediction can match either truth, second only the first.
    pairs = matching([[.51, .3], [.491, .3]], [[.5, .3], [.525, .3]])
    assert len(pairs) == 2
    assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == 2
    assert not matching([[.6, .3]], [[.5, .3]])


@pytest.mark.parametrize('correct,passed', [(19, False), (20, True), (24, True)])
def test_quantity_gate_keeps_eighty_percent_of_all_fixed_samples(correct, passed):
    rows = [human_row(f'{i:05d}.jpg') for i in range(24)]
    predictions = {r['_file']: prediction(r['_file'], [] if i < correct else [[.7, .6]])
                   for i, r in enumerate(rows)}
    metrics, _ = score(rows, predictions)
    assert metrics['enemy_count']['correct'] == correct
    assert metrics['enemy_count']['total'] == 24
    assert metrics['quantity_ownership_gate'] is passed


def test_friendly_leak_fails_gate_even_with_correct_counts():
    row = human_row(count=1); row['enemy']['friendly_positions'] = [[.51, .3]]
    metrics, details = score([row], {row['_file']: prediction(points=[[.509, .3]])})
    assert metrics['enemy_count']['correct'] == 1
    assert metrics['friendly_as_enemy'] == 1
    assert metrics['friend_window_overlap'] == 1
    assert metrics['enemy_recalled'] == 1  # a true match does not suppress a nearer friend
    assert details[0]['enemy']['friendly_as_enemy_indices'] == [0]
    assert not metrics['quantity_ownership_gate']


def test_crowded_enemy_friend_windows_remain_a_conservative_flag_for_review():
    row = human_row(count=1)
    row['enemy']['friendly_positions'] = [[.518, .32]]
    metrics, details = score([row], {row['_file']: prediction(points=[[.5, .3]])})
    assert metrics['enemy_recalled'] == 1
    assert metrics['unmatched_predictions'] == 0
    assert metrics['friendly_as_enemy'] == 0
    assert metrics['friend_window_overlap'] == 1
    assert details[0]['enemy']['friend_window_overlap_indices'] == [0]
    check = details[0]['enemy']['ownership_checks'][0]
    assert check['nearest_enemy']['distance'] == 0
    assert check['nearest_friend']['distance'] > 0
    assert metrics['quantity_ownership_gate']  # overlap alone is not an ownership gate


@pytest.mark.parametrize('point,enemies,friends,false_count,overlap', [
    ([.5, .3], [[.515, .3]], [[.502, .3]], 1, 1),
    ([.5, .3], [[.502, .3]], [[.515, .3]], 0, 1),
    ([.5, .5], [[.484375, .5]], [[.515625, .5]], 0, 1),  # exact binary tie
    ([0, .5], [[.015, .5]], [[.02, .5]], 0, 0),  # strict dx boundary
    ([.5, 0], [[.5, .01]], [[.5, .03]], 0, 0),  # strict dy boundary
    ([0, 0], [[.1, .1]], [[.2, .2]], 0, 0),  # infinity tie
    ([.5, .3], [[.8, .3]], [[.501, .3]], 1, 1),
    ([.5, .3], [], [[.501, .3]], 1, 1),
    ([.5, .3], [[.501, .3]], [], 0, 0),
    ([.5, .3], [[.505, .3]], [[.51, .3], [.501, .3]], 1, 1),
])
def test_ownership_matches_original_scorer_with_ties_and_strict_windows(point, enemies, friends, false_count, overlap):
    from scripts.eval_opponent import score as original_score
    row = human_row(count=len(enemies))
    row['enemy'].update(positions=enemies, friendly_positions=friends)
    pred = prediction(points=[point])
    metrics, details = score([row], {row['_file']: pred})
    original, false = original_score(pred['prediction']['opponent'], dict(
        board_count=len(enemies), enemy_positions=[dict(x=x, y=y) for x, y in enemies],
        friendly_positions=[dict(x=x, y=y) for x, y in friends]))
    assert metrics['friendly_as_enemy'] == original['friendly_as_enemy'][0] == len(false) == false_count
    assert metrics['friend_window_overlap'] == overlap
    checks = details[0]['enemy']['ownership_checks']
    assert len(checks[0]['enemy_distances']) == len(enemies)
    assert len(checks[0]['friend_distances']) == len(friends)


def test_duplicate_predictions_keep_legacy_ownership_count_and_one_to_one_enemy_recall():
    row = human_row(count=1); row['enemy']['friendly_positions'] = [[.51, .3]]
    metrics, details = score([row], {row['_file']: prediction(points=[[.509, .3], [.509, .3]])})
    assert metrics['enemy_recalled'] == 1
    assert metrics['unmatched_predictions'] == 1
    assert metrics['friendly_as_enemy'] == metrics['friend_window_overlap'] == 2
    assert details[0]['enemy']['friendly_as_enemy_indices'] == [0, 1]


def test_real_01352_enemy_is_nearer_despite_friend_window_overlap():
    data = json.loads((Path(__file__).parent/'fixtures/vision_session_b9_truth.json').read_text())
    row = next(r for r in data['frames'] if r['_file'] == '01352.jpg')
    metrics, details = score([row], {row['_file']: prediction(row['_file'], [[.708, .280]])})
    assert metrics['friendly_as_enemy'] == 0
    assert metrics['friend_window_overlap'] == 1
    check = details[0]['enemy']['ownership_checks'][0]
    assert check['nearest_enemy']['index'] == check['nearest_friend']['index'] == 6
    assert check['nearest_enemy']['distance'] == pytest.approx(.00277)
    assert check['nearest_friend']['distance'] == pytest.approx(.03712)


def test_names_are_coverage_only_with_separate_player_and_unit_denominators():
    rows = [prediction(points=[[.5, .3], [.6, .4]]), prediction()]
    rows[0]['prediction']['opponent'].update(name='player')
    rows[0]['prediction']['opponent']['board'][0]['name'] = '?1'
    rows[0]['prediction']['opponent']['board'][1]['name'] = 'named unit'
    result = name_coverage(rows)
    assert result['player_named_frames'] == result['unit_named_count'] == 1
    assert result['opponent_structures'] == result['detected_units'] == 2
    assert result['name_accuracy'] is None


def test_duplicate_human_points_cannot_inflate_one_to_one_recall():
    row = human_row(count=2)
    with pytest.raises(ValueError, match='duplicate human positions'):
        validate_truth(truth(row), [entry()])


def test_time_series_keeps_null_stage_stale_spans_and_backward_source_time():
    rows = []
    for i, stage in enumerate(['2-3', None, None, '2-2'], 1):
        row = prediction(f'{i:05d}.jpg')
        row.update(index=i, source_t=1 if i == 4 else i)
        row['prediction'].update(stage=stage, units_stale=i in (2, 3))
        rows.append(row)
    result = time_series(rows)
    assert result['missing_stage_frames'] == ['00002.jpg', '00003.jpg']
    assert result['units_stale_spans'][0]['frames'] == 2
    assert {r.get('anomaly') for r in result['transitions']} >= {'stage_backward', 'source_time_not_increasing'}


def test_existing_output_evidence_is_never_modified_on_cli_error(tmp_path, monkeypatch):
    from scripts.eval_vision_session import main
    output = tmp_path/'old'; output.mkdir()
    (output/'keep.txt').write_text('keep')
    monkeypatch.setattr('sys.argv', ['eval_vision_session.py', 'prepare', '--session', str(tmp_path),
                                   '--selection', str(tmp_path/'missing.json'), '--output', str(output)])
    assert main() == 2
    assert {p.name for p in output.iterdir()} == {'keep.txt'}


def test_run_refuses_output_inside_frozen_session_before_writing(tmp_path):
    from argparse import Namespace
    from scripts.eval_vision_session import run
    session = tmp_path/'session'; session.mkdir()
    with pytest.raises(ValueError, match='frozen input'):
        run(Namespace(session=session, gallery=tmp_path/'gallery', output=session/'new'))
    assert list(session.iterdir()) == []


def test_committed_human_truth_covers_fixed_37_and_24_enemy_frames():
    path = Path(__file__).parent/'fixtures/vision_session_b9_truth.json'
    data = json.loads(path.read_text())
    entries = [dict(file=r['_file'], sha256=r['sha256'],
                    reasons=['opponent_manual' if r['opponent_audit'] else 'other']) for r in data['frames']]
    rows = validate_truth(data, entries)
    assert len(rows) == 37
    assert sum(r['opponent_audit'] for r in rows) == 24
    assert {r['phase']['value'] for r in rows[-3:]} == {'ended'}
