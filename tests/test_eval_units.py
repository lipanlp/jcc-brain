"""独立验证单位名评测分母、坐标匹配、卡池读取和图库写入隔离。"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from scripts.eval_units import (combine, copied_gallery, evaluate, frame_key, load_truth,
                                match_units, pool_heroes, validate_truth)

ROOT = Path(__file__).resolve().parents[1]
POOL = {'阿狸', '奥恩'}


def score(units, state):
    return evaluate({'frame.jpg': state}, {'frame.jpg': {'units': units}}, POOL)


def test_bench_slots_and_board_positions_match_once_independent_of_input_order():
    expected = [{'zone': 'bench', 'slot': 3}, {'zone': 'board', 'x': .5, 'y': .3},
                {'zone': 'board', 'x': .51, 'y': .3}]
    predicted = [{'zone': 'board', 'x': .501, 'y': .301}, {'zone': 'bench', 'slot': 3},
                 {'zone': 'bench', 'slot': 4}]
    pairs, missing, extra = match_units(expected, predicted)
    assert set(pairs) == {(0, 1), (1, 0)}
    assert missing == [2] and extra == [2]


@pytest.mark.parametrize('x,y,matched', [(.019, .029, True), (.02, 0., False),
                                         (0., .03, False), (.021, .01, False)])
def test_board_thresholds_are_strict_window_proportions(x, y, matched):
    pairs, _, _ = match_units([{'zone': 'board', 'x': 0., 'y': 0.}],
                              [{'zone': 'board', 'x': x, 'y': y}])
    assert bool(pairs) == matched


@pytest.mark.parametrize('name', ['?1', '?42', None, '', '?'])
def test_unknown_or_missing_prediction_name_is_unnamed(name):
    result = score([{'zone': 'bench', 'slot': 0, 'name': '阿狸'}],
                   {'bench': [{'slot': 0, 'name': name}]})
    assert result['counts']['heroes'] == 1
    assert result['counts']['named'] == result['counts']['correct'] == 0
    assert result['accuracy'] is None and result['coverage'] == 0
    assert result['errors'][0]['reasons'] == ['unnamed']


def test_unknown_truth_and_wild_do_not_enter_hero_accuracy_or_coverage():
    result = score([{'zone': 'bench', 'slot': 0, 'name': '阿狸'},
                    {'zone': 'board', 'x': .5, 'y': .3, 'name': '野怪'},
                    {'zone': 'bench', 'slot': 1, 'name': '?'}],
                   {'bench': [{'slot': 0, 'name': '阿狸'}, {'slot': 1, 'name': '卡池外名字'}],
                    'board': [{'x': .5, 'y': .3, 'name': '奥恩'}]})
    c = result['counts']
    assert c['heroes'] == c['named'] == c['correct'] == c['wild'] == c['unknown'] == 1
    assert result['accuracy'] == result['coverage'] == 1
    assert c['wild_as_hero'] == c['outside_pool'] == 1


def test_outside_pool_names_count_even_on_unmatched_predictions():
    result = score([{'zone': 'bench', 'slot': 0, 'name': '阿狸'}],
                   {'bench': [{'slot': 0, 'name': '旧赛季英雄'}, {'slot': 2, 'name': '另一旧赛季英雄'}]})
    assert result['counts']['outside_pool'] == 2
    assert result['counts']['named'] == 1 and result['accuracy'] == result['coverage'] == 0
    assert result['counts']['unmatched_prediction'] == result['unmatched_units'] == 1
    assert result['errors'][0]['reasons'] == ['wrong_name', 'outside_pool']


def test_missing_positions_lower_coverage_and_report_both_sides():
    result = score([{'zone': 'bench', 'slot': 0, 'name': '阿狸'}],
                   {'board': [{'x': .3, 'y': .4, 'name': '奥恩'}]})
    assert result['counts']['unmatched_truth'] == result['counts']['unmatched_prediction'] == 1
    assert result['unmatched_units'] == 2 and result['coverage'] == 0 and result['accuracy'] is None


def test_no_evaluable_heroes_is_na_instead_of_perfect_accuracy():
    result = score([{'zone': 'bench', 'slot': 0, 'name': '?'}], {'bench': [{'slot': 0, 'name': '阿狸'}]})
    assert result['accuracy'] is result['coverage'] is None


def test_combining_repeats_sums_denominators_and_marks_error_runs():
    one = score([{'zone': 'bench', 'slot': 0, 'name': '阿狸'}], {'bench': [{'slot': 0, 'name': '奥恩'}]})
    result = combine([one, one, one])
    assert result['frames'] == 1 and result['frame_evaluations'] == 3 and result['repeat'] == 3
    assert result['counts']['named'] == result['counts']['heroes'] == 3
    assert [e['run'] for e in result['errors']] == [1, 2, 3]


def test_relative_frame_paths_do_not_mix_standard_and_track_with_same_basename(tmp_path):
    root = tmp_path / 'samples'
    key = frame_key(root / 'track/0021.jpg', root)
    assert key == 'track/0021.jpg'
    result = evaluate({'0021.jpg': {'bench': [{'slot': 0, 'name': '奥恩'}]},
                       key: {'bench': [{'slot': 0, 'name': '阿狸'}]}},
                      {key: {'units': [{'zone': 'bench', 'slot': 0, 'name': '阿狸'}]}}, POOL)
    assert result['frames'] == result['counts']['correct'] == 1


def test_gallery_copy_is_fresh_each_run_and_never_writes_back(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    index = source / 'index.json'
    index.write_text('[]')
    (source / 'unit.npy').write_bytes(b'original feature')
    for _ in range(2):
        with copied_gallery(source) as copied:
            assert (copied / 'index.json').read_text() == '[]'
            assert (copied / 'unit.npy').read_bytes() == b'original feature'
            (copied / 'index.json').write_text('[{"name":"自动标注"}]')
            (copied / 'unit.npy').write_bytes(b'mutated')
            (copied / 'new.npy').write_bytes(b'new')
        assert not copied.exists()
        assert index.read_text() == '[]' and (source / 'unit.npy').read_bytes() == b'original feature'
        assert not (source / 'new.npy').exists()


def test_pool_query_filters_current_season_pool_and_cost_without_mutating_database(tmp_path):
    db = tmp_path / 'pool.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE heroes (season_id TEXT,id TEXT,name TEXT,cost INTEGER,traits TEXT,in_pool INTEGER)')
        conn.executemany('INSERT INTO heroes VALUES (?,?,?,?,?,?)',
                         [('18_S19', 'a', '阿狸', 4, '[]', 1), ('18_S19', 'b', '召唤物', 1, '[]', 0),
                          ('16_S19', 'c', '旧赛季', 1, '[]', 1), ('18_S19', 'd', '特殊零费', 0, '[]', 1),
                          ('18_S19', 'e', '特殊六费', 6, '[]', 1), ('18_S19', 'f', '阿狸', 4, '[]', 1)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert [(row['name'], row['cost']) for row in pool_heroes(db)] == [('阿狸', 4)]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_reference_names_are_all_in_frozen_current_pool():
    db = Path('/tmp/jcc-vf-db.sqlite3')
    if not db.exists():
        pytest.skip('冻结卡池副本不存在')
    validate_truth(load_truth(), {row['name'] for row in pool_heroes(db)})


def test_truth_notes_align_and_low_confidence_names_are_unscored():
    truth = load_truth()
    notes = load_truth(ROOT / 'tests/fixtures/unit_truth_notes.jsonl')
    assert set(truth) == set(notes)
    for file, frame in truth.items():
        assert len(frame['units']) == len(notes[file]['units'])
        for unit, note in zip(frame['units'], notes[file]['units']):
            assert unit == {key: note[key] for key in unit}
            assert note['basis'] in ['shop', 'traits', 'visual', 'user', 'replay'] and note['confidence'] in ['high', 'low']
            if note['confidence'] == 'low':
                assert unit['name'] == '?'
            if unit['zone'] == 'board':
                assert all(0 <= unit[key] <= 1 and unit[key] == round(unit[key], 3) for key in ['x', 'y'])
            else:
                assert 0 <= unit['slot'] <= 8


@pytest.mark.parametrize('units', [False, True])
def test_cli_units_replays_all_frames_but_keeps_legacy_truth_scoring(monkeypatch, tmp_path, capsys, units):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import eval_units
    import eval_vision

    samples = tmp_path / 'samples'
    samples.mkdir()
    for file in ['0001.jpg', '0002.jpg', '0003.jpg']:
        (samples / file).touch()
    vision_truth = tmp_path / 'vision.jsonl'
    vision_truth.write_text(json.dumps({'_file': '0002.jpg', 'bench': [], 'board': []}) + '\n')
    monkeypatch.setattr(eval_vision, 'DEFAULT_SAMPLES', samples)
    monkeypatch.setattr(eval_vision, 'TRUTH', {'truth': vision_truth})
    monkeypatch.setattr(eval_units, 'pool_heroes', lambda _: [dict(name='阿狸', cost=4)])
    monkeypatch.setattr(eval_units, 'load_truth', lambda: {f'000{i}.jpg': {'units': []} for i in range(1, 4)})
    seen = []

    def fake_run(files, args):
        seen.extend(f.name for f in files)
        return {f.name: {'bench': [], 'board': []} for f in files}, {
            'total': [1.] * len(files), 'first': [1.], 'ocr_cells': [0.] * len(files)}

    monkeypatch.setattr(eval_vision, 'run', fake_run)
    output = tmp_path / 'out.json'
    args = ['eval_vision.py', '--samples', str(samples), '--json', str(output)]
    if units:
        args += ['--units', '--db', str(tmp_path / 'db.sqlite3')]
    monkeypatch.setattr(sys, 'argv', args)
    assert eval_vision.main() == 0
    result = json.loads(output.read_text())
    assert seen == (['0001.jpg', '0002.jpg', '0003.jpg'] if units else ['0002.jpg'])
    assert result['total'] == [5, 5]  # 商店开关、两区数量与两区星级，仍只评分那一帧。
    assert ('units_name' in result) == units
    if units:
        assert result['units_name']['frames'] == 3
    capsys.readouterr()
