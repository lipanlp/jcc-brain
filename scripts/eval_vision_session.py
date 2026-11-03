"""Ordered CG replay and separate, provenance-checked human pixel audit.

run never scores against the recorder's machine states. prepare creates blind
review images; audit scores only explicit human labels, including unknowns.
All commands require a new output directory so failed evidence is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.eval_vision_b8 import differences, manifest, process_snapshot
from scripts.eval_opponent import score as score_opponent

SELECTION = ROOT / 'var/vision-module-optimization/batch-B9/preparation/audit-selection.json'
DX, DY = .02, .03  # Unchanged opponent truth tolerances, strict inequalities.


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_files(session):
    files = sorted(p for p in (session / 'frames').iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise ValueError('no frames')
    numbers = [int(p.stem) if p.stem.isdigit() else -1 for p in files]
    if numbers != list(range(1, len(files) + 1)):
        raise ValueError('frame numbering missing, duplicated, unordered or not 1-based')
    return files


def read_rows(path):
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def align(files, rows, *, predictions=False):
    names = [p.name for p in files]
    actual = [r.get('_file') for r in rows]
    if actual != names or len(set(actual)) != len(actual):
        raise ValueError('frame alignment: missing/duplicate/reordered files')
    if predictions:
        for i, (p, row) in enumerate(zip(files, rows), 1):
            if row.get('index') != i or row.get('sha256') != sha(p):
                raise ValueError(f'prediction index/hash mismatch: {p.name}')
            if row.get('error') or row.get('schema_error') or not isinstance(row.get('prediction'), dict):
                raise ValueError(f'incomplete prediction: {p.name}')


def source_inputs(args):
    return dict(session=manifest(args.session), db=manifest(args.db), gallery=manifest(args.gallery),
                item_icons=manifest(ROOT / 'var/item_icons'),
                augment_icons=manifest(ROOT / 'var/augment_icons'),
                legacy_seed=manifest(ROOT / 'game-eye/gallery'))


def board(pred):
    return (pred.get('opponent') or {}).get('board', [])


def name_coverage(rows):
    opponents = [r['prediction']['opponent'] for r in rows
                 if isinstance(r.get('prediction'), dict) and r['prediction'].get('opponent') is not None]
    units = [unit for opponent in opponents for unit in opponent.get('board', [])]
    named = sum(bool(unit.get('name')) and not unit['name'].startswith('?') for unit in units)
    players = sum(bool(opponent.get('name')) for opponent in opponents)
    return dict(opponent_structures=len(opponents), player_named_frames=players,
                player_name_coverage=players/len(opponents) if opponents else None,
                detected_units=len(units), unit_named_count=named,
                unit_name_coverage=named/len(units) if units else None,
                name_accuracy=None, source='machine output coverage only, no human name labels')


def run(args):
    for source in (args.session, args.gallery):
        if args.output.resolve().is_relative_to(source.resolve()):
            raise ValueError('output must not be inside a frozen input')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=args.output / 'reader.log', level=logging.INFO)
    before = source_inputs(args)
    write_json(args.output / 'inputs-before.json', before)
    files = frame_files(args.session)
    old = read_rows(args.session / 'states.jsonl')
    align(files, old)
    home = args.output / 'runtime'
    home.mkdir()
    shutil.copy2(args.db, home / 'db.sqlite3')
    shutil.copytree(args.gallery, home / 'gallery')
    for name in ('item_icons', 'augment_icons'):
        shutil.copytree(ROOT / 'var' / name, home / name)
    os.environ['JCC_HOME'] = str(home.resolve())
    from jccbrain.data.db import GameDB
    from jccbrain.core.models import GameState
    from jccbrain.vision.pyeye import build_reader
    from jccbrain.vision.eye import capture, reader as reader_module
    db = GameDB(home / 'db.sqlite3')
    # Exactly one reader for the entire session; no phase resets or reordering.
    reader = build_reader(db, home / 'gallery', args.ocr)
    write_json(args.output / 'runtime-before.json', manifest(home))
    metadata = dict(command=sys.argv, cwd=str(Path.cwd()), python=sys.executable,
                    head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    reader_import=reader_module.__file__, capture_import=capture.__file__,
                    decoder='CoreGraphics load_image', ocr_requested=args.ocr, ocr=reader.ocr.name,
                    one_reader=True, source_sha=manifest(ROOT / 'src/jccbrain/vision'),
                    script_sha256=sha(Path(__file__)), jcc_home=str(home.resolve()),
                    state_logs='machine-to-machine clues only; never accuracy truth')
    shutil.copy2(Path(__file__), args.output / 'run-script.py')
    write_json(args.output / 'metadata.json', metadata)
    phases, stages, sizes, diff_counts = Counter(), Counter(), Counter(), Counter()
    anomalies, errors, snapshots, read_ms = [], [], [], []
    opponent_frames = opponent_units = named_units = 0
    previous = None
    started = time.perf_counter()
    snapshots.append(dict(at='before_read', **process_snapshot()))
    with (args.output / 'predictions.jsonl').open('w') as out, \
            (args.output / 'machine-differences.jsonl').open('w') as diffs, \
            (args.output / 'progress.jsonl').open('w') as progress:
        for i, (path, original) in enumerate(zip(files, old), 1):
            row = dict(_file=path.name, index=i, sha256=sha(path), source_t=original.get('_t'),
                       prediction=None, error=None, schema_error=None)
            t0 = time.perf_counter()
            try:
                img = capture.load_image(path)
                t1 = time.perf_counter()
                h, w = img.shape[:2]
                sizes[f'{w}x{h}'] += 1
                row['dimensions'] = [w, h]
                pred = reader.read(img)
                t2 = time.perf_counter()
                row['prediction'] = pred
                row['timings_ms'] = dict(decode=(t1-t0)*1000, read=(t2-t1)*1000,
                                         stages=dict(reader.last_ms))
                read_ms.append((t2-t1)*1000)
                try:
                    GameState.from_dict(pred)
                except Exception:
                    row['schema_error'] = traceback.format_exc()
                    errors.append(dict(_file=path.name, type='schema', detail=row['schema_error']))
                phases[str(pred.get('phase'))] += 1
                stages[str(pred.get('stage'))] += 1
                units = board(pred)
                opponent_frames += bool(units)
                opponent_units += len(units)
                named_units += sum(bool(u.get('name')) and not u['name'].startswith('?') for u in units)
                if pred.get('phase') != 'combat' and pred.get('opponent') is not None:
                    anomalies.append(dict(_file=path.name, type='noncombat_opponent', value=pred['opponent']))
                if pred.get('phase') in (None, 'unknown', 'spectate'):
                    anomalies.append(dict(_file=path.name, type='phase_requires_review', phase=pred.get('phase')))
                stage = pred.get('stage')
                if not re.fullmatch(r'\d+-\d+', stage or ''):
                    anomalies.append(dict(_file=path.name, type='unreadable_stage', value=stage))
                elif previous:
                    a, b = tuple(map(int, previous.split('-'))), tuple(map(int, stage.split('-')))
                    if b < a:
                        anomalies.append(dict(_file=path.name, type='stage_backward', old=previous, new=stage))
                if stage and re.fullmatch(r'\d+-\d+', stage):
                    previous = stage
                old_prediction = {k: v for k, v in original.items() if not k.startswith('_')}
                for field, a, b in differences(old_prediction, pred):
                    key = '.'.join(map(str, field))
                    diffs.write(line(dict(_file=path.name, index=i, field=key, old=a, new=b,
                                          comparison='machine-to-machine; not accuracy')))
                    diff_counts[key.split('.')[0]] += 1
            except Exception:
                row['error'] = traceback.format_exc()
                errors.append(dict(_file=path.name, type='read_or_decode', detail=row['error']))
            out.write(line(row))
            if i % 50 == 0 or i == len(files):
                out.flush(); diffs.flush()
                progress.write(line(dict(done=i, total=len(files), errors=len(errors),
                                         elapsed_s=time.perf_counter()-started)))
                progress.flush()
            if i % 200 == 0:
                snapshots.append(dict(at=f'after_{i}', **process_snapshot()))
    snapshots.append(dict(at='after_read', **process_snapshot()))
    write_json(args.output / 'processes.json', snapshots)
    after = source_inputs(args)
    write_json(args.output / 'inputs-after.json', after)
    write_json(args.output / 'runtime-after.json', manifest(home))
    rows = read_rows(args.output / 'predictions.jsonl')
    alignment_error = None
    try:
        align(files, rows, predictions=True)
    except ValueError as exc:
        alignment_error = str(exc)
    unchanged = before == after
    passed = not errors and unchanged and alignment_error is None
    summary = dict(integrity_passed=passed, quality_status='not scored; run audit separately',
                   input_count=len(files), output_count=len(rows), unique_count=len({r['_file'] for r in rows}),
                   decoded_count=sum('dimensions' in r for r in rows),
                   read_errors=sum(r['type'] == 'read_or_decode' for r in errors),
                   schema_errors=sum(r['type'] == 'schema' for r in errors), errors=errors,
                   alignment_error=alignment_error, inputs_unchanged=unchanged,
                   dimensions=dict(sizes), phases=dict(phases), stages=dict(stages),
                   opponent_nonempty_frames=opponent_frames, opponent_detected_units=opponent_units,
                   opponent_named_units=named_units, name_coverage=named_units/opponent_units if opponent_units else None,
                   name_accuracy=None, naming_coverage=name_coverage(rows), anomalies=anomalies,
                   machine_difference_fields=dict(diff_counts), machine_comparison='NOT accuracy',
                   timing_status='未测（环境不净）' if any(s['suspects'] for s in snapshots) else 'measured',
                   raw_read_ms=dict(mean=sum(read_ms)/len(read_ms), maximum=max(read_ms)) if read_ms else None,
                   elapsed_s=time.perf_counter()-started)
    write_json(args.output / 'summary.json', summary)
    print(line(summary))
    return 0 if passed else 1


def selected(args):
    data = json.loads(args.selection.read_text())
    frames = data['frames']
    if len({r['file'] for r in frames}) != len(frames):
        raise ValueError('duplicate selection')
    return frames


def review_images(session, entries, output):
    """No reader, detector, prediction or machine state is used here."""
    import cv2
    import numpy as np
    from jccbrain.vision.eye.capture import load_image
    output.mkdir(parents=True, exist_ok=False)
    thumbs = []
    for entry in entries:
        path = session / 'frames' / entry['file']
        if sha(path) != entry['sha256']:
            raise ValueError(f'selection hash mismatch: {path.name}')
        image = load_image(path)
        h, w = image.shape[:2]
        shutil.copy2(path, output / path.name)  # byte-identical unannotated source
        # Normalized crops only help human visibility; no image detection.
        crops = {'board': (.12, .08, .87, .78), 'stage': (.37, .025, .60, .13),
                 'economy': (.27, .83, .65, .99)}
        for name, (l, t, r, b) in crops.items():
            crop = image[int(t*h):int(b*h), int(l*w):int(r*w)]
            cv2.imwrite(str(output / f'{path.stem}-{name}.png'),
                        cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
        thumb = cv2.resize(image, (710, 454))
        cv2.putText(thumb, path.name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        thumbs.append(thumb)
    for i in range(0, len(thumbs), 6):
        batch = thumbs[i:i+6] + [np.zeros_like(thumbs[0])] * (6-len(thumbs[i:i+6]))
        sheet = np.vstack([np.hstack(batch[j:j+2]) for j in range(0, 6, 2)])
        cv2.imwrite(str(output / f'sheet-{i//6:02d}.jpg'), sheet)
    write_json(output / 'manifest.json', dict(decoder='CoreGraphics load_image', predictions_used=False,
                                             files=entries, generated_at=time.time()))


def prepare(args):
    review_images(args.session, selected(args), args.output)
    return 0


def label_value(label):
    if not isinstance(label, dict) or label.get('status') not in ('known', 'unknown') or not label.get('evidence'):
        raise ValueError('human labels need known/unknown status and pixel evidence/reason')
    if label['status'] == 'unknown':
        if label.get('value') is not None:
            raise ValueError('unknown label cannot contain a value')
        return None
    if label.get('value') is None:
        raise ValueError('known label missing value')
    return label['value']


def validate_truth(truth, entries):
    if not isinstance(truth, dict) or truth.get('schema') != 'vision-session-human-v1' or \
            truth.get('provenance') != 'human_pixel_review' or not truth.get('reviewer'):
        raise ValueError('only explicit human pixel truth accepted; machine states are not truth')
    rows = truth['frames']
    if [r.get('_file') for r in rows] != [r['file'] for r in entries]:
        raise ValueError('truth must cover the entire fixed selection, in order')
    for row, entry in zip(rows, entries):
        if row.get('sha256') != entry['sha256']:
            raise ValueError('truth hash mismatch')
        for field in ('phase', 'stage', 'gold', 'level'):
            value = label_value(row[field])
            if value is not None:
                expected_type = int if field in ('gold', 'level') else str
                if type(value) is not expected_type:
                    raise ValueError('human field type mismatch')
        expected = any(reason.startswith('opponent_') for reason in entry['reasons'])
        if row.get('opponent_audit') is not expected:
            raise ValueError('opponent audit selection changed')
        if expected:
            enemy = row['enemy']
            count = label_value(enemy)
            if count is not None and (type(count) is not int or count != len(enemy['positions'])):
                raise ValueError('enemy count and explicit positions disagree')
            for category in ('positions', 'friendly_positions', 'uncertain_positions'):
                if len({tuple(p) for p in enemy[category]}) != len(enemy[category]):
                    raise ValueError('duplicate human positions')
                for p in enemy[category]:
                    if len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p):
                        raise ValueError('invalid normalized human position')
            if not enemy.get('exclusions'):
                raise ValueError('enemy audit needs friendly/legend/decoration/occlusion review')
    return rows


def matching(predicted, actual):
    """Maximum one-to-one cardinality, not greedy proximity or reusable truth."""
    edges = [[j for j, q in enumerate(actual) if abs(p[0]-q[0]) < DX and abs(p[1]-q[1]) < DY]
             for p in predicted]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predicted)):
        augment(i, set())
    return sorted((i, j) for j, i in owner.items())


def ownership_checks(units, enemies, friends):
    """Use the established scorer; keep proximity evidence separate from errors.

    Enemy recall is one-to-one. Ownership retains eval_opponent.score's per
    prediction nearest eligible friend < nearest eligible enemy, with ties
    excluded. A friend window overlap alone is only a review lead.
    """
    _, false = score_opponent(dict(board=units), dict(
        board_count=len(enemies),  # unused here: only the ownership result is taken
        enemy_positions=[dict(x=x, y=y) for x, y in enemies],
        friendly_positions=[dict(x=x, y=y) for x, y in friends]))
    false_ids = {id(unit) for unit in false}

    def distances(unit, positions):
        result = []
        for i, (x, y) in enumerate(positions):
            dx, dy = abs(unit['x']-x), abs(unit['y']-y)
            result.append(dict(index=i, position=[x, y], dx=dx, dy=dy,
                               distance=dx+dy, in_window=dx < DX and dy < DY))
        return result

    def nearest(points):
        return min((p for p in points if p['in_window']), key=lambda p: p['distance'], default=None)

    checks = []
    for i, unit in enumerate(units):
        enemy_points, friend_points = distances(unit, enemies), distances(unit, friends)
        nearest_friend = nearest(friend_points)
        checks.append(dict(index=i, prediction=[unit['x'], unit['y']],
                           enemy_distances=enemy_points, friend_distances=friend_points,
                           nearest_enemy=nearest(enemy_points), nearest_friend=nearest_friend,
                           friendly_as_enemy=id(unit) in false_ids,
                           friend_window_overlap=nearest_friend is not None))
    return checks


def score(rows, predictions):
    counters = {k: Counter() for k in ('phase', 'stage', 'gold', 'level', 'enemy_count')}
    details = []
    recall = true_units = unmatched = friendly = overlap = detected = unknown_enemy = 0
    for row in rows:
        pred = predictions[row['_file']]['prediction']
        result = dict(_file=row['_file'], fields={})
        for field in ('phase', 'stage', 'gold', 'level'):
            value = label_value(row[field])
            status = ('unknown' if value is None else 'correct'
                      if type(pred.get(field)) is type(value) and pred.get(field) == value else 'wrong')
            counters[field][status] += 1
            result['fields'][field] = dict(status=status, truth=value, prediction=pred.get(field))
        if row['opponent_audit']:
            enemy = row['enemy']
            value = label_value(enemy)
            points = [[u['x'], u['y']] for u in board(pred)]
            pairs = matching(points, enemy['positions'])
            ownership = ownership_checks(board(pred), enemy['positions'], enemy['friendly_positions'])
            false_indices = [check['index'] for check in ownership if check['friendly_as_enemy']]
            overlap_indices = [check['index'] for check in ownership if check['friend_window_overlap']]
            uncertain_pairs = matching(points, enemy['uncertain_positions'])
            status = 'unknown' if value is None else 'correct' if len(points) == value else 'wrong'
            counters['enemy_count'][status] += 1
            recall += len(pairs); true_units += len(enemy['positions'])
            unmatched += len(points)-len(pairs); friendly += len(false_indices); detected += len(points)
            overlap += len(overlap_indices)
            unknown_enemy += len(enemy['uncertain_positions'])
            result['enemy'] = dict(status=status, truth=value, predicted=len(points), matched=len(pairs),
                                   pairs=pairs, unmatched_indices=sorted(set(range(len(points)))-{i for i, _ in pairs}),
                                   friendly_as_enemy=len(false_indices), friendly_as_enemy_indices=false_indices,
                                   friend_window_overlap=len(overlap_indices),
                                   friend_window_overlap_indices=overlap_indices, ownership_checks=ownership,
                                   uncertain_matches=uncertain_pairs)
        details.append(result)
    metrics = {}
    for field, counter in counters.items():
        metrics[field] = dict(correct=counter['correct'], decidable=counter['correct']+counter['wrong'],
                              wrong=counter['wrong'], unknown=counter['unknown'], total=sum(counter.values()))
    quantity = metrics['enemy_count']
    # All fixed samples stay in the denominator: unknowns never help the gate.
    gate = quantity['total'] > 0 and quantity['correct']*5 >= quantity['total']*4 and friendly == 0
    metrics.update(enemy_recalled=recall, true_enemy_units=true_units, predicted_enemy_units=detected,
                   enemy_recall=recall/true_units if true_units else None, unmatched_predictions=unmatched,
                   friendly_as_enemy=friendly, friend_window_overlap=overlap, unknown_enemy_positions=unknown_enemy,
                   quantity_ownership_gate=gate, phase_all_correct=metrics['phase']['correct'] == len(rows))
    return metrics, details


def time_series(rows):
    """Review leads from every prediction, not automatically inferred truth."""
    transitions, spans = [], []
    active = {}
    last_stage = last_time = None
    for row in rows:
        pred = row['prediction']
        name = row['_file']
        for field in ('phase', 'stage'):
            value = pred.get(field)
            if not transitions or field not in active or active[field]['value'] != value:
                if field in active:
                    spans.append(active[field])
                active[field] = dict(field=field, value=value, first=name, last=name, frames=0)
                transitions.append(dict(_file=name, field=field, value=value))
            active[field]['last'] = name
            active[field]['frames'] += 1
        stage = pred.get('stage')
        if stage and re.fullmatch(r'\d+-\d+', stage):
            key = tuple(map(int, stage.split('-')))
            if last_stage is not None and key < last_stage:
                transitions.append(dict(_file=name, anomaly='stage_backward', old=last_stage, new=key))
            last_stage = key
        timestamp = row.get('source_t')
        if timestamp is not None and last_time is not None and timestamp <= last_time:
            transitions.append(dict(_file=name, anomaly='source_time_not_increasing', old=last_time, new=timestamp))
        last_time = timestamp
    spans.extend(active.values())
    stale_spans = []
    for row in rows:
        if row['prediction'].get('units_stale'):
            if not stale_spans or stale_spans[-1]['end_index'] != row['index']-1:
                stale_spans.append(dict(first=row['_file'], last=row['_file'], frames=0, end_index=row['index']))
            stale_spans[-1].update(last=row['_file'], frames=stale_spans[-1]['frames']+1, end_index=row['index'])
    return dict(label='machine time-series review leads, NOT human errors', transitions=transitions,
                spans=spans, units_stale_spans=stale_spans,
                missing_stage_frames=[r['_file'] for r in rows if not r['prediction'].get('stage')],
                noncombat_opponent=[r['_file'] for r in rows if r['prediction'].get('phase') != 'combat'
                                    and r['prediction'].get('opponent') is not None])


def overlay(session, truth_rows, predictions, output):
    import cv2
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye import opponent
    output.mkdir()
    diagnostics = []
    for row in truth_rows:
        image = load_image(session / 'frames' / row['_file'])
        h, w = image.shape[:2]
        pred = predictions[row['_file']]['prediction']
        categories = []
        if row['opponent_audit']:
            categories.extend([(row['enemy']['positions'], (0, 255, 255), 'T'),
                               (row['enemy']['friendly_positions'], (0, 255, 0), 'F'),
                               (row['enemy']['uncertain_positions'], (255, 0, 255), 'U')])
            # Read-only diagnostics on the same pixels, AFTER human truth is
            # sealed. This is not a second GameReader or a replacement prediction.
            bars, _ = opponent.enemy_bars(image)
            seeds = []
            opponent.recover_enemy_bars(image, audit=seeds)
            points = [[round(b.cx, 3), round(b.cy, 3)] for b in bars]
            actual = [[u['x'], u['y']] for u in board(predictions[row['_file']]['prediction'])]
            diagnostics.append(dict(_file=row['_file'], sha256=row['sha256'], seeds=seeds,
                                    product_bars=[vars(b) for b in bars],
                                    diagnostic_positions_equal_run=points == actual))
        categories.append(([[u['x'], u['y']] for u in board(predictions[row['_file']]['prediction'])],
                           (0, 0, 255), 'P'))
        for points, color, prefix in categories:
            for i, (x, y) in enumerate(points):
                p = (round(x*w), round(y*h))
                cv2.circle(image, p, 12, color, 2)
                cv2.putText(image, f'{prefix}{i}', (p[0]+13, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        text = (f"{row['_file']} phase T={row['phase']['value']} P={pred.get('phase')}  "
                f"stage T={row['stage']['value']} P={pred.get('stage')}")
        cv2.rectangle(image, (0, h-27), (min(w, 1200), h), (0, 0, 0), -1)
        cv2.putText(image, text, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.imwrite(str(output / row['_file']), image)
    write_json(output.parent / 'bar-diagnostics.json', dict(opponent_sha256=sha(Path(opponent.__file__)),
                                                         frames=diagnostics))


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    files = frame_files(args.session)
    predictions = read_rows(args.predictions)
    align(files, predictions, predictions=True)
    entries = selected(args)
    truth = json.loads(args.truth.read_text())
    if truth.get('selection_sha256') != sha(args.selection):
        raise ValueError('fixed selection fingerprint mismatch')
    rows = validate_truth(truth, entries)
    by_file = {r['_file']: r for r in predictions}
    if any(by_file[r['file']]['sha256'] != r['sha256'] for r in entries):
        raise ValueError('selection does not match actual frame pixels')
    metrics, details = score(rows, by_file)
    write_json(args.output / 'time-series.json', time_series(predictions))
    noncombat = [r['_file'] for r in predictions if r['prediction'].get('phase') != 'combat'
                 and r['prediction'].get('opponent') is not None]
    spectate = [r['_file'] for r in predictions if r['prediction'].get('phase') == 'spectate']
    # New spectate frames require extra human labels rather than automatic acceptance.
    reviewed = {r['_file'] for r in rows}
    unreviewed_spectate = sorted(set(spectate)-reviewed)
    passed = (metrics['quantity_ownership_gate'] and metrics['phase_all_correct']
              and not noncombat and not unreviewed_spectate)
    overlay(args.session, rows, by_file, args.output / 'overlays')
    write_json(args.output / 'details.json', details)
    summary = dict(quality_passed=passed, metrics=metrics, sample_count=len(rows),
                   naming_coverage=name_coverage(predictions),
                   opponent_sample_count=sum(r['opponent_audit'] for r in rows),
                   noncombat_opponent=noncombat, spectate_frames=spectate,
                   unreviewed_spectate=unreviewed_spectate, truth_sha256=sha(args.truth),
                   predictions_sha256=sha(args.predictions), selection_sha256=sha(args.selection),
                   script_sha256=sha(Path(__file__)),
                   ownership_metric='eval_opponent.score: nearest in-window friend dx+dy < nearest in-window enemy dx+dy; ties excluded',
                   overlap_metric='friend_window_overlap: prediction inside any friend window; review lead only, not part of the quality gate',
                   accuracy_source='human CG pixel review only; machine logs excluded')
    write_json(args.output / 'summary.json', summary)
    print(line(summary))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'prepare', 'audit'):
        p = sub.add_parser(name)
        p.add_argument('--session', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True, help='new, nonexisting evidence directory')
        if name == 'run':
            p.add_argument('--db', type=Path, required=True, help='frozen input, copied before opening')
            p.add_argument('--gallery', type=Path, required=True, help='frozen input, copied before reading')
            p.add_argument('--ocr', choices=('rapid', 'vision'), default='rapid')
        else:
            p.add_argument('--selection', type=Path, default=SELECTION, help='precommitted sampling evidence')
        if name == 'audit':
            p.add_argument('--predictions', type=Path, required=True)
            p.add_argument('--truth', type=Path, required=True, help='human_pixel_review JSON, never states.jsonl')
        p.set_defaults(action=globals()[name])
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        return args.action(args)
    except Exception:
        error = traceback.format_exc()
        if not existed and args.output.is_dir():
            (args.output / 'fatal-error.txt').write_text(error)
        print(error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
