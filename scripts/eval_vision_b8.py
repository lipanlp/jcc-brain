"""B8 可复跑的串行真实回放、严格基线比较与 A→B→A 隔离验收。

suite 每组 repeat3，保留命令、日志、评分和全部三遍完整输出。
基线源码可为 git archive 快照；子进程从 --repo 导入 src/scripts。
所有数据库和图库均由既有评测器复制后打开；输入本身仅校验哈希。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'var/vision-module-optimization/batch-B8'
REPLAY = ROOT / 'var/vision-module-optimization/replay-samples'
MISSING = {'__missing__': True}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def manifest(path):
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file()}


def differences(old, new, path=()):
    if encoded(old) == encoded(new):
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            yield from differences(old.get(key, MISSING), new.get(key, MISSING), path + (key,))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            yield from differences(a, b, path + (i,))
    else:
        yield path, old, new


def compare_frames(before, after):
    """只应用 TASK 的固定豁免；返回逐帧逐字段旧新值，越界项 rule=None。"""
    import re
    changes = []
    if len(before) != len(after):
        changes.append(dict(frame=None, field='frame_count', old=len(before), new=len(after), rule=None))
    seen_spectate = False
    previous_phase = None
    previous_rep = None
    for index, (a, b) in enumerate(zip(before, after)):
        rep = a.get('rep', 0)
        if rep != previous_rep:
            seen_spectate, previous_phase = False, None
        previous_rep = rep
        old, new = a.get('prediction', a), b.get('prediction', b)
        frame = a.get('_file')
        for key in ('_file', 'rep'):
            if encoded(a.get(key, MISSING)) != encoded(b.get(key, MISSING)):
                changes.append(dict(frame=frame, index=index, rep=rep, field=key,
                                    old=a.get(key, MISSING), new=b.get(key, MISSING), rule=None))
        phase = old.get('phase')
        same_phase = encoded(old.get('phase', MISSING)) == encoded(new.get('phase', MISSING))
        spectate = same_phase and phase == 'spectate'
        following_scene = (same_phase and previous_phase == 'spectate'
                           and phase in ('carousel', 'dialog', 'reconnect'))
        for path, x, y in differences(old, new):
            root = path[0] if path else None
            rule = None
            if spectate and root in ('board', 'bench', 'units_stale') and new.get('units_stale') is False:
                rule = 'spectate_current_units'
            elif spectate and path == ('auto_labeled',) and y == MISSING:
                rule = 'remove_foreign_auto_labeled'
            elif (following_scene and root in ('board', 'bench', 'units_stale')
                  and not new.get('board') and not new.get('bench') and not new.get('units_stale')):
                rule = 'clear_immediately_following_non_own_scene'
            elif ((seen_spectate or spectate) and path and path[-1] == 'name'
                  and ('items' in path or root == 'item_bench')
                  and isinstance(x, str) and isinstance(y, str)
                  and re.fullmatch(r'\?i\d+', x) and re.fullmatch(r'\?i\d+', y)):
                rule = 'unknown_item_number_after_isolation'
            changes.append(dict(frame=frame, index=index, rep=rep, field='.'.join(map(str, path)),
                                old=x, new=y, rule=rule))
        # Metadata besides the prediction is also fixed, excluding measurements.
        for key in sorted((set(a) | set(b)) - {'prediction', '_file', 'rep', 'ms'}):
            if 'prediction' in a and encoded(a.get(key, MISSING)) != encoded(b.get(key, MISSING)):
                changes.append(dict(frame=frame, index=index, rep=rep, field=key,
                                    old=a.get(key, MISSING), new=b.get(key, MISSING), rule=None))
        seen_spectate |= spectate
        previous_phase = phase
    return changes


def valid_opponent(value):
    if not isinstance(value, dict) or set(value) != {'name', 'board', 'mirrored'}:
        return False
    if value['mirrored'] is not True or not isinstance(value['board'], list):
        return False
    if value['name'] is not None and not isinstance(value['name'], str):
        return False
    return all(isinstance(unit, dict) and set(unit) <= {'name', 'star', 'x', 'y'}
               and all(isinstance(unit.get(axis), (float, int)) and not isinstance(unit[axis], bool)
                       and 0 <= unit[axis] <= 1 for axis in ('x', 'y')) for unit in value['board'])


def opponent_update_gate(report, audit):
    """The switch cannot bypass repeat3 quantity, ownership or pixel review."""
    counts = report.get('counts') or {}
    correct, total = counts.get('board_count', [0, 0])
    false, _ = counts.get('friendly_as_enemy', [1, 0])
    metrics_pass = total == 36 and correct >= 30 and false == 0
    audit_pass = (audit.get('frame_count') == 53 and audit.get('removed_count') == 0
                  and audit.get('reviewed_count') == audit.get('added_count')
                  and audit.get('unreviewed_count') == 0 and audit.get('false_new_count') == 0
                  and audit.get('uses_product_enemy_bars') is True)
    errors = []
    if report.get('enabled') and not (metrics_pass and audit_pass):
        errors.append('enabled_without_quantity_ownership_and_pixel_audit_gate')
    for row in report.get('frames') or []:
        expected = row.get('prediction') if report.get('enabled') else None
        if encoded(row.get('production')) != encoded(expected):
            errors.append(f"production_candidate_mismatch:{row.get('rep')}:{row.get('_file')}")
    return dict(enabled=report.get('enabled') is True, metrics_pass=metrics_pass,
                audit_pass=audit_pass, passed=metrics_pass and audit_pass, errors=errors)


def compare_opponent_frames(before, after, *, group, enabled, gate_passed):
    """S3b is stricter than S2: only the explicit opponent surface may change."""
    changes = []
    if len(before) != len(after):
        changes.append(dict(frame=None, field='frame_count', old=len(before), new=len(after), rule=None))
    for index, (a, b) in enumerate(zip(before, after)):
        old, new = a['prediction'], b['prediction']
        frame, rep = a.get('_file'), a.get('rep')
        special = group == 'opponent'
        if not special and enabled and new.get('phase') == 'combat' and not valid_opponent(new.get('opponent')):
            changes.append(dict(frame=frame, index=index, rep=rep, field='opponent',
                                old=old.get('opponent'), new=new.get('opponent'), rule=None))
        for path, x, y in differences(old, new):
            rule = None
            if special and valid_opponent(new):
                rule = 'explicit_opponent_candidate_update'
            elif (path and path[0] == 'opponent' and enabled and gate_passed
                  and old.get('phase') == new.get('phase') == 'combat'
                  and old.get('opponent', MISSING) is None and valid_opponent(new.get('opponent'))):
                rule = 'enabled_combat_opponent_after_gate'
            changes.append(dict(frame=frame, index=index, rep=rep, field='.'.join(map(str, path)),
                                old=x, new=y, rule=rule))
        for key in sorted((set(a) | set(b)) - {'prediction', 'ms'}):
            x, y = a.get(key, MISSING), b.get(key, MISSING)
            if encoded(x) != encoded(y):
                rule = ('derived_opponent_'+key if special and key in ('false_friendly', 'production') else None)
                changes.append(dict(frame=frame, index=index, rep=rep, field=key, old=x, new=y, rule=rule))
    return changes


def validate_opponent_report(report, truth):
    """Recompute the existing scorer, not its metric definition, from raw rows."""
    sys.path.insert(0, str(ROOT/'scripts'))
    from eval_opponent import score
    errors = []
    rows = report.get('frames') or []
    keys = [(row['rep'], row['_file']) for row in rows]
    expected_keys = {(rep, file) for rep in range(3) for file in truth}
    if len(keys) != 36 or set(keys) != expected_keys:
        errors.append('opponent_repeat3_frame_identity')
    counts = {'board_count': [0, 0], 'friendly_as_enemy': [0, 0]}
    for row in rows:
        if row['_file'] not in truth or not valid_opponent(row['prediction']):
            errors.append('invalid_opponent_prediction')
            continue
        wanted = truth[row['_file']]
        scored, false = score(row['prediction'], wanted)
        if encoded(false) != encoded(row.get('false_friendly')) or row.get('truth_count') != wanted['board_count']:
            errors.append('opponent_derived_metadata')
        for key in counts:
            for i in (0, 1):
                counts[key][i] += scored[key][i]
    if encoded(counts) != encoded(report.get('counts')):
        errors.append('opponent_counts_disagree_with_raw_frames')
    return errors


# Run the selected repository's existing evaluator, retaining *all* repeats (its
# legacy --dump overwrites each repeat). ROOT is input provenance, never code.
SUITE_WORKER = r'''
import json, sys
from pathlib import Path
repo, inputs, samples, frames = map(Path, sys.argv[1:5])
sys.path[:0] = [str(repo/'src'), str(repo/'scripts')]
import eval_vision as ev
from jccbrain.vision.eye import reader
affected = Path(reader.__file__).resolve()
assert affected.is_relative_to((repo/'src').resolve()), affected
print('reader_import=' + str(affected), flush=True)
ev.ROOT = inputs
ev.DEFAULT_SAMPLES = samples
original = ev.run
all_frames = []
def retain(files, args):
    preds, times = original(files, args)
    rep = len(all_frames) // len(files)
    all_frames.extend(dict(rep=rep, _file=f.name, prediction=preds[f.name]) for f in files)
    return preds, times
ev.run = retain
sys.argv = ['eval_vision.py'] + sys.argv[5:]
code = ev.main()
if all_frames:
    frames.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in all_frames))
else:
    # scout/opponent already preserve their repeat3 dumps.
    dump = Path(sys.argv[sys.argv.index('--dump')+1])
    frames.write_bytes(dump.read_bytes())
raise SystemExit(code)
'''


def cases(args):
    sample, replay = args.samples, args.replay
    return [
        ('standard-items-phase', sample, ['--items', '--phase']),
        ('track-items', sample/'track', ['--items']),
        ('units-standard', sample, ['--units']),
        ('units-track', sample/'track', ['--units']),
        ('units-gallery-standard', sample, ['--units', '--gallery', str(args.gallery)]),
        ('units-gallery-track', sample/'track', ['--units', '--gallery', str(args.gallery)]),
        ('augments-old', sample, ['--augments']),
        ('augments-sequence', args.augment_samples, ['--augments', '--augment-truth', str(args.augment_truth)]),
        ('digits', replay/'digits', ['--digits']),
        *[(f'phase-{name}', replay/name, ['--phase']) for name in ('carousel', 'spectate', 'dialog')],
        ('scout', sample, ['--scout', '--gallery', str(args.gallery)]),
        ('opponent', sample, ['--opponent', '--gallery', str(args.gallery)]),
    ]


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def spectate_coverage(rows):
    truth = {Path(row['_file']).name: row for row in load_rows(ROOT/'tests/fixtures/scout_truth.jsonl')
             if '/spectate/' in row['_file'] and row.get('board_count') is not None}
    observations = []
    for row in rows:
        expected = truth.get(row['_file'])
        pred = row['prediction']
        if expected is None or pred.get('phase') != 'spectate':
            continue
        board = pred.get('board') or []
        named = sum(bool(unit.get('name')) and not unit['name'].startswith('?') for unit in board)
        observations.append(dict(rep=row['rep'], _file=row['_file'], expected=expected['board_count'],
                                 detected=len(board), named_predictions=named,
                                 name_truth_count=len(expected.get('board_names') or [])))
    return dict(board_count=[sum(row['expected'] == row['detected'] for row in observations), len(observations)],
                named_predictions=sum(row['named_predictions'] for row in observations),
                detected=sum(row['detected'] for row in observations),
                name_truth_count=sum(row['name_truth_count'] for row in observations),
                name_accuracy=None, frames=observations)


def classify_processes(raw, own_pid=None, own_parent=None):
    """Classify saved or current ps output without changing any process."""
    lines = raw.splitlines()
    parents = {int(line.split(None, 2)[0]): int(line.split(None, 2)[1]) for line in lines}
    ancestors = {os.getpid() if own_pid is None else own_pid}
    parent = os.getppid() if own_parent is None else own_parent
    while parent and parent not in ancestors:
        ancestors.add(parent)
        parent = parents.get(parent, 0)
    suspects = [line for line in lines if any(word in line.lower() for word in
                ('jccbrain', '金铲铲大脑.app/contents/macos', 'pytest', 'pyinstaller',
                 'eval_vision', 'eval_scout', 'eval_opponent', 'record_assistant.py',
                 'bench_autoroll', 'scripts/build_app.sh', 'capture_switch', 'click_latency',
                 'jkchess', 'playcover.app/contents/macos/playcover'))
                and int(line.split()[0]) not in ancestors]
    return dict(raw=raw, suspects=suspects,
                timing_status='未测（环境不净）' if suspects else 'measured')


def process_snapshot():
    # Observe only; never stop another workload. Keep raw ps evidence even if dirty.
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,%cpu=,command='], capture_output=True, text=True, check=True)
    return classify_processes(result.stdout)


def suite(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    inputs = dict(db=manifest(args.db), gallery=manifest(args.gallery))
    write_json(output/'inputs-before.json', inputs)
    report = {'repo': str(args.repo), 'groups': [], 'changes': [], 'violations': []}
    for name, samples, flags in cases(args):
        snapshot = process_snapshot()
        write_json(output/f'{name}-processes.json', snapshot)
        cmd = [sys.executable, '-c', SUITE_WORKER, str(args.repo), str(ROOT), str(args.samples),
               str(output/f'{name}-frames.jsonl'), '--samples', str(samples), '--db', str(args.db),
               '--repeat', '3', '--json', str(output/f'{name}.json'), '--dump', str(output/f'{name}.jsonl'), *flags]
        env = dict(os.environ, PYTHONPATH=str(args.repo/'src'), JCC_HOME=str(args.repo/'var'))
        with (output/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        group = dict(name=name, returncode=result.returncode, command=cmd, timing_status=snapshot['timing_status'])
        report['groups'].append(group)
        print(f'{name}: exit={result.returncode}', flush=True)
        if result.returncode:
            report['violations'].append(dict(group=name, error='subprocess_failed'))
        else:
            metrics = json.loads((output/f'{name}.json').read_text())
            group['metrics'] = {key: ({k: v for k, v in value.items() if k != 'frames'}
                                      if isinstance(value, dict) else value)
                                for key, value in metrics.items()
                                if key in ('total', 'items', 'phase', 'digits', 'augments_taken', 'scout', 'opponent')}
            if 'units_name' in metrics:
                group['units_name'] = {key: value for key, value in metrics['units_name'].items()
                                       if key in ('counts', 'accuracy', 'coverage')}
            if name == 'phase-spectate':
                group['current_units'] = spectate_coverage(load_rows(output/f'{name}-frames.jsonl'))
        if not result.returncode and args.baseline and not args.allow_opponent_update:
            changes = compare_frames(load_rows(args.baseline/f'{name}-frames.jsonl'),
                                     load_rows(output/f'{name}-frames.jsonl'))
            report['changes'].extend(dict(group=name, **row) for row in changes)
            report['violations'].extend(dict(group=name, **row) for row in changes if row['rule'] is None)
        write_json(output/'summary.json', report)
    if args.allow_opponent_update:
        opponent_report = json.loads((output/'opponent.json').read_text())['opponent']
        audit = json.loads(args.opponent_audit.read_text()) if args.opponent_audit.exists() else {}
        gate = opponent_update_gate(opponent_report, audit)
        truth = {row['_file']: row for row in load_rows(ROOT/'tests/fixtures/opponent_truth.jsonl')}
        gate['errors'].extend(validate_opponent_report(opponent_report, truth))
        if gate['enabled'] and audit.get('product_sha256') != manifest(args.repo/'src/jccbrain/vision/eye/opponent.py'):
            gate['errors'].append('opponent_audit_source_changed')
        report['opponent_gate'] = gate
        report['violations'].extend(dict(group='opponent', error=error) for error in gate['errors'])
        if args.baseline:
            for group in report['groups']:
                name = group['name']
                if group['returncode']:
                    continue
                changes = compare_opponent_frames(load_rows(args.baseline/f'{name}-frames.jsonl'),
                          load_rows(output/f'{name}-frames.jsonl'), group=name, enabled=gate['enabled'],
                          gate_passed=gate['passed'] and not gate['errors'])
                report['changes'].extend(dict(group=name, **row) for row in changes)
                report['violations'].extend(dict(group=name, **row) for row in changes if row['rule'] is None)
    current = dict(db=manifest(args.db), gallery=manifest(args.gallery))
    write_json(output/'inputs-after.json', current)
    report['inputs_unchanged'] = inputs == current
    if inputs != current:
        report['violations'].append(dict(error='input_manifest_changed'))
    write_json(output/'summary.json', report)
    return int(bool(report['violations']))


def state_hashes(reader):
    """All recognition state, including pre-existing auxiliary readers; timing omitted."""
    names = ('cache', 'cell_cache', '_frame_id', 'augment_desc_cache', 'phase', 'last_stage',
             'last_timer', 'roster', 'last_players')
    states = {name: getattr(reader, name) for name in names}
    states['round_bar'] = reader.round_bar.__dict__
    states['units'] = {k: v for k, v in reader.units.__dict__.items() if k not in ('gallery', 'items')}
    states['gallery'] = {k: v for k, v in reader.units.gallery.__dict__.items() if k != 'root'}
    states['items'] = {k: v for k, v in reader.items.__dict__.items() if k != 'last_ms'} if reader.items else None
    states['augments'] = reader.augments_taken.__dict__ if reader.augments_taken else None
    states['ocr'] = {name: getattr(reader.ocr, name, None) for name in ('cache', 'words')}
    for name in ('_scout_reader', '_opponent_reader'):
        auxiliary = getattr(reader, name, None)
        if auxiliary:
            target = getattr(auxiliary, 'reader', auxiliary)
            states[name] = (state_hashes(target) if hasattr(target, 'round_bar') else
                            dict(memory={k: v for k, v in target.__dict__.items()
                                         if k not in ('gallery', 'ocr', 'last_ms')},
                                 ocr_cache=getattr(target.ocr, 'cache', None),
                                 ocr_words=getattr(target.ocr, 'words', None)))
    return {name: hashlib.sha256(pickle.dumps(value)).hexdigest() for name, value in states.items()}


def sequence_worker(args):
    sys.path[:0] = [str(args.repo/'src'), str(args.repo/'scripts')]
    from eval_vision import build
    from eval_units import copied_gallery
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.reader import GameReader
    from jccbrain.vision.eye.opponent import OpponentReader
    assert Path(sys.modules[GameReader.__module__].__file__).resolve().is_relative_to(args.repo/'src')
    own = load_image(args.samples/'0027.jpg')
    foreign = [load_image(args.replay/'spectate'/f'{i:04d}.png') for i in (1, 2, 6)]
    inputs = dict(db=manifest(args.db), gallery=manifest(args.gallery))
    runs = []
    for inserted in (False, True):
        with copied_gallery(args.gallery) as gallery:
            reader = build(args.db, gallery, 'rapid')
            try:
                first = reader.read(own)
                reader.read_scout(own)  # pre-existing child reader must also remain frozen
                reader._opponent_reader = OpponentReader(reader.units.gallery, reader.ocr)
                frozen = state_hashes(reader)
                gallery_before = manifest(gallery)
                observations = []
                if inserted:
                    for i, image in zip((1, 2, 6), foreign):
                        prediction = reader.read(image)
                        hashes = state_hashes(reader)
                        gallery_hashes = manifest(gallery)
                        observations.append(dict(_file=f'spectate/{i:04d}.png', prediction=prediction,
                                                 hashes=hashes, frozen=hashes == frozen,
                                                 gallery_hashes=gallery_hashes,
                                                 gallery_unchanged=gallery_hashes == gallery_before))
                returned = reader.read(own)
                runs.append(dict(first=first, returned=returned, before_hashes=frozen,
                                 foreign=observations, gallery_before=gallery_before,
                                 gallery_after_foreign=observations[-1]['gallery_hashes'] if observations else gallery_before))
            finally:
                reader.ocr.pool.shutdown(wait=True)
    checks = dict(first_equal=runs[0]['first'] == runs[1]['first'],
                  returned_equal=runs[0]['returned'] == runs[1]['returned'],
                  frozen=all(row['frozen'] and row['gallery_unchanged'] for row in runs[1]['foreign']),
                  current=all(row['prediction'].get('phase') == 'spectate'
                              and row['prediction'].get('units_stale') is False
                              and len(row['prediction'].get('board', [])) == count
                              for row, count in zip(runs[1]['foreign'], (6, 8, 4))),
                  inputs_unchanged=inputs == dict(db=manifest(args.db), gallery=manifest(args.gallery)))
    write_json(args.output/'sequence.json', dict(repo=str(args.repo), runs=runs, checks=checks,
                                               inputs_before=inputs,
                                               inputs_after=dict(db=manifest(args.db), gallery=manifest(args.gallery))))
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return int(not all(checks.values()))


def sequence(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '_sequence_worker',
               '--repo', str(args.repo), '--output', str(args.output), '--samples', str(args.samples),
               '--replay', str(args.replay), '--db', str(args.db), '--gallery', str(args.gallery)]
    write_json(args.output/'command.json', command)
    with (args.output/'sequence.log').open('w') as log:
        result = subprocess.run(command, cwd=args.repo, env=dict(os.environ, PYTHONPATH=str(args.repo/'src'),
                                JCC_HOME=str(args.repo/'var')), stdout=log, stderr=subprocess.STDOUT)
    print((args.output/'sequence.log').read_text(), end='')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('suite', 'sequence', '_sequence_worker'):
        sub = commands.add_parser(name, help={'suite': '14组串行repeat3及可选严格基线比较',
                                             'sequence': '真实0027→观战0001/2/6→0027与0027→0027对照',
                                             '_sequence_worker': argparse.SUPPRESS}[name])
        sub.add_argument('--repo', type=Path, required=True, help='源码仓库/固定git archive快照，优先导入其src')
        sub.add_argument('--output', type=Path, required=True, help='新建证据目录；已存在时报错，防止抹掉失败轮次')
        sub.add_argument('--samples', type=Path, default=ROOT/'game-eye/samples', help='原标准样本，只读')
        sub.add_argument('--replay', type=Path, default=REPLAY, help='本工作树回放副本根目录')
        sub.add_argument('--db', type=Path, default=EVIDENCE/'data/jcc-vf-db.sqlite3', help='冻结数据库，评测内部复制后打开')
        sub.add_argument('--gallery', type=Path, default=EVIDENCE/'data/jcc-vf-gallery', help='冻结图库，评测内部复制后打开')
        if name == 'suite':
            sub.add_argument('--baseline', type=Path, help='另一suite证据目录；任何越界输出差异exit1')
            sub.add_argument('--allow-opponent-update', action='store_true',
                             help='显式S3b规则：只准专项candidate及派生字段、过门槛后combat.opponent变化；其他字段严格相同')
            sub.add_argument('--opponent-audit', type=Path, default=EVIDENCE/'S3b-analysis/verified/audit.json',
                             help='启用所需53帧逐新增像素审计，默认本工作树S3b-analysis/verified/audit.json')
            sub.add_argument('--augment-samples', type=Path, default=ROOT/'var/vision-module-optimization/batch-B4/baseline/augment-sequence')
            sub.add_argument('--augment-truth', type=Path, default=ROOT/'var/vision-module-optimization/batch-B4/baseline/sequence-truth.jsonl')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return {'suite': suite, 'sequence': sequence, '_sequence_worker': sequence_worker}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
