"""装备像素专项：54帧候选格的有无、名字精度/覆盖率、角标；不把战斗阵容记忆当当前像素。

    .venv/bin/python scripts/eval_items.py --repeat 3 --json /tmp/items.json

使用与 GameReader 相同的装备识别器和血条定位，逐帧评测当前可见图标；
整条 GameReader 的阶段/阵容记忆另由 eval_vision.py 和端到端测试验证。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def evaluate(samples: Path, truth_file: Path, repeat: int = 1, catalog=None) -> dict:
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.items import ItemCatalog, ItemReader
    from jccbrain.vision.eye.units import Gallery, UnitTracker

    truth = [json.loads(line) for line in truth_file.read_text('utf-8').splitlines()]
    counts = dict(tp=0, fp=0, fn=0, tn=0, named=0, correct_names=0, named_present=0, present=0,
                  badge_ok=0, badge_total=0)
    timing, diffs, predictions = [], [], []
    catalog = catalog or ItemCatalog.from_db()
    with tempfile.TemporaryDirectory(prefix='jcc-items-eval-') as temporary:
        tracker = UnitTracker(Gallery(Path(temporary)))
        for _ in range(repeat):
            reader = ItemReader(catalog)
            predictions = []
            for frame in truth:
                image = load_image(samples / frame['_file'])
                bench = reader.bench(image)
                units = tracker.detect(image)[1]
                observed = [(u, reader.on_unit(image, u.bar)) for u in units]
                timing.append(reader.last_ms)
                predictions.append({'_file': frame['_file'], 'item_bench': bench, 'units': [
                    {'zone': u.zone, 'slot': u.slot, 'x': u.x, 'y': u.y, 'items': items} for u, items in observed]})
                pairs = [('item_bench', bench, frame['item_bench'], 10)]
                for target in frame['units']:
                    near = [(u, items) for u, items in observed if u.zone == target['zone']
                            and abs(u.x - target['x']) < .001 and abs(u.y - target['y']) < .001]
                    found = near[0][1] if near else []
                    key = f"{target['zone']}:{target['x']:.3f},{target['y']:.3f}"
                    pairs.append((key, found, target['items'], 3))
                for zone, found, wanted, slots in pairs:
                    pred, expected = ({r['slot']: r for r in rows} for rows in [found, wanted])
                    for slot in range(slots):
                        a, b = pred.get(slot), expected.get(slot)
                        counts['present'] += b is not None
                        label = 'tp' if a and b else 'fp' if a else 'fn' if b else 'tn'
                        counts[label] += 1
                        if bool(a) != bool(b):
                            diffs.append([frame['_file'], zone, slot, 'presence', bool(b), bool(a)])
                        if a and not a['name'].startswith('?'):
                            counts['named'] += 1
                            counts['named_present'] += b is not None
                            ok = bool(b and a['name'] == b['name'])
                            counts['correct_names'] += ok
                            if not ok:
                                diffs.append([frame['_file'], zone, slot, 'name', (b or {}).get('name'), a['name']])
                        if b and 'badge' in b:
                            counts['badge_total'] += 1
                            counts['badge_ok'] += bool(a and a.get('badge') == b['badge'])
    total = sum(counts[k] for k in ['tp', 'fp', 'fn', 'tn'])
    return {'frames': len(truth), 'repeat': repeat, 'counts': counts,
            'presence_accuracy': (counts['tp'] + counts['tn']) / total if total else None,
            'presence_recall': counts['tp'] / counts['present'] if counts['present'] else None,
            'name_accuracy': counts['correct_names'] / counts['named'] if counts['named'] else None,
            'coverage': counts['named_present'] / counts['present'] if counts['present'] else None,
            'items_ms': {'mean': statistics.mean(timing), 'median': statistics.median(timing), 'max': max(timing)},
            'diffs': diffs, 'predictions': predictions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=Path('game-eye/samples'))
    ap.add_argument('--truth', type=Path, default=ROOT / 'tests/fixtures/item_truth.jsonl')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    report = evaluate(args.samples, args.truth, args.repeat)
    print(json.dumps({k: v for k, v in report.items() if k not in ['predictions', 'diffs']}, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
