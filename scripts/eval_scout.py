"""侦察专项：人工真值；每张清空侦察缓存，单列不支持布局，未知真值不计分。"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


def score(pred: dict, truth: dict) -> dict[str, tuple[int, int]]:
    if truth.get('_supported') is False:
        return {}
    counts = {}
    for key in ('traits', 'level', 'viewing'):
        want = truth.get(key)
        if want is not None or (key == 'viewing' and truth.get('viewing_known')):
            got = pred.get(key)
            counts[key] = (int(got == want), 1)
    wanted = truth.get('traits')
    if wanted is not None:
        got = {t['name']: t for t in pred.get('traits') or []}
        counts['traits_names'] = (sum(t['name'] in got for t in wanted), len(wanted))
        counts['traits_values'] = (sum(got.get(t['name']) == t for t in wanted), len(wanted))
    if truth.get('board_count') is not None:
        counts['board_count'] = (int(len(pred.get('board') or []) == truth['board_count']), 1)
    units = truth.get('board_names') or []
    good = 0
    available = set(range(len(pred.get('board') or [])))
    for unit in units:
        index = min((i for i in available if abs(pred['board'][i].get('x', -1)-unit['x']) < .02
                     and abs(pred['board'][i].get('y', -1)-unit['y']) < .03),
                    key=lambda i: abs(pred['board'][i]['x']-unit['x']) + abs(pred['board'][i]['y']-unit['y']), default=None)
        match = pred['board'][index] if index is not None else {}
        if index is not None:
            available.remove(index)
        good += match.get('name') == unit['name']
    counts['board_names'] = (good, len(units))
    return counts


def run(args) -> int:
    from eval_vision import DEFAULT_SAMPLES, ROOT, build
    from eval_units import copied_gallery
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.scout import supported_layout

    truth = [json.loads(line) for line in args.scout_truth.read_text().splitlines() if line]
    files = [(row, ROOT / row['_file'] if row['_file'].startswith(('var/', 'tests/'))
              else DEFAULT_SAMPLES / row['_file']) for row in truth]
    images = [(row, load_image(path)) for row, path in files]
    stats = defaultdict(lambda: [0, 0])
    rows = []
    for rep in range(args.repeat):
        with copied_gallery(args.gallery) as gallery:
            reader = build(args.db, gallery, args.ocr)
            try:
                for row, image in images:
                    if scout := getattr(reader, '_scout_reader', None):
                        scout.reset()
                    start = time.perf_counter()
                    pred = reader.read_scout(image)
                    elapsed = (time.perf_counter()-start)*1000
                    supported = row.get('_supported', True) and supported_layout(image)
                    for key, value in (score(pred, row) if supported else {}).items():
                        stats[key][0] += value[0]
                        stats[key][1] += value[1]
                    rows.append(dict(rep=rep, _file=row['_file'], ms=elapsed,
                                     supported_layout=supported, prediction=pred))
            finally:
                reader.ocr.pool.shutdown(wait=True)
    supported = [row['ms'] for row in rows if row['supported_layout']]
    report = dict(counts=dict(stats), accuracy={k: a/b if b else None for k, (a, b) in stats.items()},
                  cold_supported_ms=dict(mean=statistics.mean(supported), max=max(supported)),
                  unsupported=sum(not row['supported_layout'] for row in rows), frames=rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps({'scout': report}, ensure_ascii=False, indent=2)+'\n')
    if args.dump:
        with args.dump.open('w') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
    return 0
