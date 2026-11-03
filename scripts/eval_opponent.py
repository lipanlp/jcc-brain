"""敌方颜色识别候选专项；即便产品禁用也评分候选，不把null算成识别正确。"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict


def score(pred, truth):
    board = pred.get('board') or []
    counts = {'board_count': (int(len(board) == truth['board_count']), 1)}
    def distance(a, b):
        dx, dy = abs(a['x']-b['x']), abs(a['y']-b['y'])
        return dx+dy if dx < .02 and dy < .03 else float('inf')
    false = []
    for unit in board:
        friend = min((distance(unit, p) for p in truth.get('friendly_positions', [])), default=float('inf'))
        enemy = min((distance(unit, p) for p in truth.get('enemy_positions', [])), default=float('inf'))
        if friend < enemy:
            false.append(unit)
    counts['friendly_as_enemy'] = (len(false), len(board))
    return counts, false


def run(args):
    from eval_vision import build
    from eval_units import copied_gallery
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.opponent import OpponentReader, ENABLED
    truth = {row['_file']: row for row in map(json.loads, args.opponent_truth.read_text().splitlines())}
    # 完整原27帧时序；不能只给候选喂战斗真值从而替代阶段检测。
    images = [(path.name, load_image(path)) for path in sorted(args.samples.glob('*.jpg'))]
    counts = defaultdict(lambda: [0, 0]);rows=[]
    for rep in range(args.repeat):
        with copied_gallery(args.gallery) as gallery:
            reader = build(args.db, gallery, args.ocr)
            candidate = OpponentReader(reader.units.gallery, reader.ocr)
            try:
                for name, image in images:
                    state = reader.read(image)
                    if name not in truth:
                        continue
                    start = time.perf_counter()
                    pred = candidate.read(image) if state.get('phase') == 'combat' else {}
                    elapsed = (time.perf_counter()-start)*1000
                    scored, false = score(pred, truth[name])
                    for key, value in scored.items():
                        counts[key][0] += value[0];counts[key][1] += value[1]
                    rows.append(dict(_file=name, rep=rep, ms=elapsed, prediction=pred,
                                     production=state.get('opponent'), false_friendly=false,
                                     truth_count=truth[name]['board_count']))
            finally:
                reader.ocr.pool.shutdown(wait=True)
    total = counts['board_count']
    report = dict(enabled=ENABLED, counts=dict(counts), board_count_accuracy=total[0]/total[1] if total[1] else None,
                  candidate_ms=dict(mean=statistics.mean(r['ms'] for r in rows), max=max(r['ms'] for r in rows)),
                  frames=rows)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2))
    if args.json:args.json.write_text(json.dumps({'opponent':report},ensure_ascii=False,indent=2)+'\n')
    if args.dump:args.dump.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    return 0
