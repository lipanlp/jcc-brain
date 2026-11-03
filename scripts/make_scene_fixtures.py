#!/usr/bin/env python3
"""从已人工核对的新场景生成小图与确认框模板；不改写人工真值。"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jccbrain.vision.eye.augments import eye_template  # noqa: E402
from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.phase import SCENE_TEMPLATES, dialog_patch, vector  # noqa: E402


def cells(image, kind):
    h, w = image.shape[:2]
    if kind == 'dialog':
        return dialog_patch(image), {}
    if kind == 'spectate':
        strip = image[round(.09*h):round(.69*h), round(.963*w):round(.983*w)]
        strip = cv2.resize(strip, (35, 682), interpolation=cv2.INTER_AREA)
        scores = cv2.matchTemplate(strip, eye_template(), cv2.TM_CCOEFF_NORMED)
        y = int(np.unravel_index(np.argmax(scores), scores.shape)[0])
        return strip[y:y + 19], {'y': y, 'score': float(scores.max())}
    cell = image[round(.032*h):round(.072*h), round(.399*w):round(.601*w)]
    return cv2.resize(cell, (179, 23), interpolation=cv2.INTER_AREA), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=ROOT / 'var/vision-module-optimization/replay-samples')
    parser.add_argument('--legacy', type=Path, default=Path('game-eye/samples'))
    args = parser.parse_args()
    template = vector(dialog_patch(load_image(args.samples / 'dialog/0001.png')))
    np.savez_compressed(SCENE_TEMPLATES, dialog=template)
    rows = [json.loads(line) for line in (ROOT / 'tests/fixtures/phase_truth.jsonl').read_text().splitlines()]
    metadata, arrays = [], {}
    for row in rows:
        if not row['_file'].endswith('.png'):
            continue
        image = load_image(args.samples / row['_file'])
        kind = row['phase']
        cell, extra = cells(image, kind)
        key = f'cell_{len(arrays)}'
        arrays[key] = cell
        metadata.append(dict(file=row['_file'], kind=kind, expected=kind, key=key, **extra))
    # 旧27帧中各取分数最高的两个负例；完整旧帧仍由正式eval_vision回归。
    for kind in ('dialog', 'spectate'):
        candidates = []
        for file in sorted(args.legacy.glob('*.jpg')):
            cell, extra = cells(load_image(file), kind)
            score = float(vector(cell) @ template) if kind == 'dialog' else extra['score']
            candidates.append((score, file.name, cell, extra))
        for score, file, cell, extra in sorted(candidates, key=lambda r: -r[0])[:2]:
            key = f'cell_{len(arrays)}'
            arrays[key] = cell
            metadata.append(dict(file=file, kind=kind, expected=None, key=key, **extra))
    arrays['metadata'] = np.array(json.dumps(metadata, ensure_ascii=False))
    output = ROOT / 'tests/fixtures/scene_cells.npz'
    np.savez_compressed(output, **arrays)
    total = output.stat().st_size + SCENE_TEMPLATES.stat().st_size
    assert total <= 300 * 1024, total
    print(json.dumps(dict(fixtures=len(metadata), bytes=total)))


if __name__ == '__main__':
    main()
