"""CoreGraphics 裁取侦察眼睛定位小夹具；正例只保留眼睛附近，负例保留完整搜索列。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jccbrain.vision.eye.augments import eye_template
from jccbrain.vision.eye.capture import load_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spectate', type=Path, default=ROOT/'var/vision-module-optimization/replay-samples/spectate')
    ap.add_argument('--own', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT/'tests/fixtures')
    args = ap.parse_args()
    slots = dict(zip(('0001', '0002', '0003', '0004', '0006', 'own'), (4, 1, 0, 2, 1, None)))
    arrays = {}
    for key, slot in slots.items():
        image = load_image(args.own if key == 'own' else args.spectate/(key+'.png'))
        h, w = image.shape[:2]
        strip = cv2.resize(image[round(.09*h):round(.69*h), round(.963*w):round(.983*w)],
                           (35, 682), interpolation=cv2.INTER_AREA)
        if slot is not None:
            _, _, _, location = cv2.minMaxLoc(cv2.matchTemplate(strip, eye_template(), cv2.TM_CCOEFF_NORMED))
            lo, hi = max(0, location[1]-10), min(682, location[1]+eye_template().shape[0]+10)
            canvas = np.zeros_like(strip)
            canvas[lo:hi] = strip[lo:hi]
            strip = canvas
        arrays[key] = strip
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'scout_eye_cells.npz', **arrays)
    (args.output/'scout_eye_cells.json').write_text(json.dumps(slots)+'\n')
    shutil.copyfile(args.reference, args.output/'scout-spectate-sample.jpg')


if __name__ == '__main__':
    main()
