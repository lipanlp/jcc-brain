"""Rebuild small B8 real-pixel recovery regressions through CoreGraphics.

Outputs are below 300KB; source screenshots and old truth are never rewritten.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jccbrain.vision.eye.capture import load_image

# Source crops were manually reviewed with the full frame. Expected points
# identify actual recovered slots; they are never passed into the detector.
ENTRIES = [
    ('tank-0033', '0033.jpg', [532,91,140,70], [599,111], True),
    ('low-0033', '0033.jpg', [868,296,140,70], [935,315], True),
    ('flash-0036', '0036.jpg', [1090,111,140,70], [1157,130], True),
    ('tank-0036', '0036.jpg', [611,181,140,70], [678,201], True),
    ('low-0036', '0036.jpg', [1069,296,140,70], [1136,315], True),
    ('plant-0063', '0063.jpg', [830,139,140,70], [897,158], True),
    ('plant-bottom-0063', '0063.jpg', [832,217,145,70], [902,236], True),
    ('front-0063', '0063.jpg', [782,309,140,70], [849,328], True),
    ('augment-icon', '0060.jpg', [520,315,80,65], None, False),
    ('coarse-nameplate', '0036.jpg', [1240,107,120,45], None, False),
    ('friendly-green', '0033.jpg', [1080,294,144,40], None, False),
    ('orange-damage', '0063.jpg', [970,335,100,65], None, False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=ROOT/'game-eye/samples')
    ap.add_argument('--output', type=Path, default=ROOT/'tests/fixtures/opponent_b8_cells', help='输出前缀，写.npz和.json')
    args = ap.parse_args()
    arrays, rows = {}, []
    for key, file, rect, point, expected in ENTRIES:
        image = load_image(args.samples/file)
        x,y,width,height = rect
        arrays[key] = image[y:y+height,x:x+width].copy()
        rows.append(dict(key=key, source=file, origin=[x,y], point=point, enemy=expected))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    size = sum(args.output.with_suffix(ext).stat().st_size for ext in ('.npz','.json'))
    print(size)
    if size > 300*1024:
        raise ValueError('fixture budget exceeded')


if __name__ == '__main__':main()
