"""CoreGraphics读取侦察短血条回归小图，保留原像素与窗口锚点。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jccbrain.vision.eye.capture import load_image


def main():
    root = ROOT/'var/vision-module-optimization/replay-samples/spectate'
    sources = [('0072', Path('game-eye/samples/0072.jpg'), 1094, 230),
               ('0004', root/'0004.png', 926, 633), ('0006', root/'0006.png', 1166, 603)]
    arrays, rows = {}, []
    for name, path, x, y in sources:
        image = load_image(path)
        arrays[name] = image[y-20:y+130, x-30:x+130].copy()
        rows.append(dict(key=name, origin=[x-30, y-20], source=str(path), width=72))
    image = load_image(root/'0002.png')
    for i, (cx, cy) in enumerate(((.664, .083), (.488, .098))):
        x0, y0 = round(cx*1774)-64, round(cy*1136)-14
        key = f'upper{i}'
        arrays[key] = image[y0:y0+28, x0:x0+128].copy()
        rows.append(dict(key=key, origin=[x0, y0], source=str(root/'0002.png')))
    target = ROOT/'tests/fixtures/scout_unit_cells.npz'
    np.savez_compressed(target, **arrays)
    target.with_suffix('.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    print(target.stat().st_size + target.with_suffix('.json').stat().st_size)


if __name__ == '__main__':
    main()
