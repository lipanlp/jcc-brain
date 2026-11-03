"""经CoreGraphics从已人工核对的原图生成回合条模板，不生成或改写参考答案。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.phase import TEMPLATES, patch, vector  # noqa: E402

SOURCES = [('augment', '0021.jpg', 1), ('pvp', '0021.jpg', 2),
           ('carousel', '0021.jpg', 3), ('pve', '0021.jpg', 6), ('pve', '0000.jpg', 6)]
# 战斗开始后PVP格显示对手头像，不能拿灰色斧头模板强行匹配。
# 只增当前格模板，下一格的cursor/icons/names保持原数组。
CURRENT_SOURCES = [('pvp', '0000.jpg', 5), ('pvp', '0030.jpg', 0), ('pvp', '0060.jpg', 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=Path('game-eye/samples'))
    ap.add_argument('--output', type=Path, default=TEMPLATES)
    args = ap.parse_args()
    cursor = vector(patch(load_image(args.samples / '0021.jpg'), 0, True), True)
    icons = np.stack([vector(patch(load_image(args.samples / file), index)) for _, file, index in SOURCES])
    portraits = np.stack([vector(patch(load_image(args.samples / file), index)) for _, file, index in CURRENT_SOURCES])
    np.savez_compressed(args.output, cursor=cursor, icons=icons,
                        names=np.array([name for name, _, _ in SOURCES]),
                        sources=np.array(json.dumps(SOURCES, ensure_ascii=False)),
                        current_icons=np.concatenate([icons, portraits]),
                        current_names=np.array([name for name, _, _ in SOURCES + CURRENT_SOURCES]),
                        current_sources=np.array(json.dumps(SOURCES + CURRENT_SOURCES, ensure_ascii=False)))
    print(args.output)


if __name__ == '__main__':
    main()
