"""只读截取当前游戏窗口；操作回放请在游戏里完成，本脚本不发送输入。

示例：.venv/bin/python scripts/capture_samples.py heroes --stage 2-3 --notes 卡蜜尔信息面板
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SCENES = ('heroes', 'carousel', 'pve', 'spectate', 'dialog', 'stagebar',
          'augment-dim', 'digits', 'population', 'augment-pick', 'augment-taken')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scene', choices=SCENES)
    ap.add_argument('--stage', default='unknown', help='人工核对的回合号')
    ap.add_argument('--notes', default='')
    args = ap.parse_args()
    import cv2

    from jccbrain.actuator.window import WindowLocator
    from jccbrain.vision.eye.capture import CaptureError, WindowCapture

    try:
        image, window = WindowCapture(WindowLocator('com.tencent.jkchess')).grab()
    except CaptureError as exc:
        ap.exit(1, f'{exc}\n')
    destination = ROOT / 'var/vision-module-optimization/replay-samples' / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    sequence = max((int(p.stem) for p in destination.glob('*.png') if p.stem.isdigit()), default=0) + 1
    path = destination / f'{sequence:04d}.png'
    if not cv2.imwrite(str(path), image):
        ap.exit(1, f'写入失败：{path}\n')
    row = dict(file=path.name, timestamp=datetime.now().astimezone().isoformat(),
               scene=args.scene, stage=args.stage, notes=args.notes,
               width=image.shape[1], height=image.shape[0], window_id=window.window_id)
    with (destination / 'manifest.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
