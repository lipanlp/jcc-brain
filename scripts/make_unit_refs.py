#!/usr/bin/env python3
"""从面板确认身份的独立回放生成随包向量；原图只读，来源和位置见 unit_refs.json。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def generate(samples: Path, directory: Path) -> dict:
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.units import Bar, feature_print, unit_crop

    metadata = json.loads((directory / 'unit_refs.json').read_text('utf-8'))
    arrays, counts = {}, Counter()
    for row in metadata['entries']:
        file = samples / row['frame']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'来源帧发生变化：{file}')
        image = load_image(file)
        if [image.shape[1], image.shape[0]] != row['window']:
            raise ValueError(f'窗口尺寸不符：{file}')
        counts[row['name']] += 1
        if counts[row['name']] > 3:
            raise ValueError(f'每英雄最多3条：{row["name"]}')
        crop = unit_crop(image, Bar(*row['bar']))
        vector = feature_print(crop) if crop is not None else None
        if vector is None or vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f'特征提取失败：{file}')
        arrays[row['key']] = vector.astype(np.float32)
    output = directory / f's{metadata["season"]}.npz'
    np.savez_compressed(output, **arrays)
    size = output.stat().st_size + (directory / 'unit_refs.json').stat().st_size
    if size > 300 * 1024:
        raise ValueError(f'参考资源超出300KB：{size}')
    return dict(heroes=len(counts), vectors=len(arrays), bytes=size,
                dimensions=sorted({v.size for v in arrays.values()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=ROOT / 'var/vision-module-optimization/replay-samples')
    parser.add_argument('--output', type=Path, default=ROOT / 'src/jccbrain/vision/eye/unit_refs')
    args = parser.parse_args()
    print(json.dumps(generate(args.samples, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
