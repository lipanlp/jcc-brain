"""按 extra_items.json 经 CoreGraphics 裁取实拍图标；不修改清单或参考答案。

    .venv/bin/python scripts/make_extra_items.py --samples /path/to/samples

清单中的 rect 是来源帧原始像素坐标，slot 与散件栏定位交叉核对。
固定 ZIP 元数据并按清单顺序保存无损 uint8 像素，重复生成的 NPZ 字节一致。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.items import bench_slots  # noqa: E402

MANIFEST = ROOT / 'src/jccbrain/vision/eye/extra_items.json'


def generate(samples: Path, manifest: Path, output: Path) -> int:
    data = json.loads(manifest.read_text('utf-8'))
    if data['version'] != 1:
        raise ValueError('不支持的通用道具清单版本')
    images, arrays = {}, {}
    for item in data['items']:
        for index, template in enumerate(item['templates']):
            source = (samples / template['frame']).resolve()
            if not source.is_relative_to(samples.resolve()):
                raise ValueError('模板来源必须在样本目录内')
            if source not in images:
                images[source] = load_image(source)
            image = images[source]
            rect = tuple(template['rect'])
            if rect != bench_slots(image)[template['slot']]:
                raise ValueError(f'{source.name}: rect 与散件栏 slot 不一致')
            x0, y0, x1, y1 = rect
            arrays[f"{item['name']}/{index}"] = image[y0:y1, x0:x1]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, pixels in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, pixels, allow_pickle=False)
            entry = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())
    return len(arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=Path('game-eye/samples'))
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--output', type=Path, default=MANIFEST.with_name('extra_icons.npz'))
    args = ap.parse_args()
    count = generate(args.samples, args.manifest, args.output)
    print(f'{count} 个实拍图标，NPZ {args.output.stat().st_size} 字节：{args.output}')


if __name__ == '__main__':
    main()
