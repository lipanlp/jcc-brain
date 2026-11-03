"""按人工 item_truth.jsonl，经 CoreGraphics 提取装备回归小格；不生成/修改参考答案。

    .venv/bin/python scripts/make_item_fixtures.py --samples /path/to/samples

包含每种已标名字、明暗遮罩、数字角标、单位模糊正例和边框得分最高的背景负例。
官方图标只保存本测试用的小子集；运行时完整目录来自 item_catalog.json / 当前赛季数据库。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.items import CATALOG, bench_slots, frame_score, unit_slots  # noqa: E402
from jccbrain.vision.eye.units import Gallery, UnitTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=Path('game-eye/samples'))
    ap.add_argument('--icons', type=Path, default=ROOT / 'var/item_icons/s18_S19')
    ap.add_argument('--output', type=Path, default=ROOT / 'tests/fixtures')
    args = ap.parse_args()
    truth = [json.loads(line) for line in (ROOT / 'tests/fixtures/item_truth.jsonl').read_text().splitlines()]
    selected, unit_cells, seen = [], [], set()
    with tempfile.TemporaryDirectory() as temporary:
        tracker = UnitTracker(Gallery(Path(temporary)))
        for frame in truth:
            image = load_image(args.samples / frame['_file'])
            h, w = image.shape[:2]
            inventory = {r['slot']: r for r in frame['item_bench']}
            dim = frame['_file'] in ['0045.jpg', '0048.jpg', 'items/g2_0280.jpg']
            for slot, rect in enumerate(bench_slots(image)):
                label = inventory.get(slot)
                # 遮罩的亮度变化用未知角标和斗篷各一格代表，避免同图案反复存整格。
                if dim and (label or {}).get('name') not in ['?', '负极斗篷']:
                    continue
                key = ((label or {}).get('name'), (label or {}).get('badge'), dim)
                if key not in seen:
                    seen.add(key)
                    selected.append(dict(file=frame['_file'], zone='item_bench', rect=rect, label=label,
                                         window=[w, h], pad=0, score=None))
            units = tracker.detect(image)[1]
            for target in frame['units']:
                unit = min(units, key=lambda u: abs(u.x - target['x']) + abs(u.y - target['y']))
                labels = {r['slot']: r for r in target['items']}
                for slot, rect in enumerate(unit_slots(unit.bar)):
                    label = labels.get(slot)
                    unit_cells.append(dict(file=frame['_file'], zone=unit.zone, rect=rect, label=label,
                                           window=[w, h], pad=3, score=frame_score(image, rect)))
    # 每种单位装备保留第一格、最清晰、最模糊各一张；另保留5个最高边框分的背景负格。
    names = {c['label']['name'] for c in unit_cells if c['label']}
    for name in sorted(names):
        cells = [c for c in unit_cells if c['label'] and c['label']['name'] == name]
        for cell in [cells[0], min(cells, key=lambda c: c['score']), max(cells, key=lambda c: c['score'])]:
            if cell not in selected:
                selected.append(cell)
    selected += sorted((c for c in unit_cells if not c['label']), key=lambda c: -c['score'])[:5]
    selected += [next(c for c in unit_cells if c['zone'] == 'bench' and not c['label'])]
    arrays, cases, images = {}, [], {}
    for i, cell in enumerate(selected):
        if cell['file'] not in images:
            images[cell['file']] = load_image(args.samples / cell['file'])
        image = images[cell['file']]
        x0, y0, x1, y1 = cell['rect']
        pad = cell['pad']
        key = f'cell{i}'
        arrays[key] = image[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
        case = {**cell, 'array': key}
        if case['window'] == [1774, 1136]:
            del case['window']
        if case['score'] is None:
            del case['score']
        cases.append(case)
    catalog = json.loads(CATALOG.read_text('utf-8'))
    wanted = {r['name'] for f in truth for r in f['item_bench'] if r['name'] != '?'}
    wanted |= names - {'?'}
    wanted |= {'光明版无尽之刃', '大亨之铠', '永恒之森纹章', '金铲铲', '金锅锅', '生命值补剂'}
    icons = []
    for row in catalog['items']:
        if row['name'] not in wanted:
            continue
        key = 'icon_' + row['id']
        arrays[key] = cv2.resize(load_image(args.icons / (row['id'] + '.png')), (16, 16), interpolation=cv2.INTER_AREA)
        icons.append({k: row[k] for k in ['id', 'name', 'kind']} | {'array': key})
    args.output.mkdir(parents=True, exist_ok=True)
    # 行差分 + 标准库 LZMA，保留每个原像素，无损减小这组 JPEG 来源的小格噪点。
    with zipfile.ZipFile(args.output / 'item_cells.npz', 'w', compression=zipfile.ZIP_LZMA) as out:
        for key, image in arrays.items():
            delta = np.diff(image, axis=1, prepend=np.zeros_like(image[:, :1]))
            assert np.array_equal(np.cumsum(delta, axis=1, dtype=np.uint8), image)
            buffer = io.BytesIO()
            np.save(buffer, delta)
            out.writestr(key + '.npy', buffer.getvalue())
    (args.output / 'item_cells.json').write_text(json.dumps(
        {'decoder': 'CoreGraphics', 'encoding': 'row-delta-u8', 'window': [1774, 1136],
         'season_id': catalog['season_id'], 'cases': cases, 'icons': icons},
        ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{len(cases)} 个真实小格、{len(icons)} 个官方图标缩略图，NPZ {(args.output / "item_cells.npz").stat().st_size} 字节')


if __name__ == '__main__':
    main()
