#!/usr/bin/env python3
"""只追加人工template帧的7/8/9，保留原字库顺序；输出裁剪/来源便于复核。"""
import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from make_digit_templates import REGIONS, canvas, expected  # noqa: E402
from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.digits import TEMPLATE_FILE, normalize_float, segment  # noqa: E402
from jccbrain.vision.eye.image import crop, load_regions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='补充前的digits.npz（只读）')
    parser.add_argument('--samples', type=Path, default=ROOT / 'var/vision-module-optimization/replay-samples')
    parser.add_argument('--truth', type=Path, default=ROOT / 'tests/fixtures/digit_truth.jsonl')
    parser.add_argument('--output', type=Path, default=TEMPLATE_FILE)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'tests/fixtures/digit_cells_789.npz')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base, allow_pickle=False) as data:
        chars, glyphs = list(str(data['chars'])), list(data['glyphs'])
    original_count = len(chars)
    fixtures = dict(base_chars=np.array(''.join(chars)), base_glyphs=np.stack(glyphs))
    sources, skipped, metadata = [], [], []
    regions = load_regions()
    for row in map(json.loads, args.truth.read_text().splitlines()):
        image = load_image(args.samples / row['_file'])
        fields = dict(row['fields'], streak={'n': row['fields']['streak.n']})
        for field in REGIONS:
            value = expected(fields, field)
            if value is None or not any(c in '789' for c in value):
                continue
            cell = crop(image, regions[field])
            key = f'cell_{len(metadata)}'
            fixtures[key] = cell
            metadata.append(dict(file=row['_file'], field=field, key=key, height=image.shape[0]))
            if row['split'] != 'template':
                continue
            gs = [g for g in segment(cell, image.shape[0]) if not (g.h < .45*15 and g.w > g.h)]
            wanted = value.replace('-', '')
            if len(gs) > len(wanted) and gs[0].x0 == 0:
                gs = gs[1:]
            if len(gs) > len(wanted) and gs[-1].x1 == cell.shape[1]:
                gs = gs[:-1]
            if len(gs) != len(wanted):
                skipped.append(dict(file=row['_file'], field=field, expected=wanted, segments=len(gs)))
                continue
            for index, (char, glyph) in enumerate(zip(wanted, gs)):
                if char not in '789':
                    continue
                candidate = canvas(glyph.mask)
                old = [normalize_float(g) for c, g in zip(chars, glyphs) if c == char]
                similarity = max((float(normalize_float(candidate) @ v) for v in old), default=-1.)
                if similarity >= .95:
                    continue
                chars.append(char)
                glyphs.append(candidate.astype(np.uint8))
                sources.append(dict(file=row['_file'], field=field, glyph=index, char=char,
                                    previous_same_char_score=similarity))
    fixtures['metadata'] = np.array(json.dumps(metadata, ensure_ascii=False))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fixtures)
    assert buffer.tell() <= 300*1024, buffer.tell()
    np.savez_compressed(args.output, chars=np.array(''.join(chars)), glyphs=np.stack(glyphs).astype(np.uint8))
    args.fixtures.write_bytes(buffer.getvalue())
    report = dict(before=dict(Counter(chars[:original_count])), after=dict(Counter(chars)),
                  added=sources, skipped=skipped, fixture_bytes=args.fixtures.stat().st_size)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
