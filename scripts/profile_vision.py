"""分开报告新 reader 的第一遍和同 reader 的缓存遍；模型累计耗时不是墙钟关键路径。"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eval_vision import DEFAULT_SAMPLES, build
from eval_units import copied_gallery
from jccbrain.vision.eye.capture import load_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument('--db', type=Path, required=True)
    ap.add_argument('--gallery', type=Path)
    ap.add_argument('--rounds', type=int, default=1, help='每轮新建 reader，清空所有缓存')
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--json', type=Path, required=True)
    args = ap.parse_args()
    files = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    images = [load_image(p) for p in files]
    rows = []
    for iteration in range(args.rounds):
        with copied_gallery(args.gallery) as gallery:
            reader = build(args.db, gallery, 'rapid')
            try:
                for rep in range(args.passes):
                    for path, image in zip(files, images):
                        reader.ocr.reset_stats()
                        start = time.perf_counter()
                        state = reader.read(image)
                        rows.append(dict(round=iteration, rep=rep, file=path.name, phase=state.get('phase'),
                                         read_ms=(time.perf_counter() - start) * 1000,
                                         ocr_ms=reader.last_ms['ocr'], model_ms=dict(reader.ocr.last_ms),
                                         counts=dict(reader.ocr.last_counts)))
            finally:
                reader.ocr.pool.shutdown(wait=True)
    summary = []
    for rep in range(args.passes):
        subset = [row for row in rows if row['rep'] == rep]
        entry = dict(rep=rep, read_mean=statistics.mean(row['read_ms'] for row in subset),
                     read_median=statistics.median(row['read_ms'] for row in subset),
                     ocr_mean=statistics.mean(row['ocr_ms'] for row in subset),
                     model_mean={k: statistics.mean(row['model_ms'][k] for row in subset)
                                 for k in ('det', 'rec', 'vision')},
                     slowest=sorted(subset, key=lambda row: -row['read_ms'])[:5])
        summary.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dict(summary=summary, frames=rows), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
