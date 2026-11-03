#!/usr/bin/env python3
"""独立回放refs + 冻结图库，在完整时序reader上扫描17×3个名字门限。

仅缓存纯像素特征及与名字门限无关的OCR结果；每个组合重新建立跟踪/聚组/自动学习状态。
先真实运行(.55,0)，再缓存重放并逐字段断言一致。缓存不含真值，不跳过stale或未匹配位置。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    from eval_gallery_loo import memory_add, select_parameters
    from eval_units import copied_gallery, evaluate, load_truth, pool_heroes, validate_truth
    from eval_vision import build
    from jccbrain.vision.eye import units as units_module
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.reader import GameReader
    from jccbrain.vision.eye.units import Gallery, UnitTracker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = load_truth()
    pool = {r['name'] for r in pool_heroes(args.db)}
    validate_truth(truth, pool)
    images = [(p.name, load_image(p)) for p in sorted(args.samples.glob('*.jpg'))]
    image_names = {id(im): name for name, im in images}
    cells_cache, feature_cache = {}, {}
    original_feature = units_module.feature_print

    def cached_feature(image):
        key = (image.shape, hashlib.sha256(image.tobytes()).digest())
        if key not in feature_cache:
            feature_cache[key] = original_feature(image)
        return feature_cache[key]

    units_module.feature_print = cached_feature
    with copied_gallery(args.gallery) as directory:
        prototype = build(args.db, directory, 'rapid')
        season = prototype.units.gallery.dir.name[1:]

        def replay(threshold, margin):
            gallery = Gallery(directory, season, pool, threshold=threshold, margin=margin)
            gallery.add = lambda name, fp, crop, copy_jpg=None: memory_add(gallery, name, fp, crop, copy_jpg)
            if prototype.items:
                prototype.items.reset()
            reader = GameReader(prototype.dict, UnitTracker(gallery, prototype.items),
                                prototype.regions, prototype.ocr, prototype.augments_taken)
            original_begin = reader._begin_cells

            def cached_cells(image, only=None, before_ocr=None):
                key = (image_names[id(image)], tuple(sorted(only)) if only is not None else None)
                if key in cells_cache:
                    cells, early = copy.deepcopy(cells_cache[key])
                    if before_ocr:
                        before_ocr(early)
                    reader._last_ocr_s = 0.
                    return lambda: cells, early
                finish, early = original_begin(image, only, before_ocr)
                cells = finish()
                cells_cache[key] = copy.deepcopy((cells, early))
                return lambda: cells, early

            reader._begin_cells = cached_cells
            return {name: reader.read(image) for name, image in images}

        try:
            baseline = replay(.55, 0.)
            assert replay(.55, 0.) == baseline, '缓存重放与真实reader输出不一致'
            results = []
            for i in range(17):
                threshold = round(.4 + i * .025, 3)
                for margin in (0., .05, .10):
                    predictions = replay(threshold, margin)
                    report = evaluate(predictions, truth, pool)
                    result = dict(threshold=threshold, margin=margin,
                                  accuracy=report['accuracy'], coverage=report['coverage'], counts=report['counts'])
                    results.append(result)
                    file = args.output / f't{threshold:.3f}-m{margin:.2f}.json'
                    file.write_text(json.dumps(dict(result=result, predictions=predictions, report=report),
                                               ensure_ascii=False, indent=2), 'utf-8')
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            selected = select_parameters(results)
            output = dict(results=results, selected=selected, cached_replay_identical=True,
                          baseline=baseline, feature_cache_entries=len(feature_cache),
                          ocr_cache_entries=len(cells_cache))
            (args.output / 'scan.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), 'utf-8')
            print('selected', selected, flush=True)
        finally:
            units_module.feature_print = original_feature
            if worker_pool := getattr(prototype.ocr, 'pool', None):
                worker_pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
