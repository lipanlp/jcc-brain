"""按 (样本目录, stage) 留一组评估单位特征；答案/裁剪仅存评测缓存。

    .venv/bin/python scripts/eval_gallery_loo.py --db /tmp/jcc-vf-db.sqlite3
    .venv/bin/python scripts/eval_gallery_loo.py --db /tmp/jcc-vf-db.sqlite3 --reuse-cache

live 使用 reader.last_units 的原 crop/fp（包括 reader 复用的旧 fp），不重算。
战斗集沿用 #4a F1：stale 帧在 last_live 备战席血条位置取当前整框特征，
不把移动中的旧棋盘坐标当作当前单位。战斗集不参与训练或选参。
指定分组不能消除跨目录重复画面/跨回合缓存 fp；报告重复数量，不能视为独立测试。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from scripts.eval_units import load_truth, match_units, pool_heroes, validate_truth

DEFAULT_SAMPLES = Path('game-eye/samples')
DEFAULT_CACHE = ROOT / 'var/vision-module-optimization/gallery-loo'
THRESHOLDS = [i / 1000 for i in range(300, 901, 25)]
MARGINS = [0., .03, .05, .08, .10]


@dataclass
class Sample:
    frame: str
    fold: tuple[str, str]
    name: str
    zone: str
    slot: int | None
    fp: np.ndarray | None
    crop: np.ndarray | None = None
    kind: str = 'live'


def fold_key(frame: str, state: dict, vision_truth: dict) -> tuple[str, str]:
    # 必须用相对全路径：track/0021.jpg 不能套标准目录 0021.jpg 的答案。
    stage = vision_truth.get(frame, {}).get('stage') or state.get('stage')
    # 结算帧可能两边都无 stage；单列缺失组，不猜回合、不混入已知组。
    return Path(frame).parent.as_posix(), str(stage) if stage else '<missing>'


def nearest(query: Sample, training: list[Sample]) -> dict:
    """同名参考图先取最小值；d2 只取另一个名字，缺失为无穷大。"""
    distances = {}
    if query.fp is not None:
        for ref in training:
            if ref.fp is not None:
                d = float(np.linalg.norm(query.fp - ref.fp))
                distances[ref.name] = min(distances.get(ref.name, float('inf')), d)
    ordered = sorted(distances.items(), key=lambda item: item[1])
    return {'name': ordered[0][0] if ordered else None,
            'd1': ordered[0][1] if ordered else float('inf'),
            'd2': ordered[1][1] if len(ordered) > 1 else float('inf'),
            'reference_available': query.name in distances}


def loo_neighbors(samples: list[Sample]) -> list[dict]:
    folds = {sample.fold for sample in samples}
    training = {fold: [s for s in samples if s.fold != fold] for fold in folds}
    return [nearest(sample, training[sample.fold]) for sample in samples]


def decision(neighbor: dict, threshold: float, margin: float) -> str | None:
    if neighbor['name'] is not None and neighbor['d1'] < threshold and neighbor['d2'] - neighbor['d1'] >= margin:
        return neighbor['name']
    return None


def score(samples: list[Sample], neighbors: list[dict], threshold: float, margin: float) -> dict:
    correct = named = 0
    heroes = {}
    for sample, neighbor in zip(samples, neighbors, strict=True):
        name = decision(neighbor, threshold, margin)
        row = heroes.setdefault(sample.name, dict(samples=0, named=0, correct=0, reference_available=0))
        row['samples'] += 1
        row['named'] += name is not None
        row['correct'] += name == sample.name
        row['reference_available'] += neighbor['reference_available']
        named += name is not None
        correct += name == sample.name
    for row in heroes.values():
        row['coverage'] = row['correct'] / row['samples']
    return dict(threshold=threshold, margin=margin, samples=len(samples), named=named, correct=correct,
                accuracy=correct / named if named else None, coverage=correct / len(samples) if samples else None,
                per_hero=dict(sorted(heroes.items())))


def select_parameters(results: list[dict]) -> dict | None:
    qualified = [r for r in results if r['accuracy'] is not None and r['accuracy'] >= .98]
    return min(qualified, key=lambda r: (-r['coverage'], r['threshold'], -r['margin'])) if qualified else None


def memory_add(gallery, name, fp, crop, copy_jpg=None) -> bool:
    """仅保留 reader 自动学习的运行时效果，不向任何图库写测试裁剪/向量。"""
    if not gallery.allows(name):
        return False
    gallery.items.append((name, fp))
    return True


def extract(db_path: Path, samples_root: Path, ocr: str, cache: Path) -> tuple[list[Sample], list[dict]]:
    from jccbrain.data.db import GameDB
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.units import feature_print, unit_crop
    from jccbrain.vision.pyeye import build_reader

    truth = load_truth()
    vision_truth = load_truth(ROOT / 'tests/fixtures/vision_truth.jsonl')
    pool = {row['name'] for row in pool_heroes(db_path)}
    validate_truth(truth, pool)
    samples, frames = [], []
    # 临时数据库和空图库仅在本工作树的评测缓存目录下创建。
    with tempfile.TemporaryDirectory(dir=cache, prefix='replay-') as temporary:
        temporary = Path(temporary)
        for sequence, directory in [('standard', samples_root), ('track', samples_root / 'track')]:
            db_copy = temporary / f'{sequence}.sqlite3'
            shutil.copyfile(db_path, db_copy)
            gallery = temporary / sequence
            # 预建赛季目录，避免 build_reader 从其他目录导入旧图库。
            (gallery / 's18_S19').mkdir(parents=True)
            reader = build_reader(GameDB(db_copy), gallery, ocr)
            reader.units.gallery.add = lambda name, fp, crop, copy_jpg=None: memory_add(
                reader.units.gallery, name, fp, crop, copy_jpg)
            try:
                for file in sorted(directory.glob('*.jpg')):
                    frame = file.relative_to(samples_root).as_posix()
                    image = load_image(file)
                    state = reader.read(image)
                    fold = fold_key(frame, state, vision_truth)
                    stale = bool(state.get('units_stale'))
                    expected = truth.get(frame, {}).get('units', [])
                    if not stale:
                        units = reader.units.last_units
                        positions = [dict(zone=u.zone, slot=u.slot, x=u.x, y=u.y) for u in units]
                        pairs, missing, extra = match_units(expected, positions)
                        for i, j in pairs:
                            wanted, unit = expected[i], units[j]
                            if wanted['name'] in pool:
                                samples.append(Sample(frame, fold, wanted['name'], unit.zone, unit.slot,
                                                      unit.fp, unit.crop))
                    else:
                        # F1 的定义来自 #4a：固定备战席位置的当前 crop 对 live 整框 fp。
                        units = reader.units.last_live[0] if reader.units.last_live else []
                        positions = [dict(zone='bench', slot=u.slot) for u in units]
                        pairs, missing, extra = match_units(expected, positions)
                        for i, j in pairs:
                            wanted, unit = expected[i], units[j]
                            if wanted['name'] in pool:
                                crop = unit_crop(image, unit.bar) if unit.bar is not None else None
                                fp = feature_print(crop) if crop is not None else None
                                samples.append(Sample(frame, fold, wanted['name'], 'bench', unit.slot,
                                                      fp, crop, 'combat'))
                    frames.append(dict(frame=frame, fold=fold, stale=stale,
                                       truth_units=len(expected), matched_units=len(pairs),
                                       unmatched_truth=len(missing), unmatched_prediction=len(extra)))
                    print(f'{frame}: fold={fold}, {"stale F1" if stale else "live"}, matched={len(pairs)}', flush=True)
            finally:
                if worker_pool := getattr(reader.ocr, 'pool', None):
                    worker_pool.shutdown(wait=True)
            if list(gallery.rglob('*.*')):
                raise RuntimeError('提取阶段意外向图库写入文件')
    return samples, frames


def save_cache(cache: Path, samples: list[Sample], frames: list[dict], provenance: dict) -> None:
    arrays, rows = {}, []
    for i, sample in enumerate(samples):
        row = {k: v for k, v in vars(sample).items() if k not in ('fp', 'crop')}
        for kind in ['fp', 'crop']:
            value = getattr(sample, kind)
            row[kind] = f'{kind}_{i}' if value is not None else None
            if value is not None:
                arrays[row[kind]] = value
        rows.append(row)
    np.savez_compressed(cache / 'features.npz', **arrays)
    (cache / 'samples.json').write_text(json.dumps(dict(samples=rows, frames=frames, provenance=provenance),
                                                  ensure_ascii=False, indent=2), 'utf-8')


def read_cache(cache: Path, provenance: dict) -> tuple[list[Sample], list[dict]]:
    saved = json.loads((cache / 'samples.json').read_text('utf-8'))
    if saved['provenance'] != provenance:
        raise ValueError('缓存来源不符，请去掉 --reuse-cache 重新提取')
    with np.load(cache / 'features.npz', allow_pickle=False) as arrays:
        samples = [Sample(**dict(row, fold=tuple(row['fold']),
                                fp=arrays[row['fp']] if row['fp'] else None,
                                crop=arrays[row['crop']] if row['crop'] else None)) for row in saved['samples']]
    return samples, saved['frames']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', type=Path, required=True)
    ap.add_argument('--samples', type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--ocr', default='rapid')
    ap.add_argument('--reuse-cache', action='store_true')
    args = ap.parse_args()
    cache = args.cache.resolve()
    cache.relative_to(DEFAULT_CACHE.parent.resolve())  # 不允许把特征缓存写到图库/主目录。
    cache.mkdir(parents=True, exist_ok=True)
    provenance = dict(samples=str(args.samples.resolve()), ocr=args.ocr, version=1,
                      hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                          args.db, ROOT / 'tests/fixtures/unit_truth.jsonl', ROOT / 'tests/fixtures/vision_truth.jsonl']})
    samples, frames = read_cache(cache, provenance) if args.reuse_cache else extract(
        args.db, args.samples, args.ocr, cache)
    if not args.reuse_cache:
        save_cache(cache, samples, frames, provenance)
    live, combat = [s for s in samples if s.kind == 'live'], [s for s in samples if s.kind == 'combat']
    neighbors = loo_neighbors(live)
    scan = [score(live, neighbors, t, m) for t in THRESHOLDS for m in MARGINS]
    selected = select_parameters(scan)
    folds = []
    for fold in sorted({tuple(f['fold']) for f in frames}):
        group_frames = [f for f in frames if tuple(f['fold']) == fold]
        group = [s for s in live if s.fold == fold]
        folds.append(dict(fold=fold, frames=len(group_frames), live_frames=sum(not f['stale'] for f in group_frames),
                          live_units=len(group), combat_units=sum(s.fold == fold for s in combat),
                          missing_fp=sum(s.fp is None for s in group), missing_crop=sum(s.crop is None for s in group)))
    ranked = sorted([r for r in scan if r['accuracy'] is not None and r['accuracy'] >= .98],
                    key=lambda r: (-r['coverage'], r['threshold'], -r['margin']))
    combat_neighbors = [nearest(s, live) for s in combat]
    result = dict(provenance=provenance, folds=folds, live_samples=len(live), combat_samples=len(combat),
                  reference_available=sum(n['reference_available'] for n in neighbors),
                  old=score(live, neighbors, .55, 0), selected=selected, scan=scan,
                  combat=score(combat, combat_neighbors, selected['threshold'], selected['margin']) if selected else None,
                  combat_scope='stale bench fixed-position current whole crop (F1); training=all live folds',
                  missing_combat_fp=sum(s.fp is None for s in combat),
                  missing_combat_crop=sum(s.crop is None for s in combat),
                  cross_fold_identical_fp=sum(any(other.fold != s.fold and other.fp is not None and s.fp is not None
                                                  and np.array_equal(s.fp, other.fp) for other in live) for s in live))
    # JSON 中无穷距离用 null，保留每个单位 d1/d2，便于检查错误/拒绝。
    def details(queries, nearest_rows):
        return [dict(frame=s.frame, fold=s.fold, zone=s.zone, slot=s.slot, expected=s.name,
                     **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in n.items()}) for s, n in zip(queries, nearest_rows, strict=True)]

    result['decisions'] = details(live, neighbors)
    result['combat_decisions'] = details(combat, combat_neighbors)
    (cache / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    top10 = [{k: v for k, v in row.items() if k != 'per_hero'} for row in ranked[:10]]
    for label, value in [('folds', folds), ('top10', top10), ('old', result['old']),
                         ('selected', selected), ('combat', result['combat'])]:
        print(label, json.dumps(value, ensure_ascii=False))
    print(f'结果: {cache / "result.json"}; 仅同一局约11英雄，外推有限。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
