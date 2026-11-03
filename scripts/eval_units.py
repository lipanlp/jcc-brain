"""单位名评测：只读卡池、临时图库副本、位置一一对齐与独立的名字指标。

不调用识别器生成答案；unit_truth.jsonl 的名字必须人工核对。
英雄准确率/覆盖率都排除 ? 和 PVE 野怪，野怪误名、越出卡池及位置遗漏另计。
"""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

SEASON_ID = '18_S19'
TRUTH_FILE = Path(__file__).resolve().parents[1] / 'tests/fixtures/unit_truth.jsonl'


def pool_heroes(db_path: Path, season_id: str = SEASON_ID) -> list[dict]:
    """与 GameDB.pool_heroes 一致：当前赛季 in_pool=1、费用1～5，按名字去重。

    直接以 mode=ro 打开冻结副本，不实例化会执行迁移的 GameDB，不写数据库。
    """
    with sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT id, name, cost, traits FROM heroes '
                            'WHERE season_id = ? AND in_pool = 1 AND cost BETWEEN 1 AND 5 '
                            'ORDER BY cost, name, id', (season_id,)).fetchall()
    return list({row['name']: dict(row) for row in rows}.values())


def load_truth(path: Path = TRUTH_FILE) -> dict[str, dict]:
    frames = {}
    for line in path.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['_file'] in frames:
            raise ValueError(f"单位真值帧重复: {row['_file']}")
        frames[row['_file']] = row
    return frames


def validate_truth(truth: dict[str, dict], pool: set[str]) -> None:
    for file, frame in truth.items():
        for unit in frame['units']:
            if unit['name'] not in pool | {'?', '野怪'}:
                raise ValueError(f"单位真值含卡池外名字: {file} {unit['name']}")


@contextmanager
def copied_gallery(source: Path | None):
    """每次回放从同一输入重新复制；自动标注只能写临时目录，绝不写回 source。"""
    with tempfile.TemporaryDirectory(prefix='jcc-eval-gallery-') as temporary:
        destination = Path(temporary) / 'gallery'
        if source is not None:
            shutil.copytree(source, destination)
        try:
            yield destination
        finally:
            # 临时副本销毁前打印本遍迁移结果，便于验证；不向输入图库写回。
            for file in sorted(destination.glob('s*/migration.json')):
                migration = json.loads(file.read_text('utf-8'))
                print(f"单位图库迁移 {file.parent.name}: {json.dumps(migration, ensure_ascii=False)}")


def gallery_summary(source: Path | None, pool: set[str]) -> dict:
    from jccbrain.core import paths

    index = source / 'index.json' if source is not None else None
    rows = json.loads(index.read_text('utf-8')) if index is not None and index.exists() else []
    names = dict(sorted(Counter(row['name'] for row in rows).items()))
    legacy = paths.legacy_gallery_dir()
    return {'source': str(source) if source is not None else None, 'entries': len(rows), 'names': names,
            'outside_pool': {name: n for name, n in names.items() if name not in pool},
            'legacy_gallery': str(legacy) if legacy is not None else None}


def match_units(expected: list[dict], predicted: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """备战席同 slot；棋盘按最近距离配对，双方各用一次。

    棋盘门限是窗口比例 |dx|<.02、|dy|<.03（任务规定），与遍历顺序无关。
    """
    candidates = []
    for i, wanted in enumerate(expected):
        for j, found in enumerate(predicted):
            if wanted['zone'] != found['zone']:
                continue
            if wanted['zone'] == 'bench':
                if wanted.get('slot') is not None and wanted.get('slot') == found.get('slot'):
                    candidates.append((0., i, j))
            elif all(k in wanted and k in found for k in ['x', 'y']):
                dx, dy = abs(wanted['x'] - found['x']), abs(wanted['y'] - found['y'])
                if dx < .02 and dy < .03:
                    candidates.append((math.hypot(dx, dy), i, j))
    used_expected, used_predicted, matches = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_expected and j not in used_predicted:
            matches.append((i, j))
            used_expected.add(i)
            used_predicted.add(j)
    return matches, [i for i in range(len(expected)) if i not in used_expected], [
        j for j in range(len(predicted)) if j not in used_predicted]


def _named(unit: dict | None) -> str | None:
    name = str((unit or {}).get('name') or '').strip()
    return name if name and not name.startswith('?') else None


def evaluate(predictions: dict[str, dict], truth: dict[str, dict], pool: set[str]) -> dict:
    """predictions 的键与真值相同，使用相对 samples 根目录的路径，不能只用 basename。"""
    counts = dict(heroes=0, wild=0, unknown=0, named=0, correct=0, named_predictions=0,
                  outside_pool=0, wild_as_hero=0, unmatched_truth=0, unmatched_prediction=0)
    errors, frames = [], 0
    for file, state in predictions.items():
        if file not in truth:
            continue
        frames += 1
        expected = truth[file]['units']
        predicted = [dict(unit, zone=zone) for zone in ['bench', 'board'] for unit in state.get(zone, [])]
        matches, missing, extra = match_units(expected, predicted)
        by_expected = dict(matches)
        counts['unmatched_truth'] += len(missing)
        counts['unmatched_prediction'] += len(extra)
        for unit in predicted:
            name = _named(unit)
            counts['named_predictions'] += name is not None
            counts['outside_pool'] += name is not None and name not in pool
        for i, wanted in enumerate(expected):
            found = predicted[by_expected[i]] if i in by_expected else None
            name = _named(found)
            reasons = ['unmatched_truth'] if found is None else []
            if wanted['name'] == '?':
                counts['unknown'] += 1
            elif wanted['name'] == '野怪':
                counts['wild'] += 1
                if name in pool:
                    counts['wild_as_hero'] += 1
                    reasons.append('wild_as_hero')
            else:
                counts['heroes'] += 1
                counts['named'] += name is not None
                counts['correct'] += name == wanted['name']
                if name != wanted['name']:
                    reasons.append('unnamed' if name is None else 'wrong_name')
            if name is not None and name not in pool:
                reasons.append('outside_pool')
            if reasons:
                errors.append({'_file': file, 'expected': wanted, 'predicted': found, 'reasons': reasons})
        for j in extra:
            unit = predicted[j]
            reasons = ['unmatched_prediction']
            if _named(unit) is not None and _named(unit) not in pool:
                reasons.append('outside_pool')
            errors.append({'_file': file, 'expected': None, 'predicted': unit, 'reasons': reasons})
    return report(frames, counts, errors)


def report(frames: int, counts: dict, errors: list) -> dict:
    return {'frames': frames, 'counts': counts,
            'accuracy': counts['correct'] / counts['named'] if counts['named'] else None,
            'coverage': counts['correct'] / counts['heroes'] if counts['heroes'] else None,
            'unmatched_units': counts['unmatched_truth'] + counts['unmatched_prediction'], 'errors': errors}


def combine(reports: list[dict]) -> dict:
    counts = {key: sum(row['counts'][key] for row in reports) for key in reports[0]['counts']}
    errors = [dict(error, run=run) for run, row in enumerate(reports, 1) for error in row['errors']]
    return dict(report(reports[0]['frames'], counts, errors), repeat=len(reports),
                frame_evaluations=sum(row['frames'] for row in reports))


def frame_key(file: Path, samples_root: Path) -> str:
    return file.resolve().relative_to(samples_root.resolve()).as_posix()


def summary(result: dict) -> str:
    def percent(value):
        return f'{value:.2%}' if value is not None else 'N/A'

    c = result['counts']
    return (f"单位名：准确率 {percent(result['accuracy'])}（{c['correct']}/{c['named']}），"
            f"覆盖率 {percent(result['coverage'])}（{c['correct']}/{c['heroes']}）；"
            f"卡池外名字 {c['outside_pool']}，野怪误名 {c['wild_as_hero']}，"
            f"位置未匹配 {result['unmatched_units']}（答案{c['unmatched_truth']}/预测{c['unmatched_prediction']}）")
