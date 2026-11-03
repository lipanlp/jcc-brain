"""由有截图权限的会话运行；比较CG/SC/CLI单张截图，不启动视频流。

    .venv/bin/python scripts/capture_check.py --backend both --n 20 --json var/capture-check.json
both/all模式并发请求同一游戏窗口，另记调用起始间隔；动态画面像素差不等于色彩误差。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from jccbrain.vision.eye.capture import CaptureError, CaptureTimeout, WindowCapture  # noqa: E402
from jccbrain.actuator.window import WindowLocator  # noqa: E402


def attempt(capture):
    started = time.perf_counter()
    try:
        image, win = capture.grab()
        result = dict(ok=True, image=image, window_id=win.window_id, shape=list(image.shape), error=None, timeout=False)
    except CaptureError as exc:
        result = dict(ok=False, image=None, window_id=None, shape=None, error=str(exc),
                      timeout=isinstance(exc, CaptureTimeout))
    return dict(result, started=started, ms=(time.perf_counter() - started) * 1000)


def check(captures: dict, n: int) -> dict:
    attempts = {name: [] for name in captures}
    pairs = []
    with ThreadPoolExecutor(max_workers=len(captures)) as pool:
        for _ in range(n):
            pending = {name: pool.submit(attempt, capture) for name, capture in captures.items()}
            rows = {name: future.result() for name, future in pending.items()}
            for first, second in combinations(rows, 2):
                a, b = rows[first], rows[second]
                same_window = a['ok'] and b['ok'] and a['window_id'] == b['window_id']
                same_size = a['ok'] and b['ok'] and a['shape'] == b['shape']
                mad = float(np.abs(a['image'].astype(np.int16) - b['image'].astype(np.int16)).mean()) if same_window and same_size else None
                pairs.append(dict(backends=[first, second], same_window=bool(same_window), same_size=bool(same_size), mean_abs_pixel_diff=mad,
                                  start_gap_ms=abs(a['started'] - b['started']) * 1000))
            for name, row in rows.items():
                attempts[name].append({k: v for k, v in row.items() if k not in ('image', 'started')})
    stats = {}
    for name, rows in attempts.items():
        successful = [row['ms'] for row in rows if row['ok']]
        stats[name] = dict(attempts=n, successes=len(successful), success_rate=len(successful) / n,
                           mean_ms=statistics.mean(row['ms'] for row in rows), max_ms=max(row['ms'] for row in rows),
                           success_mean_ms=statistics.mean(successful) if successful else None,
                           timeouts=sum(row['timeout'] for row in rows))
    return dict(backends=stats, pairs=pairs, attempts=attempts,
                note='both/all为并发请求，逐对比较，不保证同一渲染帧；像素差包含动画和调用时间差。')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', choices=['cg', 'sc', 'cli', 'both', 'all'], default='both')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    if args.n <= 0:
        ap.error('--n必须大于0')
    locator = WindowLocator('com.tencent.jkchess')
    names = ['cg', 'sc', 'cli'] if args.backend == 'all' else ['cg', 'sc'] if args.backend == 'both' else [args.backend]
    report = check({name: WindowCapture(locator, backend=name) for name in names}, args.n)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if all(row['successes'] == args.n for row in report['backends'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
