"""顺序循环回放截图，统计包含帧间等待的进程 CPU（100%=一个核）。

    .venv/bin/python scripts/bench_cpu.py --samples /path/to/samples/track --seconds 30 --interval .25
    .venv/bin/python scripts/bench_cpu.py --samples /path/to/samples --seconds 30 --adaptive

截图经 CoreGraphics 预读，初始化和预热不进入测量；间隔按两次读取开始时间算，
单帧均值仅计 reader.read。CPU 用 process_time / 墙钟时间，包含等待时 ORT 忙等。
不会写正式图库/数据库，只在本工作树 var 下创建自动清除的临时副本。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=30.)
    ap.add_argument('--interval', type=float, default=.25)
    ap.add_argument('--adaptive', action='store_true')
    ap.add_argument('--ort-threads', type=int, choices=[0, 1, 2, 4],
                    help='候选1/2/4；0配合--spin对照原来的ORT默认自动线程')
    ap.add_argument('--spin', action='store_true', help='启用 ORT 忙等作对照')
    ap.add_argument('--db', type=Path, default=Path('/tmp/jcc-vf-db.sqlite3'))
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    if args.seconds <= 0 or args.interval < 0:
        ap.error('--seconds 必须大于0，--interval 不能为负')

    from jccbrain.data.db import GameDB
    from jccbrain.vision.eye import rapid
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader, frame_interval

    files = sorted(args.samples.glob('*.jpg'))
    if not files:
        ap.error('样本目录没有 jpg')
    images = [load_image(file) for file in files]
    cache = ROOT / 'var/vision-module-optimization/cpu'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bench-', dir=cache) as temporary:
        temporary = Path(temporary)
        db = temporary / 'db.sqlite3'
        shutil.copyfile(args.db, db)
        gallery = temporary / 'gallery'
        (gallery / 's18_S19').mkdir(parents=True)
        # 只在构造本次 reader 时覆盖参数；不改设置文件或其他 reader。
        override = patch.object(rapid, 'RapidOCR', partial(rapid.RapidOCR, ort_threads=args.ort_threads,
                                                         spin=args.spin)) if args.ort_threads is not None or args.spin else nullcontext()
        with override:
            reader = build_reader(GameDB(db), gallery, 'rapid')
        try:
            opts = reader.ocr.det.infer.session.get_session_options()
            try:
                spinning = opts.get_session_config_entry('session.intra_op.allow_spinning')
            except RuntimeError:
                spinning = 'default'
            reader.ocr.warmup()
            frames, read_seconds, phase = 0, 0., None
            started = time.perf_counter()
            cpu_started = time.process_time()
            deadline, previous_start = started + args.seconds, None
            while time.perf_counter() < deadline:
                interval = frame_interval(phase, args.interval) if args.adaptive else args.interval
                if previous_start is not None:
                    wait = min(previous_start + interval, deadline) - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    if time.perf_counter() >= deadline:
                        break
                previous_start = time.perf_counter()
                state = reader.read(images[frames % len(images)])
                read_seconds += time.perf_counter() - previous_start
                phase = state.get('phase')
                frames += 1
            elapsed_cpu = time.process_time() - cpu_started
            wall = time.perf_counter() - started
            result = dict(frames=frames, fps=frames / wall, mean_ms=read_seconds * 1000 / frames if frames else None,
                          cpu_percent=elapsed_cpu / wall * 100, cpu_seconds=elapsed_cpu, wall_seconds=wall,
                          interval=args.interval, adaptive=args.adaptive, ort_threads=opts.intra_op_num_threads,
                          spin=spinning, samples=str(args.samples.resolve()), initialization_included=False)
        finally:
            reader.ocr.pool.shutdown(wait=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
