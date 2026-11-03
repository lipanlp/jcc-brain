"""复现 PyObjC 懒加载竞态：多个线程在新进程里同时首次调用 load_image，偶发 KeyError('符号名')。

    .venv/bin/python scripts/repro_decode_race.py tests/fixtures/frames/0003.jpg --runs 30
    .venv/bin/python scripts/repro_decode_race.py tests/fixtures/frames/0003.jpg --runs 30 --warm

每轮都是新子进程（符号只在首次访问时解析）。10-22 本机：不预热 9/30 轮失败，--warm 0/30。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading

THREADS = 8


def once(path: str, warm: bool) -> list[str]:
    from jccbrain.vision.eye.capture import load_image, warm_decoder

    if warm:
        warm_decoder()
    errors, barrier = [], threading.Barrier(THREADS)

    def run():
        barrier.wait()
        try:
            load_image(path)
        except Exception as error:
            errors.append(repr(error))

    threads = [threading.Thread(target=run) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        errors = once(args.image, args.warm)
        print(errors or "")
        return 1 if errors else 0
    failed = 0
    for _ in range(args.runs):
        cmd = [sys.executable, __file__, args.image, "--child"] + (["--warm"] if args.warm else [])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            failed += 1
            print(result.stdout.strip() or result.stderr.strip().splitlines()[-1])
    print(f"{failed}/{args.runs} 轮出现并发解码失败（{'已' if args.warm else '未'}预热）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
