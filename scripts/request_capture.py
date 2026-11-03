"""向已在有权限会话运行的截图代办请求一帧；不启动代办，不自动重发。

.venv/bin/python scripts/request_capture.py heroes --stage 2-3 --notes '点击面板确认：卡蜜尔'
超时后按报出的id继续查 capture-results.jsonl，不换id自动重试。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_samples import SCENES  # noqa: E402


def request_capture(root: Path, scene: str, stage: str, notes: str, *, request_id=None, timeout=10.):
    if scene not in SCENES or not 0 < timeout <= 10:
        raise ValueError('scene无效或等待时间不在(0,10]秒内')
    row = dict(id=request_id or str(uuid.uuid4()), scene=scene, stage=stage, notes=notes)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'capture-requests.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')
        file.flush()
        os.fsync(file.fileno())
    deadline = time.monotonic() + timeout
    path = root / 'capture-results.jsonl'
    while True:
        if path.exists():
            with path.open(encoding='utf-8') as file:
                for line in file:
                    if not line.endswith('\n'):
                        break
                    result = json.loads(line)
                    if result.get('id') == row['id']:
                        return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"等待截图结果超时；请求id={row['id']}，请继续查原id，未重发")
        time.sleep(min(.1, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', choices=SCENES)
    parser.add_argument('--stage', default='unknown')
    parser.add_argument('--notes', default='')
    args = parser.parse_args()
    try:
        row = request_capture(ROOT / 'var/vision-module-optimization', args.scene, args.stage, args.notes)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
