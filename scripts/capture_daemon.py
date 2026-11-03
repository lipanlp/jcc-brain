"""由有录屏权限的会话运行的截图代办；只读游戏，不发送输入。

.venv/bin/python scripts/capture_daemon.py
向 var/vision-module-optimization/capture-requests.jsonl 追加带换行的
{"id":"唯一请求ID","scene":"heroes","stage":"2-3","notes":"卡蜜尔信息面板"}。
对应结果在 capture-results.jsonl。请求文件只追加；客户端最多等10秒后按id查结果，
不要自动换id重发超时请求。--once只处理当前完整行，便于离线检查。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.capture_samples import SCENES  # noqa: E402


def append_json(path: Path, row: dict):
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')
        file.flush()
        os.fsync(file.fileno())


class CaptureDaemon:
    """偏移+在途日志提供至多一次截图；替身capture与backend用于无权限测试。"""
    def __init__(self, root: Path, capture, backend):
        self.root = root.resolve()
        self.capture, self.backend = capture, backend
        self.requests = self.root / 'capture-requests.jsonl'
        self.results = self.root / 'capture-results.jsonl'
        self.state_file = self.root / 'capture-daemon-state.json'
        self._lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / 'capture-daemon.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else dict(offset=0)
            self.completed = set()
            # 崩溃留下的半行不是已提交结果；截掉后从inflight恢复，避免拼成坏JSON。
            if self.results.exists():
                with self.results.open('r+b') as file:
                    while line := file.readline():
                        if not line.endswith(b'\n'):
                            file.truncate(file.tell() - len(line))
                            break
                        row = json.loads(line)
                        if isinstance(row.get('id'), str):
                            self.completed.add(row['id'])
            self._recover()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _save_state(self):
        temporary = self.state_file.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as file:
            json.dump(self.state, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.state_file)

    def _advance(self, offset):
        self.state['offset'] = offset
        self.state.pop('inflight', None)
        self._save_state()

    def _manifest(self, pending):
        path = Path(pending['file'])
        manifest = path.parent / 'manifest.jsonl'
        if manifest.exists():
            # 与结果日志一样修复进程中断时留下的半行。
            with manifest.open('r+b') as file:
                while line := file.readline():
                    if not line.endswith(b'\n'):
                        file.truncate(file.tell() - len(line))
                        break
                    if json.loads(line).get('id') == pending['request']['id']:
                        return
        append_json(manifest, dict(pending['request'], file=path.name, **pending['metadata']))

    def _finish(self, pending, error=None):
        request, path = pending['request'], pending.get('file')
        if error is None:
            self._manifest(pending)
        row = dict(id=request.get('id'), ok=error is None,
                   file=path if error is None else None,
                   backend=pending.get('backend', self.backend()), error=error)
        append_json(self.results, row)
        if isinstance(row['id'], str):
            self.completed.add(row['id'])
        self._advance(pending['offset'])
        return row

    def _recover(self):
        pending = self.state.get('inflight')
        if pending is None:
            return
        if pending['request'].get('id') in self.completed:
            self._advance(pending['offset'])
        elif pending.get('file') and Path(pending['file']).is_file():
            self._finish(pending)
        else:
            self._finish(pending, '上次进程在处理该请求时中断；为避免重复截图，未自动重试')

    @staticmethod
    def _validate(row):
        if not isinstance(row, dict):
            raise ValueError('请求必须是JSON对象')
        if not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError('id必须是非空字符串')
        if row.get('scene') not in SCENES:
            raise ValueError(f'scene必须为{SCENES}之一')
        if any(not isinstance(row.get(key, ''), str) for key in ('stage', 'notes')):
            raise ValueError('stage/notes必须是字符串')

    def process_pending(self):
        if self._lock is None:
            raise RuntimeError('必须在with CaptureDaemon内处理请求')
        if not self.requests.exists():
            return []
        output = []
        with self.requests.open('rb') as file:
            stat = os.fstat(file.fileno())
            source = [stat.st_dev, stat.st_ino]
            if self.state.get('source') != source or stat.st_size < self.state['offset']:
                self.state.update(source=source, offset=0)
                self._save_state()
            file.seek(self.state['offset'])
            while line := file.readline():
                if not line.endswith(b'\n'):
                    break
                offset = file.tell()
                row = {}
                try:
                    row = json.loads(line)
                    self._validate(row)
                except (ValueError, UnicodeError) as exc:
                    row = row if isinstance(row, dict) else {}
                    output.append(self._finish(dict(request=row, offset=offset), str(exc)))
                    continue
                if row['id'] in self.completed:
                    self._advance(offset)
                    continue
                destination = self.root / 'replay-samples' / row['scene']
                destination.mkdir(parents=True, exist_ok=True)
                sequence = max((int(p.stem) for p in destination.glob('*.png') if p.stem.isdigit()), default=0) + 1
                path = destination / f'{sequence:04d}.png'
                pending = dict(request=row, offset=offset, file=str(path), backend=self.backend())
                self.state['inflight'] = pending
                self._save_state()  # 必须在原生截图之前落盘，重启绝不重新截图。
                try:
                    image, window = self.capture()
                    pending.update(backend=self.backend(), metadata=dict(
                        timestamp=datetime.now().astimezone().isoformat(), width=image.shape[1],
                        height=image.shape[0], window_id=window.window_id))
                    self._save_state()
                    temporary = path.with_suffix('.tmp.png')
                    if not cv2.imwrite(str(temporary), image):
                        raise OSError(f'写入图片失败：{path}')
                    with temporary.open('rb') as image_file:
                        os.fsync(image_file.fileno())
                    os.replace(temporary, path)
                except Exception as exc:
                    pending['backend'] = self.backend()
                    output.append(self._finish(pending, str(exc)))
                else:
                    output.append(self._finish(pending))
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'var/vision-module-optimization')
    parser.add_argument('--poll', type=float, default=.1, help='轮询间隔秒，默认0.1')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll <= 0:
        parser.error('--poll必须大于0')
    from jccbrain.actuator.window import WindowLocator
    from jccbrain.vision.eye.capture import WindowCapture
    capture = WindowCapture(WindowLocator('com.tencent.jkchess'))
    try:
        with CaptureDaemon(args.root, capture.grab, lambda: capture.backend) as daemon:
            while True:
                for row in daemon.process_pending():
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if args.once:
                    break
                time.sleep(args.poll)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        parser.exit(1, f'截图代办停止：{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
