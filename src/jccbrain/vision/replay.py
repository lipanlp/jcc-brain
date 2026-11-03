"""回放后端：逐行读取 game-eye 输出的 jsonl（`game-eye file samples/*.jpg > x.jsonl` 即可生成）。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from jccbrain.core.models import GameState
from jccbrain.vision.base import VisionError, VisionProvider


class ReplayProvider(VisionProvider):
    name = "replay"

    def __init__(self, path: str | Path, interval_s: float = 0.8, loop: bool = True):
        self.path = Path(path)
        self.interval_s = interval_s
        self.loop = loop
        self._states: list[dict] = []
        self._i = -1
        self._lock = threading.Lock()
        self._last_t = 0.0

    def start(self) -> None:
        if self._states:
            return
        if not self.path.is_file():
            raise VisionError(f"回放文件不存在：{self.path}")
        self._states = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self._states:
            raise VisionError(f"回放文件为空：{self.path}")

    def read(self, fresh: bool = True) -> GameState:
        self.start()
        with self._lock:
            if fresh or self._i < 0:
                wait = self.interval_s - (time.monotonic() - self._last_t)
                if wait > 0 and self._i >= 0:
                    time.sleep(wait)
                self._i += 1
                if self._i >= len(self._states):
                    if not self.loop:
                        raise VisionError("回放结束")
                    self._i = 0
                self._last_t = time.monotonic()
            return GameState.from_dict(self._states[self._i])

    def status(self) -> dict:
        return {"backend": self.name, "file": str(self.path), "frame": self._i, "total": len(self._states)}
