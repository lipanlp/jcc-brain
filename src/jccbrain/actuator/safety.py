"""安全保护：任何自动操作执行前都要过 SafetyGuard.check()。

- 急停：UI 按钮 / 菜单 / 快捷键调用 stop()
- 甩角落：鼠标移到主屏左上角（5 像素内）立即停止，和 pyautogui 的 failsafe 一样
- 限速：每秒最多 N 个操作
- 窗口检查：找不到游戏窗口时不操作
"""
from __future__ import annotations

import threading
import time
from collections import deque


DEFAULT_MAX_ACTIONS_PER_SEC = 6.0


class AbortError(RuntimeError):
    """自动操作被中止。"""


class SafetyGuard:
    def __init__(self, max_actions_per_sec: float = DEFAULT_MAX_ACTIONS_PER_SEC, failsafe_corner: bool = True):
        self.max_actions_per_sec = max_actions_per_sec
        self.failsafe_corner = failsafe_corner
        self._stop = threading.Event()
        self._recent: deque[float] = deque()
        self.reason = ""

    def stop(self, reason: str = "手动停止") -> None:
        self.reason = reason
        self._stop.set()

    def reset(self) -> None:
        self.reason = ""
        self._stop.clear()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self.failsafe_corner:
            from jccbrain.actuator.window import mouse_location
            x, y = mouse_location()
            if x <= 5 and y <= 5:
                self.stop("鼠标移到左上角，触发急停")
        if self._stop.is_set():
            raise AbortError(self.reason or "已停止")
        # 限速
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 1.0:
            self._recent.popleft()
        limit = self.max_actions_per_sec
        if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not limit > 0:
            limit = DEFAULT_MAX_ACTIONS_PER_SEC     # 0 / 负数 / 非数字不当作「不限速」，回到默认限速
        if len(self._recent) >= limit:
            self.sleep(max(1.0 - (now - self._recent[0]), 0))
        if self._stop.is_set():
            raise AbortError(self.reason or "已停止")
        self._recent.append(time.monotonic())

    def sleep(self, seconds: float) -> None:
        """可被急停打断的等待。"""
        if self._stop.wait(seconds):
            raise AbortError(self.reason or "已停止")
