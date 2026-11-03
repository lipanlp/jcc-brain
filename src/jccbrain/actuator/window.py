"""定位游戏窗口（Quartz 窗口列表），并把相对坐标换算成屏幕坐标（点）。"""
from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowInfo:
    owner: str
    title: str
    pid: int
    x: float
    y: float
    width: float
    height: float
    window_id: int = 0

    def to_screen(self, rx: float, ry: float) -> tuple[float, float]:
        return (self.x + rx * self.width, self.y + ry * self.height)

    def to_relative(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.x) / self.width, (sy - self.y) / self.height)


class WindowLocator:
    """按进程名（包含匹配）找游戏窗口，结果缓存 1 秒（窗口可能被拖动）。"""

    def __init__(self, owner: str = "金铲铲"):
        self.owner = owner
        self._cache: tuple[float, WindowInfo | None] = (0.0, None)

    def find(self, max_age_s: float = 1.0) -> WindowInfo | None:
        t, info = self._cache
        if info and time.monotonic() - t < max_age_s:
            return info
        info = self._query()
        self._cache = (time.monotonic(), info)
        return info

    def _query(self) -> WindowInfo | None:
        import Quartz

        opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        best = None
        for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
            owner = w.get("kCGWindowOwnerName") or ""
            title = w.get("kCGWindowName") or ""
            pid = int(w.get("kCGWindowOwnerPID", 0))
            # 打包后本进程名「金铲铲大脑」包含「金铲铲」，必须排除自身窗口。
            if pid == os.getpid():
                continue
            # 只按进程名 / 包名匹配：按标题会误中浏览器里标题带「金铲铲之战」的网页
            if w.get("kCGWindowLayer") != 0 or not (self.owner in owner or self.owner in _bundle_id(pid)):
                continue
            b = w.get("kCGWindowBounds")
            if float(b["Width"]) < 200:
                continue
            info = WindowInfo(owner, title, pid, float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]),
                              int(w.get("kCGWindowNumber", 0)))
            if best is None or info.width * info.height > best.width * best.height:
                best = info
        return best

    def activate(self) -> bool:
        """把游戏窗口切到前台。"""
        info = self.find(0)
        if not info:
            return False
        from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(info.pid)
        return bool(app and app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))


def _bundle_id(pid: int) -> str:
    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(app.bundleIdentifier() or "") if app else ""
    except Exception:  # noqa: BLE001 - 取不到包名时只按进程名匹配
        return ""


def accessibility_trusted() -> bool:
    """当前进程是否有「辅助功能」权限（发送鼠标事件需要）。"""
    try:
        lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except OSError:
        return False


def mouse_location() -> tuple[float, float]:
    import Quartz

    p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return (p.x, p.y)
