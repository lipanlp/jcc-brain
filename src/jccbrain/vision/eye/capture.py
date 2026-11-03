"""按windowID截图：CG→ScreenCaptureKit→CLI；原生后端2秒，CLI5秒。

注意：不要用 ScreenCaptureKit 的视频流（SCStream），本机会报 -3805 并卡死后续截图（见 game-eye 清单）。
需要「录屏」权限。
"""
from __future__ import annotations

import logging
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from jccbrain.actuator.window import WindowInfo, WindowLocator


class CaptureError(RuntimeError):
    pass


# PyObjC 首次访问框架符号时才懒加载；多线程同时首次访问同一符号偶发抛出以符号名为内容的
# KeyError（10-22 启动日志与 8 线程复现脚本均可见）。并发解码前在单线程里先解析一遍。
_DECODE_SYMBOLS = ("CGImageSourceCreateWithURL", "CGImageSourceCreateImageAtIndex", "CGImageGetWidth",
                   "CGImageGetHeight", "CGBitmapContextCreate", "CGColorSpaceCreateDeviceRGB",
                   "kCGImageAlphaPremultipliedFirst", "kCGBitmapByteOrder32Little", "CGContextDrawImage",
                   "CGRectMake", "CGBitmapContextGetBytesPerRow", "CGBitmapContextGetData")
_decoder_lock = threading.Lock()
_decoder_ready = False


def warm_decoder() -> None:
    """预先解析 load_image / cgimage_to_bgr 用到的 Quartz/Foundation 符号，之后可多线程解码。"""
    global _decoder_ready
    with _decoder_lock:
        if _decoder_ready:
            return
        import Quartz
        from Foundation import NSURL

        for name in _DECODE_SYMBOLS:
            getattr(Quartz, name)
        NSURL.fileURLWithPath_
        _decoder_ready = True


class CaptureTimeout(CaptureError):
    def __init__(self):
        super().__init__('截图超时')


CAPTURE_TIMEOUT = 2.0
CLI_TIMEOUT = 5.0
CG_RETRY_INTERVAL = 60.0
log = logging.getLogger(__name__)


@dataclass
class _Job:
    call: Callable
    done: threading.Event = field(default_factory=threading.Event)
    value: object = None
    error: BaseException | None = None


class _CaptureWorker:
    """每后端一个daemon线程、至多一个在途请求；卡死时绝不排队或扩张线程池。"""
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._jobs = queue.Queue(maxsize=1)
        self._thread = None
        self._pending = None

    def _work(self):
        while True:
            job = self._jobs.get()
            try:
                job.value = job.call()
            except BaseException as exc:
                job.error = exc
            finally:
                job.done.set()

    def call(self, fn: Callable, timeout: float):
        with self._lock:
            if self._pending is not None and not self._pending.done.is_set():
                raise CaptureTimeout()
            job = self._pending = _Job(fn)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name=self.name, daemon=True)
                self._thread.start()
            self._jobs.put_nowait(job)
        if not job.done.wait(timeout):
            raise CaptureTimeout()
        if job.error is not None:
            if isinstance(job.error, CaptureError):
                raise job.error
            raise CaptureError(f'截图失败：{job.error}') from job.error
        return job.value


# 全进程共享：创建多个WindowCapture也不能绕过卡住的原生调用累积线程。
_CG_WORKER = _CaptureWorker('capture-cg')
_SC_WORKER = _CaptureWorker('capture-sc')
_CLI_WORKER = _CaptureWorker('capture-cli')


def _await_sc(register: Callable, deadline: float, convert: Callable = lambda value: value):
    """仅在SC后台线程等待回调。主调用由worker超时保护，迟到回调不返回旧图。

    回调始终没到时此请求保持在途，后续快速失败；不能超时后不断注册新回调。
    PyObjC需要run loop派发完成事件，轮询之间等待以避免忙等，并逐次清autorelease池。
    """
    done = threading.Event()
    result = {}

    def completion(value, error):
        try:
            if time.monotonic() >= deadline:
                raise CaptureTimeout()
            if error is not None:
                raise CaptureError(f'ScreenCaptureKit截图失败：{error}')
            if value is None:
                raise CaptureError('ScreenCaptureKit没有返回截图/窗口列表')
            result['value'] = convert(value)
        except BaseException as exc:
            result['error'] = exc
        finally:
            done.set()

    register(completion)
    if not done.is_set():
        import objc
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop

        loop = NSRunLoop.currentRunLoop()
        while not done.is_set():
            with objc.autorelease_pool():
                loop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(.01))
            done.wait(.01)
    if 'error' in result:
        raise result['error']
    if time.monotonic() >= deadline:
        raise CaptureTimeout()
    return result['value']


def cgimage_to_bgr(cg) -> np.ndarray:
    """CGImage → BGR。画进 DeviceRGB 位图（与 game-eye 的 Pixels 完全一致），
    让 CoreGraphics 做色彩空间转换，阈值才和实测值对得上。"""
    import Quartz

    w, h = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)
    ctx = Quartz.CGBitmapContextCreate(
        None, w, h, 8, 0, Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaPremultipliedFirst | Quartz.kCGBitmapByteOrder32Little)
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, w, h), cg)
    bpr = Quartz.CGBitmapContextGetBytesPerRow(ctx)
    buf = Quartz.CGBitmapContextGetData(ctx).as_buffer(bpr * h)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w]
    return np.ascontiguousarray(arr[..., :3])      # 内存顺序 B G R A


def load_image(path) -> np.ndarray:
    """读本地截图（ImageIO 解码 + 同样的色彩转换），用于回放与回归测试。"""
    import Quartz
    from Foundation import NSURL

    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None) if src else None
    if cg is None:
        raise CaptureError(f"读不了图片：{path}")
    return cgimage_to_bgr(cg)


class WindowCapture:
    def __init__(self, locator: WindowLocator, backend: str = 'auto'):
        if backend not in ('auto', 'cg', 'sc', 'cli'):
            raise ValueError('截图后端只能是auto/cg/sc/cli')
        self.locator = locator
        self.backend = 'cg' if backend == 'auto' else backend
        self._requested = backend
        self._cg_failures = 0
        self._sc_failures = 0
        self._last_cg_retry = 0.
        self._sc_window_id = None
        self._sc_window = None
        self._lock = threading.Lock()

    def _window(self) -> WindowInfo:
        import Quartz

        if not Quartz.CGPreflightScreenCaptureAccess():
            Quartz.CGRequestScreenCaptureAccess()
            raise CaptureError("没有录屏权限：系统设置 → 隐私与安全性 → 屏幕与系统音频录制，勾选运行本程序的终端或 金铲铲大脑.app")
        win = self.locator.find()
        if not win or not win.window_id:
            raise CaptureError(f"找不到游戏窗口（{self.locator.owner}），游戏是否已打开且未最小化？")
        return win

    def _cg_image(self, win: WindowInfo) -> np.ndarray:
        import objc
        import Quartz

        with objc.autorelease_pool():
            return self._cg_image_in_pool(Quartz, win)

    @staticmethod
    def _cg_image_in_pool(Quartz, win):
        cg = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, win.window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution)
        if cg is None:
            raise CaptureError("截图失败（窗口可能刚关闭或最小化）")
        return cgimage_to_bgr(cg)

    def _sc_image(self, win: WindowInfo, deadline: float) -> np.ndarray:
        import objc
        import ScreenCaptureKit as SC

        with objc.autorelease_pool():
            if self._sc_window_id != win.window_id:
                content = _await_sc(lambda callback:
                    SC.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                        True, True, callback), deadline)
                window = next((w for w in content.windows() if w.windowID() == win.window_id), None)
                if window is None:
                    raise CaptureError(f'ScreenCaptureKit找不到窗口：{win.window_id}')
                self._sc_window_id, self._sc_window = win.window_id, window
            content_filter = SC.SCContentFilter.alloc().initWithDesktopIndependentWindow_(self._sc_window)
            config = SC.SCStreamConfiguration.alloc().init()
            config.setWidth_(max(1, round(win.width)))
            config.setHeight_(max(1, round(win.height)))
            config.setShowsCursor_(False)
            config.setIgnoreShadowsSingleWindow_(True)
            config.setIgnoreGlobalClipSingleWindow_(True)
            try:
                return _await_sc(lambda callback:
                    SC.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
                        content_filter, config, callback), deadline, cgimage_to_bgr)
            except CaptureError as exc:
                if not isinstance(exc, CaptureTimeout):
                    self._sc_window_id, self._sc_window = None, None
                raise

    def _cli_image(self, win: WindowInfo) -> np.ndarray:
        import cv2

        with tempfile.TemporaryDirectory(prefix='jcc-capture-') as directory:
            file = Path(directory) / 'window.png'
            try:
                subprocess.run(['screencapture', '-x', '-o', '-l', str(win.window_id), str(file)],
                               check=True, timeout=CLI_TIMEOUT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except subprocess.TimeoutExpired as exc:
                raise CaptureTimeout() from exc
            except (OSError, subprocess.CalledProcessError) as exc:
                raise CaptureError(f'CLI截图失败：{exc}') from exc
            image = load_image(file)  # CoreGraphics解码/色彩空间，不用cv2.imread。
            width, height = max(1, round(win.width)), max(1, round(win.height))
            if image.shape[1] != width:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            return image

    def _image(self, backend: str, win: WindowInfo) -> np.ndarray:
        if backend == 'cli':
            return _CLI_WORKER.call(lambda: self._cli_image(win), CLI_TIMEOUT)
        deadline = time.monotonic() + CAPTURE_TIMEOUT
        worker = _CG_WORKER if backend == 'cg' else _SC_WORKER
        call = (lambda: self._cg_image(win)) if backend == 'cg' else (lambda: self._sc_image(win, deadline))
        return worker.call(call, CAPTURE_TIMEOUT)

    def grab(self) -> tuple[np.ndarray, WindowInfo]:
        if not self._lock.acquire(blocking=False):
            raise CaptureError('截图正在进行')
        try:
            win = self._window()
            if self._requested != 'auto':
                return self._image(self.backend, win), win
            if self.backend != 'cg' and time.monotonic() - self._last_cg_retry >= CG_RETRY_INTERVAL:
                self._last_cg_retry = time.monotonic()
                try:
                    image = self._image('cg', win)
                except CaptureError:
                    pass
                else:
                    log.warning('[CAPTURE] %s → cg：原生截图恢复', self.backend)
                    self.backend, self._cg_failures, self._sc_failures = 'cg', 0, 0
                    return image, win
            while True:
                backend = self.backend
                try:
                    image = self._image(backend, win)
                except CaptureError:
                    if backend == 'cli':
                        raise
                    field = '_cg_failures' if backend == 'cg' else '_sc_failures'
                    failures = getattr(self, field) + 1
                    setattr(self, field, failures)
                    if failures < 2:
                        raise
                    self.backend = 'sc' if backend == 'cg' else 'cli'
                    if backend == 'cg':
                        self._last_cg_retry = time.monotonic()
                    log.warning('[CAPTURE] %s → %s：连续两次截图失败', backend, self.backend)
                else:
                    if backend == 'cg':
                        self._cg_failures = 0
                    elif backend == 'sc':
                        self._sc_failures = 0
                    return image, win
        finally:
            self._lock.release()
