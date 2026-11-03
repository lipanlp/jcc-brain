"""可选的 macOS 原生窗口样式；不影响其他平台启动。"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Any

log = logging.getLogger(__name__)


def hud_level(overlay: bool, modal_level: int = 8) -> int:
    """面板只比浮层高一级，二者均低于系统菜单。"""
    return int(modal_level) + (0 if overlay else 1)


class NativeDragRegions:
    """CSS 只上报命中区域；鼠标按下后由 AppKit 完成整个窗口拖动。"""

    def __init__(self, get_window):
        self._get_window = get_window
        self._regions = {}
        self._monitor = None
        self._handler = None
        self._closed = False
        self._drag_panel = None
        self._drag_timer = None

    @staticmethod
    def _on_main(action):
        if sys.platform != 'darwin':
            return
        try:
            from PyObjCTools import AppHelper
            if threading.current_thread() is threading.main_thread():
                action()
            else:
                AppHelper.callAfter(action)
        except ImportError:
            return

    def install(self):
        def apply():
            if self._closed or self._monitor is not None:
                return
            from AppKit import NSEvent, NSEventMaskLeftMouseDown
            self._handler = self._handle
            self._monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskLeftMouseDown, self._handler)

        self._on_main(apply)

    def close(self):
        self._closed = True

        def apply():
            self._cancel_panel_drag()
            if self._monitor is not None:
                from AppKit import NSEvent
                NSEvent.removeMonitor_(self._monitor)
            self._monitor = self._handler = None
            self._regions.clear()

        self._on_main(apply)

    def update(self, key, regions, exclusions):
        if self._closed or key not in ('main', 'hud_panel'):
            return

        def rects(values):
            result = []
            for value in values:
                rect = tuple(float(value[name]) for name in ('x', 'y', 'w', 'h'))
                if all(math.isfinite(v) for v in rect) and rect[2] > 0 and rect[3] > 0:
                    result.append(rect)
            return tuple(result)

        # 原子替换快照；桥线程不读取或修改 NSWindow。
        self._regions[key] = (rects(regions), rects(exclusions))

    @staticmethod
    def _contains(rects, x, y):
        return any(rx <= x < rx + w and ry <= y < ry + h for rx, ry, w, h in rects)

    @staticmethod
    def _native_button(native, point):
        from AppKit import NSWindowCloseButton, NSWindowMiniaturizeButton, NSWindowZoomButton
        getter = getattr(native, 'standardWindowButton_', None)
        if getter is None:
            return False
        for kind in (NSWindowCloseButton, NSWindowMiniaturizeButton, NSWindowZoomButton):
            button = getter(kind)
            if button is not None and not button.isHidden():
                p = button.convertPoint_fromView_(point, None)
                b = button.bounds()
                if (b.origin.x <= p.x < b.origin.x + b.size.width
                        and b.origin.y <= p.y < b.origin.y + b.size.height):
                    return True
        return False

    def _handle(self, event):
        if self._closed:
            return event
        native = event.window()
        if native is None:
            return event
        key = next((key for key in ('main', 'hud_panel')
                    if getattr(self._get_window(key), 'native', None) == native), None)
        if key is None:
            return event
        regions, exclusions = self._regions.get(key, ((), ()))
        point = event.locationInWindow()
        x, y = point.x, native.contentView().frame().size.height - point.y
        if (not self._contains(regions, x, y) or self._contains(exclusions, x, y)
                or self._native_button(native, point)):
            return event
        if key == 'main' and event.clickCount() == 2:
            from Foundation import NSGlobalDomain, NSUserDefaults
            prefs = NSUserDefaults.standardUserDefaults().persistentDomainForName_(NSGlobalDomain)
            action = (prefs or {}).get('AppleActionOnDoubleClick')
            if action == 'Minimize':
                native.miniaturize_(None)
            elif action != 'None':
                native.zoom_(None)
        else:
            if key == 'hud_panel':
                self._begin_panel_drag(native)
            try:
                native.performWindowDragWithEvent_(event)
            except Exception:
                self._end_panel_drag()
                raise
            if key == 'hud_panel':
                self._end_panel_drag_if_released()
        return None


    def _begin_panel_drag(self, native):
        from Foundation import NSRunLoop, NSRunLoopCommonModes, NSTimer

        self._end_panel_drag()
        _record_panel_drag(native)
        self._drag_panel = native
        # performWindowDragWithEvent_ 在部分 macOS 上立即返回，实际拖动仍在继续。
        # 原生主线程定时检查鼠标释放，不能把该方法返回误当作拖动结束。
        self._drag_timer = NSTimer.timerWithTimeInterval_repeats_block_(
            .05, True, lambda _: self._end_panel_drag_if_released())
        NSRunLoop.mainRunLoop().addTimer_forMode_(self._drag_timer, NSRunLoopCommonModes)

    def _cancel_panel_drag(self):
        if self._drag_timer is not None:
            self._drag_timer.invalidate()
            self._drag_timer = None
        self._drag_panel = None

    def _end_panel_drag(self):
        native = self._drag_panel
        self._cancel_panel_drag()
        if native is not None:
            _restore_panel_drag(native)

    def _end_panel_drag_if_released(self):
        from AppKit import NSEvent

        if self._drag_panel is not None and not NSEvent.pressedMouseButtons() & 1:
            self._end_panel_drag()


def hide_panel(window: Any, remember) -> None:
    """在隐藏之前读取最终原点，避免从桥线程访问 AppKit 或保存过期位置。"""
    if sys.platform != 'darwin':
        window.hide()
        return

    def apply(native):
        origin = native.frame().origin
        remember((origin.x, origin.y))
        native.orderOut_(None)

    _hud_main(window, apply)


def restore_panel_origin(window: Any, origin: tuple[float, float] | None) -> None:
    if sys.platform == 'darwin' and origin is not None:
        _hud_main(window, lambda native: native.setFrameOrigin_(origin))


def _hud_main(window: Any, action) -> None:
    """所有 HUD 的 AppKit 操作都在主线程排队，避免 WebKit/桥线程碰窗口。"""
    from PyObjCTools import AppHelper

    def apply():
        try:
            native = getattr(window, 'native', None)
            if native is not None:
                action(native)
        except Exception:
            log.exception('无法更新 macOS HUD 窗口')

    AppHelper.callAfter(apply)


def _style_hud(window: Any, overlay: bool) -> None:
    if sys.platform != 'darwin':
        return
    import AppKit
    from AppKit import (
        NSColor, NSModalPanelWindowLevel, NSWindowStyleMaskBorderless,
        NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary, NSWindowCollectionBehaviorIgnoresCycle,
    )

    def apply(native):
        native.setIgnoresMouseEvents_(overlay)
        native.setOpaque_(False)
        native.setBackgroundColor_(NSColor.clearColor())
        native.setHasShadow_(False)
        native.setHidesOnDeactivate_(False)
        native.setExcludedFromWindowsMenu_(True)
        native.setLevel_(hud_level(overlay, NSModalPanelWindowLevel))
        native.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                      | NSWindowCollectionBehaviorFullScreenAuxiliary
                                      | NSWindowCollectionBehaviorStationary
                                      | NSWindowCollectionBehaviorIgnoresCycle
                                      | getattr(AppKit, 'NSWindowCollectionBehaviorCanJoinAllApplications', 0))
        # WindowHost.canBecomeKeyWindow 读取 focus；移除 titled 等样式后，
        # NSWindow 默认的 canBecomeMainWindow 也为 False。
        if overlay:
            frame = native.frame()
            native.setStyleMask_(NSWindowStyleMaskBorderless)
            native.setFrame_display_(frame, True)
        native.focus = not overlay
        from webview.platforms.cocoa import BrowserView
        view = BrowserView.instances.get(window.uid)
        if view is not None:
            view.webview.setValue_forKey_(True, 'drawsTransparentBackground')
            view.webview.setFrame_(native.contentView().bounds())

    _hud_main(window, apply)


def make_overlay(window: Any) -> None:
    _style_hud(window, True)


def make_panel(window: Any) -> None:
    _style_hud(window, False)


def show_hud(window: Any) -> None:
    if sys.platform == 'darwin':
        # pywebview.show() 会 activateIgnoringOtherApps，即使 focus=False 也抢前台。
        _hud_main(window, lambda native: native.orderFrontRegardless())
    else:
        window.show()


def clear_toast_visibility(window):
    window._jcc_toast_visible = (False, 0.)


def toast_visible(window):
    # 最终授权读取主线程上的真实可见性；绝不拿上一次跟随缓存当确认。
    if sys.platform != 'darwin':
        return False
    ready = threading.Event()
    result = [False]
    def apply(native):
        try:
            _toast_visibility(window, native, getattr(window, '_jcc_toast_pid', None))
            result[0] = window._jcc_toast_visible[0]
        finally:
            ready.set()
    if threading.current_thread() is threading.main_thread():
        native = getattr(window, 'native', None)
        if native is not None:
            apply(native)
    else:
        _hud_main(window, apply)
        ready.wait(.15)  # 主线程忙碌/销毁时 fail closed，不无限等待。
    return ready.is_set() and result[0]


def _toast_visibility(window, native, game_pid):
    import time
    from AppKit import NSWorkspace
    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    window._jcc_toast_visible = (bool(front and int(front.processIdentifier()) == game_pid
        and native.isVisible() and native.isOnActiveSpace() and native.alphaValue() > 0), time.monotonic())


def show_toast(window, game_pid):
    """唯一自动预告窗；排队完成后才记录真正可见，不激活任何应用。"""
    clear_toast_visibility(window)
    window._jcc_toast_pid = game_pid
    if sys.platform != 'darwin':
        return  # 无原生可见性保证时不授权自动输入。
    def apply(native):
        native.orderFrontRegardless()
        _toast_visibility(window, native, game_pid)
    _hud_main(window, apply)


def check_toast_visibility(window, game_pid):
    if sys.platform == 'darwin':
        _hud_main(window, lambda native: _toast_visibility(window, native, game_pid))


def ensure_hud_above(window: Any, overlay: bool, game_pid: int) -> None:
    if sys.platform != 'darwin':
        return
    import os
    from AppKit import NSWorkspace

    def apply(native):
        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is None:
            return
        pid = int(front.processIdentifier())
        if pid != os.getpid():
            # 原生鼠标按下可能先激活本应用，保留点击前的外部前台。
            native._jcc_front_app = front
        if pid == game_pid and (not native.isVisible() or native.level() != hud_level(overlay)
                                or not native.isOnActiveSpace()):
            native.setLevel_(hud_level(overlay))
            native.orderFrontRegardless()

    _hud_main(window, apply)


def _record_panel_drag(native: Any) -> None:
    import os
    from AppKit import NSWorkspace

    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    if front is not None and int(front.processIdentifier()) == os.getpid():
        front = getattr(native, '_jcc_front_app', None)
    native._jcc_drag_app = front


def _restore_panel_drag(native: Any) -> None:
    import os
    from AppKit import NSApplicationActivateIgnoringOtherApps

    app = getattr(native, '_jcc_drag_app', None)
    native._jcc_drag_app = None
    if app is not None and int(app.processIdentifier()) != os.getpid():
        if not app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps):
            # 开发进程中原生激活可能返回 False；备用进程只激活记录的 PID。
            threading.Thread(target=_activate_by_pid, args=(int(app.processIdentifier()),),
                             name='hud-restore-focus', daemon=True).start()


def panel_drag_start(window: Any) -> None:
    if sys.platform == 'darwin':
        _hud_main(window, _record_panel_drag)


def panel_drag_end(window: Any) -> None:
    if sys.platform == 'darwin':
        _hud_main(window, _restore_panel_drag)


def _activate_by_pid(pid: int) -> bool:
    """开发环境的激活备用路径；打包程序的 executable 不是 Python。"""
    if sys.platform != 'darwin' or getattr(sys, 'frozen', False):
        return False
    import os
    import subprocess

    if pid == os.getpid():
        return False
    code = ('import sys; from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps; '
            'app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(sys.argv[1])); '
            'raise SystemExit(0 if app and app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps) else 1)')
    try:
        result = subprocess.run([sys.executable, '-c', code, str(int(pid))],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=5, check=False)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        log.warning('恢复前台应用失败（PID %s）', pid, exc_info=True)
        return False


def place_hud(window: Any, frame: tuple[int, int, int, int]) -> None:
    x, y, width, height = frame
    if sys.platform != 'darwin':
        window.resize(width, height)
        window.move(x, y)
        return
    from AppKit import NSMakeRect, NSScreen

    def apply(native):
        # Quartz 全局坐标以主屏左上角为原点；AppKit 以主屏左下角为原点。
        # 不能用 pywebview 的 window.screen（它会把副屏 origin 再加一次）。
        primary_height = NSScreen.screens()[0].frame().size.height
        native.setFrame_display_(NSMakeRect(x, primary_height - y - height, width, height), True)

    _hud_main(window, apply)


def screen_panel_frame() -> tuple[int, int, int, int]:
    if sys.platform == 'darwin':
        from AppKit import NSScreen
        screen = NSScreen.mainScreen().visibleFrame()
        primary_height = NSScreen.screens()[0].frame().size.height
        return (round(screen.origin.x + screen.size.width - 320),
                round(primary_height - screen.origin.y - screen.size.height + 40), 300, 220)
    import webview
    screen = webview.screens[0]
    return (screen.x + screen.width - 320, screen.y + 80, 300, 220)


class MainWindowStartup:
    """主窗口样式完成、AppKit 启动完成后，仅协调一次启动输入焦点。"""

    def __init__(self, window: Any):
        self._window = window
        self._lock = threading.Lock()
        self._started = False
        self._closed = threading.Event()
        self._done = False
        self._queued = False
        self._app = self._center = self._observer = self._expiry = None
        self._front_pid = None

    def shown(self) -> None:
        with self._lock:
            if self._started or self._closed.is_set():
                return
            self._started = True
        if sys.platform != 'darwin':
            self._window = None
            return
        try:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._prepare)
        except Exception:
            self._window = None
            log.warning('主窗口启动激活不可用', exc_info=True)

    def close(self) -> None:
        # 先撤销资格；排在主线程队列中的回调也不能在关闭后请求激活。
        self._closed.set()
        if sys.platform != 'darwin':
            self._window = None
            return
        try:
            from PyObjCTools import AppHelper
            if threading.current_thread() is threading.main_thread():
                self._finish('closed')
            else:
                AppHelper.callAfter(self._finish, 'closed')
        except Exception:
            log.warning('主窗口启动激活清理不可用', exc_info=True)

    def _prepare(self) -> None:
        if self._closed.is_set():
            self._finish('closed')
            return
        if getattr(self._window, 'uid', None) != 'master':
            self._finish('not-main-window')
            return
        try:
            from AppKit import NSApplication, NSWorkspace
            self._app = NSApplication.sharedApplication()
            front = NSWorkspace.sharedWorkspace().frontmostApplication()
            self._front_pid = int(front.processIdentifier()) if front is not None else None
            style_main_window(self._window, self._styled)
        except Exception:
            log.warning('主窗口启动激活准备失败', exc_info=True)
            self._finish('native-unavailable')

    def _native(self):
        from webview.platforms.cocoa import BrowserView
        view = BrowserView.instances.get(getattr(self._window, 'uid', None))
        return view, getattr(view, 'window', None)

    def _skip_reason(self, native) -> str | None:
        window = self._window
        if self._closed.is_set():
            return 'closed'
        if window is None or window.uid != 'master':
            return 'not-main-window'
        if window.hidden:
            return 'hidden'
        if not window.focus:
            return 'focus-disabled'
        if native is None or not native.isVisible():
            return 'window-unavailable'
        if window.minimized or native.isMiniaturized():
            return 'miniaturized'
        from AppKit import NSApplicationActivationPolicyProhibited
        if self._app.activationPolicy() == NSApplicationActivationPolicyProhibited:
            return 'activation-prohibited'
        return None

    def _styled(self, applied: bool) -> None:
        try:
            if not applied:
                self._finish('style-unavailable')
                return
            _, native = self._native()
            reason = self._skip_reason(native)
            if reason:
                self._finish(reason)
                return
            from AppKit import NSApplicationDidFinishLaunchingNotification, NSRunningApplication
            from Foundation import NSNotificationCenter, NSRunLoop, NSRunLoopCommonModes, NSTimer
            running = NSRunningApplication.currentApplication()
            log.info('主窗口启动激活已安排：%s', self._state(native))
            if running.isFinishedLaunching():
                self._queue_activation()
                return
            self._center = NSNotificationCenter.defaultCenter()
            self._observer = self._center.addObserverForName_object_queue_usingBlock_(
                NSApplicationDidFinishLaunchingNotification, self._app, None,
                lambda _: self._queue_activation())
            # 此计时器只撤销过期资格，不用固定延迟猜测何时可以激活。
            self._expiry = NSTimer.timerWithTimeInterval_repeats_block_(
                5., False, lambda _: self._finish('launch-timeout'))
            NSRunLoop.mainRunLoop().addTimer_forMode_(self._expiry, NSRunLoopCommonModes)
            if running.isFinishedLaunching():
                self._queue_activation()
        except Exception:
            log.warning('主窗口启动激活监听失败', exc_info=True)
            self._finish('native-unavailable')

    def _queue_activation(self) -> None:
        with self._lock:
            if self._closed.is_set() or self._done or self._queued:
                return
            self._queued = True
        try:
            from PyObjCTools import AppHelper
            # 即使通知已在主线程，也排到后续 runloop，避开 SDK 首次显示/启动栈。
            AppHelper.callAfter(self._activate)
        except Exception:
            log.warning('主窗口启动激活排队失败', exc_info=True)
            self.close()

    def _remove_callbacks(self) -> None:
        center, observer, expiry = self._center, self._observer, self._expiry
        self._center = self._observer = self._expiry = None
        try:
            if observer is not None:
                center.removeObserver_(observer)
        finally:
            if expiry is not None:
                expiry.invalidate()

    def _finish(self, reason: str) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._remove_callbacks()
        except Exception:
            log.warning('主窗口启动激活监听清理失败', exc_info=True)
        finally:
            self._window = self._app = None
        log.info('主窗口启动激活结束：%s', reason)

    def _state(self, native) -> dict:
        view, _ = self._native()
        return {'active': bool(self._app.isActive()), 'key': bool(native.isKeyWindow()),
                'main': bool(native.isMainWindow()),
                'focus': self._has_web_focus(native, getattr(view, 'webview', None))}

    @staticmethod
    def _has_web_focus(native, webview) -> bool:
        responder = native.firstResponder()
        return webview is not None and (responder == webview or (responder is not None
            and getattr(responder, 'isDescendantOf_', lambda _: False)(webview)))

    def _activate(self) -> None:
        if self._done:
            return
        try:
            view, native = self._native()
            reason = self._skip_reason(native)
            if reason:
                self._finish(reason)
                return
            from AppKit import NSRunningApplication, NSWorkspace
            if not NSRunningApplication.currentApplication().isFinishedLaunching():
                self._finish('launch-not-finished')
                return
            front = NSWorkspace.sharedWorkspace().frontmostApplication()
            pid = int(front.processIdentifier()) if front is not None else None
            if pid is None or self._front_pid is None or pid not in (self._front_pid, os.getpid()):
                self._finish('foreground-changed')
                return
            # 先移除观察者和过期回调；激活引起的原生通知不能再安排第二次。
            self._remove_callbacks()
            before = self._state(native)
            log.info('主窗口启动激活请求前：%s', before)
            if not before['active']:
                activate = getattr(self._app, 'activate', None)
                if activate is not None:
                    activate()
                else:
                    self._app.activateIgnoringOtherApps_(True)
            if not before['key'] or not before['active']:
                native.makeKeyAndOrderFront_(None)
            if not native.isMainWindow() and native.canBecomeMainWindow():
                native.makeMainWindow()
            webview = getattr(view, 'webview', None)
            if webview is not None and not self._has_web_focus(native, webview):
                native.makeFirstResponder_(webview)
            # 激活请求可能异步完成；只报告此刻观测，不声称系统已接受。
            log.info('主窗口启动激活请求后观测：%s', self._state(native))
        except Exception:
            log.warning('主窗口启动激活请求失败', exc_info=True)
        finally:
            self._finish('finished')


def style_main_window(window: Any, on_styled=None) -> None:
    if sys.platform != "darwin":
        return
    try:
        from AppKit import (
            NSColor,
            NSViewHeightSizable,
            NSViewWidthSizable,
            NSWindowStyleMaskFullSizeContentView,
            NSWindowTitleHidden,
        )
        from PyObjCTools import AppHelper
        from webview.platforms.cocoa import BrowserView

        def apply() -> None:
            applied = False
            try:
                view = BrowserView.instances.get(window.uid)
                native = getattr(view, "window", None)
                if native is None:
                    return
                frame = native.frame()
                native.setTitlebarAppearsTransparent_(True)
                native.setTitleVisibility_(NSWindowTitleHidden)
                native.setStyleMask_(native.styleMask() | NSWindowStyleMaskFullSizeContentView)
                # cocoa 后端会单独设置标题栏视图背景，需清除才能透出页面。
                titlebar = native.contentView().superview().subviews().lastObject()
                if (titlebar is not None and titlebar != native.contentView()
                        and titlebar.respondsToSelector_("setBackgroundColor:")):
                    titlebar.setBackgroundColor_(NSColor.clearColor())
                # AppKit 切换内容区域样式时会重算高度，恢复原始窗口外框。
                native.setFrame_display_(frame, True)
                webview = getattr(view, "webview", None)
                if webview is not None:
                    webview.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
                    webview.setFrame_(native.contentView().bounds())
                applied = True
            except Exception:
                log.debug("无法调整 macOS 标题栏", exc_info=True)
            finally:
                if on_styled is not None:
                    on_styled(applied)

        AppHelper.callAfter(apply)
    except Exception:
        log.debug("macOS 窗口样式不可用", exc_info=True)
        if on_styled is not None:
            on_styled(False)
