"""用可控原生生命周期验证启动时序；不调用真实 AppKit 激活。"""
from __future__ import annotations

import logging
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from jccbrain.webui import macos


@pytest.fixture
def cocoa(monkeypatch):
    queue, trace, observers, timers = [], [], [], []

    def record(name):
        assert threading.current_thread() is threading.main_thread(), name
        trace.append(name)

    class App:
        active = False
        policy = 0
        requests = 0
        reject = False
        def activationPolicy(self):
            record('policy')
            return self.policy
        def isActive(self):
            record('active')
            return self.active
        def activate(self):
            record('activate')
            self.requests += 1
            if self.reject:
                raise RuntimeError('native activation unavailable')

    app = App()
    running = SimpleNamespace(finished=False)
    def finished():
        record('launch-state')
        return running.finished
    workspace = SimpleNamespace(pid=12345)
    def front():
        record('foreground')
        return SimpleNamespace(processIdentifier=lambda: workspace.pid) if workspace.pid is not None else None

    class Center:
        def addObserverForName_object_queue_usingBlock_(self, name, obj, operation_queue, callback):
            record('observe')
            assert name == 'did-finish-launching' and obj is app and operation_queue is None
            observers.append(callback)
            return callback
        def removeObserver_(self, token):
            record('remove-observer')
            observers.remove(token)

    center = Center()
    class Timer:
        def __init__(self, seconds, repeats, callback):
            self.seconds, self.repeats, self.callback = seconds, repeats, callback
            self.valid = True
            timers.append(self)
        def invalidate(self):
            record('invalidate-timer')
            self.valid = False
        def fire(self):
            record('expiry')
            self.callback(self)

    def add_timer(timer, mode):
        record('timer')
        assert mode == 'common' and timer.seconds == 5 and timer.repeats is False

    web = SimpleNamespace(setAutoresizingMask_=lambda value: record('web-autoresize'),
                          setFrame_=lambda frame: record('web-frame'))
    content = SimpleNamespace(bounds=lambda: 'bounds',
                              superview=lambda: SimpleNamespace(subviews=lambda: SimpleNamespace(lastObject=lambda: None)))
    class Native:
        visible = True
        minimized = False
        key = main = False
        responder = None
        fail_style = False
        def frame(self):
            record('frame')
            return 'frame'
        def contentView(self):
            record('content')
            return content
        def setTitlebarAppearsTransparent_(self, value):
            record('transparent-titlebar')
            assert value
        def setTitleVisibility_(self, value):
            record('title-hidden')
            assert value == 1
        def styleMask(self):
            record('style-read')
            return 7
        def setStyleMask_(self, value):
            record('style')
            assert value == 7 | 1024
            if self.fail_style:
                raise RuntimeError('style failed')
        def setFrame_display_(self, frame, display):
            record('restore-frame')
            assert frame == 'frame' and display
        def isVisible(self):
            record('visible')
            return self.visible
        def isMiniaturized(self):
            record('miniaturized')
            return self.minimized
        def isKeyWindow(self):
            record('key-state')
            return self.key
        def isMainWindow(self):
            record('main-state')
            return self.main
        def canBecomeMainWindow(self):
            record('can-main')
            return True
        def firstResponder(self):
            record('focus-state')
            return self.responder
        def makeKeyAndOrderFront_(self, sender):
            record('make-key')
            assert 'style' in trace and running.finished
            self.key = True
        def makeMainWindow(self):
            record('make-main')
            self.main = True
        def makeFirstResponder_(self, view):
            record('make-focus')
            assert view is web
            self.responder = view
            return True

    native = Native()
    window = SimpleNamespace(uid='master', hidden=False, focus=True, minimized=False, native=native)
    views = {'master': SimpleNamespace(window=native, webview=web)}
    appkit = SimpleNamespace(NSApplication=SimpleNamespace(sharedApplication=lambda: app),
        NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(frontmostApplication=front)),
        NSRunningApplication=SimpleNamespace(currentApplication=lambda: SimpleNamespace(isFinishedLaunching=finished)),
        NSApplicationActivationPolicyProhibited=2, NSApplicationDidFinishLaunchingNotification='did-finish-launching',
        NSColor=SimpleNamespace(clearColor=lambda: None), NSWindowTitleHidden=1,
        NSWindowStyleMaskFullSizeContentView=1024, NSViewWidthSizable=2, NSViewHeightSizable=16)
    foundation = SimpleNamespace(NSNotificationCenter=SimpleNamespace(defaultCenter=lambda: center),
        NSRunLoop=SimpleNamespace(mainRunLoop=lambda: SimpleNamespace(addTimer_forMode_=add_timer)),
        NSRunLoopCommonModes='common', NSTimer=SimpleNamespace(timerWithTimeInterval_repeats_block_=Timer))
    monkeypatch.setattr(macos, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setitem(sys.modules, 'AppKit', appkit)
    monkeypatch.setitem(sys.modules, 'Foundation', foundation)
    monkeypatch.setitem(sys.modules, 'PyObjCTools', SimpleNamespace(AppHelper=SimpleNamespace(
        callAfter=lambda fn, *args: queue.append(lambda: fn(*args)))))
    monkeypatch.setitem(sys.modules, 'webview.platforms.cocoa', SimpleNamespace(BrowserView=SimpleNamespace(instances=views)))
    def drain():
        while queue:
            queue.pop(0)()
    def launch():
        record('launch-finished')
        running.finished = True
        for callback in list(observers):
            callback(None)
    return SimpleNamespace(app=app, running=running, workspace=workspace, native=native, web=web,
        window=window, trace=trace, observers=observers, timers=timers, queue=queue, drain=drain,
        launch=launch, views=views, center=center, foundation=foundation)


def test_launch_notification_and_style_complete_before_one_main_thread_request(cocoa, caplog):
    c = cocoa
    startup = macos.MainWindowStartup(c.window)
    thread = threading.Thread(target=startup.shown)
    thread.start()
    thread.join()
    assert c.trace == []  # shown 工作线程不碰 AppKit。
    c.drain()
    assert c.app.requests == 0 and c.observers and c.timers[0].valid
    assert c.trace.index('style') < c.trace.index('observe')
    with caplog.at_level(logging.INFO):
        c.launch()
        assert c.app.requests == 0  # 通知栈内仍不能请求激活。
        c.drain()
    assert c.app.requests == 1 and c.native.key and c.native.main and c.native.responder is c.web
    assert c.trace.index('launch-finished') < c.trace.index('activate') < c.trace.index('make-key')
    assert not c.observers and not c.timers[0].valid
    assert "'active': False" in caplog.text and '请求后观测' in caplog.text
    startup.shown()
    c.launch()
    c.timers[0].fire()  # 已失效回调即使晚到也不能再次激活。
    c.drain()
    assert c.app.requests == 1


def test_launch_already_finished_uses_next_main_queue_and_no_observer(cocoa):
    c = cocoa
    c.running.finished = True
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.queue.pop(0)()  # prepare 安排真实样式。
    c.queue.pop(0)()  # 样式完成，仅安排下一轮激活。
    assert 'style' in c.trace and c.app.requests == 0 and not c.observers
    c.drain()
    assert c.app.requests == 1 and c.native.responder is c.web


@pytest.mark.parametrize('gate,reason', [('hidden', 'hidden'), ('focus', 'focus-disabled'),
    ('minimized', 'miniaturized'), ('native-minimized', 'miniaturized'), ('policy', 'activation-prohibited'),
    ('hud', 'not-main-window'), ('invisible', 'window-unavailable')])
def test_ineligible_window_does_not_observe_or_activate(cocoa, caplog, gate, reason):
    c = cocoa
    if gate == 'hidden': c.window.hidden = True
    elif gate == 'focus': c.window.focus = False
    elif gate == 'minimized': c.window.minimized = True
    elif gate == 'native-minimized': c.native.minimized = True
    elif gate == 'policy': c.app.policy = 2
    elif gate == 'invisible': c.native.visible = False
    else:
        c.window.uid = 'child_hud'
        c.views['child_hud'] = c.views.pop('master')
    with caplog.at_level(logging.INFO):
        s = macos.MainWindowStartup(c.window)
        s.shown()
        c.drain()
        c.launch()
        c.drain()
    assert c.app.requests == 0 and 'make-key' not in c.trace and 'make-focus' not in c.trace
    assert not c.observers and not c.timers and reason in caplog.text
    assert ('style' in c.trace) == (gate != 'hud')  # 不碰 HUD 样式，其余保留原有标题栏。


@pytest.mark.parametrize('stage', ['before-prepare', 'waiting-launch', 'queued'])
def test_close_revokes_pending_startup_and_cleans_observer(cocoa, stage):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    if stage != 'before-prepare': c.drain()
    if stage == 'queued': c.launch()
    thread = threading.Thread(target=s.close)
    thread.start()
    thread.join()
    c.launch()
    c.drain()
    assert c.app.requests == 0 and 'make-key' not in c.trace
    assert not c.observers and all(not t.valid for t in c.timers)
    assert s._window is None


@pytest.mark.parametrize('front', [None, 54321])
def test_external_foreground_change_cancels_instead_of_stealing_back(cocoa, caplog, front):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    c.workspace.pid = front
    c.launch()
    with caplog.at_level(logging.INFO): c.drain()
    assert c.app.requests == 0 and 'make-key' not in c.trace and 'foreground-changed' in caplog.text
    assert not c.observers and not c.timers[0].valid


def test_self_becoming_frontmost_still_repairs_missing_key_and_web_focus(cocoa):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    c.app.active = True
    c.workspace.pid = os.getpid()
    c.launch()
    c.drain()
    assert c.app.requests == 0 and c.native.key and c.native.main and c.native.responder is c.web


def test_already_healthy_window_keeps_current_web_descendant_focus(cocoa):
    c = cocoa
    c.app.active = c.native.key = c.native.main = True
    child = SimpleNamespace(isDescendantOf_=lambda parent: parent is c.web)
    c.native.responder = child
    c.running.finished = True
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    assert c.app.requests == 0 and 'make-key' not in c.trace and 'make-focus' not in c.trace
    assert c.native.responder is child


def test_missing_launch_notification_expires_and_cannot_activate_later(cocoa):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    c.timers[0].fire()
    c.launch()
    c.drain()
    assert not c.observers and not c.timers[0].valid and s._window is None
    assert c.app.requests == 0


@pytest.mark.parametrize('failure', ['style', 'timer', 'activation', 'destroyed'])
def test_native_failure_cleans_pending_qualification_without_breaking_startup(cocoa, failure):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    if failure == 'style': c.native.fail_style = True
    if failure == 'timer':
        def broken(*args): raise RuntimeError('timer unavailable')
        c.foundation.NSTimer.timerWithTimeInterval_repeats_block_ = broken
    s.shown()
    c.drain()
    if failure == 'activation': c.app.reject = True
    if failure == 'destroyed': c.views.clear()
    c.launch()
    c.drain()
    assert not c.observers and all(not t.valid for t in c.timers)
    assert s._window is None
    assert c.app.requests == (1 if failure == 'activation' else 0)
    assert 'make-key' not in c.trace


def test_old_macos_activation_api_is_only_used_when_modern_api_unavailable(cocoa):
    c = cocoa
    c.app.activate = None
    def old(ignore):
        assert ignore is True
        assert threading.current_thread() is threading.main_thread()
        c.app.requests += 1
    c.app.activateIgnoringOtherApps_ = old
    c.running.finished = True
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    assert c.app.requests == 1 and c.native.key


def test_other_platform_is_noop_without_importing_appkit(monkeypatch):
    import builtins
    monkeypatch.setattr(macos, 'sys', SimpleNamespace(platform='linux'))
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ('AppKit', 'Foundation', 'PyObjCTools')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    s = macos.MainWindowStartup(SimpleNamespace(uid='master'))
    s.shown()
    s.shown()
    s.close()
    assert s._window is None


@pytest.mark.parametrize('change', ['hidden', 'minimized', 'prohibited'])
def test_safety_gates_are_checked_again_at_execution(cocoa, change):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    c.launch()
    if change == 'hidden': c.window.hidden = True
    elif change == 'minimized': c.native.minimized = True
    else: c.app.policy = 2
    c.drain()
    assert c.app.requests == 0 and 'make-key' not in c.trace
    assert not c.observers and not c.timers[0].valid


def test_notification_without_finished_launch_state_never_activates(cocoa):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    c.observers[0](None)
    c.drain()
    assert not c.running.finished and c.app.requests == 0
    assert not c.observers and not c.timers[0].valid


def test_app_start_exception_closes_startup_coordination(monkeypatch):
    from unittest.mock import Mock
    from jccbrain.webui import app
    class Signal:
        def __iadd__(self, callback): return self
    ctx, api, bus, startup = Mock(), Mock(), Mock(), Mock()
    window = SimpleNamespace(events=SimpleNamespace(shown=Signal(), loaded=Signal(), closed=Signal()))
    monkeypatch.setattr(app, 'AppContext', lambda: ctx)
    monkeypatch.setattr(app, 'Api', lambda *args, **kwargs: api)
    monkeypatch.setattr(app, 'EventBus', lambda: bus)
    monkeypatch.setattr(app, 'HudController', lambda *args: None)
    monkeypatch.setattr(app, 'MainWindowStartup', lambda win: startup if win is window else None)
    monkeypatch.setattr(app.webview, 'create_window', lambda *args, **kwargs: window)
    def broken(**kwargs): raise RuntimeError('native start failed')
    monkeypatch.setattr(app.webview, 'start', broken)
    with pytest.raises(RuntimeError, match='native start failed'):
        app.run_webui()
    assert startup.close.call_count == 2  # 主线程兜底与原 cleanup 都安全撤销。
    api._shutdown.assert_called_once()
    bus.close.assert_called_once()
    ctx.close.assert_called_once()


def test_main_thread_cleanup_does_not_need_the_stopped_runloop(cocoa):
    c = cocoa
    s = macos.MainWindowStartup(c.window)
    s.shown()
    c.drain()
    thread = threading.Thread(target=s.close)
    thread.start()
    thread.join()
    assert c.observers  # 模拟 closed 工作线程排队后，主 runloop 已退出。
    s.close()  # run_webui finally 的主线程兜底不再依赖队列。
    assert not c.observers and not c.timers[0].valid and s._window is None
    c.launch()
    c.drain()
    assert c.app.requests == 0
