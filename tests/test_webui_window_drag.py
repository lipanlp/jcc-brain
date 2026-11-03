"""原生拖动命中、系统双击偏好和监听生命周期；不访问真实窗口。"""
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import AppKit
import Foundation
from PyObjCTools import AppHelper
import pytest

from jccbrain.webui import macos


class NativeWindow:
    def __init__(self):
        self.performWindowDragWithEvent_ = Mock()
        self.zoom_ = Mock()
        self.miniaturize_ = Mock()
        self.standardWindowButton_ = Mock(return_value=None)

    def contentView(self):
        return SimpleNamespace(frame=lambda: SimpleNamespace(size=SimpleNamespace(height=820)))


def event(native, x=240, top=26, clicks=1):
    return SimpleNamespace(window=lambda: native,
                           locationInWindow=lambda: SimpleNamespace(x=x, y=820-top),
                           clickCount=lambda: clicks)


@pytest.fixture
def native_drag(monkeypatch):
    monkeypatch.setattr(macos.sys, 'platform', 'darwin')
    monkeypatch.setattr(macos, '_record_panel_drag', Mock())
    monkeypatch.setattr(macos, '_restore_panel_drag', Mock())
    monkeypatch.setattr(AppKit, 'NSEvent', SimpleNamespace(pressedMouseButtons=Mock(return_value=1)))
    monkeypatch.setattr(Foundation, 'NSTimer', SimpleNamespace(timerWithTimeInterval_repeats_block_=Mock(return_value=Mock())))
    monkeypatch.setattr(Foundation, 'NSRunLoop', SimpleNamespace(mainRunLoop=Mock(return_value=Mock())))
    windows = {key: SimpleNamespace(native=NativeWindow()) for key in ('main', 'hud_panel')}
    drag = macos.NativeDragRegions(windows.get)
    for key in windows:
        drag.update(key, [{'x': 0, 'y': 0, 'w': 1280, 'h': 52}],
                    [{'x': 400, 'y': 10, 'w': 100, 'h': 32}])
    prefs = {}
    defaults = SimpleNamespace(persistentDomainForName_=Mock(return_value=prefs))
    monkeypatch.setattr(Foundation, 'NSUserDefaults', SimpleNamespace(standardUserDefaults=lambda: defaults))
    return drag, windows, prefs, defaults


@pytest.mark.parametrize('key', ['main', 'hud_panel'])
def test_region_hit_drags_and_consumes_event(native_drag, key):
    drag, windows, _, _ = native_drag
    native = windows[key].native
    down = event(native)
    assert drag._handle(down) is None
    native.performWindowDragWithEvent_.assert_called_once_with(down)


@pytest.mark.parametrize('x,top', [(450, 26), (240, 53), (240, 794), (-1, 26), (1280, 26)])
def test_exclusions_bounds_and_y_flip_pass_through(native_drag, x, top):
    drag, windows, _, _ = native_drag
    native = windows['main'].native
    down = event(native, x, top)
    assert drag._handle(down) is down
    native.performWindowDragWithEvent_.assert_not_called()


@pytest.mark.parametrize('preference,action', [(None, 'zoom_'), ('Maximize', 'zoom_'),
                                             ('Minimize', 'miniaturize_'), ('None', None)])
def test_double_click_respects_global_system_preference(native_drag, preference, action):
    drag, windows, prefs, defaults = native_drag
    if preference is not None:
        prefs['AppleActionOnDoubleClick'] = preference
    native = windows['main'].native
    assert drag._handle(event(native, clicks=2)) is None
    defaults.persistentDomainForName_.assert_called_once_with(Foundation.NSGlobalDomain)
    for name in ('zoom_', 'miniaturize_'):
        assert getattr(native, name).call_count == int(name == action)
    native.performWindowDragWithEvent_.assert_not_called()


def test_panel_double_click_still_uses_native_drag(native_drag):
    drag, windows, _, defaults = native_drag
    native = windows['hud_panel'].native
    down = event(native, clicks=2)
    assert drag._handle(down) is None
    native.performWindowDragWithEvent_.assert_called_once_with(down)
    native.zoom_.assert_not_called()
    defaults.persistentDomainForName_.assert_not_called()


def test_foreign_missing_and_unregistered_windows_pass_through(native_drag):
    drag, windows, _, _ = native_drag
    for native in (None, NativeWindow()):
        down = event(native)
        assert drag._handle(down) is down
    drag.update('unknown', [{'x': 0, 'y': 0, 'w': 1280, 'h': 820}], [])
    assert 'unknown' not in drag._regions
    windows['main'].native = None
    down = event(NativeWindow())
    assert drag._handle(down) is down
    drag.update('hud_panel', [], [])
    down = event(windows['hud_panel'].native)
    assert drag._handle(down) is down


def test_native_traffic_light_is_not_swallowed(native_drag):
    drag, windows, _, _ = native_drag
    native = windows['main'].native
    native.standardWindowButton_.return_value = SimpleNamespace(
        isHidden=lambda: False,
        convertPoint_fromView_=lambda *_: SimpleNamespace(x=5, y=5),
        bounds=lambda: SimpleNamespace(origin=SimpleNamespace(x=0, y=0),
                                       size=SimpleNamespace(width=14, height=14)))
    down = event(native, x=20)
    assert drag._handle(down) is down
    native.performWindowDragWithEvent_.assert_not_called()


def test_monitor_installs_once_on_main_and_removes_before_exit(native_drag, monkeypatch):
    drag, windows, _, _ = native_drag
    queue = []
    monkeypatch.setattr(AppHelper, 'callAfter', lambda fn: queue.append(fn))
    monitor = SimpleNamespace(addLocalMonitorForEventsMatchingMask_handler_=Mock(return_value=object()),
                              removeMonitor_=Mock())
    monkeypatch.setattr(AppKit, 'NSEvent', monitor)
    thread = threading.Thread(target=lambda: (drag.install(), drag.install()))
    thread.start()
    thread.join()
    monitor.addLocalMonitorForEventsMatchingMask_handler_.assert_not_called()
    for apply in queue:
        apply()
    monitor.addLocalMonitorForEventsMatchingMask_handler_.assert_called_once_with(
        AppKit.NSEventMaskLeftMouseDown, drag._handler)
    token = drag._monitor
    drag.close()
    drag.close()
    drag.install()
    monitor.removeMonitor_.assert_called_once_with(token)
    assert drag._handler is None and not drag._regions
    down = event(windows['main'].native)
    assert drag._handle(down) is down


def test_other_platform_does_not_install(native_drag, monkeypatch):
    drag, _, _, _ = native_drag
    monkeypatch.setattr(macos.sys, 'platform', 'linux')
    add = Mock()
    monkeypatch.setattr(AppKit, 'NSEvent', SimpleNamespace(addLocalMonitorForEventsMatchingMask_handler_=add))
    drag.install()
    drag.close()
    add.assert_not_called()


def test_panel_position_is_read_before_native_hide(monkeypatch):
    monkeypatch.setattr(macos.sys, 'platform', 'darwin')
    queue, order = [], []
    monkeypatch.setattr(AppHelper, 'callAfter', lambda fn: queue.append(fn))
    native = SimpleNamespace(
        frame=lambda: SimpleNamespace(origin=SimpleNamespace(x=-400, y=200)),
        orderOut_=lambda _: order.append('hide'), setFrameOrigin_=Mock())
    win = SimpleNamespace(native=native)
    macos.hide_panel(win, lambda origin: order.append(origin))
    assert not order
    queue.pop(0)()
    assert order == [(-400, 200), 'hide']
    macos.restore_panel_origin(win, (-400, 200))
    queue.pop(0)()
    native.setFrameOrigin_.assert_called_once_with((-400, 200))


@pytest.mark.parametrize('key', ['main', 'hud_panel'])
@pytest.mark.parametrize('fails', [False, True])
def test_focus_restore_brackets_native_panel_drag_even_on_error(native_drag, key, fails):
    drag, windows, _, _ = native_drag
    native = windows[key].native
    calls = []
    macos._record_panel_drag.side_effect = lambda win: calls.append(('start', win))
    macos._restore_panel_drag.side_effect = lambda win: calls.append(('end', win))
    def move(_):
        calls.append(('drag', native))
        if fails:
            raise RuntimeError('drag stopped')
    native.performWindowDragWithEvent_.side_effect = move
    if fails:
        with pytest.raises(RuntimeError, match='drag stopped'):
            drag._handle(event(native))
    else:
        drag._handle(event(native))
    expected = ['drag'] if key == 'main' else ['start', 'drag'] + (['end'] if fails else [])
    assert calls == [(name, native) for name in expected]
    if key == 'hud_panel' and not fails:
        timer = drag._drag_timer
        # AppKit 方法已返回，但按键仍按着时不能提前归还前台。
        drag._end_panel_drag_if_released()
        macos._restore_panel_drag.assert_not_called()
        AppKit.NSEvent.pressedMouseButtons.return_value = 0
        drag._end_panel_drag_if_released()
        assert calls == [(name, native) for name in ('start', 'drag', 'end')]
        timer.invalidate.assert_called_once_with()
        drag._end_panel_drag_if_released()
        macos._restore_panel_drag.assert_called_once_with(native)


def test_close_cancels_native_drag_timer_without_activating_other_apps(native_drag):
    drag, windows, _, _ = native_drag
    drag._handle(event(windows['hud_panel'].native))
    timer = drag._drag_timer
    drag.close()
    timer.invalidate.assert_called_once_with()
    assert drag._drag_panel is None and drag._drag_timer is None
    macos._restore_panel_drag.assert_not_called()


def test_synchronous_native_drag_restores_after_mouse_release(native_drag):
    drag, windows, _, _ = native_drag
    native = windows['hud_panel'].native
    native.performWindowDragWithEvent_.side_effect = lambda _: setattr(AppKit.NSEvent.pressedMouseButtons, 'return_value', 0)
    drag._handle(event(native))
    macos._restore_panel_drag.assert_called_once_with(native)
    assert drag._drag_timer is None
