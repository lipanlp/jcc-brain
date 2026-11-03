"""HUD 控制器使用假的窗口工厂；不创建 webview 或操作桌面。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jccbrain.actuator.window import WindowInfo
from jccbrain.webui.hud import HudController


class Event:
    def __init__(self):
        self.handlers = []
    def __iadd__(self, callback):
        self.handlers.append(callback)
        return self
    def fire(self):
        for callback in self.handlers:
            callback()


@pytest.fixture
def hud(monkeypatch):
    from jccbrain.webui import macos
    import webview
    windows = []
    def create(title, url, **options):
        win = SimpleNamespace(title=title, url=url, options=options, events=SimpleNamespace(shown=Event(), loaded=Event(), closed=Event()),
                              evaluate_js=Mock(), hide=Mock(), destroy=Mock())
        windows.append(win)
        return win
    monkeypatch.setattr(webview, 'create_window', create)
    natives = {name: Mock() for name in ('make_overlay', 'make_panel', 'show_hud', 'place_hud',
                                       'ensure_hud_above', 'restore_panel_origin', 'hide_panel')}
    for name, mock in natives.items():
        monkeypatch.setattr(macos, name, mock)
    monkeypatch.setattr(macos, 'show_toast', lambda win,pid:setattr(win,'toast_visible',True))
    monkeypatch.setattr(macos, 'check_toast_visibility', lambda *a:None)
    monkeypatch.setattr(macos, 'toast_visible', lambda win:getattr(win,'toast_visible',False))
    monkeypatch.setattr(macos, 'clear_toast_visibility', lambda win:setattr(win,'toast_visible',False))
    monkeypatch.setattr(macos, 'screen_panel_frame', lambda: (900, 80, 300, 220))
    info = WindowInfo('Preview', '截图', 1, 50, 80, 1000, 700, 42)
    ctx = SimpleNamespace(locator=SimpleNamespace(find=Mock(return_value=info)))
    bus = SimpleNamespace(attach=Mock(), detach=Mock())
    controller = HudController(ctx, bus, '/static/index.html')
    # 模拟跟随线程已得到的快照；窗口事件不再同步查询 Quartz。
    controller._game_info = info
    monkeypatch.setattr(controller, '_start_follow', Mock())
    yield controller, windows, ctx, bus, natives
    controller.close()
    if controller._thread:
        assert not controller._thread.is_alive()


def test_overlay_lazy_creation_follow_hide_restore_and_reuse(hud):
    controller, windows, ctx, bus, native = hud
    assert not windows
    controller.toggle_overlay(True)
    win = windows[0]
    assert win.url == '/static/index.html#/hud'
    assert all(win.options[key] for key in ('frameless', 'transparent', 'on_top', 'hidden'))
    assert not win.options['focus']
    win.events.shown.fire()
    win.events.loaded.fire()
    bus.attach.assert_called_with(win.evaluate_js, key='hud')
    native['place_hud'].assert_called_with(win, (50, 80, 1000, 700))
    native['make_overlay'].assert_called_once_with(win)
    ctx.locator.find.return_value = None
    controller._follow(ctx.locator.find(0))
    win.hide.assert_called_once()
    assert controller.status['overlay']  # 暂时找不到仍保持用户开关，稍后恢复
    ctx.locator.find.return_value = WindowInfo('Preview', '', 1, 90, 120, 900, 600, 43)
    controller._follow(ctx.locator.find(0))
    native['place_hud'].assert_called_with(win, (90, 120, 900, 600))
    assert native['show_hud'].call_count == 2
    controller.toggle_overlay(False)
    controller.toggle_overlay(True)
    assert len(windows) == 1
    controller.close()
    bus.detach.assert_called_with('hud')
    win.destroy.assert_called_once()


def test_panel_remembers_position_and_does_not_repeat_placement(hud):
    controller, windows, _, _, native = hud
    controller.toggle_panel(True)
    win = windows[0]
    win.events.shown.fire()
    win.events.loaded.fire()
    native['place_hud'].assert_called_once_with(win, (730, 160, 300, 220))
    controller.toggle_panel(False)
    native['hide_panel'].assert_called_once_with(win, controller._remember_panel_position)
    native['hide_panel'].call_args.args[1]((-400, 200))  # 主线程在隐藏前读回实际原点。
    controller.toggle_panel(True)
    assert len(windows) == 1
    assert native['place_hud'].call_count == 1 and native['show_hud'].call_count == 2
    native['restore_panel_origin'].assert_called_once_with(win, (-400, 200))
    assert controller._drag_window() is win


def test_panel_fallback_and_late_loaded_after_close(hud):
    controller, windows, ctx, bus, native = hud
    ctx.locator.find.return_value = None
    controller._follow(None)
    controller.toggle_panel(True)
    win = windows[0]
    win.events.shown.fire()
    native['place_hud'].assert_called_with(win, (900, 80, 300, 220))
    controller.close()
    win.events.loaded.fire()
    bus.attach.assert_not_called()
    assert controller.status == {'overlay': False, 'panel': False}


def test_follow_interval_and_locator_run_in_background(hud):
    import threading
    from jccbrain.webui.hud import FOLLOW_INTERVAL

    controller, _, ctx, _, _ = hud
    assert 0 < FOLLOW_INTERVAL <= .5
    waited = Mock(side_effect=[False, True])
    controller._stopped = SimpleNamespace(wait=waited, is_set=lambda: False, set=Mock())
    called_from = []
    ctx.locator.find.side_effect = lambda age: called_from.append(threading.get_ident()) or None
    worker = threading.Thread(target=controller._run)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert called_from == [worker.ident] and worker.ident != threading.get_ident()
    ctx.locator.find.assert_called_once_with(0)
    assert waited.call_args_list == [((FOLLOW_INTERVAL,),), ((FOLLOW_INTERVAL,),)]


def test_missing_game_hides_both_and_restores_without_moving_panel(hud):
    controller, windows, _, _, native = hud
    controller.toggle_overlay(True)
    controller.toggle_panel(True)
    for win in windows:
        win.events.shown.fire()
        win.events.loaded.fire()
    info = controller._game_info
    native['place_hud'].reset_mock()
    native['show_hud'].reset_mock()
    controller._follow(None)
    windows[0].hide.assert_called_once()
    native['hide_panel'].assert_called_once_with(windows[1], controller._remember_panel_position)
    native['hide_panel'].call_args.args[1]((-250, 160))
    assert controller.status == {'overlay': True, 'panel': True}
    controller._follow(info)
    assert native['show_hud'].call_count == 2
    native['restore_panel_origin'].assert_called_once_with(windows[1], (-250, 160))
    native['place_hud'].assert_not_called()
    # 恢复 shown 事件也不能把用户拖过的位置重置。
    windows[1].events.shown.fire()
    native['place_hud'].assert_not_called()


def test_panel_closed_while_game_missing_is_not_restored(hud):
    controller, windows, _, _, native = hud
    controller.toggle_panel(True)
    controller._start_follow.assert_called_once()
    win = windows[0]
    win.events.shown.fire()
    win.events.loaded.fire()
    info = controller._game_info
    controller._follow(None)
    native['hide_panel'].assert_called_once_with(win, controller._remember_panel_position)
    controller.toggle_panel(False)
    native['show_hud'].reset_mock()
    controller._follow(info)
    native['show_hud'].assert_not_called()


def test_window_callbacks_use_cached_game_info_without_exposing_business_api(hud):
    controller, windows, ctx, _, _ = hud
    controller.toggle_overlay(True)
    controller.toggle_panel(True)
    for win in windows:
        win.events.shown.fire()
        win.events.loaded.fire()
    ctx.locator.find.assert_not_called()
    for win in windows:
        bridge = win.options['js_api']
        assert bridge is None  # 一般HUD没有侦察授权入口


@pytest.mark.parametrize('overlay,expected', [(True, 8), (False, 9)])
def test_hud_level_is_modal_plus_panel_offset(overlay, expected):
    from jccbrain.webui.macos import hud_level
    assert hud_level(overlay) == expected


@pytest.fixture
def native_hud(monkeypatch):
    import sys
    from jccbrain.webui import macos

    front = SimpleNamespace(processIdentifier=Mock(return_value=123456), activateWithOptions_=Mock())
    workspace = SimpleNamespace(frontmostApplication=Mock(return_value=front))
    appkit = SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace),
                             NSApplicationActivateIgnoringOtherApps=2)
    monkeypatch.setitem(sys.modules, 'AppKit', appkit)
    monkeypatch.setattr(macos, 'sys', SimpleNamespace(platform='darwin'))
    queued = []
    monkeypatch.setattr(macos, '_hud_main', lambda window, action: queued.append(lambda: action(window.native)))
    native = SimpleNamespace(isVisible=Mock(return_value=True), level=Mock(return_value=8),
                             isOnActiveSpace=Mock(return_value=True), setLevel_=Mock(), orderFrontRegardless=Mock())
    return macos, SimpleNamespace(native=native), front, workspace, queued, appkit


@pytest.mark.parametrize('visible,level,space', [(False, 8, True), (True, 3, True), (True, 8, False)])
def test_native_follower_repairs_level_or_visibility_without_activation(native_hud, visible, level, space):
    macos, win, front, _, queued, _ = native_hud
    win.native.isVisible.return_value = visible
    win.native.level.return_value = level
    win.native.isOnActiveSpace.return_value = space
    macos.ensure_hud_above(win, True, 123456)
    win.native.orderFrontRegardless.assert_not_called()  # 主线程队列尚未执行
    queued.pop(0)()
    win.native.setLevel_.assert_called_once_with(8)
    win.native.orderFrontRegardless.assert_called_once()
    front.activateWithOptions_.assert_not_called()


def test_native_follower_leaves_frontmost_other_app_alone(native_hud):
    macos, win, front, _, queued, _ = native_hud
    win.native.isVisible.return_value = False
    macos.ensure_hud_above(win, False, 999999)
    queued.pop(0)()
    win.native.orderFrontRegardless.assert_not_called()
    front.activateWithOptions_.assert_not_called()


@pytest.mark.parametrize('recorded', ['external', 'self', 'none'])
def test_drag_end_only_activates_recorded_external_app_once(native_hud, recorded):
    import os
    macos, win, front, _, queued, _ = native_hud
    front.processIdentifier.return_value = os.getpid() if recorded == 'self' else 123456
    win.native._jcc_drag_app = None if recorded == 'none' else front
    macos.panel_drag_end(win)
    queued.pop(0)()
    assert front.activateWithOptions_.call_count == (1 if recorded == 'external' else 0)
    if recorded == 'external':
        front.activateWithOptions_.assert_called_once_with(2)
    macos.panel_drag_end(win)
    queued.pop(0)()
    assert front.activateWithOptions_.call_count == (1 if recorded == 'external' else 0)


def test_drag_start_remembers_front_before_click_activation(native_hud):
    import os
    macos, win, game, workspace, queued, _ = native_hud
    macos.ensure_hud_above(win, False, 123456)
    queued.pop(0)()
    workspace.frontmostApplication.return_value = SimpleNamespace(processIdentifier=lambda: os.getpid())
    macos.panel_drag_start(win)
    queued.pop(0)()
    assert win.native._jcc_drag_app is game
    macos.panel_drag_end(win)
    queued.pop(0)()
    game.activateWithOptions_.assert_called_once_with(2)


def test_native_style_sets_borderless_overlay_and_compatible_collection(native_hud, monkeypatch):
    import sys
    macos, win, _, _, queued, appkit = native_hud
    for name, value in {'NSModalPanelWindowLevel': 8, 'NSWindowStyleMaskBorderless': 0,
                        'NSWindowCollectionBehaviorCanJoinAllSpaces': 1,
                        'NSWindowCollectionBehaviorFullScreenAuxiliary': 256,
                        'NSWindowCollectionBehaviorStationary': 16,
                        'NSWindowCollectionBehaviorIgnoresCycle': 64}.items():
        setattr(appkit, name, value)
    appkit.NSColor = SimpleNamespace(clearColor=lambda: 'clear')
    monkeypatch.setitem(sys.modules, 'webview.platforms.cocoa', SimpleNamespace(
        BrowserView=SimpleNamespace(instances={})))
    native = win.native
    win.uid = 'fake'
    for name in ('setIgnoresMouseEvents_', 'setOpaque_', 'setBackgroundColor_', 'setHasShadow_',
                 'setHidesOnDeactivate_', 'setExcludedFromWindowsMenu_', 'setCollectionBehavior_',
                 'frame', 'setStyleMask_', 'setFrame_display_'):
        setattr(native, name, Mock())
    macos.make_overlay(win)
    queued.pop(0)()
    native.setLevel_.assert_called_once_with(8)
    native.setIgnoresMouseEvents_.assert_called_once_with(True)
    native.setStyleMask_.assert_called_once_with(0)
    native.setHidesOnDeactivate_.assert_called_once_with(False)
    native.setCollectionBehavior_.assert_called_once_with(337)  # 旧系统缺常量时仍可用
    assert native.focus is False
    appkit.NSWindowCollectionBehaviorCanJoinAllApplications = 1 << 18
    macos.make_panel(win)
    queued.pop(0)()
    assert native.setLevel_.call_args.args == (9,)
    assert native.setCollectionBehavior_.call_args.args == (337 | 1 << 18,)
    assert native.focus is True


def test_drag_and_level_helpers_are_noop_off_macos(monkeypatch):
    from jccbrain.webui import macos
    monkeypatch.setattr(macos, 'sys', SimpleNamespace(platform='linux'))
    queue = Mock()
    monkeypatch.setattr(macos, '_hud_main', queue)
    macos.panel_drag_start(object())
    macos.panel_drag_end(object())
    macos.ensure_hud_above(object(), True, 1)
    queue.assert_not_called()


def test_hud_check_without_game_writes_skip_report(native_hud, monkeypatch, tmp_path, capsys):
    import json
    import runpy
    import sys
    from pathlib import Path
    from jccbrain.actuator import window

    script = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/check_hud_level.py'))
    monkeypatch.setitem(script['main'].__globals__, 'sys', SimpleNamespace(platform='darwin'))
    output = tmp_path / 'no-game.json'
    monkeypatch.setattr(sys, 'argv', ['check_hud_level.py', '--out', str(output)])
    monkeypatch.setattr(window, 'WindowLocator', lambda owner: SimpleNamespace(find=lambda age: None))
    assert script['main']() == 2
    report = json.loads(output.read_text())
    assert report['exit_code'] == 2 and report['skipped'] == '未找到游戏窗口'
    assert report['cycles'] == []
    assert '未找到游戏窗口' in capsys.readouterr().out


def test_hud_check_rejects_less_than_five_cycles(monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    script = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/check_hud_level.py'))
    monkeypatch.setattr(sys, 'argv', ['check_hud_level.py', '--cycles', '4'])
    with pytest.raises(SystemExit) as exc:
        script['main']()
    assert exc.value.code == 2


def test_drag_end_falls_back_to_recorded_pid_when_cocoa_refuses(native_hud, monkeypatch):
    macos, win, front, _, queued, _ = native_hud
    front.activateWithOptions_.return_value = False
    win.native._jcc_drag_app = front
    activate = Mock(return_value=True)
    monkeypatch.setattr(macos, '_activate_by_pid', activate)
    monkeypatch.setattr(macos, 'threading', SimpleNamespace(Thread=lambda **kw: SimpleNamespace(
        start=lambda: kw['target'](*kw['args']))))
    macos.panel_drag_end(win)
    queued.pop(0)()
    activate.assert_called_once_with(123456)
    assert win.native._jcc_drag_app is None


def test_activation_helper_only_calls_appkit_for_target_pid(native_hud, monkeypatch):
    import os
    import subprocess
    macos, _, _, _, _, _ = native_hud
    monkeypatch.setattr(macos, 'sys', SimpleNamespace(platform='darwin', executable='test-python'))
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(subprocess, 'run', run)
    assert not macos._activate_by_pid(os.getpid())
    run.assert_not_called()
    assert macos._activate_by_pid(123456)
    command = run.call_args.args[0]
    assert command[:2] == ['test-python', '-c'] and command[-1] == '123456'
    assert 'NSRunningApplication' in command[2] and 'activateWithOptions_' in command[2]
    assert run.call_args.kwargs['timeout'] == 5
    assert 'shell' not in run.call_args.kwargs


def test_scout_bridge_only_accepts_loaded_active_visible_toast_and_close(hud):
    controller, windows, _, _, _ = hud
    ack, cancel = Mock(return_value=True), Mock(return_value=True)
    controller.set_scout_notice(ack, cancel)
    controller.set_notice({'id':'id'})
    win = windows[0]; bridge = win.options['js_api']
    identity = controller._toast_identity
    assert not controller.notice_ready(identity)
    assert bridge.scout_notice_shown('id')['data'] is False
    win.events.loaded.fire()
    assert controller.notice_ready(identity)
    assert not controller.notice_ready('hud') and not controller.notice_ready('hud_panel')
    assert not controller._notice_call('scout_toast','id',False,'forged')
    assert bridge.scout_notice_shown('id')['data'] is True
    ack.assert_called_once_with('id', identity)
    assert bridge.scout_notice_cancel('id')['data'] is True
    cancel.assert_called_once_with('id', identity)
    assert bridge.scout_notice_shown(None)['data'] is False
    controller.set_notice(None)
    assert bridge.scout_notice_shown('id')['data'] is False
    controller.set_notice({'id':'new'})
    controller._follow(None)
    assert not controller.notice_ready(identity)
    assert bridge.scout_notice_cancel('new')['data'] is False
    controller.close()
    assert not controller.notice_ready(identity)
    assert bridge.scout_notice_shown('new')['data'] is False


def test_toast_is_independent_of_normal_toggles_and_native_close_recreates_identity(hud):
    controller,windows,ctx,bus,native=hud
    controller.set_notice({'id':'one'})
    first=windows[-1];identity=controller._toast_identity
    assert controller.status=={'overlay':False,'panel':False}
    assert first.url.endswith('#/hud-toast') and not first.options['focus']
    assert first.options['transparent'] and first.options['frameless']
    first.events.loaded.fire()
    assert controller.notice_ready(identity)
    native['make_overlay'].assert_called_with(first)
    controller.toggle_panel(True);windows[-1].events.loaded.fire()
    controller.toggle_panel(False)
    assert controller.notice_ready(identity)
    first.toast_visible=False
    assert not controller.notice_ready(identity)
    first.toast_visible=True
    first.events.closed.fire()
    assert not controller.notice_ready(identity)
    controller.set_notice({'id':'two'})
    second=windows[-1];second.events.loaded.fire()
    assert second is not first and controller._toast_identity!=identity
    assert not controller.notice_ready(identity)
    first.events.loaded.fire()  # late old window load cannot attach itself again
    assert controller._windows['scout_toast'] is second
    assert controller.notice_ready(controller._toast_identity)
    controller.set_notice(None)
    assert not controller.notice_ready(controller._toast_identity)
    controller.set_notice({'id':'three'})
    controller._follow(None)
    assert not controller.notice_ready(controller._toast_identity)


def test_toast_frame_does_not_consume_overlay_resize(hud):
    controller,windows,ctx,bus,native=hud
    controller.set_notice({'id':'one'});toast=windows[-1];toast.events.loaded.fire()
    controller.toggle_overlay(True);overlay=windows[-1];overlay.events.loaded.fire()
    native['place_hud'].assert_any_call(overlay,(50,80,1000,700))
