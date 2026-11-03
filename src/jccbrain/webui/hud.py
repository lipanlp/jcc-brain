"""惰性创建的透明 HUD；原生显示不激活应用，窗口生命周期从属于主窗口。"""
from __future__ import annotations

import logging
import threading
from uuid import uuid4

from jccbrain.webui import macos

log = logging.getLogger(__name__)
FOLLOW_INTERVAL = 0.3


class _ScoutNoticeBridge:
    """子窗仅有这两个业务入口；窗口身份由 Python 绑定，JS 不能指定。"""
    def __init__(self, controller, key, identity):
        self._controller, self._key, self._identity = controller, key, identity

    def scout_notice_shown(self, notice_id):
        return {'ok': True, 'data': self._controller._notice_call(self._key, notice_id, False, self._identity)}

    def scout_notice_cancel(self, notice_id):
        return {'ok': True, 'data': self._controller._notice_call(self._key, notice_id, True, self._identity)}


class HudController:
    def __init__(self, ctx, bus, base_url):
        self._ctx, self._bus, self._base = ctx, bus, base_url.split('#', 1)[0]
        self._lock = threading.RLock()
        self._windows = {}
        self._loaded = set()
        self._active = {'hud': False, 'hud_panel': False, 'scout_toast': False}
        self._visible = {'hud': False, 'hud_panel': False, 'scout_toast': False}
        self._toast_identity = None
        self._toast_id = None
        self._toast_window = None
        self._game_info = None
        self._frame = None
        self._panel_origin = None
        self._refresh = None
        self._notice_ack = self._notice_cancel = None
        self._stopped = threading.Event()
        self._thread = None

    @property
    def status(self):
        with self._lock:
            return {'overlay': self._active['hud'], 'panel': self._active['hud_panel']}

    def set_refresh(self, callback):
        self._refresh = callback

    def set_scout_notice(self, ack, cancel):
        self._notice_ack, self._notice_cancel = ack, cancel

    def notice_capable(self):
        return not self._stopped.is_set()

    def set_notice(self, notice):
        with self._lock:
            if self._stopped.is_set():
                return
            if notice and notice['id'] != self._toast_id:
                self._create('scout_toast')
                self._toast_id = notice['id']
                info = self._ctx.locator.find(0)
                self._toast_window = (info.pid, info.window_id) if info else None
                self._active['scout_toast'] = True
                self._start_follow()
                self._follow(info)
            elif not notice:
                self._toast_id = None
                self._active['scout_toast'] = False
                self._follow(self._game_info)

    def notice_ready(self, identity=None):
        with self._lock:
            win = self._windows.get('scout_toast')
            notice_id = self._toast_id
            ready = bool(identity == self._toast_identity and identity is not None
                         and not self._stopped.is_set() and notice_id
                         and 'scout_toast' in self._loaded and self._active['scout_toast']
                         and self._visible['scout_toast'] and win)
        # 不持控制器锁等待 AppKit；关闭/加载回调不能形成反向锁等待。
        if not ready or not macos.toast_visible(win):
            return False
        with self._lock:
            return bool(not self._stopped.is_set() and self._windows.get('scout_toast') is win
                        and self._toast_id == notice_id and self._toast_identity == identity
                        and self._active['scout_toast'] and self._visible['scout_toast'])

    def _notice_call(self, key, notice_id, cancel, identity=None):
        if key != 'scout_toast' or not isinstance(notice_id, str) or not self.notice_ready(identity):
            return False
        with self._lock:
            if notice_id != self._toast_id:
                return False
        callback = self._notice_cancel if cancel else self._notice_ack
        # 不持原生窗口锁调用模式，避免 UI 刷新回调反向取锁。
        return bool(callback and callback(notice_id, identity))

    def _drag_window(self):
        with self._lock:
            return self._windows.get('hud_panel')

    def _remember_panel_position(self, origin):
        with self._lock:
            self._panel_origin = origin

    def _create(self, key):
        import webview

        if key in self._windows:
            return self._windows[key]
        overlay = key != 'hud_panel'
        toast = key == 'scout_toast'
        if toast:
            self._toast_identity = 'scout_toast:' + uuid4().hex
        win = webview.create_window(
            '自动侦察提示' if toast else '金铲铲大脑 HUD' if overlay else '目标卡概率',
            self._base + ('#/hud-toast' if toast else '#/hud' if overlay else '#/hud-panel'),
            width=300, height=220, min_size=(1, 1), frameless=True, transparent=True,
            on_top=True, focus=False, hidden=True, easy_drag=False, resizable=False,
            js_api=_ScoutNoticeBridge(self, key, self._toast_identity) if toast else None,
        )
        self._windows[key] = win
        win.events.shown += lambda: (macos.make_overlay(win) if overlay else macos.make_panel(win))
        win.events.loaded += lambda: self._ready(key, win)
        if toast:
            win.events.closed += lambda: self._toast_closed(win)
        return win

    def _toast_closed(self, win):
        with self._lock:
            if self._windows.get('scout_toast') is not win:
                return
            self._windows.pop('scout_toast', None)
            self._loaded.discard('scout_toast')
            self._visible['scout_toast'] = self._active['scout_toast'] = False
            self._toast_identity = None
        self._bus.detach('scout_toast')

    def _ready(self, key, win):
        with self._lock:
            if self._stopped.is_set() or self._windows.get(key) is not win:
                return
            if key == 'scout_toast':
                macos.make_overlay(win)
            self._bus.attach(win.evaluate_js, key=key)
            self._loaded.add(key)
            self._follow(self._game_info)
        if self._refresh:
            self._refresh()

    def _start_follow(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='webui-hud-follow', daemon=True)
            self._thread.start()

    def toggle_overlay(self, on, info=None):
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError('窗口正在关闭')
            if on:
                self._create('hud')
            self._active['hud'] = on
            if on:
                self._start_follow()
                self._follow(info if info is not None else self._game_info)
            elif 'hud' in self._loaded:
                self._windows['hud'].hide()
                self._visible['hud'] = False

    def toggle_panel(self, on):
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError('窗口正在关闭')
            if on and 'hud_panel' not in self._windows:
                win = self._create('hud_panel')
                # shown 只触发首次放置，以后隐藏/恢复保留用户拖动后的位置。
                placed = False
                def place_once():
                    nonlocal placed
                    with self._lock:
                        if placed or self._stopped.is_set():
                            return
                        placed = True
                        info = self._game_info
                        frame = ((round(info.x + info.width - 320), round(info.y + 80), 300, 220)
                                 if info else macos.screen_panel_frame())
                        macos.place_hud(win, frame)
                win.events.shown += place_once
            self._active['hud_panel'] = on
            if on:
                self._start_follow()
            self._follow(self._game_info)

    def _follow(self, info):
        self._game_info = info
        for key in self._loaded:
            win = self._windows[key]
            invalid_toast = key == 'scout_toast' and (info is None or (info.pid, info.window_id) != self._toast_window)
            if not self._active[key] or info is None or invalid_toast:
                if self._visible[key]:
                    if key == 'hud_panel':
                        macos.hide_panel(win, self._remember_panel_position)
                    else:
                        win.hide()
                        if key == 'scout_toast':
                            macos.clear_toast_visibility(win)
                    self._visible[key] = False
                continue
            if key in ('hud', 'scout_toast'):
                frame = tuple(round(value) for value in (info.x, info.y, info.width, info.height))
                if key == 'scout_toast' or frame != self._frame:
                    macos.place_hud(win, frame)
                    if key == 'hud':
                        self._frame = frame
            if not self._visible[key]:
                if key == 'hud_panel' and self._panel_origin is not None:
                    macos.restore_panel_origin(win, self._panel_origin)
                if key == 'scout_toast':
                    macos.show_toast(win, info.pid)
                else:
                    macos.show_hud(win)
                self._visible[key] = True
            if key == 'scout_toast':
                macos.check_toast_visibility(win, info.pid)
            else:
                macos.ensure_hud_above(win, key == 'hud', info.pid)

    def _run(self):
        while not self._stopped.wait(FOLLOW_INTERVAL):
            try:
                # Quartz 查询只在跟随线程执行；loaded 回调使用缓存。
                info = self._ctx.locator.find(0)
                with self._lock:
                    if self._stopped.is_set():
                        return
                    self._follow(info)
            except Exception:
                log.exception('跟随游戏窗口失败')

    def close(self):
        self._stopped.set()
        if self._thread and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.2)
        with self._lock:
            windows = list(self._windows.items())
            self._windows.clear()
            self._loaded.clear()
            self._active = {'hud': False, 'hud_panel': False, 'scout_toast': False}
        for key, win in windows:
            self._bus.detach(key)
            win.destroy()
