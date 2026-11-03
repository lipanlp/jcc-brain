"""图形界面入口：主窗口、HUD 和应用生命周期。"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from time import perf_counter

import webview

from jccbrain.context import AppContext
from jccbrain.webui.api import Api
from jccbrain.webui.events import EventBus, WebLogHandler
from jccbrain.webui.hud import HudController
from jccbrain.webui.macos import MainWindowStartup

log = logging.getLogger(__name__)


def run_webui() -> int:
    started = perf_counter()

    def timeline(stage, level=logging.DEBUG):
        log.log(level, '界面启动 +%.0fms %s', (perf_counter() - started) * 1000, stage)

    dev = os.environ.get('JCC_WEB_DEV') == '1'
    index = Path(__file__).with_name('static') / 'index.html'
    if not dev and not index.is_file():
        raise RuntimeError('界面尚未构建：请先运行 make web（打包版不应出现此错误）')
    timeline('AppContext 创建开始')
    ctx = AppContext()
    timeline('AppContext 创建完成')
    bus = EventBus()
    base_url = 'http://localhost:5173' if dev else str(index)
    hud = HudController(ctx, bus, base_url)
    api = Api(ctx, bus, hud=hud)
    handler = WebLogHandler(bus)
    logging.getLogger().addHandler(handler)
    closed = threading.Event()
    cleanup_lock = threading.Lock()
    load_lock = threading.Lock()
    timer = None
    startup = None

    def auto_sync():
        if not closed.is_set() and ctx.settings.data.auto_sync:
            timeline('auto_sync 开始', logging.INFO)
            log.info('后台检查各赛季数据是否需要更新…')
            result = api.sync(True, all_seasons=True)
            if result.get('ok') and result.get('data', {}).get('started'):
                worker = api._sync_thread
                if worker is not None:
                    worker.join()
            timeline('auto_sync 结束', logging.INFO)

    def loaded():
        nonlocal timer
        if closed.is_set():
            return
        timeline('loaded', logging.INFO)
        try:
            bus.attach(window.evaluate_js)
            api._emit_status()
            timeline('首次 status 已推送', logging.INFO)
        except Exception:
            timeline('首次 status 推送失败', logging.ERROR)
            log.exception('初始界面状态读取失败，界面可重新读取')
        finally:
            with load_lock:
                if timer is None and not closed.is_set():
                    timer = threading.Timer(1.5, auto_sync)
                    timer.daemon = True
                    timer.start()
                    timeline('后台同步检查已安排（1.5s）', logging.INFO)

    def cleanup():
        with cleanup_lock:
            if closed.is_set():
                return
            closed.set()
            if startup is not None:
                startup.close()
            bus.detach('main')
            with load_lock:
                if timer is not None:
                    timer.cancel()
            try:
                api._shutdown()
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
                bus.close()
                ctx.close()

    try:
        timeline('窗口创建开始')
        window = webview.create_window(
            '金铲铲大脑', base_url, js_api=api,
            width=1280, height=820, min_size=(960, 640),
        )
        timeline('窗口创建完成', logging.INFO)
        api.set_window(window)
        startup = MainWindowStartup(window)
        window.events.loaded += loaded
        window.events.shown += startup.shown
        window.events.closed += cleanup
        webview.start(debug=dev)
    finally:
        # closed 的工作线程可能只来得及排队；runloop 停止后在主线程兜底撤销观察者。
        if startup is not None:
            startup.close()
        cleanup()
    return 0
