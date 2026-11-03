import logging
import subprocess
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from jccbrain.actuator.window import WindowInfo
from jccbrain.vision.eye import capture as C
from jccbrain.vision.pyeye import PyEyeProvider

WIN = WindowInfo('game', 'game', 1, 0, 0, 80, 60, 123)
IMAGE = np.zeros((60, 80, 3), np.uint8)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(C, '_CG_WORKER', C._CaptureWorker('test-cg'))
    monkeypatch.setattr(C, '_SC_WORKER', C._CaptureWorker('test-sc'))
    monkeypatch.setattr(C, '_CLI_WORKER', C._CaptureWorker('test-cli'))
    monkeypatch.setattr(C.WindowCapture, '_window', lambda self: WIN)
    monkeypatch.setattr(C.WindowCapture, '_cg_image', lambda self, win: IMAGE)
    monkeypatch.setattr(C.WindowCapture, '_sc_image', lambda self, win, deadline: IMAGE + 1)
    monkeypatch.setattr(C.WindowCapture, '_cli_image', lambda self, win: IMAGE + 2)


def test_timeout_is_bounded_and_pending_call_never_starts_another(native, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def stuck(self, win):
        calls.append(win.window_id)
        entered.set()
        release.wait()
        return IMAGE
    monkeypatch.setattr(C.WindowCapture, '_cg_image', stuck)
    monkeypatch.setattr(C, 'CAPTURE_TIMEOUT', .03)
    capture = C.WindowCapture(None, backend='cg')
    start = time.monotonic()
    try:
        with pytest.raises(C.CaptureError, match='截图超时'):
            capture.grab()
        assert entered.is_set() and time.monotonic() - start < .5
        worker_thread = C._CG_WORKER._thread
        for _ in range(10):
            with pytest.raises(C.CaptureTimeout):
                C.WindowCapture(None, backend='cg').grab()
        assert calls == [123] and C._CG_WORKER._thread is worker_thread and worker_thread.daemon
    finally:
        release.set()
        assert C._CG_WORKER._pending.done.wait(1)
    monkeypatch.setattr(C.WindowCapture, '_cg_image', lambda self, win: IMAGE + 7)
    assert capture.grab()[0][0, 0, 0] == 7


def test_two_failures_switch_once_to_sc_and_60_seconds_retry_restores_cg(native, monkeypatch, caplog):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(C.time, 'monotonic', lambda: clock.now)
    attempts = []
    def cg(self, win):
        attempts.append(clock.now)
        if len(attempts) <= 2:
            raise C.CaptureError('CG failed')
        return IMAGE
    monkeypatch.setattr(C.WindowCapture, '_cg_image', cg)
    capture = C.WindowCapture(None)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(C.CaptureError, match='CG failed'):
            capture.grab()
        assert capture.grab()[0][0, 0, 0] == 1 and capture.backend == 'sc'
        clock.now = 159.9
        capture.grab()
        assert len(attempts) == 2
        clock.now = 160.
        assert capture.grab()[0][0, 0, 0] == 0 and capture.backend == 'cg'
    assert sum('[CAPTURE]' in row.message for row in caplog.records) == 2


def test_cg_success_resets_consecutive_failure_count(native, monkeypatch):
    results = iter([C.CaptureError('fail'), IMAGE, C.CaptureError('fail')])
    def cg(self, win):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(C.WindowCapture, '_cg_image', cg)
    capture = C.WindowCapture(None)
    with pytest.raises(C.CaptureError):
        capture.grab()
    capture.grab()
    with pytest.raises(C.CaptureError):
        capture.grab()
    assert capture.backend == 'cg'


def test_reprobe_failure_stays_sc_and_sc_failure_propagates(native, monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(C.time, 'monotonic', lambda: clock.now)
    attempts = []
    def fail(self, win):
        attempts.append(clock.now)
        raise C.CaptureError('CG failed')
    monkeypatch.setattr(C.WindowCapture, '_cg_image', fail)
    capture = C.WindowCapture(None)
    with pytest.raises(C.CaptureError):
        capture.grab()
    capture.grab()
    clock.now = 160.
    capture.grab()
    clock.now = 160.1
    capture.grab()
    assert attempts == [100., 100., 160.] and capture.backend == 'sc'
    def fail_sc(self, win, deadline):
        raise C.CaptureError('SC failed')
    monkeypatch.setattr(C.WindowCapture, '_sc_image', fail_sc)
    with pytest.raises(C.CaptureError, match='SC failed'):
        capture.grab()


def test_status_exposes_selected_capture_backend(native, tmp_path):
    provider = PyEyeProvider(gallery_dir=tmp_path)
    assert provider.status()['capture_backend'] == 'cg'
    provider._capture = C.WindowCapture(SimpleNamespace(find=lambda: WIN), backend='sc')
    assert provider.status()['capture_backend'] == 'sc'


def test_sc_missing_callback_stays_in_flight_and_late_callback_is_discarded():
    worker = C._CaptureWorker('test-sc-callback')
    callbacks = []
    def request():
        return C._await_sc(callbacks.append, time.monotonic() + .03)
    try:
        with pytest.raises(C.CaptureTimeout):
            worker.call(request, .05)
        for _ in range(10):
            with pytest.raises(C.CaptureTimeout):
                worker.call(request, .05)
        assert len(callbacks) == 1 and worker._thread.daemon
    finally:
        if callbacks:
            callbacks[0]('late image', None)
            assert worker._pending.done.wait(1)
    assert isinstance(worker._pending.error, C.CaptureTimeout)
    assert worker.call(lambda: 'fresh image', .5) == 'fresh image'


def test_sc_uses_same_window_id_point_size_cache_and_coregraphics_conversion(monkeypatch):
    calls = SimpleNamespace(content=0, filters=[], configs=[], converted=[])
    windows = [SimpleNamespace(windowID=lambda: 123), SimpleNamespace(windowID=lambda: 456)]
    def content(a, b, callback):
        calls.content += 1
        callback(SimpleNamespace(windows=lambda: windows), None)
    def filter_window(window):
        calls.filters.append(window.windowID())
        return window
    class Config:
        def __init__(self): calls.configs.append(self)
        def setWidth_(self, value): self.width = value
        def setHeight_(self, value): self.height = value
        def setShowsCursor_(self, value): self.cursor = value
        def setIgnoreShadowsSingleWindow_(self, value): self.shadows = value
        def setIgnoreGlobalClipSingleWindow_(self, value): self.clip = value
    module = SimpleNamespace(
        SCShareableContent=SimpleNamespace(getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_=content),
        SCContentFilter=SimpleNamespace(alloc=lambda: SimpleNamespace(initWithDesktopIndependentWindow_=filter_window)),
        SCStreamConfiguration=SimpleNamespace(alloc=lambda: SimpleNamespace(init=Config)),
        SCScreenshotManager=SimpleNamespace(captureImageWithFilter_configuration_completionHandler_=lambda f, c, callback: callback('CGImage', None)))
    monkeypatch.setitem(sys.modules, 'ScreenCaptureKit', module)
    def convert(image):
        calls.converted.append(image)
        return IMAGE
    monkeypatch.setattr(C, 'cgimage_to_bgr', convert)
    capture = C.WindowCapture(None, backend='sc')
    for win in [WIN, WIN, replace(WIN, window_id=456, width=90)]:
        assert capture._sc_image(win, time.monotonic() + 1) is IMAGE
    assert calls.content == 2 and calls.filters == [123, 123, 456]
    assert [(c.width, c.height, c.cursor, c.shadows, c.clip) for c in calls.configs] == [
        (80, 60, False, True, True), (80, 60, False, True, True), (90, 60, False, True, True)]
    assert calls.converted == ['CGImage'] * 3


def test_capture_check_reports_pair_difference_timeout_and_size_mismatch():
    from scripts.capture_check import check
    class Fake:
        def __init__(self, rows): self.rows = iter(rows)
        def grab(self):
            value = next(self.rows)
            if isinstance(value, Exception): raise value
            return value, WIN
    report = check({'cg': Fake([IMAGE, C.CaptureTimeout(), IMAGE]),
                    'sc': Fake([IMAGE + 3, IMAGE, IMAGE[:20]])}, 3)
    assert report['backends']['cg']['success_rate'] == 2 / 3
    assert report['backends']['cg']['timeouts'] == 1
    assert report['pairs'][0]['same_size'] and report['pairs'][0]['mean_abs_pixel_diff'] == 3.
    assert report['pairs'][1]['mean_abs_pixel_diff'] is None and not report['pairs'][2]['same_size']


def test_cg_sc_cli_consecutive_failures_and_cg_recovery(native, monkeypatch, caplog):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(C.time, 'monotonic', lambda: clock.now)
    calls = []
    def fail_cg(self, win):
        calls.append('cg')
        if clock.now < 160:
            raise C.CaptureError('cg')
        return IMAGE
    def fail_sc(self, win, deadline):
        calls.append('sc')
        raise C.CaptureError('sc')
    monkeypatch.setattr(C.WindowCapture, '_cg_image', fail_cg)
    monkeypatch.setattr(C.WindowCapture, '_sc_image', fail_sc)
    capture = C.WindowCapture(None)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(C.CaptureError, match='cg'): capture.grab()
        with pytest.raises(C.CaptureError, match='sc'): capture.grab()
        assert capture.grab()[0][0, 0, 0] == 2 and capture.backend == 'cli'
        assert calls == ['cg', 'cg', 'sc', 'sc']
        clock.now = 159.9
        capture.grab()
        assert len(calls) == 4
        clock.now = 160
        assert capture.grab()[0][0, 0, 0] == 0 and capture.backend == 'cg'
    assert len([r for r in caplog.records if '[CAPTURE]' in r.message]) == 3


@pytest.mark.parametrize('failure', [None, 'timeout', 'exit', 'decode'])
def test_cli_arguments_coregraphics_resize_and_cleanup(monkeypatch, failure):
    files = []
    def run(args, **kwargs):
        assert args[:5] == ['screencapture', '-x', '-o', '-l', '123']
        assert kwargs['timeout'] == 5 and kwargs['check']
        file = Path(args[5]);files.append(file)
        file.write_bytes(b'test')
        if failure == 'timeout': raise subprocess.TimeoutExpired(args, 5)
        if failure == 'exit': raise subprocess.CalledProcessError(1, args)
    def load(file):
        assert file == files[0] and file.exists()
        if failure == 'decode': raise C.CaptureError('decode')
        return np.full((120, 160, 3), 73, np.uint8)
    monkeypatch.setattr(C.subprocess, 'run', run)
    monkeypatch.setattr(C, 'load_image', load)
    capture = C.WindowCapture(None, backend='cli')
    if failure:
        with pytest.raises(C.CaptureTimeout if failure == 'timeout' else C.CaptureError):
            capture._cli_image(WIN)
    else:
        image = capture._cli_image(WIN)
        assert image.shape == (60, 80, 3) and np.all(image == 73)
    assert not files[0].parent.exists()


def test_cli_backend_status_and_all_backend_pairs(native, tmp_path):
    from scripts.capture_check import check
    provider = PyEyeProvider(gallery_dir=tmp_path)
    provider._capture = C.WindowCapture(SimpleNamespace(find=lambda: WIN), backend='cli')
    assert provider.status()['capture_backend'] == 'cli'
    captures = {name: C.WindowCapture(None, backend=name) for name in ('cg', 'sc', 'cli')}
    report = check(captures, 2)
    assert len(report['pairs']) == 6
    assert all(row['successes'] == 2 for row in report['backends'].values())
    assert {tuple(row['backends']) for row in report['pairs']} == {('cg','sc'), ('cg','cli'), ('sc','cli')}


def test_sc_success_resets_its_failure_streak(native, monkeypatch):
    results = iter([C.CaptureError('sc'), IMAGE, C.CaptureError('sc')])
    def sc(self, win, deadline):
        value = next(results)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(C.WindowCapture, '_sc_image', sc)
    capture = C.WindowCapture(None)
    capture.backend = 'sc'
    capture._last_cg_retry = time.monotonic()
    with pytest.raises(C.CaptureError): capture.grab()
    capture.grab()
    with pytest.raises(C.CaptureError): capture.grab()
    assert capture.backend == 'sc'
