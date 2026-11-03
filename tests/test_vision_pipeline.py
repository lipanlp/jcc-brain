"""视觉流水线：并行顺序、失败收尾、启动预热与真实帧状态等价。"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary

FRAMES = Path(__file__).parent / "fixtures" / "frames"
mac = pytest.mark.skipif(sys.platform != "darwin", reason="需要 CoreGraphics / Apple Vision")


def _pending_cells(pool, read):
    def begin(image, **kwargs):
        future = pool.submit(read, image)
        return future.result, {}
    return begin


def test_units_and_ocr_overlap_but_observe_waits_for_shop():
    ocr_started, units_started = threading.Event(), threading.Event()
    prepared = ({}, [])

    def cells(img):
        ocr_started.set()
        assert units_started.wait(5), "OCR 与单位预处理应重叠"
        return {"stage": [[Line("2-1", 1, 0, 0, 1, 1)]],
                "shop_buttons": [[Line("刷新", 1, 0, 0, 1, 1)]]}

    def prepare(img, features=True):
        units_started.set()
        assert ocr_started.wait(5)
        return prepared

    observe = Mock(return_value={"bench": [], "board": []})
    units = SimpleNamespace(prepare=prepare, observe=observe)
    reader = GameReader(Dictionary({}, []), units)
    image = np.zeros((1136, 1774, 3), np.uint8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader._begin_cells = _pending_cells(pool, cells)
        state = reader.read(image)
    assert state["stage"] == "2-1"
    assert state["shop"]["open"]
    observe.assert_called_once_with(image, "planning", [None] * 5, prepared=prepared)


def test_unit_failure_drains_ocr_before_read_exits():
    ocr_started, units_failed, release, ocr_finished = (threading.Event() for _ in range(4))

    def cells(img):
        ocr_started.set()
        assert release.wait(5)
        ocr_finished.set()
        return {}

    def prepare(img, features=True):
        assert ocr_started.wait(5)
        units_failed.set()
        raise ValueError("单位预处理失败")

    reader = GameReader(Dictionary({}, []), SimpleNamespace(prepare=prepare))
    with ThreadPoolExecutor(max_workers=1) as ocr_pool, ThreadPoolExecutor(max_workers=1) as pool:
        reader._begin_cells = _pending_cells(ocr_pool, cells)
        future = pool.submit(reader.read, np.zeros((1136, 1774, 3), np.uint8))
        try:
            assert units_failed.wait(5)
            assert not future.done(), "不能带着仍在写缓存的 OCR 任务退出 read"
        finally:
            release.set()
        with pytest.raises(ValueError, match="单位预处理失败"):
            future.result(timeout=5)
    assert ocr_finished.is_set()


def test_ocr_failure_drains_other_batch_before_read_exits():
    from jccbrain.vision.eye.image import Region

    failed, other_started, release, other_finished = (threading.Event() for _ in range(4))

    class OCR:
        name = "vision"

        def prepare(self, img):
            return img

        def recognize_batch(self, imgs, languages, groups):
            if languages == ["en-US"]:
                failed.set()
                raise ValueError("数字 OCR 失败")
            other_started.set()
            assert release.wait(5)
            other_finished.set()
            return [[] for _ in imgs]

    units = SimpleNamespace(prepare=lambda img, features=True: ({}, []), observe=Mock())
    reader = GameReader(Dictionary({}, []), units,
                        {"n": Region(0, 0, .5, 1, numeric=True), "t": Region(.5, 0, .5, 1)}, ocr=OCR())
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reader.read, np.zeros((16, 16, 3), np.uint8))
        try:
            assert failed.wait(5) and other_started.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match="数字 OCR 失败"):
            future.result(timeout=5)
    assert other_finished.is_set() and not reader.cache
    units.observe.assert_not_called()


def test_detector_failure_drains_other_cells_before_batch_exits():
    from jccbrain.vision.eye.rapid import RapidOCR

    other_started, failed, release, finished = (threading.Event() for _ in range(4))

    def detect(cells):
        if cells[0][0] == 0:
            assert other_started.wait(5)
            failed.set()
            raise ValueError("检测失败")
        other_started.set()
        assert release.wait(5)
        finished.set()
        return []

    engine = RapidOCR.__new__(RapidOCR)
    engine._detect_group = detect
    with ThreadPoolExecutor(max_workers=2) as detection_pool, ThreadPoolExecutor(max_workers=1) as pool:
        engine.pool = detection_pool
        future = pool.submit(engine.recognize_batch, [np.zeros((16, 16, 3), np.uint8)] * 2)
        try:
            assert failed.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match="检测失败"):
            future.result(timeout=5)
    assert finished.is_set()


def test_bar_runs_keep_gap_limit_and_separate_rows():
    from jccbrain.vision.eye.units import detect_bars

    image = np.full((1136, 1774, 3), 20, np.uint8)
    image[400:406, 800:840] = (60, 200, 90)
    image[400:406, 845:880] = (60, 200, 90)  # 差 6：合并
    image[420:426, 800:840] = (60, 200, 90)
    image[420:426, 846:880] = (60, 200, 90)  # 差 7：独立
    assert [(b.min_x, b.max_x, b.min_y, b.max_y) for b in detect_bars(image)] == [
        (800, 879, 400, 405), (800, 839, 420, 425), (846, 879, 420, 425)]


def test_missing_prepared_feature_is_not_requested_twice(tmp_path, monkeypatch):
    from jccbrain.vision.eye import units

    tracker = units.UnitTracker(units.Gallery(tmp_path))
    unit = units.TrackedUnit("bench", 0, .25, .65, 1, crop=np.zeros((24, 24, 3), np.uint8))
    monkeypatch.setattr(tracker, "detect", lambda image: ({}, [unit]))
    extract = Mock(return_value=None)
    monkeypatch.setattr(units, "feature_print", extract)
    image = np.zeros((1136, 1774, 3), np.uint8)
    prepared = tracker.prepare(image)
    tracker.observe(image, "planning", None, prepared=prepared)
    assert extract.call_count == 1
    assert tracker.last_fp_count == 0


@mac
@pytest.mark.parametrize("name,bounds", [
    ("0003.jpg", [(759, 824, 125, 130), (668, 721, 136, 141), (617, 650, 187, 192), (371, 439, 616, 625)]),
    ("0069.jpg", [(1198, 1234, 178, 184), (1055, 1127, 282, 287), (301, 332, 621, 631)]),
    ("result_2nd.jpg", []),
])
def test_vectorized_bar_detection_preserves_real_bounds(name, bounds):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.units import detect_bars

    assert [(b.min_x, b.max_x, b.min_y, b.max_y) for b in detect_bars(load_image(FRAMES / name))] == bounds


@pytest.mark.parametrize("replay", [False, True])
def test_provider_start_retries_failed_warmup_and_publishes_once(tmp_path, monkeypatch, replay):
    from jccbrain.vision import pyeye

    warmup = Mock(side_effect=[RuntimeError("预热失败"), None])
    reader = SimpleNamespace(ocr=SimpleNamespace(warmup=warmup))
    build = Mock(return_value=reader)
    monkeypatch.setattr(pyeye, "build_reader", build)
    (tmp_path / "frame.jpg").touch()
    provider = (pyeye.ImageReplayProvider(tmp_path, gallery_dir=tmp_path / "gallery") if replay
                else pyeye.PyEyeProvider(gallery_dir=tmp_path / "gallery"))
    with pytest.raises(RuntimeError, match="预热失败"):
        provider.start()
    assert provider._reader is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: provider.start(), range(2)))
    assert provider._reader is reader
    if not replay:
        assert provider._capture is not None
    assert build.call_count == warmup.call_count == 2


@mac
def test_rapid_warmup_is_idempotent_and_keeps_text_caches_empty():
    from jccbrain.vision.eye.rapid import RapidOCR

    ocr = RapidOCR()
    ocr.det = Mock(wraps=ocr.det)
    ocr.rec = Mock(wraps=ocr.rec)
    ocr.warmup()
    ocr.warmup()
    assert ocr.det.call_count == 2
    assert ocr.rec.call_count == 1
    assert not ocr.cache and not ocr.words
    ocr.pool.shutdown()


@mac
def test_parallel_preparation_preserves_real_frame_states_and_gallery(tmp_path, monkeypatch):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader

    parallel = build_reader(None, tmp_path / "parallel")
    serial = build_reader(None, tmp_path / "serial")
    # 不预处理即为原有路径：等 OCR 得到 phase/shop 后，observe 完成全部单位识别。
    monkeypatch.setattr(serial.units, "prepare", lambda img, features=True: None)
    parallel.ocr.warmup()
    try:
        for frame in sorted(FRAMES.glob("*.jpg")):
            image = load_image(frame)
            assert parallel.read(image) == serial.read(image), frame.name
        assert parallel.units.gallery.names == serial.units.gallery.names
    finally:
        parallel.ocr.pool.shutdown()
        serial.ocr.pool.shutdown()

