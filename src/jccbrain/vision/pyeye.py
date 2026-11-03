"""纯 Python 视觉后端（默认）：截图 → OCR/取色/单位识别 → GameState，全部在本进程内完成。"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from jccbrain.core import paths
from jccbrain.core.models import GameState
from jccbrain.vision.base import VisionError, VisionProvider

log = logging.getLogger(__name__)


def _ocr_status(reader, requested: str) -> dict[str, Any]:
    actual = getattr(getattr(reader, 'ocr', None), 'name', None)
    return {'ocr': actual, 'ocr_requested': requested,
            'ocr_fallback': (requested == 'rapid' and actual == 'vision') if actual else None}


def frame_interval(phase: str | None, base_s: float) -> float:
    """上一帧准备阶段保持配置速度；其他阶段减少无必要的连续识别。"""
    if not phase or phase == 'unknown':
        return max(base_s, 2.0)
    if phase == 'planning':
        return base_s
    return max(base_s, 1.0)


class PyEyeProvider(VisionProvider):
    name = "pyeye"

    def __init__(self, window_owner: str = "金铲铲", min_interval_s: float = 0.25, db=None,
                 gallery_dir: Path | None = None, ocr: str = "rapid"):
        self.window_owner = window_owner
        self.ocr = ocr
        self.min_interval_s = min_interval_s
        self.db = db
        self.gallery_dir = gallery_dir or paths.var_dir() / "gallery"
        self._reader = None
        self._capture = None
        self._lock = threading.Lock()
        self._last: GameState | None = None
        self._last_img: np.ndarray | None = None
        self._last_t = 0.0
        self._capture_failed_t = 0.0
        self.last_ms: dict[str, int] = {}
        self._learning = None
        self._learning_source = None
        self._learning_frame = None
        self.last_learning = None
        self._learning_cancel = threading.Event()
        self._learning_context = None
        self._learning_status = {'mode':'off'}
        self._learning_record_on_read = True
        self._augment_diagnostics = None
        self._augment_diagnostics_failed = False

    def start(self) -> None:
        with self._lock:
            if self._reader is not None and self._capture is not None:
                return
            from jccbrain.actuator.window import WindowLocator
            from jccbrain.vision.eye.capture import WindowCapture
            reader = self._reader or build_reader(self.db, self.gallery_dir, self.ocr)
            if items := getattr(reader, "items", None):
                items.catalog.prepare()
            if augments := getattr(reader, 'augments_taken', None):
                augments.catalog.prepare()
            warmup = getattr(reader.ocr, "warmup", None)
            if warmup:
                warmup()
            self._capture = WindowCapture(WindowLocator(self.window_owner))
            self._reader = reader

    def read(self, fresh: bool = True) -> GameState:
        from jccbrain.vision.eye.capture import CaptureError

        self.start()
        with self._lock:
            if not fresh and self._last and time.monotonic() - self._last_t < self.min_interval_s:
                return self._last
            phase = self._last.phase if self._last and self._capture_failed_t <= self._last_t else None
            interval = (self.interval_override_s if self.interval_override_s is not None
                        else frame_interval(phase, self.min_interval_s))
            wait = interval - (time.monotonic() - max(self._last_t, self._capture_failed_t))
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            try:
                img, _win = self._capture.grab()
            except CaptureError as e:
                self._capture_failed_t = time.monotonic()
                if self._learning is not None:
                    self._learning_frame = None
                    self._learning.reset()
                raise VisionError(str(e)) from e
            t1 = time.perf_counter()
            # Capture is authoritative and happens exactly once. Receipt creation
            # is optional and must not lose the successful image on failure.
            learning_ticket = None
            self._learning_frame = None
            if self._learning is not None:
                try:
                    learning_ticket = self._learning_source._capture_ticket(img, _win)
                except Exception as exc:
                    self._learning_error(exc, 'capture_receipt')
            try:
                d = self._reader.read(img)
            except Exception:
                if self._learning is not None:
                    self._learning_frame = None
                    self._learning.reset()
                raise
            t2 = time.perf_counter()
            self.last_ms = {"capture": int((t1 - t0) * 1000), "read": int((t2 - t1) * 1000),
                            "ocr_cells": self._reader.last_ocr_cells}
            d["_ms"] = self.last_ms
            try:
                state = GameState.from_dict(d)
            except Exception:
                if self._learning is not None:
                    self._learning.reset()
                raise
            self._last, self._last_img, self._last_t = state, img, time.monotonic()
            if self._learning is not None and learning_ticket is not None:
                try:
                    self._learning_frame = self._learning_source.frame_from_capture(
                        learning_ticket, d.get('phase', 'unknown'), d.get('viewing'), d.get('stage'))
                    if self._learning.mode == 'record' and self._learning_record_on_read:
                        self.last_learning = self._learning.process(self._learning_frame, check=self._learning_check)
                        if self.last_learning.get('status') in ('error','cancelled'):
                            self._learning_frame = None
                except Exception as exc:
                    self._learning_error(exc, 'capture_learning')
            self._observe_augment_diagnostics(img, d, _win)
            return self._last

    def _observe_augment_diagnostics(self, image, data, window):
        """Normal successful read only; never called by scout/learning samples."""
        if self._augment_diagnostics_failed:
            return
        try:
            if self._augment_diagnostics is None:
                from jccbrain.vision.eye.augment_diagnostics import AugmentDiagnostics
                self._augment_diagnostics = AugmentDiagnostics(paths.var_dir()/'augment-diagnostics')
            self._augment_diagnostics.observe(image, data, window)
        except Exception as exc:
            self._augment_diagnostics_failed = True
            log.warning('海克斯诊断不可用，正常识别继续：%s', str(exc)[:160])

    def _learning_error(self, exc, operation):
        self._learning_frame = None
        self.last_learning = self._learning.fail(exc, operation)
        return self.last_learning

    def configure_unit_learning(self, directory: Path, *, mode: str = 'record', automatic_record: bool = True):
        """Explicit opt-in only; reuse this reader/capture. Does not start a capture or click.

        Panel processing is explicit so the caller can supply its before/click
        transaction. Record mode processes each new read, never cached reads.
        """
        from jccbrain.vision.eye.unit_learning import FrameSource, UnitLearningStore, UnitLearningController
        if mode not in ('record', 'panel'):
            raise ValueError('unsupported non-click mode')
        with self._lock:
            if self._reader is None:
                self._reader = build_reader(self.db, self.gallery_dir, self.ocr)
            gallery = self._reader.units.gallery
            if gallery.season is None or not gallery.pool:
                raise ValueError('unit learning requires a current season/card pool')
            source = FrameSource()
            store = UnitLearningStore(directory, gallery.season, gallery.pool)
            controller = UnitLearningController(self._reader, store, source, mode=mode)
            self._learning_source, self._learning = source, controller
            self._learning_record_on_read = automatic_record
            self._learning_cancel.clear()
            self._learning_context = None
            self._learning_frame = None
            self.last_learning = None
            return self._learning

    def _learning_check(self, check=None):
        from jccbrain.vision.eye.unit_learning import LearningCancelled
        if self._learning_cancel.is_set():
            raise LearningCancelled('learning_stopped')
        if check:
            check()

    def process_unit_learning(self, transaction=None, *, sample=None, check=None):
        """Process only the already captured current frame, with no input actions."""
        with self._lock:
            frame = sample.frame if sample is not None else self._learning_frame
            if self._learning is None or frame is None:
                return {'status': 'disabled', 'reason': 'no_learning_capture'}
            try:
                self.last_learning = self._learning.process(frame, transaction,check=lambda:self._learning_check(check))
                if self.last_learning.get('status') == 'error':
                    self._learning_frame = None
            except Exception as exc:
                self._learning_error(exc, 'explicit_learning')
            return self.last_learning

    def unit_learning_sample(self):
        """Receipt/context of the last ordinary read; never capture or stabilize."""
        from jccbrain.vision.eye.unit_learning import LearningSample
        with self._lock:
            frame = self._learning_frame
            if (self._learning is None or frame is None or self._last is None
                    or not self._learning_source.validates(frame)):
                return None
            return LearningSample(frame,dict(self._last.raw))

    def sample_unit_learning(self, check=None):
        """One bounded transaction capture on the SAME capturer, entirely private.

        Minimal current pixel scene/ownership/timer reading uses a foreign context;
        it cannot update main tracking, OCR, GameState, advice or scout memory.
        """
        from jccbrain.vision.eye.unit_learning import LearningSample, LearningCancelled
        with self._lock:
            if self._learning is None or self._capture is None:
                return None
            try:
                self._learning_check(check)
                start = time.monotonic()
                image, window, ticket = self._learning_source.grab(self._capture)
                captured = time.monotonic()
                self._learning_check(check)
                if self._learning_context is None:
                    self._learning_context = self._reader._foreign_context()
                context = self._learning_context._read_learning_scene(image)
                read = time.monotonic()
                self._learning_check(check)
                frame = self._learning_source.frame_from_capture(ticket,context.get('phase','unknown'),
                                                                 context.get('viewing'),context.get('stage'))
                return LearningSample(frame, context, dict(capture_ms=(captured-start)*1000,
                    scene_ms=(read-captured)*1000, receipt_ms=(time.monotonic()-read)*1000))
            except LearningCancelled:
                self._learning.reset()
                raise
            except Exception as exc:
                self._learning_error(exc,'temporary_capture')
                return None

    def finish_unit_learning_transaction(self):
        with self._lock:
            self._learning_context = None
            try:
                if self._learning is not None:
                    self._learning.reset()
            finally:
                self._learning_frame = None

    def cancel_unit_learning(self):
        # Cancellation must reach promotion while the provider lock is held by
        # slow OCR. The processing/commit hook observes this thread-safe event.
        self._learning_cancel.set()

    def end_unit_learning(self):
        self.cancel_unit_learning()
        with self._lock:
            try:
                if self._learning is not None:
                    self._learning.reset()
            finally:
                self._learning = self._learning_source = self._learning_frame = self._learning_context = None

    def stop(self):
        # Cancellation must reach an in-flight learning/commit check BEFORE
        # waiting for the read lock. Diagnostics then closes after that read.
        self.cancel_unit_learning()
        with self._lock:
            diagnostics=self._augment_diagnostics
            self._augment_diagnostics=None
            self._augment_diagnostics_failed=False
            if diagnostics is not None:
                try:diagnostics.close()
                except Exception as exc:
                    log.warning('海克斯诊断关闭失败，Provider正常停止：%s',str(exc)[:160])

    def unit_learning_status(self, value):
        self._learning_status = dict(value)  # telemetry does not acquire the OCR lock

    def current_learning_frame(self):
        """Capture receipt for the existing image; obtaining it never re-samples."""
        with self._lock:
            return self._learning_frame

    def screenshot(self, width: int = 1024) -> bytes | None:
        from jccbrain.vision.eye.image import encode_jpeg

        if self._last_img is None:
            self.read()
        return encode_jpeg(self._last_img, width) if self._last_img is not None else None

    def read_scout(self, image: np.ndarray) -> dict:
        """识别调用者提供的 BGR 图，不截图、不节流、不更新实时状态或截图缓存。"""
        with self._lock:
            if self._reader is None:
                self._reader = build_reader(self.db, self.gallery_dir, self.ocr)
            return self._reader.read_scout(image)

    def label_units(self, bench: dict[int, str] | None = None, board: list[dict[str, Any]] | None = None) -> list[str]:
        self.start()
        with self._lock:
            return self._reader.units.label(bench, board)

    def status(self) -> dict[str, Any]:
        if self._reader:
            gallery_dir = self._reader.units.gallery.dir
        else:
            import json
            from jccbrain.vision.eye.items import CATALOG, season_key

            season = season_key(getattr(self.db, "season_id", getattr(
                self.db, "set_id", json.loads(CATALOG.read_text("utf-8"))["season_id"])))
            gallery_dir = self.gallery_dir / ("s" + season)
        out: dict[str, Any] = {"backend": self.name, "gallery": str(gallery_dir), "last_ms": self.last_ms,
                               "capture_backend": getattr(self._capture, 'backend', 'cg'),
                               **_ocr_status(self._reader, self.ocr)}
        out['unit_learning'] = dict(self._learning_status, result=self.last_learning)
        out['augment_diagnostics'] = (self._augment_diagnostics.status() if self._augment_diagnostics
                                      else {'errors': int(self._augment_diagnostics_failed)})
        if self._reader:
            out["gallery_names"] = self._reader.units.gallery.names
        if self._capture:
            win = self._capture.locator.find()
            out["window"] = win and f"{win.owner} {win.width:.0f}×{win.height:.0f}"
        return out


def build_reader(db=None, gallery_dir: Path | None = None, ocr: str = "rapid"):
    """组装 GameReader：词典优先用游戏数据库，图库首次运行时从 game-eye 导入；ocr 选 OCR 后端（rapid / vision）。"""
    from jccbrain.vision.eye.ocr import create_engine
    from jccbrain.vision.eye.reader import GameReader
    from jccbrain.vision.eye.text import Dictionary
    from jccbrain.vision.eye.units import Gallery, UnitTracker
    from jccbrain.vision.eye.items import ItemCatalog, ItemReader, season_key
    from jccbrain.vision.eye.augments import AugmentCatalog, AugmentReader

    dictionary = Dictionary.from_db(db) if db is not None and not db.is_empty else Dictionary.from_file()
    catalog = ItemCatalog.from_db(db)
    pool = {hero.name for hero in db.pool_heroes()} if db is not None and not db.is_empty else None
    if pool is None:
        log.warning("没有当前赛季卡池，单位名字暂不输出")
    gallery = Gallery(gallery_dir or paths.var_dir() / "gallery", season_key(catalog.season_id), pool,
                      seed_from=paths.legacy_gallery_dir())
    # Existing, confirmed samples remain usable after disabling learning or
    # restarting. No store/directory/receipt is created in the default empty case.
    learning_root = paths.var_dir() / 'unit-learning'
    if (learning_root / ('s'+str(gallery.season)) / 'verified' / 'index.json').is_file():
        from jccbrain.vision.eye.unit_learning import UnitLearningStore
        gallery.verified_store = UnitLearningStore(learning_root, gallery.season, pool, read_only=True)
    # 实时 provider.start() 下载缺图；文件/离线回放只读缓存，不依赖网络可用性。
    return GameReader(dictionary, UnitTracker(gallery, ItemReader(catalog)), ocr=create_engine(ocr),
                      augments=AugmentReader(AugmentCatalog.from_db(db)))


class ImageReplayProvider(VisionProvider):
    """逐张识别一个截图目录（按文件名排序），跑完整的 PyEye 流程，用于离线调试和回归。"""

    name = "image-replay"

    def __init__(self, directory: str | Path, interval_s: float = 0.8, loop: bool = True, db=None,
                 gallery_dir: Path | None = None, ocr: str = "rapid"):
        self.dir = Path(directory)
        self.ocr = ocr
        self.interval_s = interval_s
        self.loop = loop
        self.db = db
        self.gallery_dir = gallery_dir
        self.files: list[Path] = []
        self._i = -1
        self._reader = None
        self._last: GameState | None = None
        self._last_t = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._reader is not None:
                return
            self.files = sorted(p for p in self.dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not self.files:
                raise VisionError(f"目录里没有截图：{self.dir}")
            import tempfile
            # 回放不污染正式图库
            reader = build_reader(self.db, self.gallery_dir or Path(tempfile.mkdtemp(prefix="jcc-gallery-")), self.ocr)
            warmup = getattr(reader.ocr, "warmup", None)
            if warmup:
                warmup()
            self._reader = reader

    def read(self, fresh: bool = True) -> GameState:
        from jccbrain.vision.eye.capture import load_image

        self.start()
        with self._lock:
            if not fresh and self._last is not None:
                return self._last
            wait = (self.interval_override_s if self.interval_override_s is not None
                    else self.interval_s) - (time.monotonic() - self._last_t)
            if wait > 0 and self._i >= 0:
                time.sleep(wait)
            self._i += 1
            if self._i >= len(self.files):
                if not self.loop:
                    raise VisionError("回放结束")
                self._i = 0
            d = self._reader.read(load_image(self.files[self._i]))
            d["_file"] = self.files[self._i].name
            self._last, self._last_t = GameState.from_dict(d), time.monotonic()
            return self._last

    def status(self) -> dict[str, Any]:
        return {"backend": self.name, "dir": str(self.dir), "frame": self._i, "total": len(self.files),
                **_ocr_status(self._reader, self.ocr)}
