"""Apple Vision 文字识别（pyobjc）。

把多个小区域竖着拼成一张图一次识别（每张拼图不超过 1600px 高，太高 Vision 会先缩小导致小字丢失），
结果再按纵向范围拆回各自区域，坐标换算成相对小图的 0~1。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr")


@dataclass
class Line:
    text: str
    conf: float
    x: float          # 左上原点，0~1
    y: float
    w: float
    h: float

    @property
    def mid_y(self) -> float:
        return self.y + self.h / 2

    @property
    def max_x(self) -> float:
        return self.x + self.w


def to_cgimage(img: np.ndarray):
    """numpy BGR/BGRA → CGImage。"""
    import Quartz

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) if img.shape[2] == 3 else img
    bgra = np.ascontiguousarray(bgra)
    h, w = bgra.shape[:2]
    data = bgra.tobytes()
    # 用 NSData 复制一份像素，避免 Python bytes 被回收后 CGImage 指向失效内存
    from Foundation import NSData
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(data, len(data)))
    return Quartz.CGImageCreate(
        w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaPremultipliedFirst | Quartz.kCGBitmapByteOrder32Little,
        provider, None, False, Quartz.kCGRenderingIntentDefault)


def recognize(img: np.ndarray, languages: list[str] | None = None, fast: bool = False) -> list[Line]:
    import Vision

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                             else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(languages or ["zh-Hans", "en-US"])
    req.setUsesLanguageCorrection_(False)
    req.setMinimumTextHeight_(0)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(to_cgimage(img), {})
    handler.performRequests_error_([req], None)
    out = []
    for o in req.results() or []:
        cands = o.topCandidates_(1)
        if not cands:
            continue
        c = cands[0]
        b = o.boundingBox()
        out.append(Line(str(c.string()), float(c.confidence()), b.origin.x, 1 - b.origin.y - b.size.height,
                        b.size.width, b.size.height))
    out.sort(key=lambda l: (round(l.y / 0.02), l.x))
    return out


def stack(imgs: list[np.ndarray], gap: int = 40) -> tuple[np.ndarray, list[tuple[int, int]]]:
    W = max(im.shape[1] for im in imgs)
    H = sum(im.shape[0] + gap for im in imgs)
    canvas = np.zeros((H, W, 3), np.uint8)
    bands, top = [], 0
    for im in imgs:
        h, w = im.shape[:2]
        canvas[top:top + h, :w] = im[..., :3]
        bands.append((top, top + h))
        top += h + gap
    return canvas, bands


def recognize_batch(imgs: list[np.ndarray], languages: list[str], max_h: int = 1600,
                    groups: list[str] | None = None) -> list[list[Line]]:
    """groups：每张小图属于哪个区域。同一区域的格子拼在一起、不和别的区域混拼——拼图内容一变 Vision
    就可能漏行（实测去掉一个格子后羁绊第一行「狂战士」整行丢失），按区域分组让每张拼图的组成稳定。"""
    found: list[list[Line]] = [[] for _ in imgs]
    if not imgs:
        return found
    chunks: list[list[int]] = [[]]
    h = 0
    for i, im in enumerate(imgs):
        new_group = groups is not None and i > 0 and groups[i] != groups[i - 1]
        if (h + im.shape[0] > max_h or new_group) and chunks[-1]:
            chunks.append([])
            h = 0
        chunks[-1].append(i)
        h += im.shape[0] + 40

    def run(idx: list[int]) -> None:
        mosaic, bands = stack([imgs[i] for i in idx])
        MH, MW = mosaic.shape[:2]
        for l in recognize(mosaic, languages):
            if l.conf < 0.3:
                continue
            cy = int(l.mid_y * MH)
            b = next((k for k, (t, btm) in enumerate(bands) if t <= cy <= btm), None)
            if b is None:
                continue
            im = imgs[idx[b]]
            ih, iw = im.shape[:2]
            top = bands[b][0]
            found[idx[b]].append(Line(l.text, l.conf, l.x * MW / iw, (l.y * MH - top) / ih,
                                      l.w * MW / iw, l.h * MH / ih))

    list(_pool.map(run, chunks))
    for lines in found:
        lines.sort(key=_line_key(lines))
    return found


def _line_key(lines: list[Line]):
    """同一行（纵向重叠超过一半）按 x 排，否则按 y 排。用分组近似 Swift 版的比较函数。"""
    rows: list[float] = []
    for l in sorted(lines, key=lambda l: l.mid_y):
        if not rows or abs(l.mid_y - rows[-1]) >= l.h / 2:
            rows.append(l.mid_y)

    def key(l: Line):
        row = min(range(len(rows)), key=lambda i: abs(rows[i] - l.mid_y)) if rows else 0
        return (row, l.x)
    return key


# ------------------------------------------------------------------ 可切换的 OCR 后端
class VisionOCR:
    """Apple Vision：小区域放大补边后竖向拼图识别。"""

    name = "vision"

    def prepare(self, img: np.ndarray) -> np.ndarray:
        from jccbrain.vision.eye.image import prep
        return prep(img)

    def recognize_batch(self, imgs: list[np.ndarray], languages: list[str],
                        groups: list[str] | None = None) -> list[list[Line]]:
        return recognize_batch(imgs, languages, groups=groups)


def create_engine(name: str = "rapid"):
    """OCR 后端：rapid（RapidOCR / PP-OCRv4，默认）| vision（Apple Vision）。rapid 不可用时退回 vision。"""
    if name == "rapid":
        try:
            from jccbrain.vision.eye.rapid import RapidOCR
            return RapidOCR()
        except ImportError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[OCR FALLBACK] RapidOCR 依赖不可用，已退回 Apple Vision（actual=vision，requested=rapid）：%s", exc)
    return VisionOCR()
