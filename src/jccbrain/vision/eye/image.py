"""图像工具。约定：图像是 numpy BGR uint8 (H, W, 3)；区域是相对窗口 0~1 的 (x, y, w, h)。"""
from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REGIONS_FILE = Path(__file__).with_name("regions.json")


@dataclass(frozen=True)
class Region:
    x: float
    y: float
    w: float
    h: float
    cols: int = 1
    rows: int = 1
    numeric: bool = False
    probe: bool = False
    when: str | None = None
    digits: str = ""            # 纯数字区域走模板匹配（eye/digits.py）：bright 亮字 / white 只取近白色字
    enhance: bool = False       # OCR 前做局部对比度增强（半透明水印字）

    def cell(self, i: int, j: int) -> "Region":
        c, r = max(self.cols, 1), max(self.rows, 1)
        return Region(self.x + self.w * i / c, self.y + self.h * j / r, self.w / c, self.h / r,
                      numeric=self.numeric, digits=self.digits, enhance=self.enhance)

    def px(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """像素矩形 (x0, y0, x1, y1)，与 CGRect.integral 一致（向外取整）。"""
        H, W = img.shape[:2]
        x0, y0 = int(np.floor(self.x * W)), int(np.floor(self.y * H))
        x1, y1 = int(np.ceil((self.x + self.w) * W)), int(np.ceil((self.y + self.h) * H))
        return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)


@lru_cache(maxsize=4)
def load_regions(path: str | None = None) -> dict[str, Region]:
    raw = json.loads(Path(path or REGIONS_FILE).read_text(encoding="utf-8"))
    out = {}
    for name, r in raw["regions"].items():
        out[name] = Region(r["x"], r["y"], r["w"], r["h"], r.get("cols") or 1, r.get("rows") or 1,
                           bool(r.get("numeric")), bool(r.get("probe")), r.get("when"),
                           "bright" if r.get("digits") is True else (r.get("digits") or ""), bool(r.get("enhance")))
    return out


def crop(img: np.ndarray, r: Region) -> np.ndarray | None:
    x0, y0, x1, y1 = r.px(img)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def resize_w(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if width <= 0 or width == w:
        return img
    nh = max(1, int(h * width / w))
    return cv2.resize(img, (width, nh), interpolation=cv2.INTER_CUBIC if width > w else cv2.INTER_AREA)


def pad(img: np.ndarray, m: int = 24) -> np.ndarray:
    """四周补黑边：贴边的窄字形（尤其是数字 1）更容易被识别。"""
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0))


PREP_HEIGHT = 120


def prep(img: np.ndarray) -> np.ndarray:
    """太小的区域放大到 120px 高并补边后识别更准。放大用 CoreGraphics 高质量插值（与 game-eye 一致，
    OpenCV 的插值会让单个数字「1」「0」更容易漏读）。"""
    h, w = img.shape[:2]
    if h >= PREP_HEIGHT:
        return img
    return pad(cg_resize(img, w * PREP_HEIGHT // max(h, 1)))


def cg_resize(img: np.ndarray, width: int) -> np.ndarray:
    try:
        import Quartz

        from jccbrain.vision.eye.capture import cgimage_to_bgr
        from jccbrain.vision.eye.ocr import to_cgimage
    except ImportError:
        return resize_w(img, width)
    h, w = img.shape[:2]
    nh = max(1, int(h * width / w))
    ctx = Quartz.CGBitmapContextCreate(
        None, width, nh, 8, 0, Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaPremultipliedFirst | Quartz.kCGBitmapByteOrder32Little)
    Quartz.CGContextSetInterpolationQuality(ctx, Quartz.kCGInterpolationHigh)
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, width, nh), to_cgimage(img))
    return cgimage_to_bgr(Quartz.CGBitmapContextCreateImage(ctx))


def enhance(img: np.ndarray) -> np.ndarray:
    """灰度 + CLAHE：棋盘中央的半透明人口水印「5/5」对比度太低，RapidOCR 检测不到（实测 0021/0024/0057 增强后才出）。"""
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def thumb(img: np.ndarray) -> np.ndarray:
    """文字掩码缩略图（原图 1/3）：只保留亮色文字像素，用来判断这格文字有没有变。"""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (max(8, w // 3), max(4, h // 3)), interpolation=cv2.INTER_AREA)
    return (small > 150).astype(np.uint8)


def changed(a: np.ndarray | None, b: np.ndarray) -> bool:
    """有 3 个以上文字像素翻转就算变了（宁可多识别一次，也不给过期数据）。"""
    if a is None or a.shape != b.shape or a.size == 0:
        return True
    return int(np.count_nonzero(a != b)) >= 3


def mean_rgb(img: np.ndarray, r: Region, skip_text: bool = False) -> tuple[float, float, float]:
    """区域平均色 (r, g, b) 0~1，隔 2 像素采样；skip_text 跳过接近白色的文字像素。"""
    x0, y0, x1, y1 = r.px(img)
    sub = img[y0:y1:2, x0:x1:2].reshape(-1, 3).astype(np.float64)
    if skip_text and len(sub):
        sub = sub[sub.min(axis=1) <= 170]
    if not len(sub):
        return (0.0, 0.0, 0.0)
    b, g, rr = sub.mean(axis=0) / 255
    return (float(rr), float(g), float(b))


def hsv(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """色相 0~360、饱和度 0~1、亮度 0~1。"""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return (h * 360, s, v)


def hsv_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整块图的 (色相 0~360, 饱和度 0~1, 亮度 0~1)，float32。"""
    f = img.astype(np.float32) / 255
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 0
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = 60 * np.fmod((g[rm] - b[rm]) / d[rm], 6)
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm] + 2)
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm] + 4)
    h[h < 0] += 360
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    return h, s, mx


def cost_from_color(rgb: tuple[float, float, float]) -> int | None:
    """商店卡底色 → 费用：灰 1 / 绿 2 / 蓝 3 / 紫 4 / 金 5。"""
    h, s, v = hsv(rgb)
    if v < 0.08:
        return None
    if s < 0.22:
        return 1
    if 80 <= h < 170:
        return 2
    if 170 <= h < 250:
        return 3
    if 250 <= h < 330:
        return 4
    if 25 <= h < 70:
        return 5
    return None


def encode_jpeg(img: np.ndarray, width: int | None = None, quality: int = 80) -> bytes:
    if width:
        img = resize_w(img, min(width, img.shape[1]))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
