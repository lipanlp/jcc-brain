"""数字模板匹配：金币、等级、经验、倒计时、回合号、连胜、费用、出卡概率。

游戏数字字体固定，切字 + 归一化相关（与 cv2.matchTemplate 的 TM_CCOEFF_NORMED 同一公式）比 OCR 快两个数量级，
也不会漏读单个「1」。流程：
1. 亮度（max(R,G,B)）Otsu 二值化，连通域；贴着上下边的（区域边框、装饰线）丢掉；
2. 横向有重叠的连通域并成一个字（「%」、断笔的「4」）；
3. 字高在窗口高度的 0.8%~1.85% 之间的才是数字（1774×1136 实测：概率 11px、经验 15px、金币 16~17px、
   等级 18px；图标 ≥22px、「级」字 23px），太扁的是「-」；
4. 每个字缩放到 28×20（保持宽高比、居中）和模板求相关，<MIN_SCORE 的不要（金币图标边、头像碎片）。

模板 digits.npz 由 scripts/make_digit_templates.py 从样本截图切出（0~9 / %），每类若干个不同写法，取最相似的。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_FILE = Path(__file__).with_name("digits.npz")
GLYPH_H, GLYPH_W = 28, 20
MIN_H, MAX_H = 0.008, 0.0185    # 字高 / 窗口高度
MIN_SCORE = 0.6                 # 实测：真数字 ≥0.80，图标碎片 ≤0.5


@dataclass
class Glyph:
    x0: int
    y0: int
    x1: int          # 不含
    y1: int
    mask: np.ndarray

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


WHITE_MIN = 170                 # white 模式：min(R,G,B) 下限（海克斯刷新按钮金底白字，实测字 ≥200、金底 B<120）


def read_dim_white_cells(cells: list[np.ndarray | None], win_h: int) -> list[str] | None:
    """仅供海克斯遮罩下的羁绊人数；统一曝光补偿，不逐格把背景提升成白字。

    1774×1136的0045/0048：激活数白字min通道最高109/111，正常帧254/255；
    原170阈值会漏读。按整列白字峰值恢复曝光，沿用字高/形状判据排除未激活行。
    """
    peak = max((int(c.min(axis=2).max()) for c in cells if c is not None and c.size), default=0)
    if peak == 0 or peak >= WHITE_MIN:
        return None
    gain = 255. / peak
    return [read_digits(np.clip(c.astype(np.float32) * gain, 0, 255).astype(np.uint8), win_h, 'white')
            if c is not None and c.size else '' for c in cells]


def text_mask(img: np.ndarray, mode: str = "bright") -> np.ndarray:
    """字掩码。bright：max(R,G,B) 大于 Otsu 阈值（至少 90，全暗区域不出字），白字、黄字、彩色概率都行；
    white：只要近白色像素（亮底按钮上的白字）。"""
    if mode == "white":
        return (img.min(axis=2) > WHITE_MIN).astype(np.uint8)
    v = img.max(axis=2)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (v > max(t, 90)).astype(np.uint8)


def segment(img: np.ndarray, win_h: int, mode: str = "bright") -> list[Glyph]:
    """切字（按 x 排序）。win_h 是整个窗口的高度，用来换算字高范围。"""
    return segment_px(img, MIN_H * win_h, MAX_H * win_h, mode)


def segment_px(img: np.ndarray, lo: float, hi: float, mode: str = "bright") -> list[Glyph]:
    """切字，数字高度范围直接给像素 [lo, hi]。"""
    m = text_mask(img, mode)
    H = m.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 4 or y == 0 or y + h >= H:          # 噪点、贴上下边的边框
            continue
        boxes.append([x, y, x + w, y + h, [i]])
    boxes.sort()
    merged: list[list] = []
    for b in boxes:                                # 横向有重叠的并成一个字（数字之间至少隔 1px）
        if merged:
            p = merged[-1]
            if min(p[2], b[2]) > max(p[0], b[0]):
                p[0], p[1], p[2], p[3] = min(p[0], b[0]), min(p[1], b[1]), max(p[2], b[2]), max(p[3], b[3])
                p[4] += b[4]
                continue
        merged.append(b)
    out = []
    for x0, y0, x1, y1, ids in merged:
        mask = np.isin(lab[y0:y1, x0:x1], ids).astype(np.uint8)
        out.append(Glyph(x0, y0, x1, y1, mask))
    digits_h = [g.h for g in out if lo <= g.h <= hi]
    ref = float(np.median(digits_h)) if digits_h else hi
    # 「-」：高度不到数字一半、比高宽；其余必须在数字高度范围内
    return [g for g in out if lo <= g.h <= hi or (g.h < 0.45 * ref and g.w > g.h and g.w < ref)]


def normalize(mask: np.ndarray) -> np.ndarray:
    """缩放到 GLYPH_H 高（保持宽高比，过宽的压到 GLYPH_W），水平居中，返回零均值单位向量。"""
    h, w = mask.shape
    nw = max(1, min(GLYPH_W, round(w * GLYPH_H / h)))
    g = cv2.resize(mask.astype(np.float32), (nw, GLYPH_H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLYPH_H, GLYPH_W), np.float32)
    x = (GLYPH_W - nw) // 2
    canvas[:, x:x + nw] = g
    v = canvas.ravel() - canvas.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@lru_cache(maxsize=1)
def templates() -> tuple[str, np.ndarray]:
    """(字符表, 归一化模板矩阵 N×(28·20))。"""
    if not TEMPLATE_FILE.exists():
        return "", np.zeros((0, GLYPH_H * GLYPH_W), np.float32)
    z = np.load(TEMPLATE_FILE)
    chars = str(z["chars"])
    mat = np.stack([normalize_float(t) for t in z["glyphs"]])
    return chars, mat


def normalize_float(t: np.ndarray) -> np.ndarray:
    v = t.astype(np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def classify(g: Glyph, ref_h: float) -> tuple[str, float]:
    if g.h < 0.45 * ref_h and g.w > g.h:
        return "-", 1.0
    chars, mat = templates()
    if not chars:
        return "?", 0.0
    scores = mat @ normalize(g.mask)
    i = int(np.argmax(scores))
    return chars[i], float(scores[i])


def read_digits(img: np.ndarray, win_h: int, mode: str = "bright", min_score: float = MIN_SCORE) -> str:
    """一个区域 → 数字串。认不出的字（图标、汉字）当分隔符，相邻字间距超过 0.6 字高也插空格。"""
    gs = segment(img, win_h, mode)
    if not gs:
        return ""
    hs = [g.h for g in gs if g.h >= MIN_H * win_h]
    ref = float(np.median(hs)) if hs else float(max(g.h for g in gs))
    cls = [classify(g, ref) for g in gs]
    out: list[str] = []
    prev: Glyph | None = None
    for k, (g, (c, s)) in enumerate(zip(gs, cls)):
        if s < min_score:
            out.append(" ")
            prev = None
            continue
        # 「-」只认夹在两个数字中间的（回合号 3-1）；区域边上的装饰短线不算
        if c == "-" and not (0 < k < len(gs) - 1 and cls[k - 1][0].isdigit() and cls[k + 1][0].isdigit()
                             and cls[k - 1][1] >= min_score and cls[k + 1][1] >= min_score):
            out.append(" ")
            prev = None
            continue
        if prev is not None and g.x0 - prev.x1 > 0.6 * ref:
            out.append(" ")
        out.append(c)
        prev = g
    return " ".join("".join(out).split())


def read_number_line(line: np.ndarray, min_score: float = 0.8) -> str | None:
    """OCR 检测出的一行，如果是「[一个图标] 1~3 位数字」（玩家血量行：✓ 82、✗ 69、••• 100），
    直接用模板读出数字串；不像就返回 None，交给识别模型。比识别模型快几十倍，且不会把 ✓ 认成 5。"""
    H = line.shape[0]
    gs = segment_px(line, 0.4 * H, 0.95 * H)
    if not gs:
        return None
    ref = float(np.median([g.h for g in gs]))
    out, icon = [], None
    for g in gs:
        c, sc = classify(g, ref)
        if sc >= min_score and c.isdigit():
            out.append((c, g))
        elif out or icon is not None or sc >= MIN_SCORE:
            return None              # 数字后面还有东西、不止一个图标、或者像数字又没把握
        else:
            icon = g
    if not 1 <= len(out) <= 3:
        return None
    if icon is not None and out[0][1].x0 - icon.x1 < 0.3 * ref:
        return None                  # 图标和数字之间要有空隙，挨着的可能是没认出的数字
    hs = [g.h for _, g in out]
    if max(hs) > 1.25 * min(hs):
        return None
    return "".join(c for c, _ in out)
