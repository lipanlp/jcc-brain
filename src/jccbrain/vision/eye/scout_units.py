"""侦察专用位置检测；不改变主UnitTracker的血条门限/裁剪。"""
from __future__ import annotations

import cv2
import numpy as np

from jccbrain.vision.eye.units import (BENCH_TOP, BENCH_X0, BENCH_STEP, Bar, TrackedUnit,
                                      color_hist, green_mask, star_of, unit_crop)

# 1774×1136：0002误检的上缘备战条cy=.083/.098，己方最高=.114，取间隔中点。
# 原27+track20未有己方单位落在此范围；只去上缘备战区，战斗单位可跨棋盘中线。
SCOUT_TOP = .106


def short_bars(image: np.ndarray, existing: list[TrackedUnit]) -> list[Bar]:
    """残血绿段只作种子；上下暗框与空血槽恢复真实右边界后才接受。"""
    h, w = image.shape[:2]
    scale = h / 1136
    gray = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
    x0, y0 = round(.17*w), round(SCOUT_TOP*h)
    mask = green_mask(image[y0:round(.7*h), x0:round(.85*w)]).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for x, y, cw, ch, area in components[1:]:
        x, y = int(x+x0), int(y+y0)
        # 三处真短条(0072、观战0004/6)：11～16×6～7px；全宽均72px=.0406W。
        # 只补原22px门限以下，颜色块须接近实心且保持原血条厚度上界。
        if not (8/1136*h <= cw < 22/1136*h and 4/1136*h <= ch <= 8/1136*h
                and area >= cw*ch*.8):
            continue
        if any(u.bar and abs((u.bar.min_y+u.bar.max_y)/2-(y+ch/2)) < 4*scale
               and u.bar.min_x-3*scale <= x <= u.bar.max_x+3*scale for u in existing):
            continue
        top, bottom = y-max(1, round(2*scale)), y+int(ch)+max(1, round(scale))
        if top < 0 or bottom >= h:
            continue
        def dark(xx):
            return gray[top, xx] < 95 and gray[bottom, xx] < 95
        if not all(dark(xx) for xx in (x, x+int(cw)//2, x+int(cw)-1)):
            continue
        end = x+int(cw)-1
        while end+1 < w and dark(end+1) and end-x < .08*w:
            end += 1
        if not .032*w <= end-x+1 <= .058*w:
            continue
        empty = gray[y:y+int(ch), x+int(cw):end+1]
        # 空血槽正例暗像素比例最低.8829；全部53帧其余候选没有同时满足框宽/暗框，
        # 不是单凭绿色长度补单位。须保留外框右端，不能以11～16px绿段宽裁英雄。
        if not empty.size or float((empty < 95).mean()) < .85:
            continue
        out.append(Bar(x, end, y, y+int(ch)-1, (x+end)/2/w, (y+(ch-1)/2)/h))
    return out


def detect_scout_units(image: np.ndarray, tracker) -> list[TrackedUnit]:
    _, detected = tracker.detect(image)
    detected = [u for u in detected if u.y >= SCOUT_TOP]
    for bar in short_bars(image, detected):
        bench = bar.cy > BENCH_TOP
        slot = round((bar.cx-BENCH_X0)/BENCH_STEP) if bench else None
        if slot is not None and not 0 <= slot <= 8:
            continue
        unit = TrackedUnit('bench' if bench else 'board', slot, bar.cx, bar.cy, star_of(image, bar), bar)
        unit.crop = unit_crop(image, bar)
        if unit.crop is not None:
            unit.hist = color_hist(unit.crop)
        detected.append(unit)
    return sorted(detected, key=lambda u: (u.y, u.x))
