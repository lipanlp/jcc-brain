"""按敌方橙红血条及血槽结构读取当前画面；只读当前帧，不推测遮挡单位。"""
from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict, deque

import cv2
import numpy as np

from jccbrain.vision.eye.units import SCAN, Bar, TrackedUnit, detect_bars, feature_print, star_of, unit_crop

# B8原12个combat repeat3：33/36=91.67%，友方误归属0/99；
# 53帧新增24处逐个CG像素审计为真血槽（combat18处），旧检测无删除。
# 残余0042前排槽仍漏报；新数据/算法必须重新检查门槛，不按等级补数。
ENABLED = True


def enemy_mask(image: np.ndarray) -> np.ndarray:
    b, g, r = (image[..., i].astype(np.int16) for i in range(3))
    # 1774×1136人工核心像素：4个橙条R-G=48～91、G-B=18～47，绿条R-G=-63。
    # 模型/伤害字的橙色与正例重叠，不能只用颜色保证单位，最终严格候选仍漏检。
    # 灰色护盾/空血槽不当颜色证据。颜色只是种子，还必须通过原暗边+全槽框检查。
    return (r >= 120) & (g >= 40) & (r-g >= 20) & (g-b >= 15) & (r-b >= 55)


def enemy_bars(image: np.ndarray) -> tuple[list[Bar], list[Bar]]:
    h, w = image.shape[:2]
    scale = h/1136
    gray = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
    raw = detect_bars(image, mask_fn=enemy_mask, min_width=8/1774)
    # 实拍名牌橙条51～77×13px；HUD图标厚条最大36px宽、植物条82×9px。
    # 两个维度都要满足，不能把左侧强化图标下的橙边定位成对手小小英雄。
    legends = [b for b in raw if 11/1136*h <= b.thick <= 14/1136*h and .025*w <= b.width <= .07*w]
    out = []
    for b in raw:
        if not 5/1136*h <= b.thick < 9/1136*h:
            continue
        top, bottom = b.min_y-max(1, round(2*scale)), b.max_y+max(1, round(2*scale))
        if top < 0 or bottom >= h:
            continue
        def dark(x):
            return gray[top, x] < 95 and gray[bottom, x] < 95
        if not all(dark(x) for x in (b.min_x, (b.min_x+b.max_x)//2, b.max_x)):
            continue
        end = b.max_x
        while end+1 < w and dark(end+1) and end-b.min_x < .08*w:
            end += 1
        # 1774×1136验收过的候选全框72～86px=.0406～.0485W。
        # .032～.058W + 85%暗边排除模型/伤害数字的短橙块；遮挡会保守漏报。
        if not .032*w <= end-b.min_x+1 <= .058*w:
            continue
        edges = (gray[top, b.min_x:end+1] < 95) & (gray[bottom, b.min_x:end+1] < 95)
        if float(edges.mean()) < .85:
            continue
        out.append(Bar(b.min_x, end, b.min_y, b.max_y, (b.min_x+end)/2/w, b.cy))
    # Keep every accepted legacy bar and its geometry. Recovery only appends
    # current-pixel candidates; it cannot replace a known positive with a guess.
    for bar in recover_enemy_bars(image):
        if not any(same_bar(bar, previous) for previous in out):
            out.append(bar)
    return out, legends


def same_bar(a: Bar, b: Bar) -> bool:
    # B8 53-frame audit: repeated segments of one slot differ by <=.008W/.003H;
    # closest distinct slots in the combat set differ by >=.030W at similar y.
    return abs(a.cx-b.cx) < .02 and abs(a.cy-b.cy) < .01


def enemy_seeds(image: np.ndarray) -> list[dict]:
    """Dense orange components seed recovery; no shield-only ownership guess."""
    h, w = image.shape[:2]
    x0, y0 = round(SCAN.x*w), round(SCAN.y*h)
    patch = image[y0:round((SCAN.y+SCAN.h)*h), x0:round((SCAN.x+SCAN.w)*w)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(enemy_mask(patch).astype(np.uint8))
    seeds = []
    for x, y, width, height, area in stats[1:]:
        fill = float(area/(width*height))
        # 1774x1136 added positives: dense cores 8..68 x 7..8px, fill>=.743;
        # orange model/text cores overlap that color range (negative fill max1).
        # Thus these limits only select seeds; every seed also needs a full slot.
        # Coarse nameplates are 11..14px tall; 0060's icon core is 19px, not a seed.
        if width >= 8/1774*w and 4/1136*h <= height <= 9/1136*h and fill >= .7:
            seeds.append(dict(rect=[int(x+x0), int(y+y0), int(width), int(height)], fill=fill))
    return seeds


def recover_enemy_bars(image: np.ndarray, audit: list | None = None) -> list[Bar]:
    """Recover paired horizontal borders and end caps near orange seed pixels.

    The stricter black border separates the slot from the dark purple backdrop;
    it never treats white/gray coverage as enemy evidence. Missing color stays
    missing. Every geometric limit scales with window dimensions.
    """
    h, w = image.shape[:2]
    scale = h/1136
    x_slack = max(1, round(w/1774))
    cap_radius = max(1, round(2/1774*w))
    top_offset, bottom_offset = max(1, round(2*scale)), max(1, round(scale))
    # 0036 right low-health slot: true cap interior max51; adjacent background
    # is78..94, leaking into the old <95 run. B8 also retains 60/70/95 failed
    # experiments; <80 plus paired edges/caps retains the audited true slots.
    dark = image.max(axis=2) < 80
    options = []
    for seed in enemy_seeds(image):
        x, y, width, height = seed['rect']
        middle = x+width//2
        candidates = []
        reason = 'no_paired_edges'
        measurements = dict(paired=0, widths=[], width_ok=0, contained=0, caps=[])
        # Verified cores can sit 1..3px below the top edge, 0..3px above bottom;
        # actual paired-edge span is10..12px at1774x1136. No expansion by y-side.
        for top in range(y-round(3*scale), y):
            for bottom in range(y+height, y+height+round(4*scale)):
                if not 0 <= top < bottom < h or not 7/1136*h <= bottom-top <= 13/1136*h:
                    continue
                joint = dark[top] & dark[bottom]
                if not joint[middle]:
                    continue
                left = right = middle
                while left > 0 and joint[left-1] and middle-left < .07*w:
                    left -= 1
                while right < w-1 and joint[right+1] and right-middle < .07*w:
                    right += 1
                measurements['paired'] += 1
                measurements['widths'].append((right-left+1)/w)
                # Same full-slot width interval as the conservative path. Added
                # verified slots are74..88px; negative model/text runs fail the
                # combined width + pair + terminal test in the53-frame audit.
                if not .032*w <= right-left+1 <= .058*w:
                    reason = 'full_width'
                    continue
                measurements['width_ok'] += 1
                if x < left-x_slack or x+width > right+cap_radius:
                    reason = 'seed_outside_slot'
                    continue
                measurements['contained'] += 1
                caps = [max(float(dark[top+1:bottom, max(0, end-cap_radius):min(w, end+cap_radius+1)][:, i].mean())
                            for i in range(min(w, end+cap_radius+1)-max(0, end-cap_radius))) for end in (left, right)]
                measurements['caps'].append(min(caps))
                # 53-frame diagnostics: accepted seed options' selected caps min.8182;
                # rejected width/containment survivors (8 trials) cap max.5455.
                # Color density overlaps (rejected max.9969); it cannot decide
                # ownership. The scaled five-column search needs both end caps.
                if min(caps) < .8:
                    reason = 'end_caps'
                    continue
                score = min(caps)+1-(abs(top-(y-top_offset))+abs(bottom-(y+height+bottom_offset)))/scale*.01
                candidates.append(dict(rect=[left, right, top+1, bottom-1], score=score, caps=caps))
        if candidates:
            best = max(candidates, key=lambda row: row['score'])
            options.append(best)
            if audit is not None:
                audit.append(dict(seed=seed, accepted=best, measurements=measurements))
        elif audit is not None:
            audit.append(dict(seed=seed, rejected=reason, measurements=measurements))
    out = []
    for option in sorted(options, key=lambda row: -row['score']):
        left, right, top, bottom = option['rect']
        bar = Bar(left, right, top, bottom, (left+right)/2/w, (top+bottom)/2/h)
        if not any(same_bar(bar, previous) for previous in out):
            out.append(bar)
    return out


class OpponentReader:
    def __init__(self, gallery, ocr):
        self.gallery = gallery
        self.ocr = copy.copy(ocr)
        if getattr(ocr, 'name', None) == 'rapid':
            self.ocr.lock = threading.Lock()
            self.ocr._stats_lock = threading.Lock()
            self.ocr.cache = OrderedDict()
            self.ocr.words = deque(maxlen=256)
            self.ocr.fallback = False
            self.ocr.reset_stats()
        self.last_ms = 0.

    def reset(self):
        if clear := getattr(self.ocr, 'clear_cache', None):
            clear()

    def read(self, image):
        start = time.perf_counter()
        bars, legends = enemy_bars(image)
        board = []
        for i, bar in enumerate(bars, 1):
            unit = TrackedUnit('board', None, bar.cx, bar.cy, star_of(image, bar), bar, group=i)
            cell = unit_crop(image, bar)
            fp = feature_print(cell) if cell is not None else None
            if fp is not None and (match := self.gallery.match(fp)):
                unit.name, unit.dist = match
            board.append(unit.to_json(self.gallery.pool))
        name = None
        # 橙色粗条是对手小小英雄；名牌跟随它的位置，不能用右侧旧名单猜。
        if len(legends) == 1:
            b = legends[0];h, w = image.shape[:2]
            cell = image[max(0, round(b.min_y-.04*h)):max(0, round(b.min_y-.005*h)),
                         max(0, round((b.cx-.11)*w)):min(w, round((b.cx+.11)*w))]
            if cell.size:
                lines = self.ocr.recognize_batch([self.ocr.prepare(cell)], ['zh-Hans', 'en-US'], ['opponent_name'])[0]
                names = [line.text.strip() for line in lines if len(line.text.strip()) >= 2
                         and any(c.isalpha() for c in line.text)]
                name = max(names, key=len, default=None)
        self.last_ms = (time.perf_counter()-start)*1000
        return dict(name=name, board=board, mirrored=True)
