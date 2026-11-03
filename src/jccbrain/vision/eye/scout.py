"""只读当前棋盘的侦察路径；独立文字格缓存，不使用己方身份锚点或自动学习。"""
from __future__ import annotations

import time
import copy
import threading
from collections import OrderedDict, deque
from dataclasses import replace

import cv2
import numpy as np

from jccbrain.vision.eye.augments import eye_template
from jccbrain.vision.eye.image import Region, hsv_image
from jccbrain.vision.eye.text import ints, edit_distance, is_cjk
from jccbrain.vision.eye.scout_units import detect_scout_units
from jccbrain.vision.eye.units import UnitTracker, feature_print

# 八行头像中心；与已验收眼睛搜索使用同一窗口比例，slot 不按 OCR 返回行数编号。
PLAYER_Y0, PLAYER_STEP = .126, .073


def correct_trait(name: str | None, dictionary) -> str | None:
    if not name or name in dictionary:
        return name
    closest = [word for word in set(dictionary) if edit_distance(name, word) == 1]
    return closest[0] if len(closest) == 1 else None


def supported_layout(image: np.ndarray) -> bool:
    # 已采集布局1576×1013～1774×1136，W/H=1.5558～1.5616；
    # 辅助大脑493×277缩图为1.7798，左栏行距/字宽也不同，不能套用此布局猜归属。
    return abs(image.shape[1] / image.shape[0] - 1774/1136) <= .02


def viewing_slot(image: np.ndarray) -> int | None:
    if not supported_layout(image):
        return None
    h, w = image.shape[:2]
    strip = image[round(.09*h):round(.69*h), round(.963*w):round(.983*w)]
    if not strip.size:
        return None
    strip = cv2.resize(strip, (35, 682), interpolation=cv2.INTER_AREA)
    template = eye_template()
    scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    if score < .60:  # 已有1774×1136标定：普通观战最低.6808，本人最高.5161。
        return None
    y = .09 + (location[1] + template.shape[0] / 2) / 682 * .60
    slot = round((y - PLAYER_Y0) / PLAYER_STEP)
    return slot if 0 <= slot <= 7 else None


class ScoutReader:
    def __init__(self, dictionary, gallery, regions, ocr):
        from jccbrain.vision.eye.reader import GameReader

        # 只共享只读模型和线程池；外人文字不能进入主 OCR 的精确/容差缓存。
        if getattr(ocr, 'name', None) == 'rapid':
            ocr = copy.copy(ocr)
            ocr.lock = threading.Lock()
            ocr.cache = OrderedDict()
            ocr.words = deque(maxlen=256)
            ocr._stats_lock = threading.Lock()
            ocr.reset_stats()
            # 快读不补认：同时禁用RapidOCR内部的弱行Vision退回，主OCR实例保持原设置。
            ocr.fallback = False
        selected = {k: regions[k] for k in ('traits', 'trait_count', 'level') if k in regions}
        self.traits_region = selected.get('traits')
        if self.traits_region:
            # 一次检测完整列，避免新观战画面文字跨原有小格边缘而被截断；
            # 分行仍使用原来的8行比例，不把其他画面文字当羁绊。
            selected['traits'] = replace(self.traits_region, x=.082, y=.09, w=.11, h=.49, rows=0)
        self.reader = GameReader(dictionary, UnitTracker(gallery), selected, ocr)
        self.last_ms: dict[str, float] = {}

    def reset(self) -> None:
        self.reader.cache.clear()
        self.reader.cell_cache.clear()
        self.reader.roster.clear()
        if clear := getattr(self.reader.ocr, 'clear_cache', None):
            clear()

    def read(self, image: np.ndarray) -> dict:
        start = time.perf_counter()
        if not supported_layout(image):
            self.last_ms = dict(ocr=0., units=0., total=(time.perf_counter()-start)*1000,
                                unsupported_layout=1.)
            return dict(traits=[], level=None, viewing=None, board=[], bench=[])
        if image.shape[0] < 1136:
            # 低分辨率输入先还原到既有字形/血条标定高度；坐标仍为窗口比例。
            # 放大只提高检测可用性，不把丢失的细节当作已确认的名字。
            image = cv2.resize(image, None, fx=1136/image.shape[0], fy=1136/image.shape[0],
                               interpolation=cv2.INTER_CUBIC)
        worker = self.reader
        slot = viewing_slot(image)
        if slot is not None:
            worker.regions['viewing_name'] = Region(.836, PLAYER_Y0 + slot*PLAYER_STEP - .035, .125, .026)
        else:
            worker.regions.pop('viewing_name', None)
        worker._frame_id += 1
        worker.cell_cache = {k: v for k, v in worker.cell_cache.items() if v[0] >= worker._frame_id - 1}
        finish, _ = worker._begin_cells(image, only=set(worker.regions))
        unit_start = time.perf_counter()
        try:
            detected = detect_scout_units(image, worker.units)
            # detect / match 均只读；不调用 observe / prepare / add，不创建历史组名。
            for i, unit in enumerate(detected, 1):
                unit.group = i
                fp = feature_print(unit.crop) if unit.crop is not None else None
                if fp is not None and (match := worker.units.gallery.match(fp)):
                    unit.name, unit.dist = match
        except BaseException:
            try:
                finish()
            except BaseException:
                pass
            raise
        unit_end = time.perf_counter()
        cells = finish()
        texts = lambda k: [' '.join(line.text for line in row) for row in cells.get(k, [])]
        levels = ints(' '.join(texts('level')))
        viewing = None
        if slot is not None:
            name = ' '.join(texts('viewing_name')).strip()
            # 沿用玩家名首尾杂字清理，roster 为侦察私有；不读写主 reader 的名单。
            name = worker._canonical_name(name)
            viewing = {'name': name if len(name) >= 2 else None, 'slot': slot}
        pool = worker.units.gallery.pool
        trait_rows = [[] for _ in range(self.traits_region.rows)] if self.traits_region else []
        if self.traits_region:
            region = worker.regions['traits']
            for line in (cells.get('traits') or [[]])[0]:
                y = region.y + line.mid_y*region.h
                row = int(np.floor((y-self.traits_region.y) / (self.traits_region.h/self.traits_region.rows)))
                if 0 <= row < len(trait_rows):
                    trait_rows[row].append(line)
        traits = []
        for row in trait_rows:
            # 主解析会丢弃单字名（实拍PP-OCR把“斗士”读成“士”）；先按本赛季词典
            # 做唯一的一次编辑纠正，保留原行数字/坐标，再交给原解析，不能凭行号补名。
            name_line = next((line for line in row if any(is_cjk(c) for c in line.text)), None)
            if name_line is not None:
                raw_name = ''.join(c for c in name_line.text if is_cjk(c))
                fixed = correct_trait(raw_name, worker.dict.traits)
                if fixed is not None and fixed != raw_name:
                    row = [replace(line, text=line.text.replace(raw_name, fixed)) if line is name_line else line
                           for line in row]
            parsed = worker._parse_traits([row])
            # 保留行及可读数字，未确认的名称显式null，不把高置信错字当正式羁绊。
            trait = parsed[0] if parsed else ({'name': None} if row else None)
            if trait is not None and trait['name'] not in worker.dict.traits:
                trait['name'] = correct_trait(trait['name'], worker.dict.traits)
            traits.append(trait)
        counts = texts('trait_count')
        h, w = image.shape[:2]
        for i, trait in enumerate(traits):
            if trait is None:
                continue
            y = .133 + i*.0565
            patch = image[round((y-.012)*h):round((y+.012)*h), round(.061*w):round(.079*w)]
            _, saturation, value = hsv_image(patch)
            # 1774×1136五张观战、25激活/14未激活：铜/金图标S>.25比例最低.807，
            # 未激活最高.171；银色激活V均值.367，未激活最高.249。按比例取中心，未看等级猜激活。
            active = float((saturation > .25).mean()) > .5 or float(value.mean()) > .32
            trait['active'] = active
            if active:
                if i < len(counts) and (numbers := ints(counts[i])):
                    trait['count'] = numbers[0]
                trait.pop('next', None)
        out = dict(traits=[trait for trait in traits if trait is not None],
                   level=levels[0] if levels and 1 <= levels[0] <= 10 else None,
                   viewing=viewing,
                   board=[u.to_json(pool) for u in sorted(detected, key=lambda u: (u.y, u.x)) if u.zone == 'board'],
                   bench=[u.to_json(pool) for u in sorted(detected, key=lambda u: u.slot or 0) if u.zone == 'bench'])
        self.last_ms = dict(ocr=worker._last_ocr_s*1000, units=(unit_end-unit_start)*1000,
                            vision=0.,
                            total=(time.perf_counter()-start)*1000)
        return out
