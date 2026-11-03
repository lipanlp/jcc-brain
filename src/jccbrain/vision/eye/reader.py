"""一帧画面 → 结构化状态（dict，字段与 game-eye 输出一致，可直接交给 GameState.from_dict）。

带跨帧记忆：回合类型（横幅只闪 1~2 秒）、玩家名单（统一 OCR 变体）、文字区域缓存、单位跟踪。
"""
from __future__ import annotations

import copy
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Callable

import numpy as np

from jccbrain.vision.eye.digits import read_digits, read_dim_white_cells
from jccbrain.vision.eye.image import Region, changed, cost_from_color, crop, enhance, hsv, load_regions, mean_rgb, thumb
from jccbrain.vision.eye.ocr import Line, VisionOCR
from jccbrain.vision.eye.phase import StagebarReader, detect_reconnect, detect_scene
from jccbrain.vision.eye.population import read_population
from jccbrain.vision.eye.shop import upgrade_region, upgrade_stars
from jccbrain.vision.eye.text import Dictionary, edit_distance, find_words, has_words, ints, is_cjk, is_word_char, norm_digits
from jccbrain.vision.eye.units import UnitTracker

log = logging.getLogger(__name__)

NUMERIC_CHARS = set("0123456789/-%")
HP_LINE = re.compile(r"[^\w\u4e00-\u9fff]*[A-Za-z]?[^\w\u4e00-\u9fff]*(\d{1,4})[^\w\u4e00-\u9fff]*")
PLACE_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-batch")


class GameReader:
    def __init__(self, dictionary: Dictionary, units: UnitTracker, regions: dict[str, Region] | None = None,
                 ocr=None, augments=None):
        self.dict = dictionary
        self.ocr = ocr or VisionOCR()
        self.units = units
        self.items = getattr(units, "items", None)
        self.augments_taken = augments
        self.regions = regions or load_regions()
        self.cache: dict[str, tuple[np.ndarray, list[Line]]] = {}
        self.cell_cache: dict[str, tuple[int, np.ndarray, list[Line]]] = {}
        self._frame_id = 0
        self.last_cell_cache_hits = 0
        self.last_cell_cache_checks = 0
        self.last_thumbnail_cache_hits = 0
        self.last_ocr_submitted = 0
        self.augment_desc_cache: tuple[str | None, tuple[str, ...], list[list[Line]]] | None = None
        self.last_augment_desc_cache_hits = 0
        self.lock = threading.Lock()
        self.last_ocr_cells = 0
        self._last_ocr_s = 0.0
        self.last_ms: dict[str, float] = {}      # 阶段墙钟耗时（ms）；ocr 与 units 重叠，不能相加当 total
        self.phase = "unknown"
        self.round_bar = StagebarReader()
        self.last_stage: str | None = None
        self.last_timer: int | None = None
        self.roster: list[str] = []
        self.last_players: list[dict] | None = None

    # ------------------------------------------------------------ OCR
    def read_cells(self, img: np.ndarray, only: set[str] | None = None,
                   reused_cells: dict[str, list[Line]] | None = None) -> dict[str, list[list[Line]]]:
        """每个区域 → 各格识别出的行。纯数字格走模板匹配；文字格没变就走缓存；其余数字格每次都 OCR。"""
        finish, _early = self._begin_cells(img, only, reused_cells=reused_cells)
        return finish()

    def _begin_cells(self, img: np.ndarray, only: set[str] | None = None,
                     before_ocr: Callable[[dict], None] | None = None,
                     reused_cells: dict[str, list[Line]] | None = None) -> tuple[Callable[[], dict], dict]:
        """主线程准备裁剪/数字模板，启动模型任务后返回收集函数及已读到的数字。"""
        start = time.perf_counter()
        jobs: list[tuple[str, str, Region]] = []
        for name in sorted(self.regions):
            r = self.regions[name]
            if r.probe or (r.when is not None if only is None else name not in only):
                continue
            for j in range(max(r.rows, 1)):
                for i in range(max(r.cols, 1)):
                    jobs.append((f"{name}#{j}#{i}", name, r.cell(i, j)))

        lines: list[list[Line]] = [[] for _ in jobs]
        early: dict[str, list[list[Line]]] = {}
        # 先读模板数字，在任何文字缓存命中或后台 OCR 开始前识别新对局。
        for k, (key, _name, region) in enumerate(jobs):
            if not region.digits:
                continue
            c = crop(img, region)
            if c is not None:
                t = read_digits(c, img.shape[0], region.digits)
                lines[k] = [Line(t, 1.0, 0, 0, 1, 1)] if t else []
            early.setdefault(_name, []).append(lines[k])
        if before_ocr:
            before_ocr(early)

        num_todo, txt_todo = [], []
        pixels = {}
        for k, (key, _name, region) in enumerate(jobs):
            if region.digits:
                continue
            if reused_cells is not None and key in reused_cells:
                lines[k] = [replace(line) for line in reused_cells[key]]
                continue
            c = crop(img, region)
            if c is None:
                continue
            # 精确原像素，不以缩略图相同推断小数字也相同；保存副本，调用者可复用画面缓冲。
            # 原有文字近似缓存保留；新路径也覆盖board_count等原来每帧OCR的格子。
            pixels[k] = c.copy()
            self.last_cell_cache_checks += 1
            hit = self.cell_cache.get(key)
            if (hit is not None and hit[0] in (self._frame_id - 1, self._frame_id)
                    and np.array_equal(hit[1], c)):
                lines[k] = [replace(line) for line in hit[2]]
                self.last_cell_cache_hits += 1
                continue
            if region.enhance:
                c = enhance(c)
            th = thumb(c)
            if not region.numeric and key in self.cache and not changed(self.cache[key][0], th):
                lines[k] = self.cache[key][1]
                self.last_thumbnail_cache_hits += 1
                continue
            (num_todo if region.numeric else txt_todo).append((k, th, self.ocr.prepare(c)))

        def recognize(todo, languages):
            found = self.ocr.recognize_batch([t[2] for t in todo], languages, [jobs[t[0]][1] for t in todo])
            return found, time.perf_counter()

        f_num = _pool.submit(recognize, num_todo, ["en-US"])
        f_txt = _pool.submit(recognize, txt_todo, ["zh-Hans", "en-US"])

        def finish():
            # 两个批次都退出后才更新缓存；任何一批失败也不会遗留仍在写 OCR 缓存的后台任务。
            wait([f_num, f_txt])
            num_found, num_end = f_num.result()
            txt_found, txt_end = f_txt.result()
            post = time.perf_counter()
            for (k, th, _), found in zip(num_todo, num_found):
                cleaned = []
                for l in found:
                    t = "".join(c for c in norm_digits(l.text) if c in NUMERIC_CHARS)
                    if t:
                        cleaned.append(Line(t, l.conf, l.x, l.y, l.w, l.h))
                lines[k] = cleaned
                self.cache[jobs[k][0]] = (th, cleaned)
            for (k, th, _), found in zip(txt_todo, txt_found):
                lines[k] = found
                self.cache[jobs[k][0]] = (th, found)
            self.last_ocr_cells = len(num_todo) + len(txt_todo)
            self.last_ocr_submitted += self.last_ocr_cells
            for k, raw in pixels.items():
                key = jobs[k][0]
                self.cell_cache[key] = self._frame_id, raw, [replace(line) for line in lines[k]]

            out: dict[str, list[list[Line]]] = {}
            for k, (_key, name, _r) in enumerate(jobs):
                out.setdefault(name, []).append(lines[k])
            self._last_ocr_s = max(num_end, txt_end) - start + time.perf_counter() - post
            return out

        return finish, early

    def _augment_cells(self, img: np.ndarray, extra: set[str], stage: str | None) -> tuple[dict, float]:
        """同一强化回合逐卡核对名字，只复用同位置同名字的非空描述。"""
        cached = self.augment_desc_cache
        if cached is not None and cached[0] == stage and 'augment_desc' in extra:
            cells = self.read_cells(img, only=extra - {'augment_desc'})
            elapsed = self._last_ocr_s
            names = tuple(' '.join(line.text for line in row) for row in cells.get('augment_names', []))
            reused = {f'augment_desc#0#{i}': cached[2][i] for i, name in enumerate(names)
                      if i < len(cached[1]) and i < len(cached[2]) and name.strip()
                      and name == cached[1][i] and cached[2][i]}
            self.last_augment_desc_cache_hits = len(reused)
            if len(reused) == len(names) == 3:
                cells['augment_desc'] = [[replace(line) for line in row] for row in cached[2]]
                return cells, elapsed
            cells.update(self.read_cells(img, only={'augment_desc'}, reused_cells=reused))
            elapsed += self._last_ocr_s
        else:
            # 首次/换回合仍按原批次并行识别，避免首次读卡增加一次串行等待。
            cells = self.read_cells(img, only=extra)
            elapsed = self._last_ocr_s
            names = tuple(' '.join(line.text for line in row) for row in cells.get('augment_names', []))
        descs = cells.get('augment_desc', [])
        if len(names) == len(descs) == 3 and any(n.strip() and desc for n, desc in zip(names, descs)):
            self.augment_desc_cache = stage, names, [[replace(line) for line in row] for row in descs]
        else:
            self.augment_desc_cache = None  # 不把尚未读全的首帧固化成空描述。
        return cells, elapsed

    def _check_new_game(self, cells: dict) -> None:
        stage = " ".join(line.text for line in (cells.get("stage") or [[]])[0])
        if re.search(r"\b1-1\b", stage) and self.last_stage not in (None, "1-1"):
            self._new_game()

    # ------------------------------------------------------------ 主流程
    def read(self, img: np.ndarray, debug: bool = False) -> dict:
        with self.lock:
            if self._is_spectate(img):
                start = time.perf_counter()
                try:
                    if not hasattr(self, '_spectate_reader'):
                        self._spectate_reader = self._foreign_context()
                    foreign = self._spectate_reader
                    result = foreign._read(img, debug)
                except BaseException:
                    # OCR tasks are joined by _read; a failed foreign session must
                    # not carry partially updated private state into the next read.
                    self.__dict__.pop('_spectate_reader', None)
                    raise
                for name in ('last_ms', 'last_ocr_cells', 'last_cell_cache_hits', 'last_cell_cache_checks',
                             'last_thumbnail_cache_hits', 'last_ocr_submitted', 'last_augment_desc_cache_hits'):
                    setattr(self, name, copy.copy(getattr(foreign, name)))
                self.last_ms['total'] = (time.perf_counter() - start) * 1000
                return result
            self.__dict__.pop('_spectate_reader', None)
            return self._read(img, debug)

    def _is_spectate(self, img: np.ndarray) -> bool:
        """Determine ownership before numeric callbacks, OCR caches or unit prepare.

        Reuse the existing scene thresholds and preserve dialog > carousel >
        spectate. The stagebar probe is private: even its stage/index must freeze.
        """
        from jccbrain.vision.eye.augments import spectate_score
        from jccbrain.vision.eye.phase import dialog_score

        if spectate_score(img) < .60 or dialog_score(img) >= .8:
            return False
        region = self.regions.get('stage')
        stage = read_digits(crop(img, region), img.shape[0], region.digits) if region else None
        _, round_type = copy.copy(self.round_bar).read_rounds(img, stage)
        return round_type != 'carousel'

    def _foreign_context(self):
        """Share read-only models/gallery matching; fork every recognition memory."""
        foreign = copy.copy(self)
        # Existing child readers hold references to *our* OCR/cache. A foreign
        # 1-1 reset must never reach them through the shallow copy.
        for name in ('_scout_reader', '_opponent_reader', '_spectate_reader'):
            foreign.__dict__.pop(name, None)
        foreign.lock = threading.Lock()
        for name in ('cache', 'cell_cache', 'roster', 'last_players', 'round_bar', 'augment_desc_cache'):
            setattr(foreign, name, copy.deepcopy(getattr(self, name)))
        foreign.ocr = copy.copy(self.ocr)
        for name in ('cache', 'words'):
            if hasattr(self.ocr, name):
                setattr(foreign.ocr, name, copy.deepcopy(getattr(self.ocr, name)))
        if getattr(self.ocr, 'name', None) == 'rapid':
            foreign.ocr.lock = threading.Lock()
            foreign.ocr._stats_lock = threading.Lock()
            foreign.ocr._warm_lock = threading.Lock()
            foreign.ocr.reset_stats()
        foreign.items = copy.deepcopy(self.items)
        foreign.augments_taken = copy.deepcopy(self.augments_taken)
        # No items connection: current foreign units cannot mutate our unknown
        # item groups or learn from shop differences / historical identity anchors.
        foreign.units = UnitTracker(self.units.gallery)
        foreign._isolated_spectate = True
        return foreign

    def read_scout(self, image: np.ndarray) -> dict:
        """输入 CoreGraphics 解码的 BGR 图；只更新独立的侦察文字缓存。"""
        from jccbrain.vision.eye.scout import ScoutReader

        with self.lock:
            if not hasattr(self, '_scout_reader'):
                self._scout_reader = ScoutReader(self.dict, self.units.gallery, self.regions, self.ocr)
            return self._scout_reader.read(image)

    def _read_learning_scene(self, img: np.ndarray) -> dict:
        """Private transaction context only; current pixels, original phase rules.

        The caller forks with _foreign_context(). Nothing here observes/matches
        units, items, shop cards, traits or augments, or updates the main reader.
        Stage/timer are still read from EVERY image, not borrowed from history.
        """
        if not getattr(self, '_isolated_spectate', False):
            raise ValueError('learning scene requires an isolated context')
        with self.lock:
            return self._read(img, False, scene_only=True)

    def _read(self, img: np.ndarray, debug: bool, *, scene_only: bool = False) -> dict:
        t0 = time.perf_counter()
        self._frame_id += 1
        self.cell_cache = {k: v for k, v in self.cell_cache.items() if v[0] >= self._frame_id - 1}
        self.last_cell_cache_hits = self.last_cell_cache_checks = 0
        self.last_thumbnail_cache_hits = self.last_ocr_submitted = 0
        self.last_augment_desc_cache_hits = 0

        # 先在调用线程完成轻量预处理并把模型任务发出，再并行做单位准备，避免两条 Python
        # 预处理循环争用 GIL；相位解析、跟踪状态与自动标注在两个阶段结束后统一完成。
        only = {'stage', 'timer', 'banner', 'result', 'players'} if scene_only else None
        finish, early_cells = self._begin_cells(img, only=only, before_ocr=self._check_new_game)
        t1 = time.perf_counter()
        try:
            hint, last_timer = self.phase, self.last_timer
            early = {name: " ".join(line.text for line in rows[0]) for name, rows in early_cells.items() if rows}
            if (stage := re.search(r"\d-\d", early.get("stage", ""))) and stage.group() != self.last_stage:
                hint, last_timer = "planning", None
            if (timer := ints(early.get("timer", ""))) and last_timer is not None:
                if timer[-1] > last_timer + 3 and hint == "planning":
                    hint = "combat"
            # 提前判断只决定是否预计算特征；最终 phase 仍按完整 OCR 结果确定。
            prepared = ({}, []) if getattr(self, '_isolated_spectate', False) else self.units.prepare(
                img, features=hint in ("planning", "unknown"))
        except BaseException:
            try:
                finish()
            except BaseException:
                pass
            raise
        t_units = time.perf_counter() - t1
        cells = finish()
        t_ocr = self._last_ocr_s
        R = self.regions

        def text(k: str) -> str:
            return " ".join(l.text for l in (cells.get(k) or [[]])[0])

        def cell_texts(k: str) -> list[str]:
            return [" ".join(l.text for l in ls) for ls in cells.get(k, [])]

        s: dict = {}
        if m := re.search(r"\d-\d", text("stage")):
            s["stage"] = m.group(0)
        if t := ints(text("timer")):
            s["timer"] = t[-1]

        # 回合类型：海克斯界面 > 横幅 > 伤害面板 > 同回合倒计时突增 > 回合号变化
        banner = text("banner")
        players_txt = "".join(l.text for l in (cells.get("players") or [[]])[0])
        st = s.get("stage")
        if st and st != self.last_stage:
            if st == "1-1" and self.last_stage not in (None, "1-1"):
                self._new_game()
                # 自定义区域若不用数字模板，只能在 OCR 后发现新局；丢弃旧缓存结果重读首帧。
                result = self._read(img, debug, scene_only=scene_only)
                self.last_ms["ocr"] += t_ocr * 1000
                self.last_ms["units"] += t_units * 1000
                self.last_ms["total"] = (time.perf_counter() - t0) * 1000
                return result
            self.last_stage, self.phase, self.last_timer = st, "planning", None
        if "timer" in s:
            if self.last_timer is not None and s["timer"] > self.last_timer + 3 and self.phase == "planning":
                self.phase = "combat"
            self.last_timer = s["timer"]
        if "战斗开始" in banner or "伤害" in players_txt:
            self.phase = "combat"
        if "准备阶段" in banner:
            self.phase = "planning"
        current = self.phase
        if "强化符文" in banner or "选择一个" in banner:
            current = "augment"
            if not scene_only:
                extra = {k for k, r in R.items() if r.when == "augment"}
                augment_cells, augment_ocr = self._augment_cells(img, extra, s.get('stage') or self.last_stage)
                cells.update(augment_cells)
                t_ocr += augment_ocr
                names, descs, refresh = cell_texts("augment_names"), cell_texts("augment_desc"), cell_texts("augment_refresh")
                augs = []
                for i, n in enumerate(names):
                    a: dict = {"name": self.dict.match_augment(n)}
                    if i < len(descs):
                        a["desc"] = descs[i].replace(" ", "")
                    if i < len(refresh) and (v := ints(refresh[i])):
                        a["refresh"] = v[-1]
                    augs.append(a)
                s["augments"] = augs

        result = text("result")
        if m := re.search(r"第([一二三四五六七八1-8])名", result):
            ch = m.group(1)
            s["placement"] = PLACE_CN.get(ch) or int(ch)
            current = "ended"
        if detect_reconnect(img):
            current = "reconnect"
        s["next_round"], s["round_type"] = self.round_bar.read_rounds(img, s.get("stage"))
        if scene := detect_scene(img, s['round_type']):
            current = scene
        s["phase"] = current
        if current != 'augment':
            self.augment_desc_cache = None
        if banner and any(is_cjk(c) for c in banner):
            s["banner"] = banner

        if scene_only:
            self._players(s, cells, img)
            self.last_ms = {"ocr": t_ocr * 1000, "units": 0., "total": (time.perf_counter() - t0) * 1000}
            return s

        # 经济
        if g := ints(text("gold")):
            s["gold"] = g[-1]
            s["interest"] = min(g[-1] // 10, 5)
        if (lv := ints(text("level"))) and 1 <= lv[0] <= 10:
            s["level"] = lv[0]
        if len(xp := ints(text("xp"))) >= 2:
            s["xp"] = [xp[0], xp[1]]
        if len(odds := ints(text("odds"))) == 5 and sum(odds) == 100:      # 「%」断笔被认成 0 时总和不对，宁可不给
            s["odds"] = odds
        if c := ints(text("xp_cost")):
            s["xp_cost"] = c[-1]
        if c := ints(text("refresh_cost")):
            s["refresh_cost"] = c[-1]

        # 连胜/连败：图标颜色定类型，旁边数字定场数
        if "streak_icon" in R and (n := ints(text("streak"))):
            h, sat, _v = hsv(mean_rgb(img, R["streak_icon"]))
            kind = None if sat < 0.3 else "win" if (h < 50 or h > 330) else "loss" if 170 < h < 260 else None
            if kind:
                s["streak"] = {"kind": kind, "n": n[-1]}

        # 商店
        buttons = text("shop_buttons")
        shop_open = "刷新" in buttons or "经验" in buttons or "refresh_cost" in s
        shop: dict = {"open": shop_open}
        if shop_open and "shop" in R:
            names, traits_txt = cell_texts("shop"), cell_texts("shop_traits")
            slots: list = []
            for i in range(5):
                t = names[i] if i < len(names) else ""
                m = self.dict.match_champion(t)
                if not m:
                    slots.append(None)
                    continue
                name, cost, known = m
                slot: dict = {"name": name}
                tr = find_words(traits_txt[i] if i < len(traits_txt) else "", self.dict.traits)
                digit = (ints(t)[-1] % 10) if ints(t) else None
                if not known and not tr:
                    # 道具卡（铜汤匙、边境村庄…）：不在英雄表里、也没有羁绊行；费用看卡上的数字（底色是金色但不是 5 费）
                    slot["item"] = True
                    if digit:
                        slot["cost"] = digit
                    slots.append(slot)
                    continue
                if not known:
                    slot["unsure"] = True
                color = cost_from_color(mean_rgb(img, R["shop"].cell(i, 0), skip_text=True))
                if (c := cost or color or digit) is not None:
                    slot["cost"] = c
                if known and self.dict.champion_traits.get(name):
                    tr = list(self.dict.champion_traits[name])       # 认出了英雄就直接用数据库的羁绊
                if tr:
                    slot["traits"] = tr
                if upgrade := upgrade_stars(crop(img, upgrade_region(R["shop"].cell(i, 0)))):
                    slot["upgrade"] = upgrade
                slots.append(slot)
            shop["slots"] = slots
        s["shop"] = shop

        trait_counts = cell_texts("trait_count")
        if current == 'augment' and (region := R.get('trait_count')):
            dim_counts = read_dim_white_cells([crop(img, region.cell(0, j)) for j in range(region.rows)], img.shape[0])
            if dim_counts is not None:
                trait_counts = dim_counts
        if t := self._parse_traits(cells.get("traits", []), trait_counts):
            s["traits"] = t

        # 单位：带羁绊的才是英雄卡；道具卡记成 "~名字"，不参与自动标注
        shop_names = None
        if shop_open and "slots" in shop:
            shop_names = [None if not d else ("~" + d["name"] if d.get("item") else d["name"]) for d in shop["slots"]]
        t1 = time.perf_counter()
        if self.items:
            s["item_bench"] = self.items.bench(img)
        s.update(self.units.read_current(img) if getattr(self, '_isolated_spectate', False)
                 else self.units.observe(img, current, shop_names, prepared=prepared))
        t_units += time.perf_counter() - t1

        bc = ints(text("board_count").replace(" ", ""))
        if not (len(bc) == 2 and 1 <= bc[1] <= 10 and bc[0] <= bc[1]) and current == 'planning':
            if region := R.get('board_count'):
                bc = read_population(crop(img, region), img.shape[0]) or bc
        if len(bc) == 2 and 1 <= bc[1] <= 10 and bc[0] <= bc[1]:
            s["board_count"] = bc

        players = self._players(s, cells, img)
        s['augments_taken'] = (self.augments_taken.remembered() if getattr(self, '_isolated_spectate', False)
                               else self.augments_taken.read(img, own_view=bool(players and len(players) >= 5))
                               ) if self.augments_taken else []
        if current == 'combat':
            from jccbrain.vision.eye.opponent import ENABLED, OpponentReader
            # 实拍数量验收未达80%时只交付分析入口，不把不完整阵容喂给消费者。
            if ENABLED:
                if not hasattr(self, '_opponent_reader'):
                    self._opponent_reader = OpponentReader(self.units.gallery, self.ocr)
                s['opponent'] = self._opponent_reader.read(img)
            else:
                s['opponent'] = None
        self.last_ms = {"ocr": t_ocr * 1000, "units": t_units * 1000, "total": (time.perf_counter() - t0) * 1000}
        if self.items:
            self.last_ms["items"] = self.items.last_ms
        if debug:
            s["_raw"] = {k: [" ".join(l.text for l in ls) for ls in v] for k, v in cells.items()}
        return s

    def _players(self, s, cells, img):
        """Original ownership parsing, remembering only in the receiving context."""
        panel, players = self._parse_players((cells.get("players") or [[]])[0])
        if panel:
            s["right_panel"] = panel
        if players:
            s["players"] = players
            self.last_players = players
        elif panel == "damage" and self.last_players:
            s["players"] = self.last_players
            s["players_stale"] = True
        me = next((p for p in s.get("players", []) if p.get("self")), None)
        if me:
            s["hp"] = me["hp"]
        from jccbrain.vision.eye.scout import viewing_slot
        viewing = viewing_slot(img)
        viewed_names = {p.get('name') for p in players or [] if p.get('slot') == viewing and p.get('name')}
        s['viewing'] = ({'name': next(iter(viewed_names)) if len(viewed_names) == 1 else None, 'slot': viewing}
                        if viewing is not None else None)
        return players

    def _new_game(self) -> None:
        """回合号回到 1-1：清空跨帧记忆。"""
        log.info("检测到新对局，重置跟踪状态")
        self.phase, self.last_stage, self.last_timer = "unknown", None, None
        self.round_bar.reset()
        self.roster.clear()
        self.last_players = None
        self.cache.clear()
        self.cell_cache.clear()
        self.augment_desc_cache = None
        if scout := getattr(self, '_scout_reader', None):
            scout.reset()
        if opponent := getattr(self, '_opponent_reader', None):
            opponent.reset()
        if not getattr(self, '_isolated_spectate', False) and (reset := getattr(self.augments_taken, 'reset', None)):
            reset()
        if clear := getattr(self.ocr, "clear_cache", None):
            clear()
        self.units.reset()

    # ------------------------------------------------------------ 羁绊
    def _parse_traits(self, rows: list[list[Line]], counts: list[str] | None = None) -> list[dict]:
        """激活时「[大数字] 名称 / 档位」，未激活时「名称 / 当前/下一档」。
        counts：每行左边激活人数的模板匹配结果（单个大数字 OCR 常漏读），有就以它为准。"""
        out = []
        for j, ls in enumerate(rows):
            name_line = next((l for l in ls if any(is_cjk(c) for c in l.text)), None)
            if not name_line:
                continue
            words = find_words(name_line.text, self.dict.traits)
            name = words[0] if words else "".join(c for c in name_line.text if is_cjk(c))
            if len(name) < 2:
                continue
            t: dict = {"name": name}
            prefix = ""
            for c in name_line.text:
                if is_cjk(c):
                    break
                prefix += c
            leading = ints(prefix)
            count_line = next((l for l in ls if l.max_x <= name_line.x + 0.02 and not has_words(l.text) and ints(l.text)), None)
            n = ints(count_line.text)[0] if count_line else (leading[0] if leading else None)
            if counts and j < len(counts) and (tc := ints(counts[j])):
                n = tc[0]
            if n is not None and n <= 10:
                t["count"], t["active"] = n, True
            else:
                below = next((l for l in ls if l.mid_y > name_line.mid_y and not has_words(l.text)), None)
                if below:
                    v = ints(below.text.replace(" ", ""))
                    if len(v) >= 2:
                        t["count"], t["next"] = v[0], v[1]
                    elif v and 10 <= v[0] < 100:       # 「1/2」被认成「12」
                        t["count"], t["next"] = v[0] // 10, v[0] % 10
                    elif v and 100 <= v[0] < 1000 and str(v[0])[1] in "17":   # 「2/3」被认成「213」「273」
                        t["count"], t["next"] = v[0] // 100, v[0] % 10
                    t["active"] = False
            out.append(t)
        return out

    # ------------------------------------------------------------ 玩家
    def _canonical_name(self, raw: str) -> str:
        """去掉首尾图标杂字，并和已见过的名字对齐（OCR 变体统一成同一个名字）。"""
        t = list(raw.strip())
        while t and not is_word_char(t[0]):
            t.pop(0)
        while t and not is_word_char(t[-1]):
            t.pop()
        if len(t) > 2 and t[0].isascii() and t[0].isdigit() and is_cjk(t[1]):
            t.pop(0)
        name = "".join(t)
        if any(is_cjk(c) for c in name):
            name = name.replace(" ", "")
        if len(name) < 2:
            return name
        if known := next((k for k in self.roster if k in name and len(name) - len(k) <= 1), None):
            return known
        if (i := next((i for i, k in enumerate(self.roster) if name in k and len(k) - len(name) == 1), None)) is not None:
            self.roster[i] = name
            return name
        nc = name.replace(" ", "")
        if known := next((k for k in self.roster if max(len(k), len(nc)) >= 4 and edit_distance(k, nc) <= 1), None):
            return known
        if name not in self.roster:
            self.roster.append(name)
        return name

    def _parse_players(self, ls: list[Line]) -> tuple[str | None, list[dict] | None]:
        """名字在上、血量在下；只有血量没有名字的是自己。战斗中可能被伤害面板盖住。"""
        if not ls:
            return None, None
        if any("伤害" in l.text for l in ls):
            return "damage", None
        # 名字、血量的行高都差不多；矮一截的是血量左边的小图标（✓ ✗ •••）被认成了「0」之类
        med = sorted(l.h for l in ls)[len(ls) // 2]
        ls = [l for l in ls if l.h >= 0.6 * med]
        used: set[int] = set()
        players = []
        for i, l in enumerate(ls):
            # 血量行：纯数字，允许前面带一个图标杂字（OCR 常读成「v100」「X 69」「%100」）
            m = HP_LINE.fullmatch(norm_digits(l.text).strip())
            if not m:
                continue
            digits = m.group(1)
            hp = int(digits)
            if hp > 100:                     # 前面的 ✓ 被认成数字：「599」「5100」
                hp = 100 if digits.endswith("100") else int(digits[-2:])
            p: dict = {"hp": hp}
            cands = [(j, n) for j, n in enumerate(ls) if j not in used and j != i and has_words(n.text)
                     and not HP_LINE.fullmatch(norm_digits(n.text).strip())
                     and n.mid_y < l.mid_y and l.mid_y - n.mid_y < 0.08]
            if cands:
                j, n = max(cands, key=lambda c: c[1].mid_y)
                used.add(j)
                p["name"] = self._canonical_name(n.text)
            else:
                p["self"] = True
            if region := self.regions.get('players'):
                from jccbrain.vision.eye.scout import PLAYER_STEP, PLAYER_Y0
                # 27帧头像中心核对图：固定8行，不按OCR剩下的行号编号；
                # HP行比头像中心低约.016H，单行漏读不挤压后续slot。
                slot = round((region.y + l.mid_y*region.h - .016 - PLAYER_Y0) / PLAYER_STEP)
                p['slot'] = slot if 0 <= slot <= 7 else None
                p['y'] = round(PLAYER_Y0 + slot*PLAYER_STEP, 4) if 0 <= slot <= 7 else None
            players.append(p)
        return "players", players
