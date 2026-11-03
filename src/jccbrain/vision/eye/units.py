"""棋盘/备战席单位：头顶绿色血条定位，血条左侧图标定星级，Vision 特征向量 + 图库认人，跨帧跟踪。

血条/星级阈值沿用 game-eye 的 1774×1136 实测值（按窗口高度 /1136 缩放）；stale 身份门限另行标定。
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from jccbrain.vision.eye.image import Region, hsv_image

log = logging.getLogger(__name__)

# bench/board 分界和备战席格距（相对窗口）
BENCH_TOP, BENCH_X0, BENCH_STEP = 0.615, 0.244, 0.063
SCAN = Region(0.17, 0.08, 0.68, 0.62)
# B3独立CLI参考图→1774×1136旧样本：最近同名距离.5120～.9805，异名.6201～1.0371。
# 51组完整时序扫描：B标准准确率≥98%的最高覆盖为40/291；并列取更小t、更大m。
MATCH_THRESHOLD = 0.60
MATCH_MARGIN = 0.10        # 最近异名的领先差；同名多个样本不互相竞争
GROUP_THRESHOLD = 0.55      # 未知聚组沿用原阈值，独立于图库名字匹配阈值
HIST_LIMIT = 0.35           # 颜色直方图距离：小于它视为同一个单位没换人
# 1774×1136 的真实跨回合/特效/待机样本：4 对同单位距离 0.5583～0.5932；
# 7 种外观的 126 对异类距离最小 0.6713。只放宽同位置的时间关联，不放宽图库/全局聚组。
TRACK_THRESHOLD = 0.60


def stale_bench_threshold(same_max: float, different_min: float) -> float:
    """可分离时取中点，否则按指令保守拒绝（只保留异类最小距离的90%）。"""
    return (same_max + different_min) / 2 if same_max < different_min else different_min * .9


# 1774×1136，CoreGraphics 原图、46个唯一同名正对 / 598个高置信异名负对：
# F1整框（与缓存 s.fp 比）：同类最大0.8872992396 / 异类最小0.6156993508；
# F2中心（左右各去15%、上去10%、下去20%，新旧两边重算）：0.8591512442 / 0.4997614622。
# 两者均不可分离，依 #4a-v2 选择异类最小更大的 F1，保守阈值为其异类最小×0.9。
STALE_BENCH_FP = stale_bench_threshold(.8872992396354675, .6156993508338928)


def assign_stale_bench(distances: list[list[float]], threshold: float = STALE_BENCH_FP) -> dict[int, int]:
    """距离升序贪心一一指派；当前格和旧单位最多各用一次，门限严格小于。"""
    candidates = sorted((d, i, j) for i, row in enumerate(distances) for j, d in enumerate(row) if d < threshold)
    matched, used = {}, set()
    for _distance, current, old in candidates:
        if current not in matched and old not in used:
            matched[current] = old
            used.add(old)
    return matched


@dataclass
class Bar:
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    cx: float          # 0~1
    cy: float

    @property
    def thick(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1


def green_mask(img: np.ndarray) -> np.ndarray:
    """己方血条的绿色：G ≥ 140，比 R 高 25 以上、比 B 高 50 以上，且 R ≥ B − 5。"""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    return (g >= 140) & (g - r >= 25) & (g - b >= 50) & (r >= b - 5)


def _dark(px) -> bool:
    return int(max(px)) < 95


def detect_bars(img: np.ndarray, region: Region = SCAN, *, mask_fn=None,
                min_width: float | None = None) -> list[Bar]:
    """每行找绿色线段（段间深色缝 ≤ gap 并起来），上下相邻、横向重叠的线段连成血条，
    最后要求上下边缘是深色边框，排除绿色地面等误检。"""
    H, W = img.shape[:2]
    s = H / 1136
    x0, x1 = int(region.x * W), int((region.x + region.w) * W)
    y0, y1 = int(region.y * H), int((region.y + region.h) * H)
    gap, min_len, max_len = max(3, int(6 * s)), int(22 * s), int(140 * s)
    if min_width is not None:
        min_len = max(1, int(min_width * W))
    mask = (mask_fn or green_mask)(img[y0:y1, x0:x1])

    # 一次取出所有绿色像素并按「换行 / 缝隙超过 gap」切段；与逐行 flatnonzero/diff
    # 相同，但避免 1774×1136 下每帧约 704 次 NumPy/Python 往返，给并行 OCR 留出 GIL。
    rows: dict[int, list[tuple[int, int]]] = {}
    ys, xs = np.nonzero(mask)
    if len(xs):
        breaks = np.flatnonzero((np.diff(ys) != 0) | (np.diff(xs) > gap))
        starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(xs) - 1]
        lengths = xs[ends] - xs[starts] + 1
        valid = (lengths >= min_len) & (lengths <= max_len)
        for a, b in zip(starts[valid], ends[valid]):
            rows.setdefault(int(ys[a]), []).append((int(xs[a]) + x0, int(xs[b]) + x0))

    open_: list[list[int]] = []     # [minX, maxX, minY, maxY]
    done: list[list[int]] = []
    for yy in range(mask.shape[0]):
        runs = rows.get(yy, [])
        nxt = []
        y = yy + y0
        for a, b in runs:
            idx = next((i for i, o in enumerate(open_) if min(o[1], b) - max(o[0], a) > (b - a) // 2), None)
            if idx is not None:
                o = open_.pop(idx)
                o[0], o[1], o[3] = min(o[0], a), max(o[1], b), y
                nxt.append(o)
            else:
                nxt.append([a, b, y, y])
        done += open_
        open_ = nxt
    done += open_

    bars = []
    for mnx, mxx, mny, mxy in done:
        t = mxy - mny + 1
        if not (max(3, int(4 * s)) <= t <= int(16 * s)):
            continue
        mx = (mnx + mxx) // 2
        if mny - 2 < 0 or mxy + 3 >= H or mx + 3 >= W:
            continue
        top_ok = _dark(img[mny - 2, mx]) or _dark(img[mny - 2, mx + 3])
        bot_ok = _dark(img[mxy + 2, mx]) or _dark(img[mxy + 2, mx + 3]) or _dark(img[mxy + 3, mx])
        if top_ok and bot_ok:
            bars.append(Bar(mnx, mxx, mny, mxy, (mnx + mxx) / 2 / W, (mny + mxy) / 2 / H))
    return bars


def star_of(img: np.ndarray, b: Bar) -> int | None:
    """血条左侧图标 → 星级。1 星金铜色小箭头；2 星银白菱形；3 星金色菱形（上下都伸出血条）。"""
    H = img.shape[0]
    s = H / 1136
    x0, x1 = max(0, b.min_x - int(24 * s)), max(0, b.min_x - 1)
    mid = (b.min_y + b.max_y) // 2
    y0, y1 = max(0, mid - int(15 * s)), min(H - 1, mid + int(15 * s))
    if x1 <= x0:
        return None
    patch = img[y0:y1 + 1, x0:x1]
    h, sat, v = hsv_image(patch)
    gold = (h >= 15) & (h <= 70) & (sat > 0.3) & (v > 0.5)
    silver = (sat < 0.2) & (v > 0.7)
    n_gold, n_silver = int(gold.sum()), int(silver.sum())
    if n_gold + n_silver < 8:
        return None
    ys = np.flatnonzero((gold | silver).any(axis=1)) + y0
    extent = (ys.max() - ys.min() + 1) / b.thick
    if n_silver > n_gold:
        return 2
    # 金色菱形会同时伸出血条上下沿；只伸出一边多半是单位身上的金色部件
    half = max(2, b.thick // 2)
    rows = gold.sum(axis=1)
    above = int(rows[: max(0, b.min_y - half - y0)].sum())
    below = int(rows[max(0, b.max_y + half + 1 - y0):].sum())
    return 3 if extent >= 2.2 and above >= 3 and below >= 3 else 1


def unit_crop(img: np.ndarray, b: Bar) -> np.ndarray | None:
    """血条下方的单位画面。"""
    H, W = img.shape[:2]
    w = b.width
    x0 = int(math.floor(b.min_x - 0.05 * w))
    y0 = b.max_y + 6
    x1 = int(math.ceil(b.min_x - 0.05 * w + w * 1.15))
    y1 = int(math.ceil(y0 + w * 1.3))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def color_hist(img: np.ndarray) -> np.ndarray:
    """4×4×4 颜色直方图（归一化）：待机动画颜色分布基本不变，换了英雄差别很大。"""
    small = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    q = small.astype(np.int32) // 64
    idx = q[..., 2] * 16 + q[..., 1] * 4 + q[..., 0]
    return np.bincount(idx.ravel(), minlength=64).astype(np.float32) / (24 * 24)


def hist_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 2.0
    return float(np.abs(a - b).sum())


def feature_print(img: np.ndarray) -> np.ndarray | None:
    """Apple Vision 图像特征向量（768 维单位向量）。"""
    import Vision

    from jccbrain.vision.eye.ocr import to_cgimage

    req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(to_cgimage(img), {})
    handler.performRequests_error_([req], None)
    res = req.results()
    if not res:
        return None
    return np.frombuffer(bytes(res[0].data()), dtype=np.float32).copy()


def fp_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ------------------------------------------------------------------ 图库
class Gallery:
    """单位图库：正式读写 root/s<season>，只接受当前卡池内的名字。

    season=None 保留旧的直接目录接口；pool=None 则关闭全部名字读写。
    """

    def __init__(self, root: Path, season: str | None = None, pool: set[str] | None = None,
                 threshold: float = MATCH_THRESHOLD, seed_from: Path | None = None,
                 refs_root: Path | None = None, margin: float = MATCH_MARGIN, verified_store=None):
        self.root = Path(root)
        self.dir = self.root / ("s" + str(season)) if season is not None else self.root
        self.pool = frozenset(pool) if pool is not None else None
        self.threshold = threshold
        self.margin = margin
        self.season = season
        self.verified_store = verified_store
        self.items: list[tuple[str, np.ndarray]] = []
        self.index: list[dict] = []
        self.refs: tuple[tuple[str, np.ndarray], ...] = ()
        if self.pool is None:
            # 无卡池启动不创建赛季目录，避免阻止以后有数据库时的首次迁移。
            return
        self.refs = self._load_refs(season, refs_root or Path(__file__).with_name('unit_refs'))
        fresh = not self.dir.exists()
        self.dir.mkdir(parents=True, exist_ok=True)
        idx_file = self.dir / "index.json"
        if season is not None and fresh and (self.root / "index.json").exists():
            self._migrate_flat()
        if idx_file.exists():
            skipped = Counter()
            for e in json.loads(idx_file.read_text(encoding="utf-8")):
                if not self.allows(e["name"]):
                    skipped[e["name"]] += 1
                    continue
                p = self.dir / e["file"]
                if p.exists():
                    self.index.append(e)
                    self.items.append((e["name"], np.load(p)))
            if skipped:
                log.info("图库加载丢弃卡池外条目 %d 条：%s", sum(skipped.values()), dict(skipped))
        elif season is not None and fresh and seed_from and (seed_from / "index.json").exists():
            self._import_game_eye(seed_from)

    def allows(self, name: str | None) -> bool:
        return self.pool is not None and name in self.pool

    def _load_refs(self, season: str | None, directory: Path) -> tuple[tuple[str, np.ndarray], ...]:
        """随包资源只读；与用户 index/items 分开，未知赛季或卡池不加载。"""
        manifest = directory / 'unit_refs.json'
        if season is None or not manifest.is_file():
            return ()
        refs = []
        try:
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            if metadata.get('version') != 1 or metadata.get('season') != str(season):
                return ()
            counts = Counter()
            with np.load(directory / f's{season}.npz', allow_pickle=False) as arrays:
                for row in metadata['entries']:
                    name = row['name']
                    if not self.allows(name) or counts[name] >= 3:
                        continue
                    vec = arrays[row['key']].astype(np.float32, copy=True)
                    if vec.ndim != 1 or not vec.size or not np.isfinite(vec).all():
                        continue
                    vec.setflags(write=False)
                    refs.append((name, vec))
                    counts[name] += 1
        except (OSError, ValueError, KeyError, TypeError):
            log.warning('随包英雄参考图库无法读取：%s', directory, exc_info=True)
            return ()
        return tuple(refs)

    def _migrate_flat(self) -> None:
        """只复制卡池内条目，保持平铺文件和文件名不变；目录存在时不再迁移。"""
        copied, skipped = [], Counter()
        for e in json.loads((self.root / "index.json").read_text(encoding="utf-8")):
            if not self.allows(e["name"]):
                skipped[e["name"]] += 1
                continue
            source = self.root / e["file"]
            if not source.is_file():
                log.warning("图库迁移缺少向量文件：%s", source)
                continue
            shutil.copy2(source, self.dir / source.name)
            jpg = source.with_suffix(".jpg")
            if jpg.is_file():
                shutil.copy2(jpg, self.dir / jpg.name)
            copied.append(e)
        (self.dir / "index.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")
        migration = {"from": "flat", "copied": len(copied), "skipped": dict(skipped)}
        (self.dir / "migration.json").write_text(json.dumps(migration, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("图库迁移至 %s：%s", self.dir, migration)

    def _import_game_eye(self, src: Path) -> None:
        """导入 game-eye 的图库（.fp 是 NSKeyedArchiver 归档的 VNFeaturePrintObservation）。"""
        try:
            import Vision
            from Foundation import NSData, NSKeyedUnarchiver
        except ImportError:
            return
        skipped = Counter()
        for e in json.loads((src / "index.json").read_text(encoding="utf-8")):
            if not self.allows(e["name"]):
                skipped[e["name"]] += 1
                continue
            data = NSData.dataWithContentsOfFile_(str(src / e["file"]))
            if data is None:
                continue
            obs, _err = NSKeyedUnarchiver.unarchivedObjectOfClass_fromData_error_(
                Vision.VNFeaturePrintObservation, data, None)
            if obs is None:
                continue
            vec = np.frombuffer(bytes(obs.data()), dtype=np.float32).copy()
            jpg = src / e["file"].replace(".fp", ".jpg")
            self.add(e["name"], vec, None, copy_jpg=jpg if jpg.exists() else None, source="legacy")
        # 即使全被拒绝，也记录一次导入完成，不在下次启动重复导入。
        (self.dir / "index.json").write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("从 game-eye 导入图库 %d 条，丢弃卡池外 %d 条：%s",
                 len(self.items), sum(skipped.values()), dict(skipped))

    def vectors(self):
        """既有用户条目来源未知时视为legacy；只读参考和已确认新样本分别保留来源。"""
        yield from self.items
        yield from self.refs
        if self.verified_store is not None and str(self.season) == self.verified_store.season:
            yield from self.verified_store.active_vectors()

    def provenance(self) -> list[dict]:
        return [dict(e, source=e.get('source', 'legacy')) for e in self.index]

    def match(self, fp: np.ndarray) -> tuple[str, float] | None:
        by_name = {}
        for name, v in self.vectors():
            if not self.allows(name) or fp.shape != v.shape:
                continue
            d = fp_distance(fp, v)
            by_name[name] = min(d, by_name.get(name, float('inf')))
        ranked = sorted(by_name.items(), key=lambda pair: pair[1])
        if not ranked or ranked[0][1] >= self.threshold:
            return None
        if len(ranked) > 1 and ranked[1][1] - ranked[0][1] < self.margin:
            return None
        return ranked[0]

    def add(self, name: str, fp: np.ndarray, crop: np.ndarray | None, copy_jpg: Path | None = None,
            *, source: str = "manual") -> bool:
        if not self.allows(name):
            log.warning("拒绝:%s(不在当前赛季卡池)", name)
            return False
        sid = uuid.uuid4().hex[:8].upper()
        file = f"{name}_{sid}.npy"
        np.save(self.dir / file, fp)
        if crop is not None:
            cv2.imwrite(str(self.dir / f"{name}_{sid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        elif copy_jpg:
            shutil.copy(copy_jpg, self.dir / f"{name}_{sid}.jpg")
        self.items.append((name, fp))
        self.index.append({"name": name, "file": file, "source": source})
        (self.dir / "index.json").write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    @property
    def names(self) -> dict[str, int]:
        extra = (self.verified_store.active_vectors() if self.verified_store is not None
                 and str(self.season) == self.verified_store.season else ())
        return dict(Counter(n for n, _ in (*self.items, *extra) if self.allows(n)))


# ------------------------------------------------------------------ 跟踪
@dataclass
class TrackedUnit:
    zone: str                       # bench / board
    slot: int | None
    x: float
    y: float
    star: int | None
    bar: Bar | None = None
    name: str | None = None
    name_source: str | None = None
    dist: float | None = None
    group: int | None = None
    stars: list[int] = field(default_factory=list)
    hist: np.ndarray | None = None
    fp: np.ndarray | None = None
    fp_prepared: bool = False        # 已尝试预计算；即使无结果也不在 observe 里重复请求
    crop: np.ndarray | None = None
    items: list[dict] | None = None

    def to_json(self, pool: set[str] | frozenset[str] | None = None) -> dict:
        j: dict = {"slot": self.slot} if self.slot is not None else {"x": round(self.x, 3), "y": round(self.y, 3)}
        if self.star is not None:
            j["star"] = self.star
        if self.name and pool is not None and self.name in pool:
            j["name"] = self.name
        elif self.group is not None:
            j["name"] = f"?{self.group}"
        if self.items is not None:
            j["items"] = self.items
        return j


def _near(a: TrackedUnit, b: TrackedUnit) -> bool:
    return a.zone == b.zone and abs(a.x - b.x) < 0.012 and abs(a.y - b.y) < 0.02


class UnitTracker:
    def read_current(self, img: np.ndarray) -> dict:
        """Read current foreign pixels without tracking, grouping or learning.

        Keep legend geometry on the original image, exactly as the main reader;
        only unit positions use the accepted scout upper-edge/short-bar path.
        """
        from jccbrain.vision.eye.scout_units import detect_scout_units

        meta, _ = self.detect(img)
        image = img
        if image.shape[0] < 1136:
            # Same low-resolution normalization as ScoutReader; no new threshold.
            image = cv2.resize(image, None, fx=1136/image.shape[0], fy=1136/image.shape[0],
                               interpolation=cv2.INTER_CUBIC)
        detected = detect_scout_units(image, self)
        for i, unit in enumerate(detected, 1):
            unit.group = i  # IDs are meaningful only within this frame.
            fp = feature_print(unit.crop) if unit.crop is not None else None
            if fp is not None and (match := self.gallery.match(fp)):
                unit.name, unit.dist = match
        return dict(meta, board=[u.to_json(self.gallery.pool) for u in sorted(detected, key=lambda u: (u.y, u.x))
                                 if u.zone == 'board'],
                    bench=[u.to_json(self.gallery.pool) for u in sorted(detected, key=lambda u: u.slot or 0)
                           if u.zone == 'bench'], units_stale=False)

    def __init__(self, gallery: Gallery, items=None):
        self.gallery = gallery
        self.items = items
        self.prev: list[TrackedUnit] = []
        self.last_live: tuple[list[TrackedUnit], list[TrackedUnit]] | None = None
        self.unknown: list[tuple[int, np.ndarray]] = []
        self.prev_shop: list[str | None] | None = None
        self.group_name: dict[int, str] = {}
        self.last_units: list[TrackedUnit] = []
        self.last_fp_count = 0
        self.last_stale_fp_count = 0
        self.last_stale_assignments: list[dict] = []
        self.learning_candidates: list[dict] = []
        self.conflicted_groups: set[int] = set()
        self._verified_revision = None

    def reset(self) -> None:
        """新对局：清空跨帧记忆（图库保留）。"""
        if self.items:
            self.items.reset()
        self.__init__(self.gallery, self.items)

    def detect(self, img: np.ndarray) -> tuple[dict, list[TrackedUnit]]:
        """提取当前帧的血条、星级与裁剪，不改跨帧跟踪状态。"""
        out: dict = {}
        s = img.shape[0] / 1136
        units: list[TrackedUnit] = []
        for b in detect_bars(img):
            if b.thick >= 9 * s:          # 小小英雄：血条更厚
                out["legend"] = {"x": round(b.cx, 3), "y": round(b.cy, 3)}
                continue
            bench = b.cy > BENCH_TOP
            slot = round((b.cx - BENCH_X0) / BENCH_STEP) if bench else None
            if slot is not None and not 0 <= slot <= 8:
                continue
            u = TrackedUnit("bench" if bench else "board", slot, b.cx, b.cy, star_of(img, b), b)
            u.crop = unit_crop(img, b)
            if u.crop is not None:
                u.hist = color_hist(u.crop)
            units.append(u)
        return out, units

    def prepare(self, img: np.ndarray, features: bool = True) -> tuple[dict, list[TrackedUnit]]:
        """与 OCR 并行准备特征；商店自动标注及跟踪状态更新留给 observe。"""
        out, units = self.detect(img)
        if not features:
            return out, units
        for u in units:
            p = next((p for p in self.prev if _near(p, u)), None)
            if p and p.fp is not None and hist_distance(p.hist, u.hist) < HIST_LIMIT:
                continue
            if u.crop is not None:
                u.fp = feature_print(u.crop)
                u.fp_prepared = True
        return out, units

    def _stale_bench(self, img: np.ndarray, bench: list[TrackedUnit], live: list[TrackedUnit]) -> list[dict]:
        """只校验备战席：不改 last_live 身份锚点，不学习商店、不把棋盘送入该流程。"""
        self.last_stale_fp_count = 0
        self.last_stale_assignments = []
        result = {}
        if live:
            # 复用正常的特征/图库识别，隔离 prev/shop/last_live，避免战斗中误判购买。
            recognizer = UnitTracker(self.gallery, self.items)
            recognizer.last_live = (bench, [])
            recognizer.unknown, recognizer.group_name = self.unknown, dict(self.group_name)
            state = recognizer.observe(img, "planning", None, prepared=({}, live))
            self.unknown, self.group_name = recognizer.unknown, recognizer.group_name
            self.last_stale_fp_count += sum(not u.fp_prepared and u.crop is not None for u in live)
            result.update((u["slot"], u) for u in state["bench"])
            self.last_stale_assignments.extend({"slot": u.slot, "live": True, "old_slot": None} for u in live)

        targets = [u for u in bench if u.slot not in result]
        distances, current_fps = [], []
        for current in targets:
            crop = unit_crop(img, current.bar) if current.bar is not None else None
            fp = feature_print(crop) if crop is not None else None
            self.last_stale_fp_count += crop is not None
            current_fps.append(fp)
            distances.append([fp_distance(fp, old.fp) if fp is not None and old.fp is not None else math.inf
                              for old in bench])
        matched = assign_stale_bench(distances)
        for i, current in enumerate(targets):
            old = bench[matched[i]] if i in matched else None
            if old is not None:
                output = replace(current, name=old.name, group=old.group, star=old.star).to_json(self.gallery.pool)
            else:
                # 保留旧的星级/装备等结构，移除不能由当前像素确认的身份。
                output = current.to_json(self.gallery.pool)
                output.pop("name", None)
                output["unsure"] = True
                # 只读当前像素的图库匹配；不学习、不写回、不改准备阶段身份锚点。
                if current_fps[i] is not None and (found := self.gallery.match(current_fps[i])):
                    output["name"] = found[0]
                    output.pop("unsure", None)
            result[current.slot] = output
            self.last_stale_assignments.append({"slot": current.slot, "live": False,
                                               "old_slot": old.slot if old is not None else None,
                                               "distance": distances[i][matched[i]] if old is not None else None})
        self.last_fp_count = self.last_stale_fp_count
        return [result[slot] for slot in sorted(result)]

    def observe(self, img: np.ndarray, phase: str, shop: list[str | None] | None,
                prepared: tuple[dict, list[TrackedUnit]] | None = None) -> dict:
        out, units = prepared if prepared is not None else self.detect(img)
        self.learning_candidates = []
        if self.gallery.verified_store is not None:
            revision = self.gallery.verified_store.revision()
            if revision != self._verified_revision:
                # A new sample or revocation also invalidates remembered/group
                # identities. Otherwise a cached unknown never sees a new vector,
                # or an old group could reintroduce a revoked name after match().
                self.group_name.clear()
                remembered = self.prev + (sum(self.last_live, []) if self.last_live else [])
                for old in remembered:
                    if old.name_source != 'manual':
                        old.name, old.name_source = None, None
                self._verified_revision = revision

        # 战斗中备战席血条隐藏、棋盘单位在跑：沿用准备阶段最后的阵容
        live = phase in ("planning", "unknown")
        if not live and self.last_live:
            bench, board = self.last_live
            out.update(bench=self._stale_bench(img, bench, [u for u in units if u.zone == "bench"]),
                       board=[u.to_json(self.gallery.pool) for u in board], units_stale=True)
            self.prev, self.prev_shop = [], None
            return out

        self.last_fp_count = 0
        # 战斗清空 prev 以阻止跨战斗误判购买/上阵，但保留最后准备阶段作为身份锚点。
        anchors = self.prev or (self.last_live[0] + self.last_live[1] if self.last_live else [])
        for u in units:
            if not self.gallery.allows(u.name):
                u.name = None
            prepared_fp, fp_prepared = u.fp, u.fp_prepared
            p = next((p for p in anchors if _near(p, u)), None)
            if p and self.prev and hist_distance(p.hist, u.hist) < HIST_LIMIT:
                u.name = p.name if self.gallery.allows(p.name) else None
                u.name_source = p.name_source
                u.group, u.fp, u.dist = p.group, p.fp, p.dist
                u.stars = (p.stars + ([u.star] if u.star else []))[-3:]
                if u.stars:
                    cnt = Counter(u.stars)
                    u.star = max(cnt, key=lambda k: (cnt[k], k))
                if u.fp is not None:
                    if self.gallery.verified_store is not None:
                        matched = self.gallery.match(u.fp)
                        u.name, u.dist = matched if matched else (None, None)
                        u.name_source = 'gallery' if matched else None
                    continue
            elif u.star:
                u.stars = [u.star]
            if u.crop is None:
                continue
            # 早期相位提示跳过特征、最终相位为准备阶段时，仍按原流程提取。
            fp = prepared_fp if fp_prepared else feature_print(u.crop)
            if fp is None:
                continue
            self.last_fp_count += 1
            u.fp = fp
            m = self.gallery.match(fp)
            if m:
                u.name, u.dist = m
                u.name_source = 'gallery'
            if (p and p.fp is not None and fp_distance(fp, p.fp) < TRACK_THRESHOLD
                    and not (u.name and p.name and u.name != p.name)):
                # 视角/特效改变颜色时仍需当前特征确认，不能只凭格子继承身份。
                u.name = u.name or (p.name if self.gallery.allows(p.name) else None)
                if not m:
                    u.name_source = p.name_source
                u.group = p.group
                if not m:
                    u.dist = p.dist
                u.stars = (p.stars + ([u.star] if u.star else []))[-3:]
                if u.stars:
                    counts = Counter(u.stars)
                    u.star = max(counts, key=lambda k: (counts[k], k))

        # 数量变化只是关联线索；必须有当前特征，且不能覆盖另一已知名字。此路径不学习。
        gone_bench = [p for p in self.prev if p.zone == "bench" and not any(u.zone == "bench" and u.slot == p.slot for u in units)]
        new_board = [u for u in units if u.zone == "board" and not any(_near(p, u) for p in self.prev)]
        if len(gone_bench) == 1 and len(new_board) == 1:
            g, nb = gone_bench[0], new_board[0]
            if (g.fp is not None and nb.fp is not None and fp_distance(g.fp, nb.fp) < TRACK_THRESHOLD
                    and not (g.name and nb.name and g.name != nb.name)):
                if nb.group is None:
                    nb.group = g.group
                if nb.name is None and self.gallery.allows(g.name):
                    nb.name = g.name
                    nb.name_source = g.name_source

        # OCR缺行/遮挡也会使商店消失；只保存候选，不命名、不传播、不写库。
        if self.prev_shop and shop and len(self.prev_shop) == len(shop) and self.prev:
            ps = self.prev_shop
            bought = [ps[i] for i in range(len(shop)) if ps[i] and not shop[i] and not ps[i].startswith("~")]
            bought_item = any(ps[i] and ps[i].startswith("~") and not shop[i] for i in range(len(shop)))
            refreshed = any(ps[i] and shop[i] and ps[i] != shop[i] for i in range(len(shop)))
            new_bench = [u for u in units if u.zone == "bench" and not any(p.zone == "bench" and p.slot == u.slot for p in self.prev)]
            if (len(bought) == 1 and self.gallery.allows(bought[0])
                    and not bought_item and not refreshed and len(new_bench) == 1):
                u = new_bench[0]
                self.learning_candidates.append({"source": "shop_disappearance", "slot": u.slot,
                                                 "candidate": bought[0], "known_name": u.name,
                                                 "reason": "unverified_inference"})

        # 没认出来的按特征聚组：同组 = 同一个英雄
        for u in units:
            if u.name or u.group is not None or u.fp is None:
                continue
            best = min(((i, fp_distance(u.fp, f)) for i, f in self.unknown), key=lambda t: t[1], default=None)
            if best and best[1] < GROUP_THRESHOLD:
                u.group = best[0]
                if best[1] > 0.25:
                    self.unknown.append((best[0], u.fp))
            else:
                gid = max((i for i, _ in self.unknown), default=0) + 1
                self.unknown.append((gid, u.fp))
                u.group = gid

        # 名字沿分组传播，旧缓存或外部注入的名字也必须通过卡池校验。
        self.group_name = {gid: name for gid, name in self.group_name.items() if self.gallery.allows(name)}
        names_by_group: dict[int, set[str]] = {}
        for gid, name in self.group_name.items():
            names_by_group.setdefault(gid, set()).add(name)
        for u in units:
            if self.gallery.allows(u.name) and u.group is not None:
                names_by_group.setdefault(u.group, set()).add(u.name)
        self.conflicted_groups.update(gid for gid, names in names_by_group.items() if len(names) > 1)
        for gid in self.conflicted_groups:
            self.group_name.pop(gid, None)
        for gid, names in names_by_group.items():
            if gid not in self.conflicted_groups:
                self.group_name[gid] = next(iter(names))
        for u in units:
            if u.group in self.conflicted_groups:
                if u.name_source == 'group':
                    u.name = None
                    u.name_source = None
                self.learning_candidates.append({"source": "group", "group": u.group,
                                                 "known_name": u.name, "reason": "conflicting_names"})
            elif not u.name and u.group is not None and u.group in self.group_name:
                u.name = self.group_name[u.group]
                u.name_source = 'group'

        if self.items:
            for u in units:
                u.items = self.items.on_unit(img, u.bar) if u.bar else []

        self.prev, self.prev_shop, self.last_units = units, shop, units
        bench = sorted((u for u in units if u.zone == "bench"), key=lambda u: u.slot or 0)
        board = sorted((u for u in units if u.zone == "board"), key=lambda u: (u.y, u.x))
        if bench or board:
            self.last_live = (bench, board)
        out["bench"] = [u.to_json(self.gallery.pool) for u in bench]
        out["board"] = [u.to_json(self.gallery.pool) for u in board]
        return out

    def label(self, bench: dict[int, str] | None = None, board: list[dict] | None = None) -> list[str]:
        """手动标注：bench 用格号，board 用坐标（取 0.05 内最近的单位）。"""
        done = []
        for slot, name in (bench or {}).items():
            if not self.gallery.allows(name):
                done.append(f"拒绝:{name}(不在当前赛季卡池)")
                continue
            u = next((u for u in self.last_units if u.zone == "bench" and u.slot == int(slot)), None)
            if u and u.fp is not None:
                self.gallery.add(name, u.fp, u.crop)
                u.name = name
                u.name_source = 'manual'
                if u.group is not None:
                    self.group_name[u.group] = name
                done.append(f"bench{slot}={name}")
        for b in board or []:
            if not self.gallery.allows(b["name"]):
                done.append(f"拒绝:{b['name']}(不在当前赛季卡池)")
                continue
            cands = [u for u in self.last_units if u.zone == "board"]
            u = min(cands, key=lambda u: math.hypot(u.x - b["x"], u.y - b["y"]), default=None)
            if u and math.hypot(u.x - b["x"], u.y - b["y"]) < 0.05 and u.fp is not None:
                self.gallery.add(b["name"], u.fp, u.crop)
                u.name = b["name"]
                u.name_source = 'manual'
                if u.group is not None:
                    self.group_name[u.group] = b["name"]
                done.append(f"board({b['x']},{b['y']})={b['name']}")
        self.prev = self.last_units
        return done
