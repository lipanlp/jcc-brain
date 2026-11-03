"""散件栏/单位装备：当前赛季官方图标、实拍通用道具、位置与未知图案编号。"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from jccbrain.core import paths
from jccbrain.vision.eye.capture import load_image, warm_decoder
from jccbrain.vision.eye.digits import read_digits
from jccbrain.vision.eye.image import hsv_image

log = logging.getLogger(__name__)
CATALOG = Path(__file__).with_name("item_catalog.json")
EXTRA_ITEMS = Path(__file__).with_name("extra_items.json")
EXTRA_ICONS = Path(__file__).with_name("extra_icons.npz")
KINDS = {"基础装备": "基础", "成型装备": "成型", "光明武器": "光明", "神器装备": "神器",
         "转职纹章": "纹章", "特殊装备": "特殊", "-1": "特殊"}
# 1774×1136，54 帧 1959 格：299 个散件栏正格最低 0.346，241 个负格最高 0；
# 海克斯遮罩下金框仍有 V>0.08，原 V>0.28 会漏 13 格。这里只降低金色明度条件。
GOLD_LIMIT = .20
# 73 个单位装备正格 0.579～1.0；1346 个负格最高 0.769。0.80 拒绝所有负格，
# 7 个变形/数字遮挡的正格漏检，专项回报单列召回率，不以样本总准确率掩盖它。
FRAME_LIMIT = .80
# 裁 8% 边框的 16×16 零均值相关：318 个已标名字正格，正确候选最低 0.507，
# 最小领先差 0.075；54 个未知正格最高 0.509。取 0.60 / 0.08，模糊格保留未知。
NAME_SCORE, NAME_MARGIN = .60, .08
# 53 个未知棕色道具（角标4/5、含遮罩）遮掉角标后同图案最低0.993，其他装备最高0.436。
UNKNOWN_SCORE = .90
# 1774×1136：仅用 0000（数量4）/0018（数量5）两帧模板，左上 .36×.36 置零；
# 全部53个拆卸器格最低0.993649，其他319个已知/未知装备格最高0.434990。
# 取两者之间0.72；51个非模板帧（含变暗遮罩）只用于验证，不加入模板。
EXTRA_SCORE = .72


def season_key(value) -> str:
    return {"18": "18_S19", "16": "16_S19"}.get(str(value), str(value))


def icon_vector(image: np.ndarray, inset: float = .08) -> np.ndarray:
    h, w = image.shape[:2]
    dx, dy = round(w * inset), round(h * inset)
    image = image[dy:h - dy, dx:w - dx]
    vector = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def masked_icon_vector(patch: np.ndarray) -> np.ndarray:
    """数量角标不参与通用道具身份；区域按格子大小的比例遮掉。"""
    image = patch.copy()
    h, w = image.shape[:2]
    image[:round(.36 * h), :round(.36 * w)] = 0
    return icon_vector(image)


class ItemCatalog:
    def __init__(self, season_id: str, entries: list[dict], directory: Path):
        self.season_id = season_key(season_id)
        self.entries = [dict(row) for row in entries if row.get("name")]
        self.directory = Path(directory) / ("s" + self.season_id)
        self.loaded: list[dict] = []
        self.matrix = np.empty((0, 768), np.float32)
        self.load_cached()

    @classmethod
    def from_db(cls, db=None, directory: Path | None = None):
        bundled = json.loads(CATALOG.read_text("utf-8"))
        sid = season_key(getattr(db, "season_id", getattr(db, "set_id", bundled["season_id"])))
        if db is None:
            entries = bundled["items"]
        else:
            # 老版 GameDB 的 Item 视图没有 picture；只在同一赛季、同一 id 下补图标 URL，
            # 名字仍取传入数据库。新官方 season_id / 旧 set_id 两种接口都能读。
            pictures = {r["id"]: r.get("picture") for r in bundled["items"]} if sid == bundled["season_id"] else {}
            entries = [{"id": item.id, "name": item.name, "kind": item.kind,
                        "picture": getattr(item, "picture", None) or pictures.get(str(item.id))}
                       for item in getattr(db, "items", {}).values()]
        return cls(sid, entries, directory or paths.var_dir() / "item_icons")

    def file(self, entry: dict) -> Path:
        # 数据库 id 不直接作为任意路径；官方装备 id 是数字，其他来源保留字母/下划线。
        key = "".join(c for c in str(entry["id"]) if c.isascii() and (c.isalnum() or c in "_-"))
        return self.directory / (key + ".png")

    def load_cached(self) -> None:
        loaded, vectors = [], []
        for entry in self.entries:
            file = self.file(entry)
            if not file.is_file():
                continue
            try:
                vectors.append(icon_vector(load_image(file), inset=0))
                loaded.append(entry)
            except Exception as error:
                log.warning("装备图标不可读 %s: %s", file.name, error)
        self.loaded = loaded
        self.matrix = np.stack(vectors) if vectors else np.empty((0, 768), np.float32)

    def prepare(self) -> None:
        """启动时下载缺失图标；识别热路径只读缓存。下载失败时保留未知，不换赛季猜名字。"""
        import httpx

        ready = {r['id'] for r in self.loaded}
        missing = [r for r in self.entries if r.get("picture") and r['id'] not in ready]
        if not missing:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        warm_decoder()      # 线程池里并发首次访问 Quartz 符号会偶发 KeyError
        offline = threading.Event()
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            def fetch(row):
                if offline.is_set():
                    return
                try:
                    response = client.get(row["picture"])
                    response.raise_for_status()
                except Exception as error:
                    if isinstance(error, httpx.TransportError):
                        # 离线时只等待已发出的至多4个请求，不给158个缺图逐批等超时。
                        offline.set()
                    log.warning("装备图标下载失败（网络）%s: %s", row["name"], error)
                    return
                temporary = self.directory / (self.file(row).stem + '.' + uuid.uuid4().hex + '.tmp')
                try:
                    temporary.write_bytes(response.content)
                    # 验证 CoreGraphics 能解码后才发布缓存，失败不会留下坏的 .png。
                    load_image(temporary)
                    temporary.replace(self.file(row))
                except Exception as error:
                    log.warning("装备图标下载失败（本地写入/解码）%s: %r", row["name"], error)
                finally:
                    temporary.unlink(missing_ok=True)
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="item-icons") as pool:
                list(pool.map(fetch, missing))
        self.load_cached()


def _rect(x, y, size) -> tuple[int, int, int, int]:
    return math.floor(x), math.floor(y), math.ceil(x + size), math.ceil(y + size)


def bench_slots(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    return [r for i in range(10) if (r := _rect(.0113 * w, (.1144 + i * .0592) * h, .052 * h))[3] < .8 * h]


def unit_slots(bar) -> list[tuple[int, int, int, int]]:
    w = bar.width
    return [_rect(bar.min_x - .12 * w + i * .40 * w, bar.max_y + .18 * w, .34 * w) for i in range(3)]


def _crop(image, rect):
    x0, y0, x1, y1 = rect
    return image[y0:y1, x0:x1] if 0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0] else None


def gold_frame(patch: np.ndarray | None, window_height: int = 1136) -> float:
    if patch is None:
        return 0.
    h, s, v = hsv_image(patch)
    gold = (h >= 20) & (h <= 70) & (s > .3) & (v > .08)
    edge, corner = max(1, round(4 * window_height / 1136)), max(1, round(4 * window_height / 1136))
    top, left = gold[:edge, corner:-corner], gold[corner:-corner, :edge]
    n = top.shape[1] + left.shape[0]
    return float((top.any(0).sum() + left.any(1).sum()) / n) if n > 0 else 0.


def frame_score(image: np.ndarray, rect) -> float:
    x0, y0, x1, y1 = rect
    scale = image.shape[0] / 1136
    outside, skip, depth = max(1, round(3 * scale)), max(1, round(2 * scale)), max(1, round(scale))
    padded = _crop(image, (x0 - outside, y0 - outside, x1 + outside, y1 + outside))
    if padded is None or min(x1 - x0, y1 - y0) <= 2 * skip + 1:
        return 0.
    v = padded.max(axis=2).astype(np.int16)
    x0 = y0 = outside
    x1, y1 = v.shape[1] - outside - 1, v.shape[0] - outside - 1
    xs, ys = slice(x0 + skip, x1 - 1), slice(y0 + skip, y1 - 1)
    edges = [np.minimum(v[y0, xs], v[y0 + depth, xs]) + 12 < v[y0 - outside, xs],
             np.minimum(v[y1, xs], v[y1 - depth, xs]) + 12 < v[y1 + outside, xs],
             np.minimum(v[ys, x0], v[ys, x0 + depth]) + 12 < v[ys, x0 - outside],
             np.minimum(v[ys, x1], v[ys, x1 - depth]) + 12 < v[ys, x1 + outside]]
    n = sum(a.size for a in edges)
    return float(sum(a.sum() for a in edges) / n) if n else 0.


def badge_number(patch: np.ndarray, window_height: int = 1136) -> int | None:
    """只读金底黑字角标，不把数字解释成道具数量/次数。"""
    h, w = patch.shape[:2]
    panel = patch[round(.05 * h):round(23 / 60 * h), round(.05 * w):round(.35 * w)]
    if not panel.size:
        return None
    hue, sat, value = hsv_image(panel)
    gold = (hue >= 20) & (hue <= 70) & (sat > .3) & (value > value.max() * .4)
    # 1774×1136：53 个角标金底覆盖率最低0.928，其余487格最高0.347；取0.65。
    if gold.mean() < .65:
        return None
    digit = patch[round(7 / 60 * h):round(.40 * h), round(5 / 60 * w):round(.35 * w)]
    inverted = 255 - digit.max(2)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pad = max(1, round(3 * window_height / 1136))
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    text = read_digits(np.repeat(mask[:, :, None], 3, 2), window_height)
    return int(text) if text.isdigit() else None


class ItemReader:
    def __init__(self, catalog: ItemCatalog):
        self.catalog = catalog
        self.extra_loaded: list[dict] = []
        self.extra_matrix = np.empty((0, 768), np.float32)
        self._load_extra()
        self.unknown: list[np.ndarray] = []
        self.last_ms = 0.

    def _load_extra(self) -> None:
        """只读随包资源；当前赛季数据库有同名条目时，连 extra 模板也不加载。"""
        names = {row['name'] for row in self.catalog.entries}
        try:
            data = json.loads(EXTRA_ITEMS.read_text('utf-8'))
            if data['version'] != 1:
                raise ValueError('不支持的通用道具清单版本')
            entries = [row for row in data['items'] if row['name'] not in names]
            loaded, vectors = [], []
            if entries:
                with np.load(EXTRA_ICONS, allow_pickle=False) as arrays:
                    for row in entries:
                        for index, _ in enumerate(row['templates']):
                            vectors.append(masked_icon_vector(arrays[f"{row['name']}/{index}"]))
                            loaded.append(row)
            self.extra_loaded = loaded
            self.extra_matrix = np.stack(vectors) if vectors else np.empty((0, 768), np.float32)
        except (OSError, ValueError, KeyError) as error:
            log.warning('通用道具资源不可读，保留未知装备: %s', error)

    def reset(self):
        self.unknown.clear()

    def identify(self, patch: np.ndarray, rect, shape, slot: int, bench: bool = False) -> dict:
        vector = icon_vector(patch)
        match, extra, feature = None, False, None
        if len(self.catalog.loaded):
            scores = self.catalog.matrix @ vector
            order = np.argsort(scores)[::-1]
            if scores[order[0]] >= NAME_SCORE and (len(order) == 1 or scores[order[0]] - scores[order[1]] >= NAME_MARGIN):
                match = self.catalog.loaded[order[0]]
        if match is None and self.extra_loaded:
            feature = masked_icon_vector(patch)
            scores = self.extra_matrix @ feature
            best = int(np.argmax(scores))
            if scores[best] >= EXTRA_SCORE:
                match, extra = self.extra_loaded[best], True
        if match:
            name = match['name']
            kind = match['kind'] if extra else KINDS.get(match.get('kind'), '特殊')
        else:
            # 未知图案用相关度聚组；角标不参与未知装备身份，4→5 不会另开组。
            if bench:
                feature = feature if feature is not None else masked_icon_vector(patch)
            else:
                feature = vector
            scores = [float(v @ feature) for v in self.unknown]
            if scores and max(scores) >= UNKNOWN_SCORE:
                gid = int(np.argmax(scores)) + 1
            else:
                self.unknown.append(feature)
                gid = len(self.unknown)
            name, kind = f"?i{gid}", None
        x0, y0, x1, y1 = rect
        out = {"slot": slot, "name": name, "kind": kind,
               "x": (x0 + x1) / 2 / shape[1], "y": (y0 + y1) / 2 / shape[0]}
        if bench and (badge := badge_number(patch, shape[0])) is not None:
            out["badge"] = badge
            # 用户已确认拆卸器角标表示持有数量；其他未知角标不猜语义。
            if name == '装备拆卸器':
                out['count'] = badge
        return out

    def bench(self, image: np.ndarray) -> list[dict]:
        start = time.perf_counter()
        found = []
        for i, rect in enumerate(bench_slots(image)):
            patch = _crop(image, rect)
            if gold_frame(patch, image.shape[0]) >= GOLD_LIMIT:
                found.append(self.identify(patch, rect, image.shape, i, bench=True))
        self.last_ms = (time.perf_counter() - start) * 1000
        return found

    def on_unit(self, image: np.ndarray, bar) -> list[dict]:
        start = time.perf_counter()
        found = []
        for i, rect in enumerate(unit_slots(bar)):
            patch = _crop(image, rect)
            if patch is not None and frame_score(image, rect) >= FRAME_LIMIT:
                found.append(self.identify(patch, rect, image.shape, i))
        self.last_ms += (time.perf_counter() - start) * 1000
        return found
