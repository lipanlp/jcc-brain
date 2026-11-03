"""RapidOCR（PP-OCRv4 检测 + 识别，onnxruntime CPU）后端。

和 Apple Vision 版的区别：
- 不放大：商店/羁绊的等尺寸格子保留黑边后按区域成组检测，其余格子独立检测（DB，线程池并行），
  每行仍从原像素裁剪再识别；玩家栏及海克斯长描述保持独立输入；
- 行级缓存：像素完全一样的行直接用上次结果；不含数字的文字行（玩家名、羁绊名…）再按白色字形容差匹配——
  玩家列表背后的龙纹每帧在动、检测框会抖 1px，名字行照样命中，每帧只剩检测（约 20ms）和血量那几行要认；
- 识别输入宽度按行的实际宽高比（默认 PaddleOCR 会补到 320 宽，短行白白多算几倍）；
- 检测到了但识别不出（置信度 < MIN_SCORE）的行，交给 Apple Vision 再认一次（实测「斗士」PP-OCRv4 读不出，
  Vision 能读）；结果同样进行级缓存，同一行只补认一次。

实测（M 系列 CPU，1774×1136）：检测 0.45MP 约 130ms（全部格子都变时），识别每行 3~6ms。
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from jccbrain.vision.eye.digits import read_number_line
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.text import has_words

PAD = 8               # 检测前四周补黑边（贴边的字检测不到）
MIN_SCORE = 0.5       # 识别置信度下限
REC_W = 96            # 识别输入最小宽度（高 48）
WHITE = 150           # 容差缓存的白字阈值：名字/血量/商店字 min(R,G,B) ≥200，背景龙纹的亮边 ≤140（1774×1136 实测）
WORD_TOL = 0.10       # 文字行字形差异容差（同一行前后帧 <0.05；不同名字 >0.3）
# #P-r：指挥2026-10-22干净实测，1774×1136、track 20帧、间隔0.25s、每档30s：
# 1线程 CPU84% / 单帧117ms；2线程100% / 93ms；4线程114% / 83ms。
# 原自动线程+忙等为151% / 96ms；仅1线程满足CPU≤原来的60%（84/151≈56%）。
# 接受单帧增加21ms以降低游戏资源竞争，117ms仍在250ms预算内。
ORT_THREADS = 1
_SESSION_OPTIONS_LOCK = threading.RLock()


@contextmanager
def ort_session_options(threads: int, spin: bool = False):
    """仅构造本后端模型期间补充 ORT 选项，退出（含异常）恢复原始描述符。"""
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    with _SESSION_OPTIONS_LOCK:
        descriptor = vars(OrtInferSession)['_init_sess_opts']
        original = OrtInferSession._init_sess_opts

        def options(config):
            opts = original(config)
            opts.intra_op_num_threads = threads
            for kind in ('intra', 'inter'):
                opts.add_session_config_entry(f'session.{kind}_op.allow_spinning', '1' if spin else '0')
            return opts

        OrtInferSession._init_sess_opts = staticmethod(options)
        try:
            yield
        finally:
            OrtInferSession._init_sess_opts = descriptor


class RapidOCR:
    name = "rapid"

    def __init__(self, cache_size: int = 2048, fallback: bool = True, *,
                 ort_threads: int | None = None, spin: bool = False):
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime.ch_ppocr_det import TextDetector
        from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
        from rapidocr_onnxruntime.utils import read_yaml

        root = Path(rapidocr_onnxruntime.__file__).parent
        cfg = read_yaml(str(root / "config.yaml"))
        for k in ("Det", "Rec"):
            cfg[k]["model_path"] = str(root / cfg[k]["model_path"])
        cfg["Det"].update(limit_type="max", limit_side_len=2000)      # 原尺寸检测，不放大
        cfg["Rec"]["rec_img_shape"] = [3, 48, REC_W]
        with ort_session_options(ORT_THREADS if ort_threads is None else ort_threads, spin):
            self.det = TextDetector(cfg["Det"])
            self.rec = TextRecognizer(cfg["Rec"])
        self.cache: OrderedDict[bytes, tuple[str, float]] = OrderedDict()
        self.cache_size = cache_size
        self.words: deque[tuple[np.ndarray, tuple[str, float]]] = deque(maxlen=256)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rapid-det")
        self.last_rec = 0          # 上次实际送去识别的行数（其余命中缓存）
        self.fallback = fallback
        self._warm_lock = threading.Lock()
        self._warmed = False
        self._stats_lock = threading.Lock()
        self.reset_stats()

    def reset_stats(self) -> None:
        """清零内部累计耗时/行数；在一帧的所有批次开始前调用，不影响识别缓存。"""
        with self._stats_lock:
            self.last_ms = {k: 0.0 for k in ('det', 'rec', 'vision')}
            self.last_counts = {k: 0 for k in ('detected_lines', 'rec_lines', 'vision_lines')}

    def _record(self, kind: str, start: float, count: int = 0) -> None:
        with self._stats_lock:
            self.last_ms[kind] += (time.perf_counter() - start) * 1000
            self.last_counts[{'det': 'detected_lines', 'rec': 'rec_lines', 'vision': 'vision_lines'}[kind]] += count

    def warmup(self) -> None:
        """启动时运行检测、识别模型一次；合成图不进入文字缓存。重复调用无副作用。"""
        with self._warm_lock:
            if self._warmed:
                return
            # 1774×1136 下玩家栏和商店单格的检测输入约为 704×192、64×224。
            for h, w in ((704, 192), (64, 224)):
                self.det(np.zeros((h, w, 3), dtype=np.uint8))
            self.rec([np.zeros((48, REC_W, 3), dtype=np.uint8)])
            self._warmed = True

    def prepare(self, img: np.ndarray) -> np.ndarray:
        return img

    def clear_cache(self) -> None:
        """新对局清除文字行记忆；调用方须先收齐上一帧任务，模型与预热状态保留。"""
        with self.lock:
            self.cache.clear()
            self.words.clear()
            self.last_rec = 0

    # ------------------------------------------------------------ 检测
    def _detect(self, i: int, im: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
        """一格检测 → [(格子序号, 行框 x0,y0,x1,y1（格内像素）, 行图)]。"""
        h, w = im.shape[:2]
        start = time.perf_counter()
        boxes, _ = self.det(cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        self._record('det', start, len(boxes) if boxes is not None else 0)
        out = []
        for b in boxes if boxes is not None else []:
            xs, ys = b[:, 0] - PAD, b[:, 1] - PAD
            x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), w)
            y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), h)
            if x1 - x0 < 3 or y1 - y0 < 4:
                continue
            out.append((i, (x0, y0, x1, y1), im[y0:y1, x0:x1]))
        return out

    # ------------------------------------------------------------ 识别
    def _detect_group(self, cells: list[tuple[int, np.ndarray]]) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
        """同一区域的等尺寸格子一起检测；每格保留黑边，识别仍从原像素裁剪。"""
        if len(cells) == 1:
            return self._detect(*cells[0])
        width = max(im.shape[1] for _, im in cells) + 2 * PAD
        height = sum(im.shape[0] + 2 * PAD for _, im in cells)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        offsets = []
        y = 0
        for i, im in cells:
            h, w = im.shape[:2]
            canvas[y + PAD:y + PAD + h, PAD:PAD + w] = im
            offsets.append((i, y + PAD, im))
            y += h + 2 * PAD
        start = time.perf_counter()
        boxes, _ = self.det(canvas)
        self._record('det', start, len(boxes) if boxes is not None else 0)
        out = []
        for b in boxes if boxes is not None else []:
            cy = float(b[:, 1].mean())
            for i, y, im in offsets:
                h, w = im.shape[:2]
                if not y <= cy < y + h:
                    continue
                xs, ys = b[:, 0] - PAD, b[:, 1] - y
                x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), w)
                y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), h)
                if x1 - x0 >= 3 and y1 - y0 >= 4:
                    out.append((i, (x0, y0, x1, y1), im[y0:y1, x0:x1]))
                break
        return out

    @staticmethod
    def _key(line: np.ndarray) -> bytes:
        return hashlib.blake2b(line.tobytes() + bytes(str(line.shape), "ascii"), digest_size=16).digest()

    @staticmethod
    def _mask(line: np.ndarray) -> np.ndarray | None:
        """近白色字形（min(R,G,B) > WHITE），裁到字形自己的外框；白字不到 3%（灰字、彩字）返回 None。"""
        m = line.min(axis=2) > WHITE
        if m.sum() < 0.03 * m.size:
            return None
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        return m[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]

    def _word_hit(self, m: np.ndarray) -> tuple[str, float] | None:
        """在最近认过的「纯文字行」里找字形几乎一样的（XOR 像素 < WORD_TOL × 字形像素）。"""
        best = None
        for wm, r in self.words:
            if abs(wm.shape[0] - m.shape[0]) > 2 or abs(wm.shape[1] - m.shape[1]) > 3:
                continue
            H, W = max(wm.shape[0], m.shape[0]), max(wm.shape[1], m.shape[1])
            a = np.zeros((H, W), bool)
            b = np.zeros((H, W), bool)
            a[:wm.shape[0], :wm.shape[1]] = wm
            b[:m.shape[0], :m.shape[1]] = m
            d = np.count_nonzero(a ^ b) / max(np.count_nonzero(a), np.count_nonzero(b))
            if d < WORD_TOL and (best is None or d < best[0]):
                best = (d, r)
        return best[1] if best else None

    def _recognize(self, lines: list[np.ndarray]) -> list[tuple[str, float]]:
        """行图 → (文字, 置信度)。依次：精确缓存（像素完全一样）→ 纯数字行直接模板匹配 → 文字行的容差缓存，
        剩下的才送识别模型。"""
        keys = [self._key(l) for l in lines]
        masks: list[np.ndarray | None] = [None] * len(lines)
        res: list[tuple[str, float] | None] = []
        todo = []
        with self.lock:
            for i, k in enumerate(keys):
                hit = self.cache.get(k)
                if hit is not None:
                    self.cache.move_to_end(k)
                elif (num := read_number_line(lines[i])) is not None:
                    hit = (num, 1.0)
                elif (m := self._mask(lines[i])) is not None:
                    masks[i] = m
                    hit = self._word_hit(m)
                if hit is None:
                    todo.append(i)
                res.append(hit)
        self.last_rec = len(todo)
        if todo:
            out = self._recognize_batches([lines[i] for i in todo])
            out = [(t, float(s)) for t, s in out]
            # 宽不到高 1.5 倍的多半是图标（✓ ✗ 头像边），不值得补认
            weak = [k for k, (t, s) in enumerate(out) if (s < MIN_SCORE or not t.strip())
                    and lines[todo[k]].shape[1] >= 1.5 * lines[todo[k]].shape[0]]
            if weak and self.fallback:
                start = time.perf_counter()
                fallback = _vision_lines([lines[todo[k]] for k in weak])
                self._record('vision', start, len(weak))
                for k, v in zip(weak, fallback):
                    if v[0]:
                        out[k] = v
            with self.lock:
                for i, r in zip(todo, out):
                    res[i] = r
                    self.cache[keys[i]] = r
                    # 只有不含数字的文字行进容差缓存：数字差一笔就是另一个值（82/83），必须重认
                    if masks[i] is not None and r[1] >= MIN_SCORE and has_words(r[0]) and not any(c.isdigit() for c in r[0]):
                        self.words.append((masks[i], r))
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return res  # type: ignore[return-value]

    def _recognize_batches(self, lines: list[np.ndarray]) -> list[tuple[str, float]]:
        """并行原模型的批次；保留 argsort、批大小和每批最大宽高比，不改变模型输入。"""
        size = self.rec.rec_batch_num
        if len(lines) <= size:
            return self._rec_batch(lines)[0]
        order = np.argsort(np.array([line.shape[1] / line.shape[0] for line in lines]))
        batches = [order[i:i + size] for i in range(0, len(order), size)]
        futures = [self.pool.submit(self._rec_batch, [lines[i] for i in batch]) for batch in batches]
        wait(futures)  # 即使某批失败，也先收齐所有任务，避免下一帧与遗留任务交叉。
        out = [('', 0.0)] * len(lines)
        for batch, future in zip(batches, futures):
            for i, value in zip(batch, future.result()[0]):
                out[i] = value
        return out

    def _rec_batch(self, lines: list[np.ndarray]):
        start = time.perf_counter()
        try:
            return self.rec(lines)
        finally:
            self._record('rec', start, len(lines))

    def recognize_batch(self, imgs: list[np.ndarray], languages: list[str] | None = None,
                        groups: list[str] | None = None) -> list[list[Line]]:
        """每格 → 行列表（坐标相对格子 0~1）。groups 用于商店/羁绊格子成组检测，languages 不区分。"""
        found: list[list[Line]] = [[] for _ in imgs]
        if not imgs:
            return found
        batches = {}
        for i, im in enumerate(imgs):
            key = groups[i] if groups and groups[i] in ("traits", "shop", "shop_traits") else i
            batches.setdefault(key, []).append((i, im))
        futures = [self.pool.submit(self._detect_group, cells) for cells in batches.values()]
        # 任一检测失败时也等其他格子退出，避免下一帧与遗留任务交叉使用模型。
        wait(futures)
        dets = [d for future in futures for d in future.result()]
        texts = self._recognize([d[2] for d in dets])
        for (i, (x0, y0, x1, y1), _), (text, score) in zip(dets, texts):
            if score < MIN_SCORE or not text.strip():
                continue
            ih, iw = imgs[i].shape[:2]
            found[i].append(Line(text, score, x0 / iw, y0 / ih, (x1 - x0) / iw, (y1 - y0) / ih))
        from jccbrain.vision.eye.ocr import _line_key
        for lines in found:
            lines.sort(key=_line_key(lines))
        return found


def _vision_lines(lines: list[np.ndarray]) -> list[tuple[str, float]]:
    """Apple Vision 补认（只在 macOS 上有）；失败就当没认出。"""
    try:
        from jccbrain.vision.eye.image import prep
        from jccbrain.vision.eye.ocr import recognize_batch
        found = recognize_batch([prep(l) for l in lines], ["zh-Hans", "en-US"])
    except Exception:          # noqa: BLE001 —— 没有 pyobjc / Vision 时静默退化
        return [("", 0.0)] * len(lines)
    return [(" ".join(l.text for l in ls), min((l.conf for l in ls), default=0.0)) for ls in found]
