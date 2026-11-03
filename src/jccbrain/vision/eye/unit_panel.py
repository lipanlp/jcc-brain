"""Read a unit detail panel and its cyan selection; no identity inference or writes."""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .image import Region, crop

# 1774x1136 and 1576x1013, L1/preflight/panel-inventory-2:
# 24 real panels: gold vertical fraction min=1; 47 old negatives max=.071722;
# 2021 session frames max=.497951. Require a continuous border, before any OCR.
PRESENCE_THRESHOLD = .90
NAME_REGIONS = (Region(.836, .116, .115, .029), Region(.836, .110, .125, .045))
COST_REGION = Region(.854, .161, .028, .027)
TRAITS_REGION = Region(.779, .201, .22, .066)
PANEL_REGION = Region(.77, .07, .23, .20)


def compact(text: str) -> str:
    return ''.join(text.split())


def presence(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    patch = image[int(.17*h):int(.60*h), int(.772*w):int(.785*w)]
    if not patch.size:
        return 0.
    b, g, r = (patch[..., i].astype(np.int16) for i in range(3))
    mask = (r-g >= 15) & (g-b >= 15) & (r >= 100) & (g >= 70) & (r <= 220)
    return float(mask.mean(axis=0).max())


@dataclass
class PanelObservation:
    present: bool
    presence_score: float
    name: str | None = None
    cost: int | None = None
    traits: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    reason: str = 'no_panel'

    @property
    def valid(self) -> bool:
        return self.reason == 'consistent_fields'

    @property
    def signature(self):
        return self.name, self.cost, tuple(sorted(self.traits))

    def to_json(self):
        return asdict(self)


class PanelReader:
    def __init__(self, dictionary, ocr):
        self.dictionary = dictionary
        self.ocr = ocr
        self.calls = 0

    def _fresh_backend(self):
        # Reuse loaded OCR models, but never use the main reader's word/pixel caches
        # to manufacture a second panel confirmation or mutate its OCR memory.
        engine = copy.copy(self.ocr)
        if hasattr(engine, 'cache'):
            engine.cache, engine.words = OrderedDict(), deque(maxlen=256)
            engine.lock = threading.Lock()
            engine._stats_lock = threading.Lock()
            engine.reset_stats()
        return engine

    def read(self, image: np.ndarray) -> PanelObservation:
        score = presence(image)
        result = PanelObservation(score >= PRESENCE_THRESHOLD, score)
        if not result.present:
            return result
        if not self.dictionary.champions:
            result.reason = 'no_current_pool'
            return result
        engine = self._fresh_backend()
        regs = (*NAME_REGIONS, COST_REGION, TRAITS_REGION)
        cells = [engine.prepare(crop(image, r)) for r in regs]
        lines = engine.recognize_batch(cells, ['zh-Hans', 'en-US'],
                                       groups=['unit-name-1', 'unit-name-2', 'unit-cost', 'unit-traits'])
        self.calls += 1
        result.raw = {key: [asdict(line) for line in ls]
                      for key, ls in zip(('name_compact', 'name_wide', 'cost', 'traits'), lines)}
        names = {compact(l.text) for ls in lines[:2] for l in ls
                 if compact(l.text) in self.dictionary.champions}
        if len(names) != 1:
            result.reason = 'missing_or_ambiguous_exact_name'
            return result
        result.name = next(iter(names))
        # A single isolated decimal token; no permissive extraction from unrelated text.
        costs = {int(compact(l.text)) for l in lines[2] if compact(l.text).isascii()
                 and compact(l.text).isdecimal() and l.conf >= .5}
        if not costs and getattr(engine, 'name', None) == 'rapid':
            from .ocr import VisionOCR
            fallback = VisionOCR()
            fallback_lines = fallback.recognize_batch([fallback.prepare(crop(image, COST_REGION))],
                                                       ['zh-Hans', 'en-US'])[0]
            result.raw['cost_vision'] = [asdict(line) for line in fallback_lines]
            costs = {int(compact(l.text)) for l in fallback_lines if compact(l.text).isascii()
                     and compact(l.text).isdecimal() and l.conf >= .5}
        if len(costs) != 1:
            result.reason = 'missing_or_ambiguous_cost'
            return result
        result.cost = next(iter(costs))
        if result.cost != self.dictionary.champions[result.name]:
            # Upgraded sell price isn't the base card cost. Until star-aware pricing
            # is independently verified, retain it as evidence instead of guessing.
            result.reason = 'cost_mismatch'
            return result
        observed = [compact(l.text) for l in lines[3]]
        result.traits = sorted(t for t in self.dictionary.champion_traits.get(result.name, [])
                               if any(compact(t) in s for s in observed))
        result.reason = 'consistent_fields' if result.traits else 'trait_mismatch'
        return result


def selection_mask(image: np.ndarray) -> np.ndarray:
    b, g, r = (image[..., i].astype(np.int16) for i in range(3))
    return ((b >= 200) & (g >= 170) & (r <= 100) & (g-r >= 100) & (b-r >= 100))


def bind_selection(image, units, target=None):
    """Require unique cyan outline support around a detected unit, never nearest HP.

    Geometry scales with the observed bar width. Preliminary L1 live a/b/c:
    selected outline pixels/bar_width^2 min .07120, unselected max .01190;
    thresholds and all old/new positives and negatives are exported by the eval.
    Multiple matches, overlapping crops or an absent bar fail closed.
    """
    h, w = image.shape[:2]
    mask = selection_mask(image)
    candidates, diagnostics = [], []
    for i, unit in enumerate(units):
        bar = unit.bar
        if bar is None:
            continue
        bw = bar.width
        x0, x1 = max(0, int(unit.x*w-.8*bw)), min(w, int(unit.x*w+.8*bw))
        y0 = bar.max_y + max(1, round(.005*h))
        y1 = min(h, y0 + round(1.9*bw))
        patch = mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(patch)
        area = len(xs)/(bw*bw)
        extent_x = (int(xs.max())-int(xs.min())+1)/bw if len(xs) else 0.
        extent_y = (int(ys.max())-int(ys.min())+1)/bw if len(xs) else 0.
        accepted = area >= .04 and extent_x >= .45 and extent_y >= .60
        diag = dict(index=i, x=unit.x, y=unit.y, zone=unit.zone, slot=unit.slot,
                    cyan_area=area, extent_x=extent_x, extent_y=extent_y, selected=accepted,
                    rect=[x0, y0, x1, y1])
        diagnostics.append(diag)
        if accepted:
            candidates.append((unit, diag))
    if len(candidates) != 1:
        return None, diagnostics, 'no_unique_selection'
    unit, diag = candidates[0]
    # The clicked point must lie inside this supported selection region. It does
    # not choose a candidate and cannot rescue an ambiguous/no-outline frame.
    if target is not None:
        x, y = target
        x0, y0, x1, y1 = diag['rect']
        if not (x0 <= x*w < x1 and y0 <= y*h < y1):
            return None, diagnostics, 'target_mismatch'
    return unit, diagnostics, 'unique_selection'
