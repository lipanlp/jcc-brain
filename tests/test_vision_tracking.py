"""T4：真实截图特征的跨回合身份、替换单位，以及新局首帧的完整隔离。"""
import json
import sys
import threading
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.units import Gallery, TrackedUnit, UnitTracker, fp_distance

FIXTURES = Path(__file__).parent / "fixtures"
DATA = json.loads((FIXTURES / "tracking_units.json").read_text("utf-8"))
mac = pytest.mark.skipif(sys.platform != "darwin", reason="需要 CoreGraphics / Apple Vision")


def unit(key):
    row = DATA["samples"][key]
    with np.load(FIXTURES / "tracking_units.npz") as arrays:
        return TrackedUnit(row["zone"], row["slot"], row["x"], row["y"], row["star"],
                           fp=arrays[f"{key}_fp"], hist=arrays[f"{key}_hist"], fp_prepared=True,
                           crop=arrays[f"{key}_crop"] if f"{key}_crop" in arrays else np.zeros((1, 1, 3), np.uint8))


def observe(tracker, units, phase="planning", shop=None):
    return tracker.observe(np.zeros((1, 1, 3), np.uint8), phase, shop, prepared=({}, units))


def test_real_association_distance_separates_effects_from_other_heroes():
    from jccbrain.vision.eye.units import TRACK_THRESHOLD

    distances = [fp_distance(unit(a).fp, unit(b).fp) for a, b in DATA["positive"]]
    assert min(distances) > .55  # 原来的全局阈值会把这四次变化另开组。
    assert max(distances) < TRACK_THRESHOLD
    negatives = [fp_distance(unit(a).fp, unit(b).fp)
                 for ca, aa in DATA["classes"].items() for cb, bb in DATA["classes"].items() if ca < cb
                 for a in aa for b in bb]
    assert len(negatives) == 126
    assert min(negatives) > TRACK_THRESHOLD


@pytest.mark.parametrize("sequence", [
    ["0009_0", "0012_0", None, "0021_0", "0025_0", "0026_0", "0027_0", "0051_0", "0076_0"],
    ["0021_0", "0025_0", "0026_0", "0027_0"],
    ["0055_5", "0056_5", "0076_5"],
], ids=["previous-round-and-combat", "gold-effect", "idle-animation"])
def test_same_real_unit_keeps_group_through_rounds_and_effects(tmp_path, sequence):
    tracker = UnitTracker(Gallery(tmp_path))
    group = None
    for key in sequence:
        if key is None:
            state = observe(tracker, [], "combat")
            assert state["units_stale"] and not tracker.prev and tracker.prev_shop is None
            continue
        current = unit(key)
        observe(tracker, [current])
        group = current.group if group is None else group
        assert current.group == group, key
        assert current.star == 1


@pytest.mark.parametrize("combat", [False, True])
def test_different_real_hero_at_same_position_does_not_inherit_group(tmp_path, combat):
    tracker = UnitTracker(Gallery(tmp_path))
    before, after = unit("0027_0"), unit("0055_5")  # 126 对异类中距离最近的一对。
    after.x, after.y = before.x, before.y
    observe(tracker, [before])
    if combat:
        observe(tracker, [], "combat")
    observe(tracker, [after])
    assert after.group != before.group


@pytest.mark.parametrize("move", ["position", "zone"])
def test_looser_distance_requires_same_position_and_zone(tmp_path, move):
    tracker = UnitTracker(Gallery(tmp_path))
    before, after = unit("0021_0"), unit("0026_0")
    observe(tracker, [before])
    if move == "position":
        after.x += .05
    else:
        after.zone, after.slot = "bench", 0
    observe(tracker, [after])
    assert after.group != before.group
    from jccbrain.vision.eye.units import GROUP_THRESHOLD
    assert GROUP_THRESHOLD == .55  # 聚组门限独立于经过校准的名字门限。


def test_gallery_recognition_of_another_name_overrides_temporal_anchor(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, season=None, pool={"英雄甲", "英雄乙"}))
    before, after = unit("0021_0"), unit("0026_0")
    tracker.gallery.items = [("英雄甲", before.fp), ("英雄乙", after.fp)]
    observe(tracker, [before])
    observe(tracker, [after])
    assert before.name == "英雄甲" and after.name == "英雄乙"


def test_combat_anchor_does_not_create_a_false_purchase_label(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, season=None, pool={"英雄甲"}))
    observe(tracker, [unit("0021_0")], shop=["英雄甲", None])
    observe(tracker, [], "combat")
    added = unit("0055_5")
    added.zone, added.slot = "bench", 0
    state = observe(tracker, [unit("0026_0"), added], shop=[None, None])
    assert "auto_labeled" not in state and not tracker.gallery.items


@mac
@pytest.mark.parametrize("key", ["0009_0", "0021_0", "0026_0", "0027_0", "0055_5", "0056_5"])
def test_native_features_from_real_crops_match_recorded_vectors(key):
    from jccbrain.vision.eye.units import feature_print

    current = unit(key)
    assert fp_distance(feature_print(current.crop), current.fp) < 1e-5


def test_new_game_clears_all_transient_state_but_preserves_gallery_and_models(tmp_path):
    from jccbrain.vision.eye.rapid import RapidOCR

    ocr = RapidOCR.__new__(RapidOCR)
    ocr.cache = OrderedDict([(b"old", ("旧局", 1.0))])
    ocr.words = deque([(np.ones((2, 2)), ("旧局", 1.0))], maxlen=256)
    ocr.lock, ocr.last_rec, ocr._warmed = threading.Lock(), 3, True
    ocr.det, ocr.rec = object(), object()
    models = ocr.det, ocr.rec
    tracker = UnitTracker(Gallery(tmp_path, season=None, pool={"英雄甲", "永久标注"}))
    observe(tracker, [unit("0021_0")], shop=["英雄甲"])
    tracker.group_name[1] = "英雄甲"
    tracker.gallery.items.append(("永久标注", unit("0021_0").fp))
    gallery = tracker.gallery
    reader = GameReader(Dictionary({}, []), tracker, ocr=ocr)
    reader.phase, reader.last_stage, reader.last_timer = "combat", "3-2", 1
    reader.roster, reader.last_players = ["旧局"], [{"name": "旧局", "hp": 1}]
    reader.cache["shop#0#0"] = (np.zeros((2, 2)), [Line("旧局", 1, 0, 0, 1, 1)])
    reader._new_game()
    assert (reader.phase, reader.last_stage, reader.last_timer) == ("unknown", None, None)
    assert not reader.roster and reader.last_players is None and not reader.cache
    assert not tracker.prev and not tracker.unknown and not tracker.group_name and not tracker.last_units
    assert tracker.last_live is None and tracker.prev_shop is None and tracker.last_fp_count == 0
    assert tracker.gallery is gallery and gallery.names == {"永久标注": 1}
    assert not ocr.cache and not ocr.words and ocr.last_rec == 0
    assert ocr._warmed and (ocr.det, ocr.rec) == models


def new_game_image():
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.digits import read_digits, segment
    from jccbrain.vision.eye.image import load_regions

    image = load_image(FIXTURES / "frames/0060.jpg")  # 真实 3-2 帧。
    with np.load(FIXTURES / "digit_cells.npz") as arrays:
        source = arrays["stage_3_1"]
    glyphs = segment(source, image.shape[0])[:3]
    one = glyphs[2]
    patch = np.zeros_like(source)
    # 用真实 3-1 ROI 中的「1」「-」「1」拼 1-1；保留字距及原尺寸，不调用字体渲染。
    for x, glyph in [(glyphs[0].x1 - one.w, one), (glyphs[1].x0, glyphs[1]), (one.x0, one)]:
        patch[glyph.y0:glyph.y1, x:x + glyph.w] = source[glyph.y0:glyph.y1, glyph.x0:glyph.x1]
    assert read_digits(patch, image.shape[0]) == "1-1"
    x0, y0, x1, y1 = load_regions()["stage"].px(image)
    assert patch.shape == image[y0:y1, x0:x1].shape
    image[y0:y1, x0:x1] = patch
    return image


@mac
@pytest.mark.parametrize("backend", ["rapid", "vision"])
def test_composed_3_2_to_1_1_first_frame_matches_fresh_reader(tmp_path, backend):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader

    continued = build_reader(None, tmp_path / "continued", ocr=backend)
    fresh = build_reader(None, tmp_path / "fresh", ocr=backend)
    try:
        assert continued.read(load_image(FIXTURES / "frames/0060.jpg"))["stage"] == "3-2"
        continued.roster, continued.last_players = ["旧局玩家"], [{"name": "旧局玩家", "hp": 1}]
        old = unit("0021_0")
        old.name = "旧局单位"
        continued.units.last_live = ([], [old])
        # 背景不变时区域缓存必然命中；Rapid 的精确/容差行缓存也写入旧局错误结果。
        for key, (thumb, _) in list(continued.cache.items()):
            continued.cache[key] = (thumb, [Line("旧局单位", 1, 0, 0, 1, 1)])
        if backend == "rapid":
            for key in continued.ocr.cache:
                continued.ocr.cache[key] = ("旧局玩家", 1.)
            continued.ocr.words = deque([(mask, ("旧局玩家", 1.)) for mask, _ in continued.ocr.words], maxlen=256)
            clear = Mock(wraps=continued.ocr.clear_cache)
            continued.ocr.clear_cache = clear
        image = new_game_image()
        expected, actual = fresh.read(image), continued.read(image)
        assert actual == expected
        assert actual["stage"] == "1-1" and "旧局" not in str(actual)
        assert "旧局玩家" not in continued.roster
        if backend == "rapid":
            assert clear.call_count == 1
            continued.read(image)
            assert clear.call_count == 1  # 同局的 1-1 不反复清缓存。
    finally:
        for reader in [continued, fresh]:
            if pool := getattr(reader.ocr, "pool", None):
                pool.shutdown()


def test_custom_stage_without_digit_templates_restarts_first_frame_after_reset():
    from jccbrain.vision.eye.image import Region

    class OCR:
        def prepare(self, image):
            return image

        def recognize_batch(self, images, languages, groups):
            return [[Line("1-1", 1, 0, 0, 1, 1)] for _ in images]

    units = SimpleNamespace(prepare=Mock(return_value=({}, [])), reset=Mock(),
                            observe=Mock(return_value={"board": [], "bench": []}))
    reader = GameReader(Dictionary({}, []), units, {"stage": Region(0, 0, 1, 1, numeric=True)}, ocr=OCR())
    reader.last_stage = "3-2"
    state = reader.read(np.zeros((32, 32, 3), np.uint8))
    assert state["stage"] == "1-1"
    units.reset.assert_called_once()
    assert units.prepare.call_count == 2
    units.observe.assert_called_once()
