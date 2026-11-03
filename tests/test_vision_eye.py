"""纯 Python 视觉模块：文字后处理、颜色判断、血条检测、图库，以及用真实截图做的端到端回归。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from jccbrain.vision.eye.digits import read_digits, read_number_line
from jccbrain.vision.eye.image import cost_from_color, thumb, changed
from jccbrain.vision.eye.reader import HP_LINE
from jccbrain.vision.eye.text import Dictionary, find_words, ints
from jccbrain.vision.eye.units import Gallery, detect_bars, star_of

FIXTURES = Path(__file__).parent / "fixtures"
FRAMES = FIXTURES / "frames"
mac = pytest.mark.skipif(sys.platform != "darwin", reason="需要 macOS（Quartz / Apple Vision）")


# ------------------------------------------------------------------ 文字
def test_ints_and_circled_digits():
    assert ints("经验 15/20") == [15, 20]
    assert ints("①②/③") == [12, 3]


def test_find_words_exact_and_fuzzy():
    traits = ["魔战士", "狂战士", "永恒之森", "约德尔人和朋友"]
    assert find_words("魔战士 1/2", traits) == ["魔战士"]          # 3 字不容错，不能认成狂战士
    assert find_words("永恒之林 重装", traits) == ["永恒之森"]       # 4 字容 1 个错字
    assert find_words("约德尔人和朋友狂战士", traits) == ["约德尔人和朋友", "狂战士"]


def test_match_champion():
    d = Dictionary({"赫卡里姆": 3, "蔚": 3, "奥恩": 1}, [])
    assert d.match_champion("赫卡里姆 ◎3") == ("赫卡里姆", 3, True)
    assert d.match_champion("赫卡里娒") == ("赫卡里姆", 3, True)
    assert d.match_champion("铜汤匙") == ("铜汤匙", None, False)
    assert d.match_champion("123") is None


def test_hp_line_pattern():
    for t in ("100", "v100", "X 69", "%100", "••• 67", "5100"):
        assert HP_LINE.fullmatch(t)
    for t in ("海盐阿姨", "G-Daydreaming", "2挖掘机的文科生"):
        assert not HP_LINE.fullmatch(t)


def test_dictionary_from_db_uses_display_trait_names():
    class Hero:
        def __init__(self, name, cost, traits):
            self.name, self.cost, self.traits = name, cost, traits

    class Aug:
        def __init__(self, name):
            self.name = name

    class DB:
        traits = {"约德尔人": None, "斗士": None}
        augments = {"1": Aug("治疗法球 I"), "2": Aug("白银命运+")}

        def pool_heroes(self):
            return [Hero("可酷伯", 1, ("约德尔人", "斗士"))]

    d = Dictionary.from_db(DB())
    assert d.champions == {"可酷伯": 1}
    assert "约德尔人和朋友" in d.traits and "约德尔人" not in d.traits       # 游戏里显示的是全称
    assert d.champion_traits["可酷伯"] == ["约德尔人和朋友", "斗士"]
    assert d.match_augment("治疗法球1") == "治疗法球 I"                     # 罗马数字 I 被认成 1
    assert d.match_augment("白银命运十") == "白银命运+"                     # 容 1 个错字
    assert d.match_augment("别再错过") == "别再错过"                         # 词典里没有就原样


def test_bundled_dictionary_is_complete():
    """dict.json 由数据库生成：本赛季商店英雄齐全，每个英雄都有羁绊（离线回放、测试用它）。"""
    d = Dictionary.from_file()
    assert len(d.champions) >= 60 and len(d.augments) >= 200
    assert {"苍蓝哨戒", "伊莉丝", "慎", "蔚"} <= set(d.champions)
    assert all(d.champion_traits.get(n) for n in d.champions)
    assert {"神谕", "猎人", "野兽之灵", "约德尔人和朋友"} <= set(d.traits)


# ------------------------------------------------------------------ 数字模板
def _digit_cells():
    cells = np.load(FIXTURES / "digit_cells.npz")
    return [(k, cells[k], *v) for k, v in json.loads((FIXTURES / "digit_cells.json").read_text("utf-8")).items()]


@pytest.mark.parametrize("name,cell,mode,want", _digit_cells(), ids=lambda v: v if isinstance(v, str) else "")
def test_digit_templates_on_real_cells(name, cell, mode, want):
    """真实截图切下的数字格（CoreGraphics 取像素后存的 npz）：单个 1、0、黄色倒计时、彩色概率、
    金底白字、羁绊人数、血量行。"""
    if mode == "line":
        assert read_number_line(cell) == want, name
    else:
        assert read_digits(cell, 1136, mode) == want, name


def test_read_digits_rejects_stray_dash():
    from jccbrain.vision.eye import digits as D

    img = np.zeros((40, 80, 3), np.uint8)
    img[18:21, 60:70] = 255                  # 单独一条短横（区域边上的装饰），不能读成「-」
    assert D.read_digits(img, 1136) == ""


# ------------------------------------------------------------------ 图像
def test_cost_from_color():
    assert cost_from_color((0.5, 0.5, 0.52)) == 1         # 灰
    assert cost_from_color((0.2, 0.6, 0.3)) == 2          # 绿
    assert cost_from_color((0.2, 0.4, 0.8)) == 3          # 蓝
    assert cost_from_color((0.6, 0.2, 0.8)) == 4          # 紫
    assert cost_from_color((0.8, 0.65, 0.2)) == 5         # 金
    assert cost_from_color((0.02, 0.02, 0.02)) is None


def test_thumb_change_detection():
    a = np.zeros((30, 90, 3), np.uint8)
    b = a.copy()
    assert not changed(thumb(a), thumb(b))
    b[10:20, 10:40] = 255
    assert changed(thumb(a), thumb(b))


def _synthetic_bar(img, x0, y0, w=68, t=6, star_color=None):
    img[y0 - 3:y0 + t + 3, x0 - 2:x0 + w + 2] = (20, 20, 20)        # 深色边框
    img[y0:y0 + t, x0:x0 + w] = (60, 200, 90)                       # 绿色血条（BGR）
    if star_color is not None:
        img[y0 - 2:y0 + t + 2, x0 - 16:x0 - 4] = star_color


def test_detect_bars_and_star_on_synthetic_frame():
    img = np.full((1136, 1774, 3), 60, np.uint8)
    _synthetic_bar(img, 800, 400, star_color=(40, 170, 220))          # 金铜色 → 1 星
    _synthetic_bar(img, 1000, 450, star_color=(235, 235, 235))        # 银白 → 2 星
    bars = sorted(detect_bars(img), key=lambda b: b.cx)
    assert len(bars) == 2
    assert bars[0].thick == 6 and abs(bars[0].cx - (800 + 33.5) / 1774) < 1e-3
    assert star_of(img, bars[0]) == 1 and star_of(img, bars[1]) == 2


def test_gallery_roundtrip(tmp_path):
    g = Gallery(tmp_path, season=None, pool={"奥恩"})
    v = np.random.default_rng(0).normal(size=768).astype(np.float32)
    v /= np.linalg.norm(v)
    g.add("奥恩", v, None)
    assert g.match(v)[0] == "奥恩"
    far = -v
    assert g.match(far) is None
    assert Gallery(tmp_path, season=None, pool={"奥恩"}).names == {"奥恩": 1}                    # 重新加载


# ------------------------------------------------------------------ 端到端（真实截图）
EXPECTED = {
    "0003.jpg": dict(stage="2-6", timer=13, phase="planning", gold=4, level=5, xp=[8, 20], hp=92,
                     streak={"kind": "win", "n": 1}, odds=[45, 33, 20, 2, 0],
                     shop=["约里克", "维迦", "瑟庄妮", None, None], board=3,
                     trait_counts=[(3, True), (3, True), (1, False), (1, False), (1, False), (1, False),
                                   (1, False), (1, False)]),
    "0021.jpg": dict(stage="3-1", timer=27, gold=24, streak={"kind": "win", "n": 2}, board_count=[5, 5],
                     shop=["绯红树怪", "韦鲁斯", "赫卡里姆", "蔚", "铜汤匙"], item=[4],
                     shop_traits={0: ["峡谷野怪", "猎人"], 3: ["野兽之灵", "主宰"]},
                     players=8),
    "0033.jpg": dict(stage="3-1", phase="planning", gold=26, level=5, xp=[15, 20], hp=92,
                     shop=[None, "韦鲁斯", "赫卡里姆", "蔚", "奥恩"], board=4, stars=[1, 1, 3, 3]),
    "0060.jpg": dict(stage="3-2", timer=5, phase="combat", gold=35, level=6, xp=[0, 36], hp=81,
                     streak={"kind": "loss", "n": 1},
                     shop=["苍蓝哨戒", "苍蓝哨戒", "伊莉丝", "慎", None],
                     shop_traits={0: ["峡谷野怪", "神谕"]}),
    "0069.jpg": dict(players=8, shop_upgrades=[2, 2, None, None, None],
                     trait_counts=[(4, True), (2, True), (2, False), (1, False), (1, False),
                                              (1, False), (1, False), (1, False)]),
    "result_2nd.jpg": dict(phase="ended", placement=2),
}


@mac
@pytest.mark.parametrize("ocr", ["rapid", "vision"])
def test_frames_end_to_end(tmp_path, ocr):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader

    reader = build_reader(None, tmp_path / "gallery", ocr)      # 用内置词典，不依赖同步过的数据库
    for f in sorted(FRAMES.glob("*.jpg")):
        d = reader.read(load_image(f))
        exp = EXPECTED.get(f.name)
        if not exp:
            continue
        for k, v in exp.items():
            if k == "shop":
                got = [(s or {}).get("name") for s in d["shop"]["slots"]]
                assert got == v, (f.name, got)
            elif k == "board":
                assert len(d["board"]) == v, f.name
            elif k == "stars":
                assert sorted(u["star"] for u in d["board"]) == v, f.name
            elif k == "item":
                assert [i for i, s in enumerate(d["shop"]["slots"]) if s and s.get("item")] == v, f.name
            elif k == "shop_traits":
                for i, tr in v.items():
                    assert d["shop"]["slots"][i]["traits"] == tr, (f.name, i)
            elif k == "shop_upgrades":
                assert [(s or {}).get("upgrade") for s in d["shop"]["slots"]] == v, f.name
            elif k == "players":
                assert len(d["players"]) == v, (f.name, d["players"])
            elif k == "trait_counts":
                assert [(t.get("count"), t["active"]) for t in d["traits"]] == v, (f.name, d["traits"])
            else:
                assert d.get(k) == v, (f.name, k, d.get(k))


# ------------------------------------------------------------------ RapidOCR 后端
@mac
def test_rapid_line_cache_survives_background_animation():
    """玩家列表背后龙纹在动：名字行按白色字形容差命中缓存，只有血量这类数字行可能重认，结果不变。"""
    pytest.importorskip("rapidocr_onnxruntime")
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.image import crop, load_regions
    from jccbrain.vision.eye.rapid import RapidOCR

    cell = crop(load_image(FRAMES / "0069.jpg"), load_regions()["players"])
    ocr = RapidOCR(fallback=False)
    first = [l.text for l in ocr.recognize_batch([cell])[0]]
    n_first = ocr.last_rec
    assert "隐意的晨星°" in first and "69" in first

    rng = np.random.default_rng(0)
    moved = cell.astype(np.int16)
    bg = cell.min(axis=2) <= 140                                 # 只动背景（非白色）像素
    moved[bg] += rng.integers(-25, 26, size=(int(bg.sum()), 1))
    moved = np.clip(moved, 0, 255).astype(np.uint8)
    second = [l.text for l in ocr.recognize_batch([moved])[0]]
    assert ocr.last_rec < n_first / 2
    names = lambda ts: [t for t in ts if len(t) >= 2 and not any(c.isdigit() for c in t)]   # noqa: E731
    assert names(second) == names(first)
