"""真实商店双星/未锁图标与合成闭锁回归；裁剪像素来自 CoreGraphics。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from jccbrain.core.models import GameState
from jccbrain.vision.eye.image import crop, load_regions
from jccbrain.vision.eye.shop import shop_locked, upgrade_region, upgrade_stars

FIXTURES = Path(__file__).parent / "fixtures"
FRAMES = json.loads((FIXTURES / "shop_cells.json").read_text("utf-8"))["frames"]
with np.load(FIXTURES / "shop_cells.npz") as _cells:
    CELLS = {name: _cells[name] for name in _cells.files}


@pytest.mark.parametrize("name", FRAMES)
def test_real_shop_icons(name):
    frame = FRAMES[name]
    assert [upgrade_stars(CELLS[key]) for key in frame["markers"]] == frame["upgrade"]
    assert shop_locked(CELLS[frame["lock"]]) is frame["locked"]


@pytest.mark.parametrize("scale", [.5, .75, 1.5, 2])
@pytest.mark.skipif(sys.platform != "darwin", reason="需要 CoreGraphics 解码真实截图")
def test_upgrade_region_tracks_real_frame_scaling(scale):
    from jccbrain.vision.eye.capture import load_image

    frame = load_image(FIXTURES / "frames/0069.jpg")
    scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shop = load_regions()["shop"]
    assert [upgrade_stars(crop(scaled, upgrade_region(shop.cell(i, 0)))) for i in range(5)] == [2, 2, None, None, None]


def test_single_star_does_not_claim_upgrade():
    marker = CELLS[FRAMES["0051.jpg"]["markers"][0]].copy()
    marker[:, 20:] = marker[25, 0]
    assert upgrade_stars(marker) is None


def test_three_stars_are_not_truncated_to_two():
    # 用真实星形合成第三颗；不代表已有真实三星商店样本。
    marker = CELLS[FRAMES["0051.jpg"]["markers"][0]].copy()
    marker[:, 36:52] = marker[:, 4:20]
    assert upgrade_stars(marker) == 3


def test_shop_upgrade_survives_state_parsing():
    state = GameState.from_dict({"shop": {"open": True, "slots": [
        {"name": "苍蓝哨戒", "cost": 1, "upgrade": 2}, {"name": "伊莉丝"}, None]}})
    assert state.shop[0].upgrade == 2
    assert state.shop[0].name == "苍蓝哨戒" and state.shop[0].cost == 1
    assert state.shop[1].upgrade is None
    assert state.shop[2] is None


def _padlock(locked):
    image = np.full((52, 90, 3), (32, 40, 44), np.uint8)
    gold = (70, 155, 190)
    if locked:
        cv2.ellipse(image, (44, 21), (8, 8), 0, 180, 360, gold, 3)
        cv2.line(image, (36, 21), (36, 26), gold, 3)
        cv2.line(image, (52, 21), (52, 26), gold, 3)
    else:
        cv2.ellipse(image, (44, 19), (8, 8), 0, 180, 320, gold, 3)
        cv2.line(image, (36, 19), (36, 26), gold, 3)
    cv2.rectangle(image, (35, 25), (54, 39), gold, -1)
    cv2.circle(image, (44, 32), 3, (32, 40, 44), -1)
    return image


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("scale", [.5, 1, 2])
def test_padlock_geometry_on_synthetic_icons(locked, scale):
    image = cv2.resize(_padlock(locked), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    assert shop_locked(image) is locked


def test_missing_or_ambiguous_icons_are_unknown():
    for image in (None, np.empty((0, 0, 3), np.uint8), np.zeros((52, 90, 3), np.uint8)):
        assert shop_locked(image) is None
        assert upgrade_stars(image) is None
    badge = np.zeros((52, 90, 3), np.uint8)
    cv2.rectangle(badge, (35, 14), (54, 39), (70, 155, 190), -1)
    assert shop_locked(badge) is None    # 没有钥匙孔，不能把普通金色图案当未锁
