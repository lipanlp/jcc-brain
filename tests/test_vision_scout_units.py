import json
from pathlib import Path

import numpy as np
import pytest

from jccbrain.vision.eye.scout import correct_trait
from jccbrain.vision.eye.scout_units import detect_scout_units, short_bars, SCOUT_TOP
from jccbrain.vision.eye.units import UnitTracker, Gallery


@pytest.mark.parametrize('raw,words,want', [('斗土',['斗士'],'斗士'), ('法士',['法师','斗士'],None),
    ('斗士',['斗士','法师'],'斗士'), ('士',['斗士','法师'],'斗士'), ('斗土',['法师'],None), ('XYZ',['斗士'],None), (None,['斗士'],None)])
def test_trait_correction_is_one_edit_unique_and_season_scoped(raw, words, want):
    assert correct_trait(raw, words) == want


def test_real_short_bars_restore_full_width_and_upper_bench_is_excluded(tmp_path):
    root = Path(__file__).parent/'fixtures'
    assert (root/'scout_unit_cells.npz').stat().st_size < 300_000
    tracker = UnitTracker(Gallery(tmp_path))
    with np.load(root/'scout_unit_cells.npz') as data:
        for row in json.loads((root/'scout_unit_cells.json').read_text()):
            image = np.zeros((1136, 1774, 3), np.uint8)
            x, y = row['origin']; patch = data[row['key']]
            image[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
            _, old = tracker.detect(image)
            found = detect_scout_units(image, tracker)
            if row['key'].startswith('upper'):
                removed = [u for u in old if u.y < SCOUT_TOP]
                assert len(removed) == 1
                assert all(u.y >= SCOUT_TOP for u in found)
            else:
                bars = short_bars(image, old)
                assert len(bars) == 1 and bars[0].width == row['width']
                new = [u for u in found if u.bar == bars[0]]
                assert len(new) == 1 and new[0].crop.shape[1] >= 72
                assert short_bars(image, found) == []  # 不重复计算同一条


def test_green_rectangle_without_dark_border_is_not_a_unit():
    image = np.full((1136, 1774, 3), 255, np.uint8)
    image[500:507, 800:814] = [40, 180, 80]
    assert short_bars(image, []) == []
