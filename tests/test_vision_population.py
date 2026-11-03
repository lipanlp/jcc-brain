from pathlib import Path

import numpy as np
import pytest

from jccbrain.vision.eye.population import read_population


@pytest.mark.parametrize('frame,want', [('0054.jpg', [6, 6]), ('0057.jpg', [6, 6]),
    ('track/0055.jpg', [6, 6]), ('0021.jpg', None), ('0051.jpg', None), ('0003.jpg', None),
    ('0045.jpg', None), ('0060.jpg', None)])
def test_real_watermark_fallback_and_non_watermark_rejection(frame, want):
    with np.load(Path(__file__).parent / 'fixtures/population_cells.npz') as z:
        assert read_population(z[frame], 1136) == want


def test_missing_population_crop_is_unknown():
    assert read_population(None, 1136) is None
    assert read_population(np.zeros((10, 10, 3), np.uint8), 1136) is None
