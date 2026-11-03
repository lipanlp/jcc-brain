from pathlib import Path

import numpy as np
import pytest

from jccbrain.vision.eye.digits import read_dim_white_cells, read_digits


@pytest.mark.parametrize('frame', ['0042.jpg', '0045.jpg', '0048.jpg', '0051.jpg'])
def test_real_augment_dim_counts_and_inactive_rows(frame):
    with np.load(Path(__file__).parent / 'fixtures/dim_trait_cells.npz') as z:
        cells = [z[f'{frame}/{i}'] for i in range(8)]
    original = [read_digits(c, 1136, 'white') for c in cells]
    recovered = read_dim_white_cells(cells, 1136)
    if frame in ['0045.jpg', '0048.jpg']:
        assert original == [''] * 8
        assert recovered == ['3', '2'] + [''] * 6
    else:
        assert original == ['3', '2'] + [''] * 6
        assert recovered is None


def test_empty_dim_cells_do_not_create_counts():
    assert read_dim_white_cells([None, np.zeros((20, 20, 3), np.uint8)], 1136) is None
