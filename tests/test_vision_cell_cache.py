from types import SimpleNamespace

import numpy as np

from jccbrain.vision.eye.image import Region
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary


class OCR:
    def __init__(self):
        self.seen = []

    def prepare(self, image):
        return image

    def recognize_batch(self, images, languages, groups):
        if images:
            self.seen.append([int(image.max()) for image in images])
        return [[Line(str(int(image.max())), 1., 0., 0., 1., 1.)] for image in images]


def reader(regions):
    units = SimpleNamespace(prepare=lambda *a, **k: ({}, []),
                            observe=lambda *a, **k: {'board': [], 'bench': []}, reset=lambda: None)
    return GameReader(Dictionary({}, []), units, regions, OCR())


def test_exact_repeated_cell_reuses_result_without_submitting_ocr():
    game = reader({'board_count': Region(0, 0, 1, 1, numeric=True)})
    image = np.zeros((20, 20, 3), np.uint8)
    first = game.read(image, debug=True)
    assert game.last_ocr_submitted == 1
    assert game.read(image, debug=True) == first
    assert game.ocr.seen == [[0]]
    assert (game.last_cell_cache_hits, game.last_cell_cache_checks, game.last_ocr_submitted) == (1, 1, 0)


def test_one_pixel_change_and_reused_input_buffer_miss_cache():
    game = reader({'board_count': Region(0, 0, 1, 1, numeric=True)})
    image = np.zeros((20, 20, 3), np.uint8)
    game.read(image)
    image[10, 10] = 9
    assert game.read(image, debug=True)['_raw']['board_count'] == ['9']
    assert game.ocr.seen == [[0], [9]]
    assert game.last_cell_cache_hits == 0


def test_same_pixels_in_different_positions_do_not_share_results():
    game = reader({'board_count': Region(0, 0, 1, 1, cols=2, numeric=True)})
    image = np.zeros((20, 20, 3), np.uint8)
    image[:, :10], image[:, 10:] = 1, 2
    game.read(image)
    image[:, :10], image[:, 10:] = 2, 1
    game.read(image)
    assert game.ocr.seen == [[1, 2], [2, 1]]
    assert game.last_cell_cache_hits == 0


def test_new_game_invalidates_cache_and_unobserved_frame_expires_entry():
    game = reader({'board_count': Region(0, 0, 1, 1, numeric=True)})
    image = np.zeros((20, 20, 3), np.uint8)
    game.read(image)
    game._new_game()
    assert game.cell_cache == {}
    game.read(image)
    assert game.ocr.seen == [[0], [0]]
    game.regions = {}
    game.read(image)
    game.regions = {'board_count': Region(0, 0, 1, 1, numeric=True)}
    game.read(image)
    assert game.ocr.seen == [[0], [0], [0]]


def test_gold_and_timer_templates_are_read_every_frame_even_if_pixels_do_not_change(monkeypatch):
    calls = []
    monkeypatch.setattr('jccbrain.vision.eye.reader.read_digits',
                        lambda *args: calls.append(args) or str(7 + len(calls) // 3))
    game = reader({'gold': Region(0, 0, 1, 1, digits='bright'),
                   'timer': Region(0, 0, 1, 1, digits='bright')})
    image = np.zeros((20, 20, 3), np.uint8)
    game.read(image)
    game.read(image)
    assert len(calls) == 4
    assert game.last_cell_cache_checks == game.last_cell_cache_hits == 0
    assert game.ocr.seen == []
