from collections import Counter
from types import SimpleNamespace

import numpy as np

from jccbrain.vision.eye.image import Region
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary


class OCR:
    def __init__(self):
        self.submitted = Counter()

    def prepare(self, image):
        return image

    def recognize_batch(self, images, languages, groups):
        result = []
        for image, group in zip(images, groups):
            self.submitted[group] += 1
            value = int(np.median(image))
            text = ('强化符文' if value == 50 else '') if group == 'banner' else (
                f'卡{value}' if group == 'augment_names' else f'说明{value % 10}')
            result.append([Line(text, 1., 0., 0., 1., 1.)] if text else [])
        return result


def setup(monkeypatch):
    stage = ['3-2']
    monkeypatch.setattr('jccbrain.vision.eye.reader.read_digits', lambda *a: stage[0])
    regions = dict(stage=Region(0, .8, 1, .1, digits='bright'),
                   banner=Region(0, 0, 1, .1),
                   augment_names=Region(0, .2, 1, .1, cols=3, when='augment'),
                   augment_desc=Region(0, .4, 1, .15, cols=3, when='augment'))
    units = SimpleNamespace(prepare=lambda *a, **k: ({}, []),
                            observe=lambda *a, **k: {'board': [], 'bench': []}, reset=lambda: None)
    game = GameReader(Dictionary({}, []), units, regions, OCR())
    image = np.zeros((100, 100, 3), np.uint8)
    image[:10] = 50
    for i, (x0, x1) in enumerate([(0, 33), (33, 66), (66, 100)]):
        image[20:30, x0:x1] = 10 * (i+1)
        image[40:55, x0:x1] = 40 + 10*i
    return game, image, stage


def test_same_card_names_reuse_descriptions_despite_background_pixel_changes(monkeypatch):
    game, image, _ = setup(monkeypatch)
    first = game.read(image)
    assert game.ocr.submitted['augment_desc'] == 3
    image[40:55] += 100  # 仅背景变化，假OCR仍会读相同的说明文字。
    second = game.read(image)
    assert second == first
    assert game.ocr.submitted['augment_desc'] == 3
    assert game.last_augment_desc_cache_hits == 3


def test_changed_card_name_forces_fresh_descriptions(monkeypatch):
    game, image, _ = setup(monkeypatch)
    game.read(image)
    image[20:30, :33] = 199  # 同时改变既有>150文字缩略图，模拟实际刷新后的新字形。
    image[40:55] += 100
    image[40:55, :33] = 249
    result = game.read(image)
    assert game.last_augment_desc_cache_hits == 2
    assert game.ocr.submitted['augment_desc'] == 4
    assert [a['desc'] for a in result['augments']] == ['说明9', '说明0', '说明0']
    for i, (x0, x1) in enumerate([(0, 33), (33, 66), (66, 100)]):
        image[20:30, x0:x1] = 201 + i
        image[40:55, x0:x1] = 151 + i
    result = game.read(image)
    assert game.last_augment_desc_cache_hits == 0
    assert [a['desc'] for a in result['augments']] == ['说明1', '说明2', '说明3']


def test_leaving_augment_changing_stage_and_new_game_invalidate_descriptions(monkeypatch):
    game, image, stage = setup(monkeypatch)
    game.read(image)
    image[:10] = 200
    game.read(image)
    assert game.augment_desc_cache is None
    image[:10] = 50
    image[40:55] += 100
    game.read(image)
    assert game.last_augment_desc_cache_hits == 0
    stage[0] = '4-2'
    image[40:55] -= 100
    game.read(image)
    assert game.last_augment_desc_cache_hits == 0
    game._new_game()
    assert game.augment_desc_cache is None
