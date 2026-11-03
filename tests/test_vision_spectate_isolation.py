"""真实解析流水线中的归属分流；插入外来帧不能更改己方状态。"""
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from jccbrain.vision.eye import reader as module, units, scout_units
from jccbrain.vision.eye.image import Region
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.units import Gallery, UnitTracker, TrackedUnit


class OCR:
    name = 'fake'
    prepare = staticmethod(lambda im: im)

    def __init__(self):
        self.cache, self.words = {}, []

    def recognize_batch(self, images, *args):
        self.cache['foreign' if images and images[0][0, 0, 0] == 2 else 'own'] = True
        return [[Line('3-1', 1, 0, 0, 1, 1)] for _ in images]

    def clear_cache(self):
        self.cache.clear()
        self.words.clear()


class Augments:
    def __init__(self):
        self.confirmed, self.pending = {0: '本人已选'}, {1: ('待确认', 1)}

    def remembered(self):
        return list(self.confirmed.values())

    def reset(self):
        self.confirmed.clear()
        self.pending.clear()

    def read(self, *args, **kwargs):
        return self.remembered()


@pytest.fixture
def scene(monkeypatch):
    from jccbrain.vision.eye import augments, phase
    monkeypatch.setattr(augments, 'spectate_score', lambda im: float(im[0, 0, 0] == 2))
    monkeypatch.setattr(phase, 'dialog_score', lambda im: -1.)
    monkeypatch.setattr(module, 'detect_scene', lambda im, *_: 'spectate' if im[0, 0, 0] == 2 else None)
    monkeypatch.setattr(phase.StagebarReader, 'read_rounds', lambda *_: ('unknown', 'pvp'))
    monkeypatch.setattr(units, 'feature_print', lambda im: np.array([1., 0.], np.float32))

    def detect(self, im):
        foreign = im[0, 0, 0] == 2
        return {'legend': {'x': .8}}, [TrackedUnit('board', None, .7 if foreign else .4, .4, 1,
                                      crop=im[:10, :10], hist=np.array([1., 0.], np.float32))]
    monkeypatch.setattr(UnitTracker, 'detect', detect)
    monkeypatch.setattr(scout_units, 'short_bars', lambda *_: [])
    return np.ones((1136, 1774, 3), np.uint8), np.full((1136, 1774, 3), 2, np.uint8)


def make(tmp_path, name):
    gallery = Gallery(tmp_path/name, pool={'甲英雄'})
    gallery.match = Mock(return_value=None)
    gallery.add = Mock(side_effect=AssertionError('不可学习'))
    return GameReader(Dictionary({}, []), UnitTracker(gallery),
                      {'stage': Region(0, 0, .01, .01, digits='stage'),
                       'banner': Region(0, 0, .01, .01)}, OCR(), Augments())


def state(r):
    return pickle.dumps((r.units.prev, r.units.last_live, r.units.unknown, r.units.group_name,
                         r.units.prev_shop, r.units.last_units, r.phase, r.last_stage, r.last_timer,
                         r.roster, r.last_players, r.cache, r.cell_cache, r._frame_id,
                         r.round_bar.__dict__, r.augment_desc_cache, r.augments_taken.__dict__,
                         r.ocr.cache, r.ocr.words))


@pytest.mark.parametrize('foreign_stage', ['4-3', '1-1'])
def test_own_foreign_own_keeps_full_output_and_state(tmp_path, monkeypatch, scene, foreign_stage):
    monkeypatch.setattr(module, 'read_digits', lambda im, *_: foreign_stage if im[0, 0, 0] == 2 else '3-1')
    own, foreign = scene
    control, r = make(tmp_path, 'control'), make(tmp_path, 'subject')
    assert control.read(own) == r.read(own)
    scout, opponent = Mock(), Mock()
    r._scout_reader, r._opponent_reader = scout, opponent
    frozen = state(r)
    for _ in range(3):
        out = r.read(foreign)
        assert out['phase'] == 'spectate' and out['units_stale'] is False
        assert out['board'] == [{'name': '?1', 'x': .7, 'y': .4, 'star': 1}]
        assert out['legend'] == {'x': .8}
        assert out['augments_taken'] == ['本人已选']
        assert state(r) == frozen
        assert not hasattr(r._spectate_reader, '_scout_reader')
        assert not hasattr(r._spectate_reader, '_opponent_reader')
    scout.reset.assert_not_called()
    opponent.reset.assert_not_called()
    assert r.read(own) == control.read(own)
    assert not hasattr(r, '_spectate_reader')
    r.units.gallery.add.assert_not_called()


def test_early_spectate_does_not_prepare_or_observe_main_units(tmp_path, monkeypatch, scene):
    r = make(tmp_path, 'subject')
    monkeypatch.setattr(module, 'read_digits', lambda *_: '1-1')
    r.units.prepare = Mock(side_effect=AssertionError('本人prepare不可执行'))
    r.units.observe = Mock(side_effect=AssertionError('本人observe不可执行'))
    # A class-level guard distinguishes the private context from the owner.
    original = GameReader._check_new_game
    def guard(self, cells):
        assert self is not r
        return original(self, cells)
    monkeypatch.setattr(GameReader, '_check_new_game', guard)
    out = r.read(scene[1])
    assert out['units_stale'] is False
    r.units.prepare.assert_not_called()
    r.units.observe.assert_not_called()


@pytest.mark.parametrize('priority', ['dialog', 'carousel'])
def test_dialog_and_carousel_keep_priority_and_do_not_create_foreign_context(tmp_path, monkeypatch, scene, priority):
    from jccbrain.vision.eye import phase
    monkeypatch.setattr(module, 'read_digits', lambda *_: '4-4')
    if priority == 'dialog':
        monkeypatch.setattr(phase, 'dialog_score', lambda *_: 1.)
    else:
        monkeypatch.setattr(phase.StagebarReader, 'read_rounds', lambda *_: ('pvp', 'carousel'))
    r = make(tmp_path, 'subject')
    r._read = Mock(return_value={'phase': priority})
    assert r.read(scene[1]) == {'phase': priority}
    r._read.assert_called_once()
    assert not hasattr(r, '_spectate_reader')


def test_exception_discards_foreign_context_and_preserves_own_state(tmp_path, monkeypatch, scene):
    monkeypatch.setattr(module, 'read_digits', lambda im, *_: '4-3' if im[0, 0, 0] == 2 else '3-1')
    r = make(tmp_path, 'subject')
    r.read(scene[0])
    frozen = state(r)
    monkeypatch.setattr(UnitTracker, 'read_current', Mock(side_effect=RuntimeError('foreign failed')))
    with pytest.raises(RuntimeError, match='foreign failed'):
        r.read(scene[1])
    assert not hasattr(r, '_spectate_reader')
    assert state(r) == frozen


def test_current_units_reuse_scout_filter_recovery_and_keep_original_legend(tmp_path, monkeypatch):
    tracker = UnitTracker(Gallery(tmp_path/'gallery', pool={'甲英雄'}))
    image = np.zeros((1013, 1576, 3), np.uint8)
    tracker.detect = Mock(return_value=({'legend': {'x': .8}}, []))
    recovered = TrackedUnit('board', None, .6, .5, 2, crop=image[:10, :10])
    detect = Mock(return_value=[recovered])
    monkeypatch.setattr(scout_units, 'detect_scout_units', detect)
    monkeypatch.setattr(units, 'feature_print', lambda *_: np.array([1., 0.], np.float32))
    tracker.gallery.match = Mock(return_value=('甲英雄', .1))
    tracker.gallery.add = Mock(side_effect=AssertionError('不可学习'))
    result = tracker.read_current(image)
    assert detect.call_count == 1 and detect.call_args.args[0].shape[0] == 1136
    assert result == dict(legend={'x': .8}, board=[dict(name='甲英雄', x=.6, y=.5, star=2)],
                          bench=[], units_stale=False)
    assert not tracker.prev and not tracker.unknown and tracker.last_live is None
    tracker.gallery.add.assert_not_called()
