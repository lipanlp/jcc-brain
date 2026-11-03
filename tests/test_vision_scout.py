import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jccbrain.vision.eye import scout
from jccbrain.vision.eye.reader import GameReader
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.units import Gallery, TrackedUnit, UnitTracker


def test_real_eye_locations_include_missing_ocr_rows_and_own_view():
    root = Path(__file__).parent / 'fixtures'
    with np.load(root/'scout_eye_cells.npz') as cells:
        for key, slot in json.loads((root/'scout_eye_cells.json').read_text()).items():
            image = np.zeros((1136, 1774, 3), np.uint8)
            y0, y1, x0, x1 = round(.09*1136), round(.69*1136), round(.963*1774), round(.983*1774)
            image[y0:y1, x0:x1] = cv2.resize(cells[key], (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
            assert scout.viewing_slot(image) == slot, key
    assert scout.viewing_slot(np.zeros((277, 493, 3), np.uint8)) is None


@pytest.mark.parametrize('name', ['英雄甲', '卡池外英雄'])
def test_scout_never_mutates_own_tracking_learning_or_phase(tmp_path, monkeypatch, name):
    class OCR:
        name = 'fake'
        prepare = staticmethod(lambda image: image)

        def recognize_batch(self, images, *args):
            return [[] for _ in images]

    gallery = Gallery(tmp_path, pool={'英雄甲'})
    gallery.match = Mock(return_value=(name, .1))
    gallery.add = Mock(side_effect=AssertionError('不得学习'))
    tracker = UnitTracker(gallery)
    anchor = TrackedUnit('board', None, .5, .4, 2, name='英雄甲', group=7)
    tracker.prev = [anchor]
    tracker.last_live = ([], [anchor])
    tracker.prev_shop = ['英雄甲']
    tracker.unknown = [(7, np.ones(4))]
    tracker.group_name = {7: '英雄甲'}
    before = pickle.dumps({k: v for k, v in tracker.__dict__.items() if k not in ('gallery', 'items')})
    monkeypatch.setattr(UnitTracker, 'detect', lambda *_: ({}, [TrackedUnit(
        'board', None, .4, .4, 1, crop=np.zeros((20, 20, 3), np.uint8))]))
    monkeypatch.setattr(scout, 'feature_print', lambda _: np.ones(4))
    reader = GameReader(Dictionary({}, []), tracker, ocr=OCR())
    reader.phase, reader.last_stage, reader.last_timer = 'planning', '3-2', 17
    reader.roster = ['自己的名字']
    reader.augments_taken = SimpleNamespace(confirmed={0: '已选'}, pending={1: ('待确认', 1)})
    reader.cache['owned'] = ('原图', [])
    image = np.zeros((1136, 1774, 3), np.uint8)
    out = reader.read_scout(image)
    assert out['board'][0]['name'] == ('英雄甲' if name == '英雄甲' else '?1')
    assert pickle.dumps({k: v for k, v in tracker.__dict__.items() if k not in ('gallery', 'items')}) == before
    assert (reader.phase, reader.last_stage, reader.last_timer) == ('planning', '3-2', 17)
    assert reader.roster == ['自己的名字'] and reader.cache == {'owned': ('原图', [])}
    assert reader._frame_id == 0 and not reader.cell_cache and reader.round_bar.stage is None
    assert reader.augments_taken.confirmed == {0: '已选'} and reader.augments_taken.pending == {1: ('待确认', 1)}
    gallery.add.assert_not_called()


def test_scout_rapid_cache_is_private_but_models_are_shared(tmp_path, monkeypatch):
    from jccbrain.vision.eye.rapid import RapidOCR
    ocr = RapidOCR()
    try:
        ocr.cache[b'owner'] = ('自己', 1.)
        reader = scout.ScoutReader(Dictionary({}, []), Gallery(tmp_path), GameReader(
            Dictionary({}, []), UnitTracker(Gallery(tmp_path))).regions, ocr)
        other = reader.reader.ocr
        assert other.det is ocr.det and other.rec is ocr.rec and other.pool is ocr.pool
        assert other.lock is not ocr.lock and other.cache is not ocr.cache and not other.cache
        assert other.fallback is False and ocr.fallback is True
        from jccbrain.vision.eye import rapid
        monkeypatch.setattr(rapid, '_vision_lines', Mock(side_effect=AssertionError('不得弱行补认')))
        monkeypatch.setattr(rapid, 'read_number_line', lambda _: None)
        other._recognize_batches = Mock(return_value=[('', .1)])
        assert other._recognize([np.zeros((10, 30, 3), np.uint8)]) == [('', .1)]
        other.cache[b'foreign'] = ('别人', 1.)
        reader.reset()
        assert ocr.cache == {b'owner': ('自己', 1.)}
    finally:
        ocr.pool.shutdown(wait=True)


def test_provider_scout_does_not_capture_or_replace_live_state(tmp_path):
    from jccbrain.vision.pyeye import PyEyeProvider
    provider = PyEyeProvider(gallery_dir=tmp_path)
    provider._reader = SimpleNamespace(read_scout=Mock(return_value={'traits': []}))
    provider._capture = SimpleNamespace(grab=Mock(side_effect=AssertionError('不得截图')))
    provider._last, provider._last_t = object(), 12.
    image = np.zeros((1136, 1774, 3), np.uint8)
    provider._last_img = image
    last = provider._last
    assert provider.read_scout(image) == {'traits': []}
    assert provider._last is last and provider._last_t == 12. and provider._last_img is image
    provider._capture.grab.assert_not_called()


def test_scout_unknown_truth_is_not_counted_as_correct():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from eval_scout import score
    counts = score(dict(traits=[], level=None, viewing=None, board=[]),
                   dict(level=None, viewing=None, board_count=None))
    assert 'level' not in counts and 'viewing' not in counts and 'board_count' not in counts
    assert score({}, dict(level=8, viewing={'name': '别人', 'slot': 4}, board_count=2))['board_count'] == (0, 1)
    assert score({'viewing': None}, {'viewing': None, 'viewing_known': True})['viewing'] == (1, 1)
    unit = dict(x=.5, y=.4, name='英雄甲')
    assert score({'board': [unit]}, {'board_names': [unit, unit]})['board_names'] == (1, 2)


def test_player_slots_follow_screen_positions_when_a_row_is_missing(tmp_path):
    from jccbrain.vision.eye.ocr import Line
    reader = GameReader(Dictionary({}, []), UnitTracker(Gallery(tmp_path)))
    lines = []
    for slot, name in [(0, '甲玩家'), (2, '乙玩家')]:
        hp_mid = (scout.PLAYER_Y0 + slot*scout.PLAYER_STEP + .016 - .09) / .6
        lines += [Line(name, 1, 0, hp_mid-.05, .6, .02), Line('99', 1, .5, hp_mid-.01, .2, .02)]
    _, players = reader._parse_players(lines)
    assert [(p['name'], p['slot'], p['y']) for p in players] == [('甲玩家', 0, .126), ('乙玩家', 2, .272)]


def test_models_accept_new_fields_without_changing_previous_semantics():
    from jccbrain.core.models import GameState
    base = dict(stage='3-1', phase='planning', shop={'open': True}, players=[{'name': '甲', 'hp': 90}])
    old = GameState.from_dict(base)
    new = GameState.from_dict(dict(base, players=[dict(base['players'][0], slot=2, y=.272)],
                                   viewing={'name': '甲', 'slot': 2}, opponent={'name': None, 'board': []}))
    assert new.summary() == old.summary() and new.can_act == old.can_act
    assert (new.players[0].name, new.players[0].hp, new.players[0].slot, new.players[0].y) == ('甲', 90, 2, .272)
    assert new.viewing == {'name': '甲', 'slot': 2} and new.opponent == {'name': None, 'board': []}


def test_live_start_after_image_only_scout_reuses_models_and_creates_capture(tmp_path, monkeypatch):
    from jccbrain.vision import pyeye
    from jccbrain.actuator import window
    from jccbrain.vision.eye import capture
    reader = SimpleNamespace(ocr=SimpleNamespace(), read_scout=Mock(return_value={}))
    build = Mock(return_value=reader)
    monkeypatch.setattr(pyeye, 'build_reader', build)
    monkeypatch.setattr(window, 'WindowLocator', lambda _: object())
    handle = object()
    monkeypatch.setattr(capture, 'WindowCapture', lambda _: handle)
    provider = pyeye.PyEyeProvider(gallery_dir=tmp_path)
    provider.read_scout(np.zeros((1136, 1774, 3), np.uint8))
    assert provider._capture is None
    provider.start()
    assert provider._capture is handle and provider._reader is reader
    build.assert_called_once()


@pytest.mark.parametrize('raw,words,want', [('法士', ['斗士', '法师'], None), ('士', ['斗士'], '斗士')])
def test_scout_does_not_call_vision_supplement_and_keeps_unknown_name(tmp_path, monkeypatch, raw, words, want):
    from jccbrain.vision.eye.ocr import Line
    from jccbrain.vision.eye import ocr as ocr_module

    class OCR:
        name = 'fake'
        prepare = staticmethod(lambda image: image)

        def recognize_batch(self, images, *args):
            return [[Line(f'{raw} 1/2', 1., 0, .07, .5, .03)] for _ in images]

    monkeypatch.setattr(ocr_module, 'recognize_batch', Mock(side_effect=AssertionError('禁止Vision补认')))
    monkeypatch.setattr(ocr_module.VisionOCR, 'recognize_batch', Mock(side_effect=AssertionError('禁止Vision补认')))
    monkeypatch.setattr(UnitTracker, 'detect', lambda *_: ({}, []))
    reader = GameReader(Dictionary({}, words), UnitTracker(Gallery(tmp_path)), ocr=OCR())
    image = np.zeros((1136, 1774, 3), np.uint8)
    out = reader.read_scout(image)
    assert out['traits'] and out['traits'][0]['name'] == want
    reader._scout_reader.reset()
    assert reader.read_scout(image) == out
    assert reader._scout_reader.last_ms['vision'] == 0.


def test_unsupported_layout_is_excluded_from_all_accuracy_counts():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from eval_scout import score
    assert score({}, {'_supported': False, 'traits': [{'name': '斗士'}], 'level': 8,
                      'viewing': {'name': '某人', 'slot': 1}, 'board_count': 5}) == {}
