import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from jccbrain.core.models import GameState
from jccbrain.vision.eye.augments import AugmentCatalog, AugmentReader, CENTERS, spectate_score
from jccbrain.vision.eye.items import icon_vector
from scripts.eval_augments import evaluate

ROOT = Path(__file__).parent / 'fixtures'
DATA = json.loads((ROOT / 'augment_cells.json').read_text())


@pytest.fixture(scope='module')
def pixels():
    with np.load(ROOT / 'augment_cells.npz', allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


@pytest.fixture
def reader(tmp_path, pixels):
    catalog = AugmentCatalog('18_S19', [], tmp_path)
    catalog.loaded = DATA['entries']
    catalog.matrix = np.stack([icon_vector(pixels['official'][row['index']], 0) for row in catalog.loaded])
    return AugmentReader(catalog)


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda c: c['file'])
def test_real_icons_reject_wrong_names_and_foreign_view(reader, pixels, case):
    image = np.zeros((1136, 1774, 3), np.uint8)
    x0, y0, x1, y1 = case['rect']
    image[y0:y1, x0:x1] = pixels[case['array']]
    if case['spectate']:
        image[102:784, 1708:1743] = pixels['spectate_strip']
        assert spectate_score(image) >= .60
    got = reader.read(image, own_view=True)
    assert all(name in case['expected'] for name in got)
    got = reader.read(image, own_view=True)  # 普通置信度须连续两帧确认。
    if case['file'].startswith('augment-taken/') or case['spectate']:
        assert got == case['expected']
    assert reader.read(image, own_view=False) == got  # 不明/观战只沿用本人已确认值。


def test_shared_icon_different_names_is_ambiguous(reader, monkeypatch):
    # 名称不同而同图（白银命运/+/++）不能靠id先后次序定名。
    monkeypatch.setattr(reader, 'candidates', lambda _: [(530/1774, '白银命运+', .99, .99)])
    assert reader.read(np.zeros((1136, 1774, 3), np.uint8), own_view=True) == []


def test_partial_cache_cannot_resolve_identical_url_names(tmp_path, monkeypatch):
    catalog = AugmentCatalog('18_S19', [dict(id='1', name='甲', picture='same'),
                                       dict(id='2', name='乙', picture='same')], tmp_path)
    reader = AugmentReader(catalog)
    monkeypatch.setattr(reader, 'candidates', lambda _: [(530/1774, '甲', .99, 0.)])
    assert reader.read(np.zeros((1136, 1774, 3), np.uint8), own_view=True) == []


def test_catalog_only_reads_selected_season_and_does_not_modify_database(tmp_path):
    path = tmp_path / 'db.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE augments(season_id TEXT,id TEXT,name TEXT,tier INTEGER,icon TEXT)')
        db.executemany('INSERT INTO augments VALUES(?,?,?,?,?)', [('18_S19','1','本赛季',2,'a'), ('16_S19','2','其他赛季',1,'b')])
    before = path.read_bytes()
    result = AugmentCatalog.from_db(SimpleNamespace(path=path, season_id='18_S19'), tmp_path/'cache')
    assert [e['name'] for e in result.entries] == ['本赛季']
    assert result.entries[0]['tier'] == 2
    assert path.read_bytes() == before
    assert AugmentCatalog.from_db(None, tmp_path/'empty').entries == []


@pytest.mark.parametrize('name', ['潘朵拉的装备 I', '潘朵拉的装备 II', '潘朵拉的装备 III'])
def test_color_ranks_same_glyph_tiers_and_close_tiers_remain_unknown(reader, pixels, name):
    row = next(row for row in DATA['entries'] if row['name'] == name)
    image = np.zeros((1136, 1774, 3), np.uint8)
    image[286:332, 507:553] = cv2.resize(pixels['official'][row['index']], (46, 46))
    assert reader.candidates(image)[2][1] == name
    # 银/彩的灰白高亮很接近：当前保守阈值拒绝二者，不应放宽阈值把等级猜错。
    expected = [name] if name.endswith(' II') else []
    assert reader.read(image, own_view=True) == expected


def test_models_and_scoring_do_not_confuse_order_missing_or_false_names():
    assert GameState.from_dict({'augments_taken': ['甲', None, '', 2]}).augments_taken == ['甲']
    assert GameState.from_dict({'augments_taken': '甲'}).augments_taken == []
    assert GameState.from_dict({}).augments_taken == []
    truth = {'a': {'augments_taken': ['甲', '乙']}, 'b': {'augments_taken': ['丙']}}
    result = evaluate({'a': {'augments_taken': ['乙', '甲', '错']}}, truth)
    assert (result['exact'], result['frames'], result['tp'], result['fp'], result['fn']) == (0, 2, 2, 1, 1)
    assert result['precision'] == result['recall'] == 2/3


def test_memory_requires_consecutive_ordinary_frames_and_returns_a_copy(reader, monkeypatch):
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(reader, 'candidates', lambda _: [(CENTERS[2], '甲', .85, .70)])
    assert reader.read(image, own_view=True) == []
    result = reader.read(image, own_view=True)
    assert result == ['甲']
    result.append('外部修改')
    monkeypatch.setattr(reader, 'candidates', lambda _: [])
    assert reader.read(image, own_view=True) == ['甲']


def test_high_confidence_conflict_requires_two_frames_and_logs(reader, monkeypatch, caplog):
    image = np.zeros((1, 1, 3), np.uint8)
    candidate = [(CENTERS[2], '甲', .96, .6)]
    monkeypatch.setattr(reader, 'candidates', lambda _: candidate)
    assert reader.read(image, own_view=True) == ['甲']
    candidate[:] = [(CENTERS[2], '乙', .99, .1)]
    assert reader.read(image, own_view=True) == ['甲']
    with caplog.at_level('INFO'):
        assert reader.read(image, own_view=True) == ['乙']
    assert '位置1' in caplog.text and '甲 → 乙' in caplog.text


def test_foreign_or_unknown_view_breaks_confirmation_without_learning(reader, monkeypatch):
    image = np.zeros((1, 1, 3), np.uint8)
    candidate = [(CENTERS[2], '甲', .85, .7)]
    monkeypatch.setattr(reader, 'candidates', lambda _: candidate)
    assert reader.read(image, own_view=True) == []
    assert reader.read(image, own_view=False) == []
    assert reader.read(image, own_view=True) == []
    assert reader.read(image, own_view=True) == ['甲']
    candidate[:] = [(CENTERS[2], '别人', .99, .1)]
    monkeypatch.setattr('jccbrain.vision.eye.augments.spectate_score', lambda _: .95)
    assert reader.read(image, own_view=True) == ['甲']
    assert reader.read(image, own_view=False) == ['甲']
    assert reader.confirmed == {0: '甲'} and reader.pending == {}


def test_selection_positions_survive_centered_layout_changes_and_new_game(reader, monkeypatch):
    from jccbrain.vision.eye.reader import GameReader
    from jccbrain.vision.eye.text import Dictionary
    image = np.zeros((1, 1, 3), np.uint8)
    candidate = [(CENTERS[2], '甲', .96, .6)]
    monkeypatch.setattr(reader, 'candidates', lambda _: candidate)
    assert reader.read(image, own_view=True) == ['甲']
    candidate[:] = [(CENTERS[1], '甲', .96, .6), (CENTERS[3], '乙', .96, .6)]
    assert reader.read(image, own_view=True) == ['甲', '乙']
    candidate[:] = [(CENTERS[0], '甲', .96, .6), (CENTERS[4], '丙', .96, .6)]
    assert reader.read(image, own_view=True) == ['甲', '乙', '丙']
    assert reader.confirmed == {0: '甲', 1: '乙', 2: '丙'}
    game = GameReader(Dictionary({}, []), SimpleNamespace(reset=lambda: None),
                      ocr=SimpleNamespace(clear_cache=lambda: None), augments=reader)
    game._new_game()
    assert reader.confirmed == reader.pending == {}
    assert reader.read(image, own_view=False) == []


def test_occlusion_breaks_pending_conflict_and_does_not_remove_confirmed_name(reader, monkeypatch):
    image = np.zeros((1, 1, 3), np.uint8)
    candidate = [(CENTERS[2], '甲', .96, .6)]
    monkeypatch.setattr(reader, 'candidates', lambda _: candidate)
    reader.read(image, own_view=True)
    candidate[:] = [(CENTERS[2], '乙', .96, .6)]
    reader.read(image, own_view=True)
    candidate.clear()
    assert reader.read(image, own_view=True) == ['甲']
    candidate[:] = [(CENTERS[2], '乙', .96, .6)]
    assert reader.read(image, own_view=True) == ['甲']
