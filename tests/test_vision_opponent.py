import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from jccbrain.vision.eye import opponent
from jccbrain.vision.eye.units import Bar


def test_enemy_color_seed_rejects_green_and_requires_border():
    pixels=np.array([[[50,120,200],[50,200,100],[180,180,180]]],np.uint8)
    assert opponent.enemy_mask(pixels).tolist()==[[True,False,False]]
    image=np.full((1136,1774,3),255,np.uint8)
    image[400:407,800:870]=[50,120,200]
    assert opponent.enemy_bars(image)[0]==[]


def test_real_orange_cells_are_enemies_and_green_is_not():
    root=Path(__file__).parent/'fixtures'
    with np.load(root/'opponent_cells.npz') as data:
        for row in json.loads((root/'opponent_cells.json').read_text()):
            im=np.zeros((1136,1774,3),np.uint8);x,y=row['origin'];patch=data[row['key']]
            im[y:y+patch.shape[0],x:x+patch.shape[1]]=patch
            if not row['enemy']:
                from jccbrain.vision.eye.units import green_mask
                assert green_mask(patch).sum() > 20
            found,_=opponent.enemy_bars(im)
            assert bool(found)==row['enemy'],row
            from jccbrain.vision.eye.units import detect_bars
            before=detect_bars(im)
            detect_bars(im,mask_fn=opponent.enemy_mask,min_width=8/1774)
            assert detect_bars(im)==before


def test_candidate_readonly_pool_guard_and_frame_local_unknowns(monkeypatch):
    gallery=SimpleNamespace(pool={'甲英雄'},match=Mock(return_value=('不在卡池',.1)),
                            add=Mock(side_effect=AssertionError('禁止学习')))
    bar=Bar(700,771,300,306,.415,.267)
    monkeypatch.setattr(opponent,'enemy_bars',lambda _ :([bar],[]))
    monkeypatch.setattr(opponent,'feature_print',lambda _:np.ones(4))
    reader=opponent.OpponentReader(gallery,SimpleNamespace(name='fake'))
    image=np.zeros((1136,1774,3),np.uint8)
    first=reader.read(image)
    assert first['mirrored'] is True and first['name'] is None
    assert first['board'][0]['name']=='?1' and first==reader.read(image)
    gallery.add.assert_not_called()


def test_opponent_scorer_counts_missing_and_friendly_false_positive():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from eval_opponent import score
    truth=dict(board_count=1,enemy_positions=[dict(x=.3,y=.3)],friendly_positions=[dict(x=.6,y=.4)])
    assert score({},truth)[0]['board_count']==(0,1)
    bad=dict(x=.6,y=.4,name='?1')
    counts,false=score({'board':[bad]},truth)
    assert counts['friendly_as_enemy']==(1,1) and false==[bad]
    assert score({'board':[dict(x=.3,y=.3)]},truth)[0]['friendly_as_enemy']==(0,1)


def test_disabled_production_field_is_null_without_candidate_inference(tmp_path,monkeypatch):
    from jccbrain.vision.eye.reader import GameReader
    from jccbrain.vision.eye.units import UnitTracker,Gallery
    from jccbrain.vision.eye.text import Dictionary
    from jccbrain.vision.eye import reader as module
    monkeypatch.setattr(opponent,'ENABLED',False)
    factory=Mock(side_effect=AssertionError('禁用时不推理'))
    monkeypatch.setattr(opponent,'OpponentReader',factory)
    monkeypatch.setattr(module,'detect_scene',lambda *_:None)
    ocr=SimpleNamespace(prepare=lambda x:x,recognize_batch=lambda *args:[])
    r=GameReader(Dictionary({},[]),UnitTracker(Gallery(tmp_path)),ocr=ocr)
    r.phase='combat'
    out=r.read(np.zeros((1136,1774,3),np.uint8))
    assert out['phase']=='combat' and out['opponent'] is None
    factory.assert_not_called()


def test_nameplate_requires_both_legend_dimensions(monkeypatch):
    good=Bar(1200,1250,100,112,.69,.093)
    icon=Bar(539,574,347,361,.31,.31)
    plant=Bar(850,931,155,163,.5,.14)
    monkeypatch.setattr(opponent,'detect_bars',lambda *a,**k:[good,icon,plant])
    assert opponent.enemy_bars(np.zeros((1136,1774,3),np.uint8))[1]==[good]


def test_opponent_name_comes_from_current_moving_plate(monkeypatch):
    from jccbrain.vision.eye.ocr import Line
    legend=Bar(1200,1280,500,512,.7,.445)
    monkeypatch.setattr(opponent,'enemy_bars',lambda _: ([],[legend]))
    ocr=SimpleNamespace(name='fake',prepare=lambda x:x,
                        recognize_batch=Mock(return_value=[[Line('对手甲',.95,0,0,1,1)]]))
    gallery=SimpleNamespace(pool=set(),match=Mock())
    reader=opponent.OpponentReader(gallery,ocr)
    assert reader.read(np.zeros((1136,1774,3),np.uint8))['name']=='对手甲'
    ocr.recognize_batch.assert_called_once()


def b8_cells():
    root = Path(__file__).parent/'fixtures'
    rows = json.loads((root/'opponent_b8_cells.json').read_text())
    with np.load(root/'opponent_b8_cells.npz') as data:
        return [(row, data[row['key']].copy()) for row in rows]


def test_b8_real_recovery_slots_and_hard_negatives():
    for row, patch in b8_cells():
        image = np.full((1136, 1774, 3), 255, np.uint8)
        x, y = row['origin']
        image[y:y+patch.shape[0],x:x+patch.shape[1]] = patch
        found = opponent.recover_enemy_bars(image)
        if row['enemy']:
            px, py = row['point']
            matching = [b for b in found if abs(b.cx-px/1774)<.01 and abs(b.cy-py/1136)<.005]
            assert len(matching) == 1, (row, found)
            assert 70 <= matching[0].width <= 90
        else:
            assert found == [], (row, found)


def test_b8_gray_shield_without_orange_cannot_seed_recovery():
    row, patch = next((row, patch) for row, patch in b8_cells() if row['key'] == 'flash-0036')
    image = np.full((1136,1774,3),255,np.uint8)
    x,y = row['origin']
    patch[opponent.enemy_mask(patch)] = [170,170,170]
    image[y:y+patch.shape[0],x:x+patch.shape[1]] = patch
    assert opponent.enemy_seeds(image) == []
    assert opponent.recover_enemy_bars(image) == []


def test_b8_recovery_deduplicates_segments_and_preserves_legacy_geometry(monkeypatch):
    root = Path(__file__).parent/'fixtures'
    with np.load(root/'opponent_cells.npz') as data:
        patch = data['enemy-shield']
    image = np.zeros((1136,1774,3),np.uint8)
    image[212:212+patch.shape[0],816:816+patch.shape[1]] = patch
    old, legends = opponent.enemy_bars(image)
    assert len(old) == 1
    legacy = old[0]
    duplicate = Bar(legacy.min_x-3,legacy.max_x,legacy.min_y,legacy.max_y,
                    legacy.cx-.001,legacy.cy)
    distinct = Bar(1100,1174,311,319,.641,.277)
    monkeypatch.setattr(opponent,'recover_enemy_bars',lambda _: [duplicate,distinct,distinct])
    result, _ = opponent.enemy_bars(image)
    assert result == [legacy,distinct]
    assert result[0] is not duplicate


def test_b8_recovery_geometry_scales_to_supported_window(monkeypatch):
    import cv2
    row, patch = next((row, patch) for row, patch in b8_cells() if row['key'] == 'low-0033')
    image = np.full((1136,1774,3),255,np.uint8)
    x,y = row['origin'];image[y:y+patch.shape[0],x:x+patch.shape[1]] = patch
    resized = cv2.resize(image,(1576,1013),interpolation=cv2.INTER_AREA)
    found = opponent.recover_enemy_bars(resized)
    assert len(found) == 1
    assert abs(found[0].cx-row['point'][0]/1774) < .005
    assert abs(found[0].cy-row['point'][1]/1136) < .005


def test_b8_real_recovered_units_use_readonly_gallery_and_do_not_mutate_pixels(monkeypatch):
    row, patch = next((row, patch) for row, patch in b8_cells() if row['key'] == 'tank-0033')
    image=np.full((1136,1774,3),255,np.uint8)
    x,y=row['origin'];image[y:y+patch.shape[0],x:x+patch.shape[1]]=patch
    before=image.copy()
    gallery=SimpleNamespace(pool={'甲英雄'},match=Mock(return_value=('甲英雄',.1)),
                            add=Mock(side_effect=AssertionError('禁止学习')))
    ocr=SimpleNamespace(name='fake',prepare=lambda x:x,recognize_batch=Mock(return_value=[[]]))
    monkeypatch.setattr(opponent,'feature_print',lambda _:np.ones(4))
    reader=opponent.OpponentReader(gallery,ocr)
    first=reader.read(image)
    assert any(unit['name']=='甲英雄' for unit in first['board'])
    assert reader.read(image)==first
    assert gallery.match.called
    gallery.add.assert_not_called()
    np.testing.assert_array_equal(image,before)
