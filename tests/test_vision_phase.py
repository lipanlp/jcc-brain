import json
import cv2
from pathlib import Path

import numpy as np
import pytest

from jccbrain.core.models import GameState, Phase
from jccbrain.vision.eye.phase import StagebarReader, detect_reconnect

FIXTURES = Path(__file__).parent / 'fixtures'


def frames():
    truth = [json.loads(line) for line in (FIXTURES / 'phase_truth.jsonl').read_text().splitlines()]
    with np.load(FIXTURES / 'stagebar_cells.npz') as data:
        h, w = data['shape']
        x0, y0 = data['origin']
        for row in truth:
            if row['_file'] not in data:
                continue  # 新场景使用独立小图夹具；旧27帧继续完整验证回合条。
            cell = cv2.resize(data[row['_file']], (round(.601*w)-x0, round(.072*h)-y0), interpolation=cv2.INTER_LINEAR)
            image = np.zeros((h, w, 3), np.uint8)
            image[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell
            yield row, image


def test_visible_next_icons_and_same_round_cursor_memory():
    reader = StagebarReader()
    for row, image in frames():
        assert reader.read(image, row.get('stage')) == row['next_round'], row['_file']
    assert reader.index is None  # 结算没有回合条，清掉旧指针。


def test_current_icons_match_manually_verified_truth_without_changing_next_round():
    reader = StagebarReader()
    for row, image in frames():
        next_round, round_type = reader.read_rounds(image, row.get('stage'))
        assert (next_round, round_type) == (row['next_round'], row['round_type']), row['_file']


def test_current_requires_cursor_and_rejects_unrecognized_icon():
    reader = StagebarReader()
    images = {r['_file']: image for r, image in frames()}
    assert reader.read_current(images['0006.jpg'], '2-6') == 'unknown'
    assert reader.read_current(images['0000.jpg'], '2-6') == 'pvp'
    assert reader.read_current(images['0006.jpg'], '2-6') == 'pvp'
    assert reader.read_current(images['0009.jpg'], '2-7') == 'pve'
    assert reader.read_current(images['0015.jpg'], '2-7') == 'pve'
    assert reader.read_current(np.zeros((1136, 1774, 3), np.uint8), '2-7') == 'unknown'
    assert reader.read_current(images['0006.jpg'], 'another-stage') == 'unknown'


@pytest.mark.parametrize('value', ['pvp', 'pve', 'carousel', 'augment', 'unknown'])
@pytest.mark.parametrize('phase', ['planning', 'combat'])
def test_round_type_parses_without_changing_phase_summary_or_action_eligibility(value, phase):
    raw = dict(phase=phase, shop={'open': True})
    before = GameState.from_dict(raw)
    state = GameState.from_dict(dict(raw, round_type=value))
    assert state.round_type == value and state.phase == before.phase
    assert state.summary() == before.summary() and state.can_act == before.can_act


@pytest.mark.parametrize('value', [None, 'bad', 7, [], {}])
def test_invalid_or_missing_current_round_is_unknown(value):
    assert GameState.from_dict({'round_type': value}).round_type == 'unknown'
    assert GameState.from_dict({}).round_type == 'unknown'


def test_phase_evaluation_reports_current_round_classes_and_missing_prediction():
    from scripts.eval_phase import evaluate
    result = evaluate({'a': {'round_type': 'pve'}, 'b': {}, 'unscored': {'round_type': 'pvp'}},
                      {'a': {'round_type': 'pve'}, 'b': {'round_type': 'pvp'}})
    assert result['round_type']['correct'] == 1 and result['round_type']['total'] == 2
    assert result['round_type']['classes']['pve']['accuracy'] == 1.
    assert dict(expected='pvp', predicted='unknown', count=1) in result['round_type']['confusion']


def test_hidden_cursor_at_start_is_unknown_and_round_change_clears_memory():
    reader = StagebarReader()
    images = {r['_file']: (r, image) for r, image in frames()}
    assert reader.read(images['0006.jpg'][1], '2-6') == 'unknown'
    assert reader.read(images['0000.jpg'][1], '2-6') == 'pve'
    assert reader.read(images['0006.jpg'][1], '2-6') == 'pve'
    assert reader.read(np.zeros((1136, 1774, 3), np.uint8), '3-1') == 'unknown'


@pytest.mark.parametrize('value', ['pve', 'spectate', 'dialog', 'reconnect'])
def test_new_phases_parse_without_changing_existing_summary(value):
    state = GameState.from_dict({'phase': value, 'next_round': 'augment'})
    assert state.phase == Phase(value) and state.next_round == 'augment'


@pytest.mark.parametrize('value', [None, 'bad', 7])
def test_invalid_next_round_is_unknown(value):
    assert GameState.from_dict({'next_round': value}).next_round == 'unknown'


def test_missing_scene_evidence_does_not_guess_reconnection():
    assert not detect_reconnect(np.zeros((1, 1, 3), np.uint8))


def scene_image(row, cell):
    image = np.zeros((1136, 1774, 3), np.uint8)
    h, w = image.shape[:2]
    if row['kind'] == 'dialog':
        x0, x1, y0, y1 = round(.35*w), round(.65*w), round(.57*h), round(.65*h)
    elif row['kind'] == 'spectate':
        strip = np.zeros((682, 35, 3), np.uint8)
        strip[row['y']:row['y'] + cell.shape[0]] = cell
        cell = strip
        x0, x1, y0, y1 = round(.963*w), round(.983*w), round(.09*h), round(.69*h)
    else:
        x0, x1, y0, y1 = round(.399*w), round(.601*w), round(.032*h), round(.072*h)
    image[y0:y1, x0:x1] = cv2.resize(cell, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    return image


def test_new_scene_crops_and_hard_legacy_negatives():
    from jccbrain.vision.eye.phase import detect_scene
    with np.load(FIXTURES / 'scene_cells.npz', allow_pickle=False) as data:
        for row in json.loads(str(data['metadata'])):
            image = scene_image(row, data[row['key']])
            _, current = StagebarReader().read_rounds(image, '4-4')
            assert detect_scene(image, current) == row['expected'], row


def test_dialog_carousel_spectate_priority_and_no_pve_phase():
    from jccbrain.vision.eye.phase import detect_scene
    with np.load(FIXTURES / 'scene_cells.npz', allow_pickle=False) as data:
        rows = json.loads(str(data['metadata']))
        eye = next(r for r in rows if r['expected'] == 'spectate')
        dialog = next(r for r in rows if r['expected'] == 'dialog')
        image = scene_image(eye, data[eye['key']])
        assert detect_scene(image, 'carousel') == 'carousel'
        image += scene_image(dialog, data[dialog['key']])
        assert detect_scene(image, 'carousel') == 'dialog'
    assert detect_scene(np.zeros((1136, 1774, 3), np.uint8), 'pve') is None


@pytest.mark.parametrize('phase', ['carousel', 'spectate', 'dialog'])
def test_new_scenes_disable_model_actions_and_engine_plans(phase):
    from types import SimpleNamespace
    from jccbrain.engine.llm import LLMClient
    from jccbrain.engine.planner import AutoPlanner
    state = GameState.from_dict(dict(stage='4-3', phase=phase, gold=50, level=8, shop={'open': True}))
    assert not state.can_act
    assert AutoPlanner(None, None).plan(state, SimpleNamespace(augments=[])) == []
    assert LLMClient.call_key(state) is None
