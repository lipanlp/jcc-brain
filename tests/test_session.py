"""换局确认：正常推进、回跳噪声、缺帧、结算及多局复用。"""
from conftest import make_state
from jccbrain.core.models import GameState
from jccbrain.engine.session import GameWatcher


def frame(stage, phase='planning'):
    return GameState.from_dict(make_state(stage=stage, phase=phase))


def test_watcher_normal_progression_never_starts_another_game():
    watcher = GameWatcher()
    assert not any(watcher.update(frame(stage)) for stage in ('1-1', '1-2', '1-3', '1-4', '2-1', '2-5', '3-2', '6-1'))


def test_watcher_requires_three_early_frames_after_late_round():
    watcher = GameWatcher()
    assert not watcher.update(frame('2-5'))
    assert not watcher.update(frame('1-2'))
    assert not watcher.update(frame('1-2'))
    assert watcher.update(frame('1-2'))


def test_watcher_missing_stage_neither_counts_nor_resets():
    watcher = GameWatcher()
    watcher.update(frame('2-5'))
    assert not watcher.update(frame('1-2'))
    assert not watcher.update(frame(None))
    assert not watcher.update(frame('1-3'))
    assert not watcher.update(frame('unreadable'))
    assert watcher.update(frame('1-4'))


def test_watcher_valid_late_frame_interrupts_early_confirmation():
    watcher = GameWatcher()
    watcher.update(frame('2-5'))
    for stage in ('1-2', '1-2', '2-6', '1-2', '1-2'):
        assert not watcher.update(frame(stage))
    assert watcher.update(frame('1-2'))


def test_watcher_ended_without_stage_confirms_next_first_stage():
    watcher = GameWatcher()
    watcher.update(frame(None, 'ended'))
    assert not watcher.update(frame('1-1'))
    assert not watcher.update(frame('1-1'))
    assert watcher.update(frame('1-1'))


def test_watcher_resets_history_and_can_detect_a_later_game():
    watcher = GameWatcher()
    watcher.update(frame('5-2', 'ended'))
    for _ in range(2):
        assert not watcher.update(frame('1-2'))
    assert watcher.update(frame('1-2'))
    assert not any(watcher.update(frame(stage)) for stage in ('1-2', '1-2', '1-2', '1-3', '2-1'))
    assert not watcher.update(frame('1-2'))
    assert not watcher.update(frame('1-2'))
    assert watcher.update(frame('1-2'))
