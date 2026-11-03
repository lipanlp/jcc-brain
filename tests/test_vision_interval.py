"""临时读帧间隔的恢复与实际等待；全部使用假时钟和假捕获。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jccbrain.core.models import GameState
from jccbrain.vision.base import VisionProvider
from jccbrain.vision.pyeye import ImageReplayProvider, PyEyeProvider


class StubVision(VisionProvider):
    def read(self, fresh=True):
        return GameState()


@pytest.mark.parametrize("previous", [None, .7])
def test_fast_frames_restores_nested_intervals(previous):
    vision = StubVision()
    vision.interval_override_s = previous
    with vision.fast_frames(.1):
        assert vision.interval_override_s == .1
        with vision.fast_frames(.02):
            assert vision.interval_override_s == .02
        assert vision.interval_override_s == .1
    assert vision.interval_override_s == previous


@pytest.mark.parametrize("previous", [None, .7])
def test_fast_frames_restores_nested_intervals_after_exception(previous):
    vision = StubVision()
    vision.interval_override_s = previous
    with pytest.raises(RuntimeError, match="识别失败"):
        with vision.fast_frames(.1):
            with pytest.raises(ValueError, match="内层失败"):
                with vision.fast_frames(.02):
                    raise ValueError("内层失败")
            assert vision.interval_override_s == .1
            raise RuntimeError("识别失败")
    assert vision.interval_override_s == previous


def test_fast_frames_does_not_change_other_provider_instances():
    first, second = StubVision(), StubVision()
    with first.fast_frames(.1):
        assert first.interval_override_s == .1
        assert second.interval_override_s is None
    assert first.interval_override_s is None


@pytest.fixture
def fake_clock(monkeypatch):
    from jccbrain.vision import pyeye

    clock = SimpleNamespace(now=10., sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(pyeye.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(pyeye.time, "sleep", sleep)
    return clock


def live_provider(tmp_path, phase):
    provider = PyEyeProvider(gallery_dir=tmp_path)
    provider.start = lambda: None
    provider._capture = SimpleNamespace(grab=Mock(return_value=(object(), None)))
    provider._reader = SimpleNamespace(read=Mock(return_value={"phase": phase}), last_ocr_cells=0)
    provider._last = GameState.from_dict({"phase": phase})
    provider._last_t = 10.
    return provider


@pytest.mark.parametrize("phase,override,expected", [
    ("combat", None, 1.), ("combat", .1, .1), ("combat", 0., 0.),
    ("unknown", None, 2.), ("unknown", .1, .1), ("planning", None, .25),
])
def test_pyeye_read_waits_for_override_or_original_phase_interval(fake_clock, tmp_path, phase, override, expected):
    provider = live_provider(tmp_path, phase)
    provider.interval_override_s = override

    provider.read()

    assert sum(fake_clock.sleeps) == pytest.approx(expected)
    assert provider._capture.grab.call_count == 1
    assert provider._reader.read.call_count == 1


def test_pyeye_override_preserves_fresh_false_cache_window(fake_clock, tmp_path):
    provider = live_provider(tmp_path, "combat")
    previous = provider._last
    fake_clock.now += .01
    with provider.fast_frames(0.):
        assert provider.read(fresh=False) is previous
    assert fake_clock.sleeps == []
    provider._capture.grab.assert_not_called()
    provider._reader.read.assert_not_called()


def replay_provider(monkeypatch, tmp_path):
    from jccbrain.vision.eye import capture

    provider = ImageReplayProvider(tmp_path, interval_s=.8)
    provider.start = lambda: None
    provider.files = [tmp_path / "fake.png"]
    provider._i = 0
    provider._last_t = 10.
    provider._last = GameState.from_dict({"phase": "combat"})
    provider._reader = SimpleNamespace(read=Mock(return_value={"phase": "combat"}))
    monkeypatch.setattr(capture, "load_image", Mock(return_value=object()))
    return provider


@pytest.mark.parametrize("override,expected", [(None, .8), (.1, .1), (0., 0.)])
def test_image_replay_read_uses_override(fake_clock, monkeypatch, tmp_path, override, expected):
    provider = replay_provider(monkeypatch, tmp_path)
    provider.interval_override_s = override

    provider.read()

    assert sum(fake_clock.sleeps) == pytest.approx(expected)
    provider._reader.read.assert_called_once()


def test_image_replay_override_preserves_fresh_false_cache(fake_clock, monkeypatch, tmp_path):
    provider = replay_provider(monkeypatch, tmp_path)
    previous = provider._last
    with provider.fast_frames(0.):
        assert provider.read(fresh=False) is previous
    assert fake_clock.sleeps == []
    provider._reader.read.assert_not_called()
