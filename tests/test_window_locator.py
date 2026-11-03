"""窗口定位只使用合成的 Quartz 列表，不查询真实桌面。"""
import os
from unittest.mock import Mock

import pytest
import Quartz

from jccbrain.actuator import window


def _window(pid, *, owner="金铲铲之战", width=1000, height=700, layer=0, number=42):
    return {
        "kCGWindowOwnerName": owner,
        "kCGWindowName": "对局",
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowBounds": {"X": 50, "Y": 80, "Width": width, "Height": height},
        "kCGWindowNumber": number,
    }


@pytest.fixture
def windows(monkeypatch):
    rows = []
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", Mock(return_value=rows))
    monkeypatch.setattr(window, "_bundle_id", Mock(return_value=""))
    return rows


def test_own_app_is_not_a_game_window(windows):
    windows.append(_window(os.getpid(), owner="金铲铲大脑"))

    assert window.WindowLocator().find(0) is None


def test_game_is_selected_even_when_own_window_is_larger(windows):
    game_pid = os.getpid() + 1
    windows.extend([
        _window(os.getpid(), owner="金铲铲大脑", width=1280, height=820, number=1),
        _window(game_pid, number=2),
    ])

    assert window.WindowLocator().find(0) == window.WindowInfo(
        "金铲铲之战", "对局", game_pid, 50, 80, 1000, 700, 2,
    )


def test_nonzero_layer_is_ignored(windows):
    pid = os.getpid() + 1
    windows.extend([
        _window(pid, width=2000, height=1500, layer=3, number=1),
        _window(pid, number=2),
    ])

    assert window.WindowLocator().find(0).window_id == 2


def test_width_below_200_is_ignored(windows):
    pid = os.getpid() + 1
    windows.extend([
        _window(pid, width=199, height=10000, number=1),
        _window(pid, width=200, height=300, number=2),
    ])

    assert window.WindowLocator().find(0).window_id == 2


def test_largest_area_is_selected(windows):
    pid = os.getpid() + 1
    windows.extend([
        _window(pid, width=900, height=600, number=1),
        _window(pid, width=1200, height=400, number=2),
        _window(pid, width=600, height=500, number=3),
    ])

    assert window.WindowLocator().find(0).window_id == 1
