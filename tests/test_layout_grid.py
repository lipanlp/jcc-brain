"""有符号错位、用户标定与录像实点；不依赖本地 var 数据。"""
import pytest

from jccbrain.actuator.layout import Layout
from jccbrain.core.models import Unit
from jccbrain.engine.board import unit_hex


@pytest.mark.parametrize('offset', [-.5, 0., .5])
def test_signed_stagger_preserves_four_calibrated_corners(offset):
    corners = {'r1c1': (.2, .3), 'r1c7': (.8, .3),
               'r4c1': (.2 + offset * .12, .6), 'r4c7': (.92 + offset * .12, .6)}
    layout = Layout(board_corners=corners, board_row_offset=offset)
    for r, c in [(1, 1), (1, 7), (4, 1), (4, 7)]:
        assert layout.hex_center(r, c) == pytest.approx(corners[f'r{r}c{c}'])
    assert layout.hex_center(2, 1) == pytest.approx((.2 + offset * (.1 + .02 / 3), .4))
    assert layout.hex_center(3, 1) == pytest.approx((.2, .5))
    for r, c in layout.all_hexes():
        x, y = layout.hex_center(r, c)
        assert unit_hex(Unit('测试', x=x, y=y - layout.bar_to_hex_dy), layout) == (r, c)


def test_custom_calibration_has_priority_on_load(tmp_path):
    layout = Layout(board_corners={'r1c1': (.25, .31), 'r1c7': (.7, .32),
                                   'r4c1': (.18, .65), 'r4c7': (.8, .66)},
                    board_row_offset=-.4, bar_to_hex_dy=.08, board_calibrated=True)
    path = tmp_path / 'layout.json'
    layout.save(path)
    loaded = Layout.load(path)
    assert loaded == layout
    assert loaded.hex_center(4, 1) == pytest.approx((.18, .65))


def test_fitted_default_keeps_uncalibrated_semantics_and_perspective(tmp_path):
    layout = Layout.load(tmp_path / 'missing.json')
    assert not layout.board_calibrated
    steps = [layout.hex_center(r, 2)[0] - layout.hex_center(r, 1)[0] for r in range(1, 5)]
    assert steps == sorted(steps) and steps[-1] / steps[0] == pytest.approx(1.1704347826)


# 原始录像 20261022-011650 第650帧的全部5个场上血条点，不更改识别结果。
@pytest.mark.parametrize('name,x,y,rc', [
    ('峡谷迅捷蟹', .614, .337, (1, 6)), ('暗影狼', .499, .355, (1, 4)),
    ('苍蓝哨戒', .738, .531, (4, 7)), ('?9', .465, .542, (4, 3)),
    ('深红锋喙鸟', .332, .548, (4, 1)),
])
def test_recorded_units_map_inside_fitted_hex(name, x, y, rc):
    layout = Layout()
    assert unit_hex(Unit(name, x=x, y=y), layout) == rc
    cx, cy = layout.hex_center(*rc)
    width = (layout.hex_center(rc[0], 2)[0] - layout.hex_center(rc[0], 1)[0]) * .9
    height = width * 2 / 3 ** .5
    # 与 HUD 同一尖顶六边形边界，不能只断言最近格存在。
    dx, dy = abs(x - cx), abs(y + layout.bar_to_hex_dy - cy)
    assert dx <= width / 2
    assert dy <= height / 2 - dx * height / (2 * width)
