"""HUD 站位的坐标、分配、阶段开关与逐帧推送；不依赖窗口/真实数据库。"""
from math import sqrt
from types import SimpleNamespace

import json
import pytest

from jccbrain.actuator.layout import Layout
from jccbrain.core.models import GameState, Phase, Unit
from jccbrain.core.settings import HudSettings, Settings
from jccbrain.engine.advisor import Advice
from jccbrain.engine.board import BoardCell, BoardPlan, hex_width, unit_hex
from jccbrain.modes.assistant import advice_signature
from jccbrain.webui import serialize, settings_schema
from jccbrain.webui.api import Api


def cell(row=4, col=7, name='绯红树怪', role='carry'):
    return BoardCell(row, col, name, 3, role == 'carry', None, 2, None, False, role=role)


def at(layout, row, col, name='绯红树怪'):
    x, y = layout.hex_center(row, col)
    return Unit(name, x=x, y=y - layout.bar_to_hex_dy)


def render(db, *, cells=None, layout=None, cfg=None, **kwargs):
    st = GameState(phase=Phase.PLANNING, **kwargs)
    adv = Advice(st.stage, st.phase, board=BoardPlan('测试', 5, cells or [cell()]))
    return serialize.hud(st, adv, db, layout or Layout(), cfg or HudSettings())


@pytest.mark.parametrize('rc', [(1, 1), (2, 4), (3, 7), (4, 1)])
def test_unit_at_hex_center(rc):
    layout = Layout()
    assert unit_hex(at(layout, *rc), layout) == rc


def test_unit_small_offset_is_accepted():
    layout = Layout()
    u = at(layout, 1, 1)
    u.x -= hex_width(layout, 1) * .59
    assert unit_hex(u, layout) == (1, 1)


def test_unit_outside_distance_limit_is_unknown():
    layout = Layout()
    u = at(layout, 1, 1)
    u.x -= hex_width(layout, 1) * .61
    assert unit_hex(u, layout) is None


@pytest.mark.parametrize('u', [Unit('A'), Unit('A', x=.3), Unit('A', y=.4),
                              Unit('A', x=float('nan'), y=.4), Unit('A', x=.4, y=float('inf')),
                              Unit('A', zone='bench', x=.4, y=.5)])
def test_unit_without_usable_board_coordinate(u):
    assert unit_hex(u, Layout()) is None


def test_even_row_offset_and_custom_calibration_are_used():
    layout = Layout(board_corners={'r1c1': (.1, .3), 'r1c7': (.7, .3),
                                   'r4c1': (.15, .6), 'r4c7': (.75, .6)},
                    board_calibrated=True, bar_to_hex_dy=.08, board_row_offset=.5)
    x, y = layout.hex_center(2, 3)
    assert x == pytest.approx(.35) and y == pytest.approx(.4)
    assert unit_hex(Unit('A', x=x, y=y - .08), layout) == (2, 3)


def test_all_four_states_roles_and_arrows(db):
    layout = Layout()
    cells = [cell(4, 7), cell(1, 5, '奥恩', 'tank'), cell(2, 3, '卡兹克', None), cell(3, 1, '远古巨龙', None)]
    result = render(db, cells=cells, layout=layout, board=[at(layout, 4, 7), at(layout, 1, 1, '奥恩')],
                    bench=[Unit('卡兹克', zone='bench', slot=3)], board_count=(2, 5))
    json.dumps(result, allow_nan=False)
    assert [p['state'] for p in result['positions']] == ['ok', 'misplaced', 'bench', 'missing']
    assert [p['role'] for p in result['positions']] == ['carry', 'tank', None, None]
    assert [a['kind'] for a in result['arrows']] == ['move', 'place']
    move, place = result['arrows']
    assert (move['x1'], move['y1']) == layout.hex_center(1, 1)
    assert (move['x2'], move['y2']) == layout.hex_center(1, 5)
    assert (place['x1'], place['y1']) == layout.bench_slot(3)
    assert (place['x2'], place['y2']) == layout.hex_center(2, 3)
    for p, c in zip(result['positions'], cells):
        assert (p['x'], p['y']) == layout.hex_center(c.row, c.col)
        assert p['w'] == pytest.approx(hex_width(layout, c.row))
        assert p['h'] == pytest.approx(p['w'] * 2 / sqrt(3))


def test_same_name_uses_nearest_board_unit(db):
    layout = Layout()
    result = render(db, board=[at(layout, 1, 1), at(layout, 4, 6)])
    assert result['positions'][0]['state'] == 'misplaced'
    a = result['arrows'][0]
    assert (a['x1'], a['y1']) == layout.hex_center(4, 6)
    result = render(db, board=[at(layout, 1, 1), at(layout, 4, 7)])
    assert result['positions'][0]['state'] == 'ok' and not result['arrows']


def test_same_name_board_takes_precedence_over_bench(db):
    result = render(db, board=[at(Layout(), 4, 6)], bench=[Unit('绯红树怪', zone='bench', slot=8)], board_count=(1, 5))
    assert result['positions'][0]['state'] == 'misplaced'
    assert [a['kind'] for a in result['arrows']] == ['move']


def test_same_name_bench_uses_nearest_slot(db):
    layout = Layout()
    result = render(db, bench=[Unit('绯红树怪', zone='bench', slot=0), Unit('绯红树怪', zone='bench', slot=8)], board_count=(0, 5))
    a = result['arrows'][0]
    assert (a['x1'], a['y1']) == layout.bench_slot(8)


@pytest.mark.parametrize('roles', [('carry', 'tank'), ('tank', None)])
def test_unit_is_used_only_once_in_role_priority(db, roles):
    # 较高优先级格故意放后面；分配顺序不能依赖源数据顺序。
    cells = [cell(1, 1, role=roles[1]), cell(4, 7, role=roles[0])]
    result = render(db, cells=cells, board=[at(Layout(), 1, 1)])
    assert [p['state'] for p in result['positions']] == ['missing', 'misplaced']
    assert len(result['arrows']) == 1


def test_bench_unit_is_not_assigned_twice(db):
    result = render(db, cells=[cell(1, 1, role=None), cell(4, 7)],
                    bench=[Unit('绯红树怪', zone='bench', slot=4)], board_count=(0, 5))
    assert [p['state'] for p in result['positions']] == ['missing', 'bench']
    assert len(result['arrows']) == 1


@pytest.mark.parametrize('count', [(5, 5), (6, 5), None])
def test_bench_without_confirmed_space_has_no_place_arrow(db, count):
    result = render(db, bench=[Unit('绯红树怪', zone='bench', slot=2)], board_count=count)
    assert result['positions'][0]['state'] == 'bench' and not result['arrows']


@pytest.mark.parametrize('slot', [None, -1, 9])
def test_unknown_bench_slot_has_no_arrow(db, slot):
    result = render(db, bench=[Unit('绯红树怪', zone='bench', slot=slot)], board_count=(0, 5))
    assert result['positions'][0]['state'] == 'bench' and not result['arrows']


def test_board_unit_with_unknown_position_has_no_arrow(db):
    result = render(db, board=[Unit('绯红树怪', x=.01, y=.01)])
    assert result['positions'][0]['state'] == 'misplaced' and not result['arrows']


@pytest.mark.parametrize('phase', [Phase.COMBAT, Phase.AUGMENT, Phase.CAROUSEL, Phase.ENDED, Phase.UNKNOWN])
def test_only_planning_shows_positions(db, phase):
    st = GameState(phase=phase)
    adv = Advice(st.stage, phase, board=BoardPlan('测试', 5, [cell()]))
    result = serialize.hud(st, adv, db, Layout(), HudSettings())
    assert result['positions'] == result['arrows'] == []


def test_disabled_guides_preserve_other_hud_content(db):
    kwargs = dict(board=[at(Layout(), 1, 1)], board_count=(1, 5))
    enabled = render(db, **kwargs)
    disabled = render(db, cfg=HudSettings(show_positions=False), **kwargs)
    assert enabled['positions'] and enabled['arrows']
    assert disabled['positions'] == disabled['arrows'] == []
    assert disabled['units'] == enabled['units'] and disabled['units']
    assert disabled['shop'] == enabled['shop'] and disabled['augments'] == enabled['augments']


@pytest.mark.parametrize('st,adv', [(None, None), (GameState(phase=Phase.PLANNING), None),
                                  (GameState(phase=Phase.PLANNING), Advice(None, Phase.PLANNING))])
def test_no_state_or_board_has_empty_guides(db, st, adv):
    result = serialize.hud(st, adv, db, Layout(), HudSettings())
    assert result['positions'] == result['arrows'] == []


def test_setting_defaults_schema_and_save(tmp_path):
    settings = Settings()
    assert settings.hud.show_positions is True
    fields = next(s['fields'] for s in settings_schema.schema(settings, full=True) if s['key'] == 'hud')
    option = next(f for f in fields if f['key'] == 'show_positions')
    assert option['label'] == '显示站位指引' and option['type'] == 'bool'
    settings.hud.show_positions = False
    path = tmp_path / 'settings.json'
    settings.save(path)
    assert Settings.load(path).hud.show_positions is False


def test_moving_units_pushes_new_hud_without_new_advice(db):
    layout = Layout()
    events = []
    ctx = SimpleNamespace(db=db, layout=layout, settings=Settings())
    hud = SimpleNamespace(status={'overlay': True, 'panel': False}, set_refresh=lambda callback: None)
    api = Api(ctx, SimpleNamespace(emit=lambda name, value: events.append((name, value))), hud=hud)
    adv = Advice('2-6', Phase.PLANNING, board=BoardPlan('测试', 5, [cell()]))
    signature = advice_signature(adv)
    try:
        api._sink.on_advice(adv)
        api._sink.on_state(GameState(phase=Phase.PLANNING, board=[at(layout, 1, 1)]))
        api._sink.on_state(GameState(phase=Phase.PLANNING, board=[at(layout, 4, 7)]))
        frames = [payload for name, payload in events if name == 'hud']
        assert frames[-2]['positions'][0]['state'] == 'misplaced' and len(frames[-2]['arrows']) == 1
        assert frames[-1]['positions'][0]['state'] == 'ok' and not frames[-1]['arrows']
        assert advice_signature(adv) == signature
        assert len([e for e in events if e[0] == 'advice']) == 1
    finally:
        api._icons.close()
