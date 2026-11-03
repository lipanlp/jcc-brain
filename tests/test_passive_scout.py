from copy import deepcopy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from jccbrain.core.models import GameState, Phase, Player, Unit
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.passive_scout import PassiveScout
from jccbrain.engine.scout import ScoutBook, ScoutShot
from jccbrain.webui.serialize import advice as serialize


@pytest.fixture
def book(db, tmp_path):
    return ScoutBook(db, directory=tmp_path/'scout')


def state(stage='4-1', name='甲', phase=Phase.COMBAT, **kw):
    return GameState(stage=stage, phase=phase, opponent={'name': name, 'board': [{'name': '绯红树怪', 'star': 2}]}, **kw)


def image(value=0):
    return np.full((160, 240, 3), value, dtype=np.uint8)


def test_once_each_round_and_source(book):
    passive = PassiveScout(book)
    assert passive.observe(state(), image()) is None
    assert passive.observe(state(), image()).source == 'passive'
    assert passive.observe(state(), image()) is None
    assert len(book.history['name:甲']) == 1
    assert passive.observe(state('4-2'), image()) is None
    assert passive.observe(state('4-2'), image()).round == '4-2'
    assert len(book.history['name:甲']) == 2
    assert all(json.loads(line)['source'] == 'passive' for line in book.path.read_text().splitlines())


def test_not_combat_or_no_opponent_no_record(book):
    passive = PassiveScout(book)
    for st in [GameState(), GameState(stage='4-1', phase=Phase.COMBAT), state(phase=Phase.PLANNING),
               GameState(stage='4-1', phase=Phase.COMBAT, opponent={'name': '甲', 'board': []})]:
        assert not passive.needs_image(st)
        assert passive.observe(st, image()) is None
    assert book.latest() == []


def test_missing_parser_and_optional_opponent():
    assert GameState.from_dict({}).opponent is None
    data = {'opponent': {'name': None, 'board': [{'name': '?1'}]}}
    st = GameState.from_dict(data)
    assert st.opponent == data['opponent']
    assert GameState.from_dict({'opponent': []}).opponent is None


def test_threshold_is_strict_and_unstable_skips(book):
    passive = PassiveScout(book)
    assert passive.observe(state(), image(0)) is None
    assert passive.observe(state(), image(6)) is None  # 差异==6不算稳定
    assert passive.observe(state(), image(11)).name == '甲'  # 差异5，首次稳定


def test_identity_or_phase_gap_requires_new_pair(book):
    passive = PassiveScout(book)
    passive.observe(state(name='甲'), image())
    assert passive.observe(state(name='乙'), image()) is None
    passive.observe(GameState(stage='4-1', phase=Phase.PLANNING), image())
    assert passive.observe(state(name='乙'), image()) is None
    assert passive.observe(state(name='乙'), image()).name == '乙'


def test_no_image_or_different_size_does_not_invent_stability(book):
    passive = PassiveScout(book)
    assert passive.observe(state(), None) is None
    assert passive.observe(state(), image()) is None
    assert passive.observe(state(), np.zeros((200, 400, 3), dtype=np.uint8)) is None
    assert not book.latest()


def test_anonymous_keeps_unknown_identity_and_does_not_double_count_pool(db, rules, book):
    passive = PassiveScout(book)
    passive.observe(state(name=None), image())
    row = passive.observe(state(name=None), image())
    assert row.name is None and row.slot is None and row.source == 'passive'
    view = book.view()[0]
    assert view['key'] == 'unknown:passive' and view['name'] is None
    # 无法确认是不是已侦察过的玩家，只展示这次匿名观察，不重复扣公共池。
    advisor = HeuristicAdvisor(db, rules, scout=book)
    assert advisor.pool(GameState()).taken == {}
    assert advisor.pool(GameState()).opponents == []
    assert serialize(advisor.advise(GameState()))['opponents'][0]['name'] is None


def test_merges_active_by_name_new_round_wins(book):
    book.update(GameState(players=[Player('甲', 50, False, 1)]))
    book.add(ScoutShot(slot=1, name='甲', round='4-1', t=1, board=[Unit('奥恩')]))
    passive = PassiveScout(book)
    passive.observe(state(stage='4-2'), image())
    passive.observe(state(stage='4-2'), image())
    assert len(book.latest()) == 1
    latest = book.latest()[0]
    assert latest.slot == 1 and latest.round == '4-2' and latest.units[0].name == '绯红树怪'
    assert book.add(ScoutShot(slot=1, name='甲', round='4-1', t=10**12)) is None


def test_new_game_clears_round_once_guard(book):
    passive = PassiveScout(book)
    passive.observe(state(), image())
    passive.observe(state(), image())
    book.reset()
    assert passive.needs_image(state())
    assert passive.observe(state(), image()) is None
    assert passive.observe(state(), image()).name == '甲'


def test_mode_consumes_enemy_field_without_overwriting_own_board(db, rules, book):
    from conftest import make_state
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.vision.base import VisionError
    raw = make_state(phase='combat', board=[{'name': '奥恩', 'star': 1}],
                     opponent={'name': '甲', 'board': [{'name': '绯红树怪', 'star': 2}]})
    raw_before = deepcopy(raw)
    ctx = Ctx(db, rules, [deepcopy(raw)] * 4)
    ctx.advisor.scout = book
    ctx.llm.ready = False
    calls = []
    jpeg = cv2.imencode('.jpg', image())[1].tobytes()
    ctx.vision.screenshot = lambda width: (calls.append(width), jpeg)[1]
    sink = AdviceSink()
    mode = AssistantMode(ctx, sink)
    with pytest.raises(VisionError):
        mode.loop()
    assert calls == [0, 0]  # 本回合记录完成后不再读取截图
    assert book.latest()[0].units[0].name == '绯红树怪'
    assert mode.stabilizer.acc['board'][0]['name'] == '奥恩'
    assert raw == raw_before


def test_legacy_mode_does_not_request_screenshot(db, rules, book):
    from conftest import make_state
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.vision.base import VisionError
    ctx = Ctx(db, rules, [make_state(phase='combat')] * 3)
    ctx.advisor.scout = book
    ctx.llm.ready = False
    def forbidden(width):
        pytest.fail('旧录像没有opponent，不应截图')
    ctx.vision.screenshot = forbidden
    with pytest.raises(VisionError):
        AssistantMode(ctx, AdviceSink()).loop()
    assert book.latest() == []


def test_screenshot_error_only_logs_and_continues(db, rules, book, caplog):
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    ctx = Ctx(db, rules, [])
    ctx.advisor.scout = book
    ctx.vision.screenshot = lambda width: (_ for _ in ()).throw(RuntimeError('capture failed'))
    mode = AssistantMode(ctx, AdviceSink())
    mode._observe_opponent(state())
    assert book.latest() == [] and '被动侦察读取失败' in caplog.text


def test_respects_settings_threshold(db, rules, book):
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    ctx = Ctx(db, rules, [])
    ctx.advisor.scout = book
    ctx.settings.scout = SimpleNamespace(stable_threshold=3)
    frames = iter([image(0), image(4), image(5)])
    ctx.vision.screenshot = lambda width: cv2.imencode('.jpg', next(frames))[1].tobytes()
    mode = AssistantMode(ctx, AdviceSink())
    mode._observe_opponent(state())
    mode._observe_opponent(state())
    assert not book.latest()
    mode._observe_opponent(state())
    assert book.latest()


def test_mode_restart_does_not_record_same_round_again(book):
    first = PassiveScout(book)
    first.observe(state(), image())
    first.observe(state(), image())
    # 停止/重开辅助模式不会重记这个回合，即使主动快照覆盖了最新画像。
    book.add(ScoutShot(name='甲', round='4-1', t=10**12, board=[Unit('奥恩')]))
    second = PassiveScout(book)
    assert not second.needs_image(state())
    assert second.observe(state(), image()) is None
    assert len(book.history['name:甲']) == 2
