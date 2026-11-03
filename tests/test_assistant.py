"""辅助大脑：状态稳定器、买卡规则、阵容推荐、海克斯名匹配、大模型提问时机、建议推送。"""
from __future__ import annotations

from ai_fakes import authorize_ai

import pytest

from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.context import AppContext
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher
from jccbrain.engine.economy import EconAction, decide
from jccbrain.engine.llm import LLMClient, LLMReply
from jccbrain.engine.stabilizer import StateStabilizer
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.base import ModeSink
from jccbrain.vision.base import VisionError, VisionProvider

from conftest import make_state


def shop(*names, cost=1):
    return {"open": True, "slots": [{"name": n, "cost": cost} if n else None for n in names]}


# ------------------------------------------------------------------ 稳定器
def test_stabilizer_carries_missing_and_ignores_single_frame_misread():
    s = StateStabilizer()
    s.update(make_state(stage="2-3", gold=12, level=4))
    out = s.update(make_state(stage="2-3", gold=None, level=None, phase="combat"))
    assert out["gold"] == 12 and out["level"] == 4                     # 战斗中商店收起读不到金币
    assert s.update(make_state(stage="2-3", gold=2, level=4))["gold"] == 12   # 只出现一帧的 2 不采用
    assert s.update(make_state(stage="2-3", gold=2, level=4))["gold"] == 2    # 连续两帧才采用


def test_stabilizer_confirms_streak_and_carries_it_across_rounds():
    s = StateStabilizer()
    loss = {"kind": "loss", "n": 4}
    win = {"kind": "win", "n": 1}
    # 第一次读到的三连以上也要确认，避免单帧误读直接切换运营。
    assert s.update(make_state(stage="2-5", streak=loss))["streak"] is None
    assert s.update(make_state(stage="2-5", streak=loss))["streak"] == loss
    assert s.update(make_state(stage="2-5", phase="combat", streak=None))["streak"] == loss
    assert s.update(make_state(stage="2-6", streak=win))["streak"] == loss
    assert s.update(make_state(stage="2-6", streak=loss))["streak"] == loss
    assert s.update(make_state(stage="2-6", streak=win))["streak"] == loss
    assert s.update(make_state(stage="2-6", streak=win))["streak"] == win
    # 开新局不把上一局的连败传下去。
    assert s.update(make_state(stage="1-2", streak=None))["streak"] is None


def test_stabilizer_single_frame_streak_does_not_change_economy(rules):
    from jccbrain.engine import economy

    s = StateStabilizer()
    base = make_state(stage="2-5", level=4, gold=26, xp=[0, 10])
    before = economy.decide(GameState.from_dict(s.update(base)), rules)
    assert before.action == economy.EconAction.LEVEL
    noisy = {**base, "streak": {"kind": "loss", "n": 4}}
    assert economy.decide(GameState.from_dict(s.update(noisy)), rules) == before
    confirmed = economy.decide(GameState.from_dict(s.update(noisy)), rules)
    assert confirmed.action == economy.EconAction.SAVE
    assert s.update({**base, "stage": "2-6", "streak": None})["streak"] == noisy["streak"]


def _rank_state(healths=(90, 80, 70, 66, 62, 60, 58, 55), **kw):
    players = [{"name": f"玩家{i}", "hp": hp, "self": i == len(healths) - 1}
               for i, hp in enumerate(healths)]
    return make_state(stage="4-3", level=7, gold=40, xp=[0, 48], hp=healths[-1], players=players, **kw)


def test_stabilizer_hp_rank_keeps_review_four_frame_economy_stable(rules):
    s = StateStabilizer()
    decisions = []
    for own_hp in (55, 56, 55, 56):
        frame = _rank_state()
        frame["players"][-1]["hp"] = own_hp
        d = decide(GameState.from_dict(s.update(frame)), rules)
        decisions.append((d.action, d.keep_gold, d.roll_budget))
    assert decisions == [(EconAction.ROLL, 20, 20)] * 4


def test_stabilizer_hp_rank_requires_two_frames_for_opponent_changes(rules):
    s = StateStabilizer()
    low = _rank_state((100, 95, 90, 85, 80, 65, 60, 70))  # 第 6/8 名
    high = _rank_state((100, 95, 90, 85, 60, 65, 60, 70))  # 对手一人降血，变第 5/8 名
    decisions = [decide(GameState.from_dict(s.update(frame)), rules)
                 for frame in (low, high, low, high, high)]
    assert [(d.action, d.keep_gold) for d in decisions] == [(EconAction.ROLL, 20)] * 4 + [(EconAction.SAVE, 40)]
    assert "血量第 6/8" in decisions[0].reason
    # 真实排名改善连续确认后，解除额外花钱，不把旧排名锁死。
    assert "血量第" not in decisions[-1].reason


@pytest.mark.parametrize("case", ["empty", "missing_hp", "no_self", "stale", "self_mismatch"])
def test_stabilizer_hp_rank_carries_unavailable_data_across_rounds(rules, case):
    s = StateStabilizer()
    before = decide(GameState.from_dict(s.update(_rank_state())), rules)
    frame = _rank_state()
    if case == "empty":
        frame["players"] = []
    elif case == "missing_hp":
        frame["players"][0]["hp"] = None
    elif case == "no_self":
        frame["players"][-1]["self"] = False
    elif case == "stale":
        frame["players_stale"] = True
    else:
        frame["players"][-1]["hp"] = 56
    for stage in ("4-3", "4-4", "4-4"):
        frame = {**frame, "stage": stage}
        after = decide(GameState.from_dict(s.update(frame)), rules)
        assert (after.action, after.keep_gold, after.roll_budget) == (before.action, before.keep_gold, before.roll_budget)
        assert "血量第 8/8" in after.reason
    # 保留排名，不把旧玩家列表伪装成当前画面。
    assert s.acc["players"] == frame["players"]


@pytest.mark.parametrize("gap", [False, True])
def test_stabilizer_hp_rank_resets_for_new_game(rules, gap):
    s = StateStabilizer()
    s.update(_rank_state())
    if gap:
        s.update(make_state(stage=None, phase="unknown", hp=None, players=[]))
    new = s.update(make_state(stage="1-2", level=None, gold=None, xp=None, hp=None, players=[]))
    assert new.get("hp_rank") is None
    frame = _rank_state()
    frame["players"] = []
    d = decide(GameState.from_dict(s.update(frame)), rules)
    assert (d.action, d.keep_gold) == (EconAction.SAVE, 40)


def test_stabilizer_hp_rank_confirms_survivor_count(rules):
    s = StateStabilizer()
    original = _rank_state()
    fewer = _rank_state((80, 70, 66, 62, 60, 58, 55))
    states = [GameState.from_dict(s.update(frame)) for frame in (original, fewer, fewer)]
    decisions = [decide(state, rules) for state in states]
    assert "血量第 8/8" in decisions[0].reason and "血量第 8/8" in decisions[1].reason
    assert "血量第 7/7" in decisions[2].reason
    assert [(d.action, d.keep_gold) for d in decisions] == [(EconAction.ROLL, 20)] * 3


def test_stabilizer_board_and_units():
    s = StateStabilizer(empty_frames=3)
    board = [{"name": "苍蓝哨戒", "star": 2, "x": 0.5, "y": 0.4}]
    bench = [{"slot": 0, "star": 1, "name": "奥恩"}]
    s.update(make_state(stage="2-2", board=board, bench=bench))
    # 横幅遮挡：棋盘读空两帧仍沿用
    assert s.update(make_state(stage="2-2", board=[], bench=bench))["board"] == board
    assert s.update(make_state(stage="2-2", board=[], bench=bench))["board"] == board
    assert s.update(make_state(stage="2-2", board=[], bench=bench))["board"] == []       # 连续 3 帧才认
    s.update(make_state(stage="2-2", board=board, bench=bench))
    # 同一位置这帧没认出名字/星级：沿用上一帧
    out = s.update(make_state(stage="2-2", board=[{"name": "?3", "star": None, "x": 0.51, "y": 0.41}],
                              bench=[{"slot": 0, "star": None, "name": "?1"}]))
    assert out["board"][0]["name"] == "苍蓝哨戒" and out["board"][0]["star"] == 2
    assert out["bench"][0]["name"] == "奥恩" and out["bench"][0]["star"] == 1


def test_stabilizer_shop_and_new_game():
    s = StateStabilizer()
    s.update(make_state(stage="2-1", shop=shop("奥恩", "绯红树怪", None, None, None)))
    out = s.update(make_state(stage="2-1", phase="combat", shop={"open": False}))
    assert out["shop"]["open"] and out["shop"]["slots"][1]["name"] == "绯红树怪"   # 同回合沿用商店
    out = s.update(make_state(stage="2-1", shop=shop("奥恩", None, None, None, None)))
    assert out["shop"]["slots"][1]["name"] == "绯红树怪"                          # 单帧漏读一格
    assert s.update(make_state(stage="2-2", phase="combat", shop={"open": False}))["shop"] == {"open": False}
    # 新的一局：不沿用上一局的金币
    out = s.update(make_state(stage="1-2", gold=None, level=None, xp=None))
    assert out["gold"] is None and out["level"] is None
    assert s.update(make_state(stage="1-3", xp=[9, 0]))["xp"] is None          # 不可能的经验读数


def test_stabilizer_keeps_augment_names():
    s = StateStabilizer()
    s.update(make_state(stage="2-1", phase="augment", augments=[{"name": "飞升"}, {"name": "存心失利"}]))
    out = s.update(make_state(stage="2-1", phase="augment", augments=[{"name": ""}, {"name": "存心失利"}]))
    assert [a["name"] for a in out["augments"]] == ["飞升", "存心失利"]


# ------------------------------------------------------------------ 买卡
def test_buys_skip_random_cards_when_board_full(db, rules):
    adv = HeuristicAdvisor(db, rules)
    # 1-3：场上 2/2 已满，商店是和羁绊无关的单卡 → 不推荐乱买；目标阵容卡照买
    st = GameState.from_dict(make_state(stage="1-3", level=2, gold=5, board_count=[2, 2],
                                        shop=shop("奥恩", "厄斐琉斯", "苍蓝哨戒", None, None)))
    assert [b.name for b in adv.advise(st).buys] == ["苍蓝哨戒"]
    # 场上有 1 个空位：补一张，优先羁绊吻合的
    st = GameState.from_dict(make_state(stage="1-3", level=2, gold=5, board_count=[1, 2],
                                        traits=[{"name": "永恒之森", "count": 1}],
                                        shop=shop("奥恩", None, None, None, None)))
    b = adv.advise(st).buys
    assert [x.name for x in b] == ["奥恩"] and "补场上空位" in b[0].reason and "永恒之森" in b[0].reason


def test_buys_skip_shop_items_and_pairs_need_synergy(db, rules):
    adv = HeuristicAdvisor(db, rules)
    st = GameState.from_dict(make_state(stage="2-2", level=4, gold=10, board_count=[4, 4],
                                        bench=[{"slot": 0, "star": 1, "name": "奥恩"}],
                                        shop=shop("铜汤匙", "奥恩", None, None, None)))
    assert adv.advise(st).buys == []           # 铜汤匙不是英雄；奥恩单张和羁绊无关
    st = GameState.from_dict(make_state(stage="2-2", level=4, gold=12, board_count=[4, 4],
                                        bench=[{"slot": 0, "star": 1, "name": "奥恩"}],
                                        traits=[{"name": "护卫", "count": 2, "active": True}],
                                        shop=shop("铜汤匙", "奥恩", None, None, None)))
    assert [b.name for b in adv.advise(st).buys] == ["奥恩"]


def test_losing_streak_skips_transition_buys_but_keeps_target_cards(db, rules):
    adv = HeuristicAdvisor(db, rules)
    state = make_state(stage="2-3", level=4, gold=19, xp=[0, 10], board_count=[3, 4],
                       traits=[{"name": "护卫", "count": 1}],
                       shop=shop("奥恩", "绯红树怪", None, None, None))
    assert "奥恩" in [b.name for b in adv.advise(GameState.from_dict(state)).buys]
    state["streak"] = {"kind": "loss", "n": 4}
    result = adv.advise(GameState.from_dict(state))
    assert result.econ.action.value == "save"
    assert [b.name for b in result.buys] == ["绯红树怪"]


# ------------------------------------------------------------------ 阵容
def test_comp_uses_traits_and_shop(db, rules):
    matcher = CompMatcher(db.comps_ranked())
    # 单位都没认出来，但羁绊栏有峡谷野怪 4：契合度仍然有
    st = GameState.from_dict(make_state(board=[{"name": "?1", "x": 0.5, "y": 0.5}] * 4,
                                        traits=[{"name": "峡谷野怪", "count": 4, "active": True}],
                                        shop=shop("绯红树怪", None, None, None, None)))
    m = matcher.match(st)[0]
    assert m.fit > 0.1
    assert any("羁绊 峡谷野怪4" in w for w in m.why) and any("商店有 绯红树怪" in w for w in m.why)


def test_early_headline_does_not_lock_comp(db, rules):
    adv = HeuristicAdvisor(db, rules)
    assert "前期按强度过渡" in adv.advise(GameState.from_dict(make_state(stage="1-3", level=2, gold=3))).headline
    assert adv.advise(GameState.from_dict(make_state(stage=None, phase="unknown"))).headline == "等待对局画面"


def test_resolve_augment_names(db, rules):
    adv = HeuristicAdvisor(db, rules)
    assert adv.resolve_augment("存心 失利") == "存心失利"
    assert adv.resolve_augment("存心失利 3") == "存心失利"
    assert adv.resolve_augment("", "飞升战斗后段增伤") == "飞升"
    assert adv.resolve_augment("", "飞并战斗后段增伤") == ""            # 太短的错字不硬匹配
    st = GameState.from_dict(make_state(phase="augment", augments=[{"name": "", "desc": "存心失利输了给钱"},
                                                                   {"name": "飞升 2"}]))
    names = [a.name for a in adv.advise(st, locked_comp="104").augments]
    assert names == ["存心失利", "飞升"]


# ------------------------------------------------------------------ 大模型提问时机
def test_llm_call_key():
    key = LLMClient.call_key
    assert key(GameState.from_dict(make_state(stage=None, phase="unknown"))) is None
    assert key(GameState.from_dict(make_state(gold=None))) is None
    planning = key(GameState.from_dict(make_state(stage="2-3")))
    assert planning == key(GameState.from_dict(make_state(stage="2-3", phase="combat")))   # 准备/战斗合并
    assert key(GameState.from_dict(make_state(stage="2-1", phase="augment",
                                              augments=[{"name": "飞升"}, {"name": ""}]))) is None


# ------------------------------------------------------------------ 辅助大脑模式
class ListVision(VisionProvider):
    name = "list"

    def __init__(self, states):
        self.states = list(states)

    def read(self, fresh=True):
        if not self.states:
            raise VisionError("回放结束")
        return GameState.from_dict(self.states.pop(0))


class FakeLLM:
    ready = True

    def __init__(self):
        self.calls = []
        self._key = None

    def should_call(self, st, adv):
        k = LLMClient.call_key(st, adv)
        return k is not None and k != self._key

    def advise(self, st, adv):
        self._key = LLMClient.call_key(st, adv)
        self.calls.append(st.stage)
        return LLMReply(f"{st.stage} 存钱", ["买奥恩"], "")


class Ctx:
    effective_lock_id = AppContext.effective_lock_id
    enabled_comp = AppContext.enabled_comp

    def __init__(self, db, rules, states):
        self.settings = Settings()
        self.game_lock_id = None
        self.game_watch_ids = []
        self.game_augments = []
        self.db = db
        self.advisor = HeuristicAdvisor(db, rules)
        self.vision = ListVision(states)
        self.llm = FakeLLM()
        self.extra_taken = {}


class AdviceSink(ModeSink):
    def __init__(self):
        self.advices = []

    def on_advice(self, advice):
        self.advices.append(advice)

    def on_log(self, text):
        pass


def test_assistant_pushes_stable_advice_and_keeps_llm_text(db, rules, monkeypatch):
    import threading

    # 大模型同步执行，便于断言
    monkeypatch.setattr(threading, "Thread", lambda target, args, daemon: type(
        "T", (), {"start": lambda self: target(*args)})())
    states = [make_state(stage="2-3", gold=20)] * 3 + [make_state(stage="2-3", gold=None, phase="combat")] * 2 \
        + [make_state(stage="2-4", gold=24)] * 2
    sink = AdviceSink()
    ctx = Ctx(db, rules, states)
    mode = AssistantMode(ctx, sink)
    try:
        mode.loop()
    except VisionError:
        pass
    assert ctx.llm.calls == ["2-3", "2-4"]                    # 每回合一次，战斗阶段不重问
    texts = [(a.stage, a.phase.value, a.llm_text) for a in sink.advices]
    # 不变准备帧保留缓存；阶段/可信上下文变化不把旧说明搬到新场景。
    assert any(t for s, p, t in texts if s == "2-3" and p == "planning")
    assert all(t is None for s, p, t in texts if s == "2-3" and p == "combat")
    assert all(t is None or t.startswith("2-4") for s, p, t in texts if s == "2-4")
    # 内容不变不重推：7 帧只推了阶段切换 + AI 回复
    assert len(sink.advices) <= 6
    active = [a for a in sink.advices if a.stage]
    assert all(a.econ and a.econ.action.value != "none" for a in active)  # 金币沿用
    assert sink.advices[-1].headline == '辅助已停止' and sink.advices[-1].econ is None


def test_mode_initializes_default_lock_on_first_frame(db, rules):
    from unittest.mock import Mock
    ctx = Ctx(db, rules, [])
    ctx.settings.assistant.default_lock_comp_id = '104'
    sink = ModeSink()
    sink.on_lock_changed = Mock()
    sink.on_log = Mock()
    mode = AssistantMode(ctx, sink)
    advice = mode.compute_advice(GameState.from_dict(make_state()))
    assert ctx.game_lock_id == '104' and advice.locked
    sink.on_lock_changed.assert_called_once_with()
    assert '开始辅助：锁定为 野怪小红' in sink.on_log.call_args.args[0]


def test_mode_does_not_override_explicit_unlock_on_first_frame(db, rules):
    ctx = Ctx(db, rules, [])
    ctx.settings.assistant.default_lock_comp_id = '104'
    ctx.game_lock_id = ''
    advice = AssistantMode(ctx, ModeSink()).compute_advice(GameState.from_dict(make_state()))
    assert ctx.game_lock_id == '' and not advice.locked


@pytest.mark.parametrize('default,manual', [('104', ''), ('', '104')])
def test_new_game_restores_default_and_resets_tracker(db, rules, default, manual):
    from unittest.mock import Mock
    ctx = Ctx(db, rules, [])
    ctx.settings.assistant.default_lock_comp_id = default
    ctx.game_lock_id = manual
    original = ctx.advisor.advise
    ctx.advisor.advise = Mock(wraps=original)
    sink = ModeSink()
    sink.on_lock_changed = Mock()
    sink.on_log = Mock()
    mode = AssistantMode(ctx, sink)
    for stage in ('2-5', '1-2', '1-2'):
        mode.compute_advice(GameState.from_dict(make_state(stage=stage)))
        assert ctx.game_lock_id == manual
    mode.tracker.reset = Mock(wraps=mode.tracker.reset)
    advice = mode.compute_advice(GameState.from_dict(make_state(stage='1-2')))
    assert ctx.game_lock_id == default and advice.locked == bool(default)
    assert ctx.advisor.advise.call_args.kwargs['tracker'] is mode.tracker
    mode.tracker.reset.assert_called_once_with()
    sink.on_lock_changed.assert_called_once_with()
    sink.on_log.assert_called_once()
    assert sink.on_log.call_args.args[0].startswith('新的一局：锁定已重置为 ')


@pytest.mark.parametrize('locked_id,locked', [('104', True), (None, False), ('missing', False)])
def test_advice_lock_flag_context_and_ai_switch_filter(db, rules, locked_id, locked):
    from jccbrain.engine.llm import build_context
    from jccbrain.webui.serialize import advice as serialize_advice
    state = GameState.from_dict(make_state())
    advice = HeuristicAdvisor(db, rules).advise(state, locked_id)
    assert advice.locked is locked and serialize_advice(advice)['locked'] is locked
    context = build_context(state, advice)
    assert context.get('locked_comp') == (advice.target.name if locked else None)
    if locked:
        assert advice.target.comp.id == locked_id and '本局锁定' in advice.target.why
    mode = AssistantMode(Ctx(db, rules, []), ModeSink())
    authorize_ai(mode, advice)
    mode._llm = (advice.stage, LLMReply('继续运营', [], '理由', switch_comp='另一套阵容'))
    mode._attach_llm(advice)
    assert any('AI 建议转' in note for note in advice.notes) is (not locked)
    assert advice.llm_text
