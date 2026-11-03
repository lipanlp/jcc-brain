import pytest

from jccbrain.actuator.actions import BenchToBoard, BuyShop, BuyXP, PickAugment, Reroll
from jccbrain.actuator.layout import Layout
from jccbrain.core.models import GameState
from jccbrain.engine import economy
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.comps import CompMatcher
from jccbrain.engine.llm import parse_reply
from jccbrain.engine.planner import AutoPlanner

from conftest import make_state


def shop(*names, cost=1):
    return {"open": True, "slots": [{"name": n, "cost": cost} if n else None for n in names]}


def test_comp_matcher_prefers_owned(db):
    st = GameState.from_dict(make_state(bench=[{"slot": 0, "star": 2, "name": "绯红树怪"}]))
    m = CompMatcher(db.comps_ranked()).match(st)[0]
    assert m.name == "野怪小红" and m.fit > 0
    assert "绯红树怪" in [h.name for h in m.owned]
    assert m.missing[0].is_subcarry or m.missing[0].is_carry or m.missing[0].is_core


def test_economy_level_on_curve(rules):
    st = GameState.from_dict(make_state(stage="3-2", level=5, xp=[16, 20], gold=30))
    d = economy.decide(st, rules, "fast8")
    assert d.action == economy.EconAction.LEVEL and d.xp_buys == 1


def test_economy_save_and_roll(rules):
    st = GameState.from_dict(make_state(stage="3-3", level=6, gold=34))
    assert economy.decide(st, rules, "fast8").action == economy.EconAction.SAVE
    st = GameState.from_dict(make_state(stage="4-2", level=8, xp=[0, 76], gold=60))
    assert economy.decide(st, rules, "fast8").action == economy.EconAction.ROLL
    st = GameState.from_dict(make_state(stage="4-5", level=7, xp=[0, 48], gold=10, hp=20))
    assert economy.decide(st, rules, "fast8").action == economy.EconAction.ALL_IN


def test_economy_reroll_rolls_at_level5(rules):
    st = GameState.from_dict(make_state(stage="3-1", level=5, xp=[0, 20], gold=60))
    d = economy.decide(st, rules, "reroll", carry_cost=1)
    assert d.action == economy.EconAction.ROLL and d.keep_gold == 50
    st = GameState.from_dict(make_state(stage="3-2", level=6, xp=[0, 36], gold=37))
    assert economy.decide(st, rules, "reroll", carry_cost=1).action == economy.EconAction.SAVE


@pytest.mark.parametrize("stage,level,gold,xp,buys,keep", [
    ("2-3", 4, 26, [2, 10], 2, 10),
    ("3-3", 6, 48, [20, 36], 4, 20),
])
def test_economy_winning_streak_levels_early(rules, stage, level, gold, xp, buys, keep):
    state = make_state(stage=stage, level=level, gold=gold, xp=xp)
    assert economy.decide(GameState.from_dict(state), rules).action == economy.EconAction.SAVE
    d = economy.decide(GameState.from_dict({**state, "streak": {"kind": "win", "n": 3}}), rules)
    assert d.action == economy.EconAction.LEVEL
    assert (d.xp_buys, d.keep_gold) == (buys, keep)
    assert "连胜 3" in d.reason and "保连胜" in d.reason


def test_economy_winning_streak_respects_budget_and_reroll_level(rules):
    state = GameState.from_dict(make_state(stage="2-3", level=4, gold=15, xp=[0, 10],
                                          streak={"kind": "win", "n": 4}))
    assert economy.decide(state, rules).action == economy.EconAction.SAVE
    # 一费 reroll 已到 5 级，连胜也不越过追三星的等级。
    state = GameState.from_dict(make_state(stage="3-3", level=5, gold=45, xp=[0, 20],
                                          streak={"kind": "win", "n": 4}))
    assert economy.decide(state, rules, "reroll", 1).action == economy.EconAction.SAVE


@pytest.mark.parametrize("stage,level,gold,xp,style", [
    ("2-5", 4, 26, [0, 10], "fast8"),
    ("3-2", 5, 35, [16, 20], "fast8"),
    ("3-1", 5, 65, [0, 20], "reroll"),
])
def test_economy_losing_streak_saves_instead_of_leveling_or_rolling(rules, stage, level, gold, xp, style):
    state = GameState.from_dict(make_state(stage=stage, level=level, gold=gold, xp=xp,
                                          streak={"kind": "loss", "n": 4}))
    d = economy.decide(state, rules, style, 1)
    assert d.action == economy.EconAction.SAVE
    assert (d.keep_gold, d.xp_buys, d.roll_budget) == (gold, 0, 0)
    assert "连败 4" in d.reason and "连败奖励" in d.reason


def test_economy_losing_streak_does_not_delay_low_hp_recovery(rules):
    state = make_state(stage="3-2", level=5, gold=35, xp=[16, 20], hp=50)
    baseline = economy.decide(GameState.from_dict(state), rules)
    assert baseline.action == economy.EconAction.LEVEL
    assert economy.decide(GameState.from_dict({**state, "streak": {"kind": "loss", "n": 4}}), rules) == baseline


@pytest.mark.parametrize("streak", [None, {}, {"kind": "win", "n": 2}, {"kind": "loss", "n": 2},
                                         {"kind": "loss"}, {"kind": "loss", "n": "4"},
                                         {"kind": "unknown", "n": 4}])
def test_economy_missing_or_short_streak_keeps_existing_behavior(rules, streak):
    state = GameState.from_dict(make_state(stage="3-3", level=6, gold=34, streak=streak))
    assert economy.decide(state, rules) == economy.EconDecision(
        economy.EconAction.SAVE, "存钱：再存 6 金币多 1 利息", keep_gold=30)


@pytest.mark.parametrize("stage,level,gold,xp", [
    ("1-3", 2, 5, [0, 2]), ("4-2", 8, 60, [0, 76]),
])
@pytest.mark.parametrize("kind", ["win", "loss"])
def test_economy_streak_only_changes_stage_two_and_three(rules, stage, level, gold, xp, kind):
    state = make_state(stage=stage, level=level, gold=gold, xp=xp)
    baseline = economy.decide(GameState.from_dict(state), rules)
    assert economy.decide(GameState.from_dict({**state, "streak": {"kind": kind, "n": 5}}), rules) == baseline


def _players():
    return [{"name": f"玩家{i}", "hp": hp, "self": i == 5}
            for i, hp in enumerate([100, 95, 90, 85, 80, 70, 65, 60])]


@pytest.mark.parametrize("stage,style,level,gold,xp,keep,budget", [
    ("4-1", "reroll", 5, 60, [0, 20], 30, 30),
    ("4-1", "fast8", 7, 30, [0, 48], 20, 10),
    ("4-2", "fast8", 8, 60, [0, 76], 0, 60),
])
def test_economy_low_hp_rank_spends_more_to_stabilize(rules, stage, style, level, gold, xp, keep, budget):
    state = make_state(stage=stage, level=level, gold=gold, xp=xp, hp=70)
    baseline = economy.decide(GameState.from_dict(state), rules, style, 1)
    # 打乱右侧列表仍按血量计算，不拿 self 所在行当名次。
    d = economy.decide(GameState.from_dict({**state, "players": list(reversed(_players()))}), rules, style, 1)
    assert d.action == economy.EconAction.ROLL
    assert (d.keep_gold, d.roll_budget) == (keep, budget)
    assert d.keep_gold < baseline.keep_gold
    assert "血量第 6/8" in d.reason and "稳血" in d.reason


def test_economy_low_hp_rank_keeps_level_curve_and_emergency_priority(rules):
    state = GameState.from_dict(make_state(stage="4-1", level=7, gold=30, xp=[44, 48], hp=70,
                                          players=_players()))
    d = economy.decide(state, rules)
    assert (d.action, d.xp_buys, d.keep_gold) == (economy.EconAction.LEVEL, 1, 20)
    assert "血量第 6/8" in d.reason
    players = _players()
    players[5]["hp"] = 20
    state = make_state(stage="4-2", level=7, gold=30, hp=20, streak={"kind": "loss", "n": 5})
    baseline = economy.decide(GameState.from_dict(state), rules)
    assert baseline.action == economy.EconAction.ALL_IN
    assert economy.decide(GameState.from_dict({**state, "players": players}), rules) == baseline


@pytest.mark.parametrize("case", ["missing", "too_many", "unknown_hp", "no_self", "two_self",
                                      "self_mismatch", "stale", "tied", "fifth"])
def test_economy_unreliable_or_healthy_rank_keeps_existing_behavior(rules, case):
    players = _players()
    if case == "missing":
        players = []
    elif case == "too_many":
        players.append({"hp": 80})
    elif case == "unknown_hp":
        players[0]["hp"] = None
    elif case == "no_self":
        players[5]["self"] = False
    elif case == "two_self":
        players[0]["self"] = True
    elif case == "self_mismatch":
        players[5]["hp"] = 71
    elif case == "tied":
        players = [{**p, "hp": 70} for p in players]
    elif case == "fifth":
        players[4]["hp"] = 70
    state = GameState.from_dict(make_state(stage="4-1", level=5, gold=60, xp=[0, 20], hp=70,
                                          players=players, players_stale=case == "stale"))
    assert economy.decide(state, rules, "reroll", 1) == economy.EconDecision(
        economy.EconAction.ROLL, "reroll 阵容：在 5 级慢 D，保留 50 金币", keep_gold=50, roll_budget=10)


def test_economy_hp_rank_only_changes_stage_four_onward(rules):
    state = make_state(stage="3-1", level=5, gold=60, xp=[0, 20], hp=70)
    baseline = economy.decide(GameState.from_dict(state), rules, "reroll", 1)
    assert economy.decide(GameState.from_dict({**state, "players": _players()}), rules, "reroll", 1) == baseline


@pytest.mark.parametrize("zero_rows", [False, True])
@pytest.mark.parametrize("healths,low_rank", [
    ([90, 80, 70, 66, 62, 60, 55], True),   # 7 人，最后一名
    ([90, 80, 70, 66, 55, 50, 40], True),   # 7 人，第 5 名，倒数三名边界
    ([90, 80, 70, 55, 50, 40, 35], False),  # 7 人，第 4 名
    ([90, 80, 55, 50], True),              # 4 人，第 3 名
    ([90, 55, 50], False),                 # 3 人，第 2 名
    ([90, 80, 55], True),                  # 3 人，最后一名
    ([90, 55], True),                      # 2 人，最后一名
    ([55, 50], False),                     # 2 人，领先者
    ([55], False),                         # 只剩自己
    ([55, 55, 55], False),                 # 并列领先
])
def test_economy_hp_rank_handles_eliminated_players(rules, zero_rows, healths, low_rank):
    # 每组中 55 血的自己故意放在中间/末尾，不能用行号作名次。
    own = healths.index(55)
    players = [{"name": f"玩家{i}", "hp": hp, "self": i == own} for i, hp in enumerate(healths)]
    if zero_rows:
        players += [{"name": f"出局{i}", "hp": 0} for i in range(8 - len(players))]
    state = GameState.from_dict(make_state(stage="4-3", level=7, gold=40, xp=[0, 48], hp=55, players=players))
    d = economy.decide(state, rules)
    expected = (economy.EconAction.ROLL, 20) if low_rank else (economy.EconAction.SAVE, 40)
    assert (d.action, d.keep_gold) == expected
    if low_rank:
        rank = 1 + sum(hp > 55 for hp in healths)
        assert f"血量第 {rank}/{len(healths)}" in d.reason


def test_advisor_buys_target_cards(db, rules):
    st = GameState.from_dict(make_state(gold=8, shop=shop("绯红树怪", "奥恩", None, "苍蓝哨戒", None)))
    adv = HeuristicAdvisor(db, rules).advise(st)
    bought = [b.name for b in adv.buys]
    assert "绯红树怪" in bought and "苍蓝哨戒" in bought
    assert bought.index("绯红树怪") < bought.index("苍蓝哨戒")      # 主 C 优先
    assert any(t.name == "绯红树怪" and t.need == 9 for t in adv.targets)
    assert adv.items["绯红树怪"] == ["无尽之刃"]


def test_advisor_rates_augments_by_comp_stats(db, rules):
    st = GameState.from_dict(make_state(phase="augment", augments=[
        {"name": "存心失利"}, {"name": "飞升"}, {"name": "没见过的海克斯"}]))
    adv = HeuristicAdvisor(db, rules).advise(st, locked_comp="104")
    best = max(adv.augments, key=lambda a: a.score)
    assert best.name == "飞升" and "飞升" in adv.headline


def test_advisor_empty_db(tmp_path, rules):
    from jccbrain.data.db import GameDB
    adv = HeuristicAdvisor(GameDB(tmp_path / "empty.sqlite3"), rules).advise(GameState.from_dict(make_state()))
    assert "同步" in adv.headline


def test_planner_steps(db, rules):
    adv = HeuristicAdvisor(db, rules)
    pl = AutoPlanner(db, Layout())
    # 1. 有目标卡先买
    st = GameState.from_dict(make_state(gold=10, shop=shop("绯红树怪", None, None, None, None)))
    assert pl.plan(st, adv.advise(st)) == [BuyShop(0)]
    # 2. 海克斯阶段选最高分
    st = GameState.from_dict(make_state(phase="augment", augments=[{"name": "存心失利"}, {"name": "飞升"}]))
    assert pl.plan(st, adv.advise(st, "104")) == [PickAugment(1)]
    # 3. 曲线落后先升级（2-1 应到 4 级，差 6 经验 = 买 2 次）
    st = GameState.from_dict(make_state(stage="2-1", level=3, xp=[0, 6], gold=30))
    assert pl.plan(st, adv.advise(st)) == [BuyXP(), BuyXP()]
    # 4. 人口没满时上阵
    st = GameState.from_dict(make_state(stage="3-3", level=6, gold=10, board_count=[5, 6],
                                        bench=[{"slot": 2, "star": 1, "name": "远古石甲虫"}]))
    act = pl.plan(st, adv.advise(st))
    assert len(act) == 1 and isinstance(act[0], BenchToBoard) and act[0].slot == 2
    assert (act[0].row, act[0].col) == (1, 5)                         # 阵容推荐站位
    # 5. D 牌点：刷新
    st = GameState.from_dict(make_state(stage="3-2", level=6, xp=[0, 36], gold=70,
                                        bench=[{"slot": 0, "star": 1, "name": "绯红树怪"}], board_count=[6, 6]))
    assert pl.plan(st, adv.advise(st, "104")) == [Reroll()]


def test_parse_llm_reply():
    r = parse_reply('好的```json\n{"headline":"D到3星","actions":["刷新"],"reason":"血量低","switch_comp":""}\n```')
    assert r.headline == "D到3星" and r.actions == ["刷新"]
    r = parse_reply("纯文本回答")
    assert r.headline == "纯文本回答" and r.actions == []


def _mock_llm(cfg, reply_content):
    import json as _json

    import httpx

    from jccbrain.engine.llm import LLMClient

    sent = []

    def handler(request):
        sent.append(_json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply_content, "reasoning_content": "想"}}]})

    client = LLMClient(cfg)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client, sent


def test_llm_request_effort_and_json_mode(db, rules):
    import pytest

    from jccbrain.core.settings import LLMSettings
    from jccbrain.engine.llm import LLMError

    cfg = LLMSettings(enabled=True, api_key="k", model="glm-5.3", reasoning_effort="low",
                      reasoning_effort_augment="high")
    client, sent = _mock_llm(cfg, '{"headline":"存钱","actions":["买奥恩"],"reason":"","switch_comp":""}')
    adv = HeuristicAdvisor(db, rules)
    st = GameState.from_dict(make_state())
    assert client.advise(st, adv.advise(st)).headline == "存钱"
    assert sent[-1]["reasoning_effort"] == "low" and sent[-1]["response_format"] == {"type": "json_object"}
    assert sent[-1]["max_tokens"] == 3000
    st = GameState.from_dict(make_state(phase="augment", augments=[{"name": "飞升"}]))
    client.advise(st, adv.advise(st))
    assert sent[-1]["reasoning_effort"] == "high"

    empty, _ = _mock_llm(cfg, "")
    with pytest.raises(LLMError):
        empty.advise(st, adv.advise(st))


def test_llm_presets():
    from jccbrain.core.settings import LLMSettings
    from jccbrain.engine.llm import PRESETS, apply_preset

    assert set(PRESETS) == {"fast", "balanced", "deep"}
    cfg = LLMSettings(api_key="k", base_url="https://x/v4", max_tokens=1000)
    apply_preset(cfg, "fast")
    assert (cfg.model, cfg.reasoning_effort, cfg.reasoning_effort_augment) == ("glm-5.3-flashx", "low", "low")
    apply_preset(cfg, "balanced")
    assert (cfg.model, cfg.reasoning_effort_augment, cfg.preset) == ("glm-5.3", "high", "balanced")
    assert cfg.api_key == "k" and cfg.base_url == "https://x/v4" and cfg.max_tokens == 3000
