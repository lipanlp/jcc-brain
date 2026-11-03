"""自动 D 牌的无进展保护与同批合成容量；不碰真实屏幕或鼠标。"""
import pytest

from jccbrain.actuator.actions import BuyShop, Reroll
from jccbrain.actuator.executor import DryRunExecutor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import SafetyGuard
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.modes.autoroll import (
    MAX_BUY_STALLS,
    MAX_GOLD_MISSES,
    MAX_REROLL_STALLS,
    AutoRollMode,
    RollTarget,
)
from jccbrain.modes.base import ModeSink
from jccbrain.vision.base import VisionProvider

from conftest import make_state

TARGET = "绯红树怪"
SECOND = "奥恩"


def snapshot(shop=(), *, gold=20, bench=(), board=()):
    return GameState.from_dict(make_state(
        gold=gold, level=5,
        shop={"open": True, "slots": [{"name": name, "cost": 1} if name else None for name in shop]},
        bench=[{"slot": i, "name": name, "star": star} for i, (name, star) in enumerate(bench)],
        board=[{"name": name, "star": star} for name, star in board],
    ))


def occupied(count):
    """未识别的其他单位仍占备战席，且不会与目标卡合成。"""
    return [(f"?{i}", 1) for i in range(count)]


class ScriptedVision(VisionProvider):
    """返回预设视觉帧，尾帧保持不变；忽略点击以构造丢失/延迟反馈。"""

    name = "scripted-autoroll"

    def __init__(self, frames):
        self.frames = list(frames)
        self.read_count = 0

    def read(self, fresh=True):
        self.read_count += 1
        assert self.read_count <= 64, "模式未在有限读取内停止"
        return self.frames[min(self.read_count - 1, len(self.frames) - 1)]


class FakeCtx:
    def __init__(self, db, rules, vision):
        self.settings = Settings()
        self.db, self.rules, self.vision = db, rules, vision
        self.guard = SafetyGuard(max_actions_per_sec=1000, failsafe_corner=False)
        self.executor = DryRunExecutor(Layout(), self.guard, delay_s=0)
        self.advisor = HeuristicAdvisor(db, rules)
        self.extra_taken = {}


@pytest.fixture
def make_mode(db, rules, monkeypatch):
    def build(frames, *, targets=None, max_rolls=10, config=None):
        vision = ScriptedVision(frames)
        ctx = FakeCtx(db, rules, vision)
        for key, value in (config or {}).items():
            setattr(ctx.settings.autoroll, key, value)
        mode = AutoRollMode(ctx, ModeSink(), targets or [RollTarget(TARGET, 2)],
                            keep_gold=0, max_rolls=max_rolls)
        # 这里只测控制流；实际概率报告由原有 runtime 测试和离线基准覆盖。
        monkeypatch.setattr(mode, "_report", lambda state, targets: None)
        return mode, vision
    return build


def test_ignored_buy_stops_after_bounded_batches(make_mode):
    st = snapshot([TARGET, TARGET], bench=[(TARGET, 1)])
    mode, vision = make_mode([st])

    assert mode.loop() == f"买卡未生效（连续 {MAX_BUY_STALLS} 次），已停止"
    assert mode.ctx.executor.history == [BuyShop(0), BuyShop(1)] * MAX_BUY_STALLS
    assert mode.bought == 0 and mode.rolls == 0
    assert vision.read_count == MAX_BUY_STALLS + 1


def test_unknown_gold_after_buy_is_not_payment(make_mode):
    known = snapshot([TARGET])
    unknown = snapshot([TARGET], gold=None)
    mode, _ = make_mode([known, unknown, known, unknown])

    assert "买卡未生效" in mode.loop()
    assert mode.ctx.executor.history == [BuyShop(0)] * MAX_BUY_STALLS
    assert mode.bought == 0


@pytest.mark.parametrize("evidence", ["gold", "owned"])
def test_effective_buy_resets_stalls_and_counts_only_effective_batch(make_mode, evidence):
    before = snapshot([TARGET])
    after = snapshot([TARGET], gold=19 if evidence == "gold" else 20,
                     bench=[(TARGET, 1)] if evidence == "owned" else [])
    frames = [before] * MAX_BUY_STALLS + [after] * (MAX_BUY_STALLS + 1)
    mode, _ = make_mode(frames)

    assert "买卡未生效" in mode.loop()
    assert len(mode.ctx.executor.history) == 2 * MAX_BUY_STALLS
    assert mode.bought == 1


def test_owned_progress_counts_even_when_new_gold_is_unknown(make_mode):
    before = snapshot([TARGET])
    after = snapshot(gold=None, bench=[(TARGET, 1)])
    mode, _ = make_mode([before, after], targets=[RollTarget(TARGET, 1)])

    assert "目标全部凑齐" in mode.loop()
    assert mode.bought == 1 and mode.rolls == 0


def test_unknown_gold_stops_after_bounded_reads(make_mode):
    mode, vision = make_mode([snapshot(gold=None)])

    assert mode.loop() == f"连续 {MAX_GOLD_MISSES} 帧读不到金币，已停止"
    assert vision.read_count <= MAX_GOLD_MISSES + 2
    assert mode.ctx.executor.history == []


def test_known_gold_resets_misses_even_when_it_enters_buy_branch(make_mode):
    unknown = snapshot([TARGET], gold=None)
    known = snapshot([TARGET])
    bought_but_unknown = snapshot(gold=None, bench=[(TARGET, 1)])
    frames = [unknown] * (MAX_GOLD_MISSES - 1) + [known]
    frames += [bought_but_unknown] * MAX_GOLD_MISSES
    mode, vision = make_mode(frames)

    assert "读不到金币" in mode.loop()
    assert vision.read_count == 2 * MAX_GOLD_MISSES
    assert mode.ctx.executor.history == [BuyShop(0)]
    assert mode.bought == 1


def test_ignored_reroll_stops_and_counts_sent_clicks(make_mode):
    mode, vision = make_mode([snapshot()])

    assert mode.loop() == f"刷新未生效（连续 {MAX_REROLL_STALLS} 次），已停止"
    assert mode.rolls == MAX_REROLL_STALLS
    assert mode.ctx.executor.history == [Reroll()] * MAX_REROLL_STALLS
    # 每次 _wait_new_shop 仍是首次读取加至多三次重读。
    assert vision.read_count == 1 + 4 * MAX_REROLL_STALLS


def test_unknown_gold_after_reroll_does_not_prove_progress(make_mode):
    known, unknown = snapshot(), snapshot(gold=None)
    frames = [known]
    for _ in range(MAX_REROLL_STALLS):
        frames += [unknown] * 4 + [known]
    mode, _ = make_mode(frames)

    assert "刷新未生效" in mode.loop()
    assert mode.rolls == MAX_REROLL_STALLS
    assert mode.ctx.executor.history == [Reroll()] * MAX_REROLL_STALLS


def test_same_shop_with_gold_spent_continues_to_roll_limit(make_mode):
    rolls = MAX_REROLL_STALLS + 1
    frames = [snapshot()]
    for i in range(1, rolls + 1):
        frames += [snapshot(gold=20 - i * 2)] * 4
    mode, _ = make_mode(frames, max_rolls=rolls)

    assert mode.loop() == f"达到最大刷新次数 {rolls}"
    assert mode.rolls == rolls
    assert mode.ctx.executor.history == [Reroll()] * rolls


@pytest.mark.parametrize("evidence", ["shop", "gold"])
def test_effective_reroll_resets_consecutive_stalls(make_mode, evidence):
    before = snapshot([SECOND])
    after = snapshot(["苍蓝哨戒"] if evidence == "shop" else [SECOND],
                     gold=18 if evidence == "gold" else 20)
    frames = [before] * (1 + 4 * (MAX_REROLL_STALLS - 1))
    frames += [after] * (1 if evidence == "shop" else 4)
    frames += [after] * (4 * MAX_REROLL_STALLS)
    mode, _ = make_mode(frames)

    assert "刷新未生效" in mode.loop()
    assert mode.rolls == 2 * MAX_REROLL_STALLS
    assert mode.ctx.executor.history == [Reroll()] * (2 * MAX_REROLL_STALLS)


def test_full_bench_merge_frees_room_for_second_target_in_same_batch(make_mode):
    st = snapshot([TARGET, SECOND], bench=[(TARGET, 1)] * 2 + occupied(7))
    targets = [RollTarget(TARGET, 2), RollTarget(SECOND, 2)]
    mode, _ = make_mode([st], targets=targets)

    assert st.bench_free == 0
    assert mode._buys(st, mode.need(st, targets)) == [BuyShop(0), BuyShop(1)]


@pytest.mark.parametrize("bench_ones", [0, 1])
def test_merge_with_board_copies_does_not_invent_bench_space(make_mode, bench_ones):
    st = snapshot([TARGET, SECOND],
                  bench=[(TARGET, 1)] * bench_ones + occupied(9 - bench_ones),
                  board=[(TARGET, 1)] * (2 - bench_ones))
    targets = [RollTarget(TARGET, 2), RollTarget(SECOND, 2)]
    mode, _ = make_mode([st], targets=targets)

    assert mode._buys(st, mode.need(st, targets)) == [BuyShop(0)]


def test_newly_bought_copy_participates_in_same_batch_merge(make_mode):
    st = snapshot([TARGET, TARGET, SECOND], bench=[(TARGET, 1)] + occupied(7))
    targets = [RollTarget(TARGET, 2), RollTarget(SECOND, 2)]
    mode, _ = make_mode([st], targets=targets)

    assert st.bench_free == 1
    assert mode._buys(st, mode.need(st, targets)) == [BuyShop(0), BuyShop(1), BuyShop(2)]


class ReportSink(ModeSink):
    def __init__(self):
        self.logs = []

    def on_log(self, text):
        self.logs.append(text)


@pytest.fixture
def make_reporting_mode(db, rules, monkeypatch):
    from jccbrain.modes import autoroll

    simulate = autoroll.simulate_rolldown

    def build(frames, *, max_rolls=10):
        vision = ScriptedVision(frames)
        sink = ReportSink()
        mode = AutoRollMode(FakeCtx(db, rules, vision), sink, [RollTarget(TARGET, 2)],
                            keep_gold=0, max_rolls=max_rolls)
        simulations = []

        def counted_simulate(*args, **kwargs):
            assert kwargs["sims"] == 600
            simulations.append((mode.rolls, dict(args[1])))
            return simulate(*args, **kwargs, seed=1)

        monkeypatch.setattr(autoroll, "simulate_rolldown", counted_simulate)
        return mode, sink, simulations

    return build


def test_nine_empty_refreshes_simulate_only_initially_and_at_five(make_reporting_mode, monkeypatch):
    from jccbrain.modes import autoroll

    frames = [snapshot(gold=50)]
    for roll in range(1, 10):
        # 商店始终为空但刷新确实扣金；原等待策略每次读取四帧。
        frames += [snapshot(gold=50 - 2 * roll)] * 4
    mode, sink, simulations = make_reporting_mode(frames, max_rolls=9)
    any_hit = autoroll.any_hit_per_roll
    hit_reports = []

    def counted_any_hit(*args, **kwargs):
        hit_reports.append(mode.rolls)
        return any_hit(*args, **kwargs)

    monkeypatch.setattr(autoroll, "any_hit_per_roll", counted_any_hit)

    assert mode.loop() == "达到最大刷新次数 9"
    assert mode.ctx.executor.history == [Reroll()] * 9
    assert mode.bought == 0
    # 初始第 0 次算一次，第 5 次算一次；尚未到第 10 次，共 1 + 9 // 5 == 2 次。
    assert simulations == [(0, {TARGET: 3}), (5, {TARGET: 3})]
    assert hit_reports == list(range(10))
    assert sink.logs[-1].endswith("（沿用第 5 次）")


def test_report_after_purchase_recalculates_changed_need(make_reporting_mode):
    frames = [snapshot(), snapshot([TARGET], gold=18),
              snapshot(gold=17, bench=[(TARGET, 1)]),
              snapshot([SECOND], gold=15, bench=[(TARGET, 1)])]
    mode, sink, simulations = make_reporting_mode(frames, max_rolls=2)

    assert mode.loop() == "达到最大刷新次数 2"
    assert mode.ctx.executor.history == [Reroll(), BuyShop(0), Reroll()]
    assert mode.bought == 1
    assert simulations == [(0, {TARGET: 3}), (2, {TARGET: 2})]
    assert "还缺 绯红树怪×2" in sink.logs[-1]
    assert "沿用第" not in sink.logs[-1]


def test_reused_report_marks_original_roll_even_when_rate_is_zero(make_reporting_mode):
    st = snapshot(gold=0)
    mode, sink, simulations = make_reporting_mode([st])

    mode._report(st, mode.targets)
    assert sink.logs[-1].endswith("全部凑齐 0%")
    mode.rolls = 1
    mode._report(st, mode.targets)

    assert simulations == [(0, {TARGET: 3})]
    assert sink.logs[-1].endswith("全部凑齐 0%（沿用第 0 次）")


def test_completed_targets_skip_report_and_simulation(make_reporting_mode, monkeypatch):
    st = snapshot(bench=[(TARGET, 2)])
    mode, sink, simulations = make_reporting_mode([st])

    def unexpected_pool(*args, **kwargs):
        pytest.fail("目标已凑齐时不应构建概率报告")

    monkeypatch.setattr(mode.ctx.advisor, "pool", unexpected_pool)
    mode._report(st, mode.targets)

    assert simulations == []
    assert sink.logs == []


@pytest.mark.parametrize("ending", ["complete", "stopped", "error"])
def test_autoroll_interval_is_active_during_reads_and_restored_on_exit(make_mode, monkeypatch, ending):
    mode, vision = make_mode([snapshot(bench=[(TARGET, 2)])], config={"read_interval_ms": 137})
    observed = []
    read = vision.read

    def checked_read(fresh=True):
        observed.append(vision.interval_override_s)
        if ending == "error":
            raise RuntimeError("视觉读取失败")
        if ending == "stopped":
            mode.stop_event.set()
        return read(fresh)

    monkeypatch.setattr(vision, "read", checked_read)
    assert vision.interval_override_s is None
    if ending == "error":
        with pytest.raises(RuntimeError, match="视觉读取失败"):
            mode.loop()
    else:
        reason = mode.loop()
        assert ("目标全部凑齐" in reason) if ending == "complete" else reason == "已停止"
    assert observed == pytest.approx([.137])
    assert vision.interval_override_s is None


def test_autoroll_interval_is_restored_when_actuator_aborts(make_mode, monkeypatch):
    from jccbrain.actuator.safety import AbortError

    mode, vision = make_mode([snapshot()])

    def abort(action):
        assert vision.interval_override_s == pytest.approx(.1)
        mode.ctx.guard.stop("测试急停")
        raise AbortError("测试急停")

    monkeypatch.setattr(mode.ctx.executor, "execute", abort)
    with pytest.raises(AbortError, match="测试急停"):
        mode.loop()
    assert mode.stopped
    assert vision.interval_override_s is None


def test_autoroll_restores_callers_interval_override(make_mode, monkeypatch):
    mode, vision = make_mode([snapshot(bench=[(TARGET, 2)])])
    read = vision.read

    def checked_read(fresh=True):
        assert vision.interval_override_s == pytest.approx(.1)
        return read(fresh)

    monkeypatch.setattr(vision, "read", checked_read)
    with vision.fast_frames(.4):
        assert "目标全部凑齐" in mode.loop()
        assert vision.interval_override_s == pytest.approx(.4)
    assert vision.interval_override_s is None


@pytest.mark.parametrize("item,still_unsure,expected_reads", [
    (False, False, 2), (False, True, 2), (True, True, 1),
])
def test_uncertain_shop_gets_at_most_one_extra_read_except_items(
        make_mode, item, still_unsure, expected_reads):
    first = snapshot([TARGET])
    first.shop[0].unsure = True
    first.shop[0].item = item
    confirmed = snapshot([TARGET])
    confirmed.shop[0].unsure = still_unsure
    mode, vision = make_mode([first, confirmed, snapshot([SECOND])])

    result = mode._wait_new_shop(snapshot([SECOND]).shop_names)

    assert vision.read_count == expected_reads
    assert result is (confirmed if expected_reads == 2 else first)


def test_stopping_during_uncertain_shop_read_skips_extra_read(make_mode, monkeypatch):
    st = snapshot([TARGET])
    st.shop[0].unsure = True
    mode, vision = make_mode([st])
    read = vision.read

    def stop_after_read(fresh=True):
        state = read(fresh)
        mode.stop_event.set()
        return state

    monkeypatch.setattr(vision, "read", stop_after_read)
    assert mode._wait_new_shop(snapshot([SECOND]).shop_names) is st
    assert vision.read_count == 1


def test_autoroll_read_interval_remains_in_validation_schema():
    from jccbrain.webui.settings_schema import _prepare, schema

    settings = Settings()
    section = next(s for s in schema(settings, full=True) if s["key"] == "autoroll")
    field = next(f for f in section["fields"] if f["key"] == "read_interval_ms")
    assert field == {"key": "read_interval_ms", "label": "D牌读画面间隔(ms)",
                     "type": "int", "choices": None, "value": 100}
    updated = _prepare(settings, {"autoroll": {"read_interval_ms": "175"}})
    assert updated.autoroll.read_interval_ms == 175
    assert settings.autoroll.read_interval_ms == 100


def test_removed_autoroll_option_is_ignored_in_legacy_settings(tmp_path):
    import json

    from jccbrain.webui.settings_schema import schema

    legacy_key = "stop_" + "on_hit"
    assert not hasattr(Settings().autoroll, legacy_key)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"autoroll": {legacy_key: False, "keep_gold": 3}}))

    settings = Settings.load(path)

    assert settings.autoroll.keep_gold == 3
    assert not hasattr(settings.autoroll, legacy_key)
    section = next(s for s in schema(settings, full=True) if s["key"] == "autoroll")
    assert legacy_key not in {f["key"] for f in section["fields"]}
    settings.save(path)
    assert legacy_key not in json.loads(path.read_text())["autoroll"]


@pytest.mark.parametrize("value,expected", [
    (True, 2), (False, 2), (None, 2), ("3", 2), (object(), 2),
    (-10, 1), (0, 1), (1, 1), (2, 2), (3, 3), (10, 3),
    (1.9, 1), (3.9, 3), (float("nan"), 2), (float("inf"), 2), (-float("inf"), 2),
])
def test_clamp_numeric_bounds_and_invalid_values(value, expected):
    from jccbrain.modes.autoroll import _clamp

    assert _clamp(value, 1, 3, 2) == expected


@pytest.mark.parametrize("source", ["settings", "arguments"])
@pytest.mark.parametrize("keep,rolls,expected_keep,expected_rolls", [
    (-1, -10, 0, 0), (5000, 10000, 100, 200),
    (True, False, 0, 30), ("3", "5", 0, 30), (None, None, 0, 30),
])
def test_constructor_clamps_settings_and_explicit_arguments(
        db, rules, source, keep, rolls, expected_keep, expected_rolls):
    ctx = FakeCtx(db, rules, ScriptedVision([snapshot()]))
    kwargs = {}
    if source == "settings":
        ctx.settings.autoroll.keep_gold = keep
        ctx.settings.autoroll.max_rolls = rolls
    else:
        kwargs = {"keep_gold": keep, "max_rolls": rolls}

    mode = AutoRollMode(ctx, ModeSink(), [RollTarget(TARGET, 2)], **kwargs)

    assert mode.keep_gold == expected_keep
    assert mode.max_rolls == expected_rolls


def test_explicit_constructor_values_override_invalid_settings(db, rules):
    ctx = FakeCtx(db, rules, ScriptedVision([snapshot()]))
    ctx.settings.autoroll.keep_gold = -1
    ctx.settings.autoroll.max_rolls = None

    mode = AutoRollMode(ctx, ModeSink(), keep_gold=7, max_rolls=9)

    assert (mode.keep_gold, mode.max_rolls) == (7, 9)
    assert ctx.settings.autoroll.keep_gold == -1
    assert ctx.settings.autoroll.max_rolls is None


def automatic_target_mode(make_mode, monkeypatch, setting_star, hero_star=None):
    from types import SimpleNamespace

    from jccbrain.data.models import CompHero

    mode, _ = make_mode([snapshot()], config={"target_star": setting_star})
    mode.targets = None
    hero = CompHero("test", TARGET, 1, is_carry=True, star_target=hero_star)
    advice = SimpleNamespace(target=SimpleNamespace(comp=SimpleNamespace(heroes=[hero])))
    monkeypatch.setattr(mode, "advise", lambda state: advice)
    return mode


@pytest.mark.parametrize("setting_star,expected", [
    (5, 3), (0, 1), (-1, 1), (None, 2), (True, 2), ("3", 2),
])
def test_automatic_target_star_uses_clamped_settings(make_mode, monkeypatch, setting_star, expected):
    mode = automatic_target_mode(make_mode, monkeypatch, setting_star)

    assert mode.resolve_targets(snapshot()) == [RollTarget(TARGET, expected)]


@pytest.mark.parametrize("hero_star,expected", [
    (5, 3), (0, 3), (-1, 3), (None, 3), (1, 1), (2, 2), (3, 3),
])
def test_comp_star_is_preserved_or_falls_back_to_clamped_setting(make_mode, monkeypatch, hero_star, expected):
    mode = automatic_target_mode(make_mode, monkeypatch, 5, hero_star)

    assert mode.resolve_targets(snapshot()) == [RollTarget(TARGET, expected)]


@pytest.mark.parametrize("star", [5, 0, -1, None, "2", 2.5])
def test_invalid_manual_target_stops_before_read_or_action(make_mode, star):
    mode, vision = make_mode([snapshot()], targets=[RollTarget(TARGET, star)])

    assert mode.loop() == f"目标星级无效：{TARGET}{star}星"
    assert mode.ctx.executor.history == []
    assert vision.read_count == 0
    assert vision.interval_override_s is None


@pytest.mark.parametrize("interval_ms,expected", [(-1, 0.), (5000, 1.)])
def test_read_interval_is_clamped_only_for_current_run(make_mode, monkeypatch, interval_ms, expected):
    mode, vision = make_mode([snapshot(bench=[(TARGET, 2)])], config={"read_interval_ms": interval_ms})
    read = vision.read

    def checked_read(fresh=True):
        assert vision.interval_override_s == expected
        return read(fresh)

    monkeypatch.setattr(vision, "read", checked_read)

    assert "目标全部凑齐" in mode.loop()
    assert vision.interval_override_s is None
    assert mode.ctx.settings.autoroll.read_interval_ms == interval_ms


@pytest.mark.parametrize("interval_ms", [None, 80, 0])
def test_cli_interval_override_is_temporary(monkeypatch, tmp_path, interval_ms):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from jccbrain import cli, modes

    path = tmp_path / "settings.json"
    settings = Settings()
    settings.autoroll.read_interval_ms = 137
    settings.save(path)
    saved = path.read_bytes()
    ctx = SimpleNamespace(settings=Settings.load(path))
    ctx.settings.save = Mock()
    mode = object()
    constructor = Mock(return_value=mode)
    runner = Mock(return_value=0)
    monkeypatch.setattr(cli, "_ctx", lambda args: ctx)
    monkeypatch.setattr(modes, "AutoRollMode", constructor)
    monkeypatch.setattr(cli, "_run_mode", runner)
    argv = ["autoroll", "--target", f"{TARGET}=3", "--keep", "7", "--max-rolls", "9"]
    if interval_ms is not None:
        argv += ["--read-interval-ms", str(interval_ms)]
    args = cli.build_parser().parse_args(argv)

    assert args.func(args) == 0

    assert ctx.settings.autoroll.read_interval_ms == (137 if interval_ms is None else interval_ms)
    assert constructor.call_args.args[0] is ctx
    assert constructor.call_args.args[2:] == ([RollTarget(TARGET, 3)], 7, 9)
    constructor.assert_called_once()
    runner.assert_called_once_with(ctx, mode)
    ctx.settings.save.assert_not_called()
    assert path.read_bytes() == saved
    assert Settings.load(path).autoroll.read_interval_ms == 137
