"""布局、执行器、模式循环（都不碰真实屏幕和鼠标）。"""
import pytest

from jccbrain.actuator.actions import BuyShop, Reroll
from jccbrain.actuator.executor import DryRunExecutor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import AbortError, SafetyGuard
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.modes.base import ModeSink
from jccbrain.vision.base import VisionProvider

from conftest import make_state


# ------------------------------------------------------------------ 布局
def test_hex_corners_and_roundtrip(tmp_path):
    lay = Layout()
    for key, (r, c) in {"r1c1": (1, 1), "r1c7": (1, 7), "r4c1": (4, 1), "r4c7": (4, 7)}.items():
        x, y = lay.hex_center(r, c)
        assert (x, y) == pytest.approx(lay.board_corners[key])
    for r, c in lay.all_hexes():
        x, y = lay.hex_center(r, c)
        assert lay.nearest_hex(x, y - lay.bar_to_hex_dy) == (r, c)
    lay.bench_x0 = 0.3
    lay.save(tmp_path / "layout.json")
    assert Layout.load(tmp_path / "layout.json").bench_x0 == 0.3


def test_free_hex_preferences():
    lay = Layout()
    assert lay.free_hex(set(), prefer=(2, 3)) == (2, 3)
    assert lay.free_hex({(2, 3)}, prefer=(2, 3)) == (1, 4)
    assert lay.free_hex(set(), back=True) == (4, 4)
    assert lay.free_hex(set(lay.all_hexes())) is None


# ------------------------------------------------------------------ 执行器与安全
def test_dry_run_and_guard():
    guard = SafetyGuard(max_actions_per_sec=100, failsafe_corner=False)
    ex = DryRunExecutor(Layout(), guard, delay_s=0)
    ex.execute_all([BuyShop(0), Reroll()])
    assert ex.history == [BuyShop(0), Reroll()]
    guard.stop("测试")
    with pytest.raises(AbortError):
        ex.execute(Reroll())


@pytest.mark.parametrize("limit", [0, -1, 0.0, None, True])
def test_guard_invalid_rate_falls_back_to_default(monkeypatch, limit):
    from jccbrain.actuator import safety

    sleeps = []
    guard = SafetyGuard(max_actions_per_sec=limit, failsafe_corner=False)
    monkeypatch.setattr(guard, "sleep", sleeps.append)   # 限速等待走可被急停打断的 guard.sleep
    for _ in range(int(safety.DEFAULT_MAX_ACTIONS_PER_SEC)):
        guard.check()
    assert sleeps == []
    guard.check()                       # 超过默认限速才等待，不报错也不变成不限速
    assert len(sleeps) == 1 and sleeps[0] > 0


# ------------------------------------------------------------------ 模式
class ScriptedVision(VisionProvider):
    """按执行器收到的动作推进的假画面：刷新 → 下一个商店，买卡 → 进备战席。"""

    name = "scripted"

    def __init__(self, shops, gold=20):
        self.shops = list(shops)
        self.gold = gold
        self.bench = []

    def state(self) -> GameState:
        slots = [{"name": n, "cost": 1} if n else None for n in self.shops[0]]
        return GameState.from_dict(make_state(gold=self.gold, level=5, shop={"open": True, "slots": slots},
                                              bench=[{"slot": i, "star": 1, "name": n} for i, n in enumerate(self.bench)]))

    def read(self, fresh=True):
        return self.state()

    def on_action(self, action, real):
        if isinstance(action, Reroll):
            self.gold -= 2
            self.shops.pop(0)
            if not self.shops:
                self.shops = [[None] * 5]
        elif isinstance(action, BuyShop):
            name = self.shops[0][action.slot]
            self.bench.append(name)
            self.shops[0][action.slot] = None
            self.gold -= 1


class FakeCtx:
    def __init__(self, db, rules, vision):
        self.settings = Settings()
        self.db, self.rules, self.vision = db, rules, vision
        self.guard = SafetyGuard(100, failsafe_corner=False)
        self.executor = DryRunExecutor(Layout(), self.guard, delay_s=0)
        self.executor.listeners.append(vision.on_action)
        self.advisor = HeuristicAdvisor(db, rules)
        self.extra_taken = {}


class ListSink(ModeSink):
    def __init__(self):
        self.logs, self.actions = [], []

    def on_log(self, text):
        self.logs.append(text)

    def on_action(self, action, real):
        self.actions.append(action)


def test_autoroll_buys_until_target_hit(db, rules):
    vision = ScriptedVision([
        [None, "奥恩", None, None, None],
        ["绯红树怪", None, "绯红树怪", None, None],
        [None, None, None, "绯红树怪", None],
    ], gold=20)
    ctx = FakeCtx(db, rules, vision)
    sink = ListSink()
    mode = AutoRollMode(ctx, sink, [RollTarget("绯红树怪", 2)], keep_gold=0, max_rolls=10)
    reason = mode.loop()
    assert "凑齐" in reason
    assert vision.bench.count("绯红树怪") == 3 and "奥恩" not in vision.bench
    assert mode.rolls == 2 and mode.bought == 3


def test_autoroll_stops_at_keep_gold(db, rules):
    vision = ScriptedVision([[None] * 5] * 20, gold=9)
    ctx = FakeCtx(db, rules, vision)
    reason = AutoRollMode(ctx, ListSink(), [RollTarget("远古巨龙", 2)], keep_gold=4, max_rolls=50).loop()
    assert "保留线" in reason and vision.gold == 5
