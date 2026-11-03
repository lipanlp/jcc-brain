"""离线运行真实 AutoRollMode，测量墙钟耗时并独立审计买卡决策。

    PYTHONPATH=src python scripts/bench_autoroll.py --runs 30 --seed 1
    PYTHONPATH=src python scripts/bench_autoroll.py --lag-frames 1 --json var/lag1.json

只使用假画面和 DryRunExecutor；原数据库以 mode=ro 打开，SQLite backup 到
本工作树 var/ 内的临时目录，GameDB 仅打开副本。初始化不计时；初始读帧/报告、
买卡后的读取、限速和 oracle 都计入 mode.loop 墙钟。pool 是 _report 的子项，
不可重复相加。帧间隔从上次读取完成时计算，与 PyEyeProvider.read 一致。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jccbrain.actuator.actions import BuyShop, Reroll
from jccbrain.actuator.executor import DryRunExecutor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import SafetyGuard
from jccbrain.core.models import GameState
from jccbrain.core.rules import load_rules
from jccbrain.core.settings import Settings
from jccbrain.data.db import GameDB
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.odds import PoolState
from jccbrain.modes import autoroll
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.modes.base import ModeSink
from jccbrain.vision.base import VisionProvider


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def database_snapshot(path: Path, season_id: str):
    """backup 同时包含已提交的 WAL；不调用原库的 GameDB 初始化/迁移。"""
    path = path.resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
        source.execute("PRAGMA query_only = ON")
        source.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        runtime = ROOT / "var"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bench-autoroll-", dir=runtime) as tmp:
            copy = Path(tmp) / "snapshot.sqlite3"
            with closing(sqlite3.connect(copy)) as destination:
                source.backup(destination)
            snapshot_hash = sha256(copy)
            db = GameDB(copy, season_id=season_id)
            try:
                yield db, snapshot_hash
            finally:
                db.close()


class FakeVision(VisionProvider):
    """真实商店立即刷新，视觉可延迟若干次 read 才显示；没有截图接口。"""

    def __init__(self, db, rules, rng, targets, read_ms, frame_ms, lag_frames):
        self.db, self.rules, self.rng = db, rules, rng
        self.level, self.gold = 7, 50
        self.read_s, self.frame_s = read_ms / 1000, frame_ms / 1000
        self.lag_frames = lag_frames
        self.bench = [(target.name, 1) for target in targets]
        self.taken = Counter(target.name for target in targets)
        self.shop = self._draw_shop()
        self.visible_shop = list(self.shop)
        self.lag_left = 0
        self.last_state = None
        self.last_read_at = 0.0
        self.reads = self.refreshes = self.purchases = self.rejected_actions = 0

    def _draw_shop(self):
        pool = PoolState.from_db(self.db, self.rules, self.taken)
        names_by_cost = {cost: sorted(n for n, h in pool.heroes.items() if h.cost == cost)
                         for cost in range(1, 6)}
        slots = []
        for _ in range(5):
            cost = self.rng.choices(range(1, 6), weights=self.rules.odds(self.level))[0]
            names = names_by_cost[cost]
            weights = [pool.remaining(n) for n in names]
            if not sum(weights):
                slots.append(None)
                continue
            name = self.rng.choices(names, weights=weights)[0]
            slots.append(name)
            # 同一商店内抽样不放回；下一次刷新归还未买的卡。
            pool.taken[name] += 1
        return slots

    def state(self):
        return GameState.from_dict({
            "phase": "planning", "stage": "4-1", "level": self.level, "gold": self.gold,
            "refresh_cost": 2, "odds": [round(p * 100) for p in self.rules.odds(self.level)],
            "shop": {"open": True, "slots": [
                {"name": n, "cost": self.db.hero(n).cost} if n else None for n in self.visible_shop]},
            "bench": [{"slot": i, "name": n, "star": star}
                      for i, (n, star) in enumerate(self.bench)],
            "board": [], "_frame": self.reads,
        })

    def read(self, fresh=True):
        self.reads += 1
        if self.reads > 1000:
            raise RuntimeError("单局超过 1000 次 read，基准中止：可能出现无进展循环")
        if not fresh and self.last_state and time.monotonic() - self.last_read_at < self.frame_s:
            return self.last_state
        interval = self.interval_override_s if self.interval_override_s is not None else self.frame_s
        wait = interval - (time.monotonic() - self.last_read_at)
        if wait > 0:
            time.sleep(wait)
        time.sleep(self.read_s)
        if self.lag_left:
            self.lag_left -= 1
        else:
            self.visible_shop = list(self.shop)
        self.last_state = self.state()
        self.last_read_at = time.monotonic()
        return self.last_state

    def on_action(self, action, real):
        if real:
            raise RuntimeError("离线基准禁止真实执行器")
        if isinstance(action, Reroll):
            if self.gold < 2:
                self.rejected_actions += 1
                return
            self.gold -= 2
            self.shop = self._draw_shop()
            self.lag_left = self.lag_frames
            self.refreshes += 1
        elif isinstance(action, BuyShop):
            name = self.shop[action.slot]
            cost = self.db.hero(name).cost if name else 0
            merges = self.bench.count((name, 1)) >= 2
            if not name or self.gold < cost or (len(self.bench) >= 9 and not merges):
                self.rejected_actions += 1
                return
            self.gold -= cost
            self.taken[name] += 1
            self.purchases += 1
            self.shop[action.slot] = None
            self.bench.append((name, 1))
            for star in range(1, 4):
                while self.bench.count((name, star)) >= 3:
                    for _ in range(3):
                        self.bench.remove((name, star))
                    self.bench.append((name, star + 1))
        else:
            raise RuntimeError(f"基准收到未支持的动作：{action}")


class OracleInventory:
    """仅按 GameState 的实体单位和金币重建容量，不调用模式的 need/_buys。"""

    def __init__(self, state):
        self.gold = state.gold or 0
        self.units = [(u.name, u.star, "board") for u in state.board]
        self.units += [(u.name, u.star, "bench") for u in state.bench]

    def copies(self, name):
        return sum(3 ** (star - 1) for n, star, _ in self.units if n == name)

    def buy(self, name, cost):
        ones = sum(n == name and star == 1 for n, star, _ in self.units)
        free = 9 - sum(zone == "bench" for _, _, zone in self.units)
        if self.gold < cost or (free <= 0 and ones < 2):
            return False
        self.gold -= cost
        self.units.append((name, 1, "bench"))
        for star in range(1, 4):
            matching = [i for i, (n, s, _) in enumerate(self.units) if n == name and s == star]
            while len(matching) >= 3:
                used = matching[:3]
                zone = "board" if any(self.units[i][2] == "board" for i in used) else "bench"
                self.units = [unit for i, unit in enumerate(self.units) if i not in used]
                self.units.append((name, star + 1, zone))
                matching = [i for i, (n, s, _) in enumerate(self.units) if n == name and s == star]
        return True


def audit_buys(state, targets, actions, db):
    """参考策略按商店从左到右逐笔扣金/合成，按名字比张数，允许等价同名格。

    不信任传给 _buys 的 need，也不使用 owned_copies/bench_free 或假画面内部状态。
    invalid 另计空格、重复格、金币/容量不足等非法动作，不混入“非目标/已够”的误买。
    """
    goals = {t.name: 3 ** (t.star - 1) for t in targets}
    reference, actual = OracleInventory(state), OracleInventory(state)
    expected, bought = Counter(), Counter()
    for slot in state.shop:
        if slot and reference.copies(slot.name) < goals.get(slot.name, 0):
            if reference.buy(slot.name, db.hero(slot.name).cost):
                expected[slot.name] += 1
    wrong = invalid = 0
    seen = set()
    for action in actions:
        if not isinstance(action, BuyShop) or not 0 <= action.slot < len(state.shop):
            invalid += 1
            continue
        slot = state.shop[action.slot]
        if action.slot in seen or slot is None:
            invalid += 1
            continue
        seen.add(action.slot)
        needed = actual.copies(slot.name) < goals.get(slot.name, 0)
        wrong += not needed
        if actual.buy(slot.name, db.hero(slot.name).cost):
            if needed:
                bought[slot.name] += 1
        else:
            invalid += 1
    return {"checks": 1, "wrong": wrong, "missed": sum((expected - bought).values()),
            "expected": sum(expected.values()), "invalid": invalid}


@dataclass
class Timing:
    calls: int = 0
    seconds: float = 0.0

    def wrap(self, fn):
        @wraps(fn)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.calls += 1
                self.seconds += time.perf_counter() - started
        return measured


def end_category(reason):
    for prefix in ("目标全部凑齐", "金币到保留线", "达到最大刷新次数", "离开可操作阶段", "已停止"):
        if reason.startswith(prefix):
            return prefix
    return reason


def run_game(db, rules, settings, args, index):
    game_seed = args.seed + index
    rng = random.Random(game_seed)
    candidates = sorted(h.name for h in db.pool_heroes() if h.cost in (3, 4))
    if len(candidates) < 2:
        raise ValueError(f"赛季 {db.season_id} 的 3~4 费卡池英雄不足两名")
    targets = [RollTarget(name, 2) for name in rng.sample(candidates, 2)]
    vision = FakeVision(db, rules, rng, targets, args.read_ms, args.frame_ms, args.lag_frames)
    guard = SafetyGuard(max_actions_per_sec=6, failsafe_corner=False)
    executor = DryRunExecutor(Layout(), guard, delay_s=0.06)
    executor.listeners.append(vision.on_action)
    advisor = HeuristicAdvisor(db, rules)
    ctx = SimpleNamespace(settings=settings, db=db, rules=rules, vision=vision,
                          guard=guard, executor=executor, advisor=advisor, extra_taken={})
    mode = AutoRollMode(ctx, ModeSink(), targets=targets, keep_gold=0, max_rolls=30)
    timings = {key: Timing() for key in ("read", "execute", "_report", "pool", "oracle")}
    oracle_counts = Counter(checks=0, wrong=0, missed=0, expected=0, invalid=0)
    examples = []
    original_buys = mode._buys
    simulate = autoroll.simulate_rolldown
    # 报告有自己的随机流，不会因报告降频而改变目标/商店序列。
    report_rng = random.Random(f"autoroll-report:{game_seed}")

    def seeded_simulate(*a, **kw):
        if kw.get("seed") is None:
            kw["seed"] = report_rng.getrandbits(64)
        return simulate(*a, **kw)

    def checked_buys(state, need):
        actions = original_buys(state, need)
        result = timings["oracle"].wrap(audit_buys)(state, targets, actions, db)
        oracle_counts.update(result)
        if (result["wrong"] or result["missed"] or result["invalid"]) and len(examples) < 5:
            examples.append({"shop": state.shop_names, "gold": state.gold,
                             "actions": [repr(a) for a in actions], **result})
        return actions

    with ExitStack() as stack:
        for obj, method, key in ((vision, "read", "read"), (executor, "execute", "execute"),
                                 (mode, "_report", "_report"), (advisor, "pool", "pool")):
            stack.enter_context(patch.object(obj, method, timings[key].wrap(getattr(obj, method))))
        stack.enter_context(patch.object(mode, "_buys", checked_buys))
        stack.enter_context(patch.object(autoroll, "simulate_rolldown", seeded_simulate))
        started = time.perf_counter()
        reason = mode.loop()
        elapsed = time.perf_counter() - started
    return {"run": index + 1, "seed": game_seed, "targets": [asdict(t) for t in targets],
            "rolls": mode.rolls, "bought": mode.bought, "actual_refreshes": vision.refreshes,
            "actual_purchases": vision.purchases, "rejected_actions": vision.rejected_actions,
            "reason": reason, "end_category": end_category(reason), "gold_left": vision.gold,
            "wall_seconds": elapsed, "timings": {k: asdict(v) for k, v in timings.items()},
            "oracle": dict(oracle_counts), "oracle_examples": examples}


def summarize(games):
    rolls = sum(g["rolls"] for g in games)
    wall = sum(g["wall_seconds"] for g in games)
    timings = {}
    for key in games[0]["timings"]:
        seconds = sum(g["timings"][key]["seconds"] for g in games)
        calls = sum(g["timings"][key]["calls"] for g in games)
        timings[key] = {"calls": calls, "seconds": seconds,
                        "ms_per_roll": seconds * 1000 / rolls if rolls else None,
                        "wall_percent": 100 * seconds / wall if wall else 0}
    counts = Counter()
    for g in games:
        counts.update(g["oracle"])
    return {"runs": len(games), "total_rolls": rolls, "wall_seconds": wall,
            "mean_ms_per_roll": wall * 1000 / rolls if rolls else None,
            "reads_per_roll": timings["read"]["calls"] / rolls if rolls else None,
            "end_reasons": dict(sorted(Counter(g["end_category"] for g in games).items())),
            "timings": timings, "oracle": dict(counts),
            "rejected_actions": sum(g["rejected_actions"] for g in games)}


def print_summary(summary):
    def number(value):
        return "N/A（无刷新）" if value is None else f"{value:.3f}"

    print(f"局数：{summary['runs']}；总刷新次数：{summary['total_rolls']}")
    print("结束原因分布：" + json.dumps(summary["end_reasons"], ensure_ascii=False, sort_keys=True))
    print(f"总墙钟耗时：{summary['wall_seconds']:.3f} s")
    print(f"每次刷新平均耗时：{number(summary['mean_ms_per_roll'])} ms（总墙钟 / 总刷新次数）")
    for key, item in summary["timings"].items():
        print(f"  {key}: {item['calls']} 次，{number(item['ms_per_roll'])} ms/刷新，"
              f"占墙钟 {item['wall_percent']:.3f}%")
    print("  pool 包含在 _report 中，不重复相加；execute 含限速、60 ms 延迟和假画面推进。")
    print(f"每次刷新平均 read 次数：{number(summary['reads_per_roll'])}")
    audit = summary["oracle"]
    print(f"误买：{audit['wrong']}；漏买：{audit['missed']}；应买：{audit['expected']}；"
          f"非法决策：{audit['invalid']}；拒绝执行：{summary['rejected_actions']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--read-ms", type=float, default=150)
    parser.add_argument("--frame-ms", type=float, default=250)
    parser.add_argument("--autoroll-interval-ms", type=int, default=Settings().autoroll.read_interval_ms)
    parser.add_argument("--lag-frames", type=int, default=0)
    parser.add_argument("--db", type=Path, default=Path("var/jcc.sqlite3"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.lag_frames < 0:
        parser.error("--runs 必须大于 0；--lag-frames 不能为负")
    if args.autoroll_interval_ms < 0:
        parser.error("--autoroll-interval-ms 不能为负")
    if any(not math.isfinite(v) or v < 0 for v in (args.read_ms, args.frame_ms)):
        parser.error("--read-ms 和 --frame-ms 必须是非负有限数")
    if not args.db.is_file():
        parser.error(f"数据库不存在：{args.db}；不会自动创建或同步数据库")
    if args.json and (args.json.resolve() == args.db.resolve()
                      or (args.json.exists() and args.json.samefile(args.db))):
        parser.error("--json 不能覆盖输入数据库")
    settings = Settings()
    settings.autoroll.read_interval_ms = args.autoroll_interval_ms
    rules = load_rules(settings.data.season_id)
    original_hash = sha256(args.db)
    with database_snapshot(args.db, settings.data.season_id) as (db, snapshot_hash):
        print(f"离线基准：赛季 {db.season_id}，seed={args.seed}，lag={args.lag_frames}，"
              f"read={args.read_ms:g} ms，frame={args.frame_ms:g} ms，"
              f"autoroll_interval={args.autoroll_interval_ms} ms", flush=True)
        games = []
        for index in range(args.runs):
            game = run_game(db, rules, settings, args, index)
            games.append(game)
            print(f"  [{index + 1}/{args.runs}] {game['reason']}，{game['wall_seconds']:.3f} s", flush=True)
        result = {"config": {**vars(args), "db": str(args.db.resolve()),
                              "json": str(args.json) if args.json else None,
                              "season_id": db.season_id, "level": 7, "gold": 50,
                              "keep_gold": 0, "max_rolls": 30, "click_delay_ms": 60,
                              "max_actions_per_sec": 6, "report_sims": 600},
                  "provenance": {"python": platform.python_version(), "platform": platform.platform(),
                                 "db_sha256": original_hash, "snapshot_sha256": snapshot_hash,
                                 "game_version": db.game_version, "synced_at": db.synced_at,
                                 "pool_counts": db.pool_count_by_cost(),
                                 "autoroll_path": str(Path(autoroll.__file__).resolve()),
                                 "autoroll_sha256": sha256(Path(autoroll.__file__)),
                                 "script_sha256": sha256(Path(__file__))},
                  "summary": summarize(games), "games": games}
    result["provenance"]["db_unchanged"] = sha256(args.db) == original_hash
    if not result["provenance"]["db_unchanged"]:
        raise RuntimeError("运行期间输入数据库发生变化，本次结果不可作为固定快照基线")
    print_summary(result["summary"])
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
