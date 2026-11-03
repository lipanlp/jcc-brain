"""把 Advice 翻译成操作序列（全自动模式）。

每次只规划「一小步」：执行完后重新读画面、重新规划，这样任何一步失败（没买到、被挡住）
都能在下一轮被纠正，不会按过期状态连续误操作。
"""
from __future__ import annotations

from jccbrain.actuator.actions import Action, BenchToBoard, BuyShop, BuyXP, PickAugment, Reroll, SellBench
from jccbrain.actuator.layout import Layout
from jccbrain.core.models import GameState, Phase
from jccbrain.data.db import GameDB
from jccbrain.engine.advisor import Advice
from jccbrain.engine.economy import EconAction


class AutoPlanner:
    def __init__(self, db: GameDB, layout: Layout):
        self.db = db
        self.layout = layout

    def plan(self, state: GameState, advice: Advice) -> list[Action]:
        if state.phase == Phase.AUGMENT and advice.augments:
            best = max(advice.augments, key=lambda a: a.score)
            return [PickAugment(best.index)]
        if state.phase not in (Phase.PLANNING, Phase.COMBAT) or not state.shop_open:
            return []

        acts: list[Action] = []
        if advice.buys and state.bench_free == 0:
            acts += [SellBench(s.slot) for s in advice.sells[:1]]
        acts += [BuyShop(b.slot) for b in advice.buys]
        if acts:
            return acts  # 买卖完先重新读状态（合成、金币变化）再决定下一步

        econ = advice.econ
        gold = state.gold or 0
        if econ and econ.action == EconAction.LEVEL and econ.xp_buys > 0:
            xp_cost = state.xp_cost or 4
            n = min(econ.xp_buys, max((gold - econ.keep_gold) // xp_cost, 0))
            if n > 0:
                return [BuyXP()] * n

        if state.phase == Phase.PLANNING:
            field = self._field_unit(state, advice)
            if field:
                return [field]

        if econ and econ.action in (EconAction.ROLL, EconAction.ALL_IN):
            refresh = state.refresh_cost if state.refresh_cost is not None else 2
            if gold - refresh >= econ.keep_gold:
                return [Reroll()]
        return []

    def _field_unit(self, state: GameState, advice: Advice) -> BenchToBoard | None:
        """场上人数没满时，把备战席最有价值的单位放上去。"""
        if not state.bench or state.units_stale:
            return None
        cap = state.board_count[1] if state.board_count else state.level
        on_board = state.board_count[0] if state.board_count else len(state.board)
        if cap is None or on_board >= cap:
            return None

        wanted = {h.name: h for h in advice.target.comp.heroes} if advice.target else {}

        def value(u) -> float:
            hero = self.db.hero(u.name)
            cost = hero.cost if hero else 1
            return (5 if u.name in wanted else 0) + u.star * 2 + cost * 0.5

        unit = max((u for u in state.bench if u.known and u.slot is not None), key=value, default=None)
        if unit is None:
            return None
        occupied = {self.layout.nearest_hex(u.x, u.y) for u in state.board if u.x is not None and u.y is not None}
        pref = wanted[unit.name].position if unit.name in wanted else None
        hero = self.db.hero(unit.name)
        ranged = bool(hero and (hero.attack_range or 1) >= 3)
        cell = self.layout.free_hex(occupied, prefer=pref, back=ranged)
        return BenchToBoard(unit.slot, *cell) if cell else None
