"""自动 D 牌（半自动）。

逻辑：
1. 目标卡：用户指定，或取目标阵容里需要升星的卡（主 C / 追三星的卡优先）。
2. 每轮：读画面 → 商店里有还缺的目标卡就买 → 没有就刷新 → 等商店换了再读。
3. 停止：没有目标 / 目标星级无效 / 全部凑齐（固定行为）/ 金币到保留线 /
   达到最大刷新次数 / 离开准备或战斗阶段 / 商店关闭 / 急停。
4. 无进展保护：连续 2 批买卡未生效 / 连续 5 帧读不到金币 / 连续 3 次刷新未生效。
5. 每次刷新报告单次命中率；蒙特卡洛在缺口变化或每 5 次刷新时重算。

AutoRollSettings 参数：
- keep_gold：保留金币，默认 0，运行时夹到 0～100。
- target_star：自动目标的默认星级，默认 2，运行时夹到 1～3；有效阵容星级优先。
- max_rolls：最多刷新次数，默认 30，运行时夹到 0～200。
- buy_pairs_only：默认 False；尚未持有的目标仅在商店能凑对时购买。
- read_interval_ms：D 牌期间读帧间隔，默认 100 ms，运行时夹到 0～1000 ms，结束后恢复。
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

from jccbrain.actuator.actions import BuyShop, Reroll
from jccbrain.core.models import COPIES_PER_STAR, GameState, Phase
from jccbrain.engine.odds import any_hit_per_roll, simulate_rolldown
from jccbrain.modes.base import Mode


MAX_BUY_STALLS = 2
MAX_GOLD_MISSES = 5
MAX_REROLL_STALLS = 3
REPORT_SIM_EVERY = 5


def _clamp(value, lo, hi, default):
    """数字先取整再夹紧；布尔值、非数字及无法取整的值使用默认值。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return default
    try:
        value = int(value)
    except (ValueError, OverflowError):
        return default
    return max(lo, min(hi, value))


@dataclass
class RollTarget:
    name: str
    star: int = 2           # 要升到几星


class AutoRollMode(Mode):
    key = "autoroll"
    title = "自动 D 牌"
    acts = True

    def __init__(self, ctx, sink, targets: list[RollTarget] | None = None,
                 keep_gold: int | None = None, max_rolls: int | None = None):
        super().__init__(ctx, sink)
        cfg = ctx.settings.autoroll
        self._cfg = deepcopy(cfg)
        self._refresh_cost = deepcopy(ctx.rules.refresh_cost)
        self.targets = deepcopy(targets)
        self._resolved_targets = None
        self.keep_gold = _clamp(cfg.keep_gold if keep_gold is None else keep_gold, 0, 100, 0)
        self.max_rolls = _clamp(cfg.max_rolls if max_rolls is None else max_rolls, 0, 200, 30)
        self.rolls = 0
        self.bought = 0
        self._sim_rate: float | None = None
        self._sim_need: dict[str, int] | None = None
        self._sim_roll = -1

    # ------------------------------------------------------------ 目标
    def resolve_targets(self, state: GameState) -> list[RollTarget]:
        if self._resolved_targets is not None:
            return deepcopy(self._resolved_targets)
        if self.targets is not None:
            self._resolved_targets = deepcopy(self.targets)
            return deepcopy(self._resolved_targets)
        adv = self.advise(state)
        if not adv.target:
            self._resolved_targets = []
            return []
        star = _clamp(self._cfg.target_star, 1, 3, 2)
        heroes = adv.target.comp.heroes
        # 优先：阵容标注要追星的卡 + 主副 C；都没有时取全部核心卡
        picked = [h for h in heroes if h.star_target or h.is_carry or h.is_subcarry] or \
                 [h for h in heroes if h.is_core]
        self._resolved_targets = [RollTarget(h.name, h.star_target if h.star_target in COPIES_PER_STAR else star)
                                  for h in picked]
        return deepcopy(self._resolved_targets)

    @staticmethod
    def need(state: GameState, targets: list[RollTarget]) -> dict[str, int]:
        owned = state.owned_copies()
        return {t.name: COPIES_PER_STAR[t.star] - owned.get(t.name, 0)
                for t in targets if COPIES_PER_STAR[t.star] - owned.get(t.name, 0) > 0}

    # ------------------------------------------------------------ 主循环
    def loop(self) -> str:
        for target in self.targets or []:
            if target.star not in COPIES_PER_STAR:
                return f"目标星级无效：{target.name}{target.star}星"
        cfg = self._cfg
        interval_ms = _clamp(cfg.read_interval_ms, 0, 1000, 100)
        with self.ctx.vision.fast_frames(interval_ms / 1000):
            return self._loop()

    def _loop(self) -> str:
        st = self.read()
        targets = self.resolve_targets(st)
        if not targets:
            return "没有目标卡（请导入当前建议阵容，或手动指定）"
        self.sink.on_log("D 牌目标：" + "、".join(f"{t.name}{t.star}星" for t in targets)
                         + f"，保留 {self.keep_gold} 金，最多刷新 {self.max_rolls} 次")
        self.sink.on_mode_changed()
        self._report(st, targets)

        buy_stalls = gold_misses = reroll_stalls = 0
        while not self.stopped:
            if st.phase not in (Phase.PLANNING, Phase.COMBAT) or not st.shop_open:
                return f"离开可操作阶段（{st.phase.value}）"
            if st.gold is not None and not self._money(st.gold):
                return "金币读数无效，已停止"
            if self._money(st.gold):
                gold_misses = 0
            need = self.need(st, targets)
            if not need:
                return f"目标全部凑齐，共刷新 {self.rolls} 次、买 {self.bought} 张"

            for slot in st.shop:
                if slot and need.get(slot.name, 0) and self._cost(slot) is None:
                    return f"目标卡成本未知或无效：{slot.name}，已停止"
            buys = self._buys(st, need)
            if buys:
                before_gold = st.gold
                self.act(buys)
                st = self.read()
                gold_decreased = (before_gold is not None and st.gold is not None
                                  and st.gold < before_gold)
                if sum(self.need(st, targets).values()) < sum(need.values()) or gold_decreased:
                    buy_stalls = 0
                    self.bought += len(buys)
                else:
                    buy_stalls += 1
                    if buy_stalls >= MAX_BUY_STALLS:
                        return f"买卡未生效（连续 {MAX_BUY_STALLS} 次），已停止"
                self.sink.on_mode_changed()
                continue

            refresh = st.refresh_cost if st.refresh_cost is not None else self._refresh_cost
            if st.gold is None:
                gold_misses += 1
                if gold_misses >= MAX_GOLD_MISSES:
                    return f"连续 {MAX_GOLD_MISSES} 帧读不到金币，已停止"
                st = self.read()
                continue
            if not self._money(refresh):
                return "刷新成本未知或无效，已停止"
            if st.gold - refresh < self.keep_gold:
                return f"金币到保留线（剩 {st.gold}），共刷新 {self.rolls} 次"
            if self.rolls >= self.max_rolls:
                return f"达到最大刷新次数 {self.max_rolls}"

            before = st.shop_names
            before_gold = st.gold
            self.act([Reroll()])
            self.rolls += 1
            self.sink.on_mode_changed()
            st = self._wait_new_shop(before)
            gold_decreased = (before_gold is not None and st.gold is not None
                              and st.gold < before_gold)
            if st.shop_names == before and not gold_decreased:
                reroll_stalls += 1
                if reroll_stalls >= MAX_REROLL_STALLS:
                    return f"刷新未生效（连续 {MAX_REROLL_STALLS} 次），已停止"
            else:
                reroll_stalls = 0
            self._report(st, targets)
        return "已停止"

    def _buys(self, st: GameState, need: dict[str, int]) -> list[BuyShop]:
        if not self._money(st.gold):
            return []
        gold = st.gold
        free = st.bench_free
        owned = st.owned_copies()
        bench1 = Counter(u.name for u in st.bench if u.known and u.star == 1)
        left = dict(need)
        out = []
        for i, s in enumerate(st.shop):
            if not s or left.get(s.name, 0) <= 0:
                continue
            cost = self._cost(s)
            if cost is None or s.item or s.unsure:
                continue
            merges = owned.get(s.name, 0) % 3 == 2      # 买下直接合成，不占格子
            if gold - cost < self.keep_gold or (free <= 0 and not merges):
                continue
            if self._cfg.buy_pairs_only and owned.get(s.name, 0) == 0 and \
                    sum(1 for x in st.shop if x and x.name == s.name) < 2:
                continue
            out.append(BuyShop(i))
            gold -= cost
            left[s.name] -= 1
            owned[s.name] += 1
            if merges:
                if bench1[s.name] >= 2:
                    free += 1
                    bench1[s.name] -= 2
            else:
                free -= 1
                bench1[s.name] += 1
        return out

    @staticmethod
    def _money(value):
        return isinstance(value, Real) and not isinstance(value, bool) and value >= 0 and value < float('inf') and int(value) == value

    def _cost(self, slot):
        hero = self.ctx.db.hero(slot.name)
        value = slot.cost if slot.cost is not None else hero.cost if hero else None
        return int(value) if self._money(value) and value > 0 else None

    @property
    def status(self):
        return {'targets': [{'name': t.name, 'star': t.star} for t in (self._resolved_targets if self._resolved_targets is not None else self.targets or [])],
                'keep': self.keep_gold, 'max_rolls': self.max_rolls, 'rolls': self.rolls, 'bought': self.bought}

    def _wait_new_shop(self, before: list[str | None]) -> GameState:
        """刷新后等画面里的商店真的换了（视觉有延迟），最多等 3 帧。"""
        st = self.read()
        for _ in range(3):
            if st.shop_names != before or self.stopped:
                break
            st = self.read()
        if not self.stopped and any(s and s.unsure and not s.item for s in st.shop):
            st = self.read()
        return st

    def _report(self, st: GameState, targets: list[RollTarget]) -> None:
        need = self.need(st, targets)
        if not need:
            return
        pool = self.ctx.advisor.pool(st, self.ctx.extra_taken)
        level = st.level or 1
        odds = [p / 100 for p in st.odds] if st.odds and sum(st.odds) else None
        p_any = any_hit_per_roll(pool, list(need), level, odds)
        reused = ""
        if (self._sim_rate is None or need != self._sim_need
                or self.rolls - self._sim_roll >= REPORT_SIM_EVERY):
            gold = int(st.gold) if self._money(st.gold) else 0
            sim = simulate_rolldown(pool, need, gold, level, self.keep_gold, sims=600, odds=odds)
            self._sim_rate = sim.all_hit_rate
            self._sim_need = dict(need)
            self._sim_roll = self.rolls
        else:
            reused = f"（沿用第 {self._sim_roll} 次）"
        self.sink.on_log(
            f"第 {self.rolls} 次刷新｜金 {st.gold}｜还缺 " + "、".join(f"{n}×{k}" for n, k in need.items())
            + f"｜下次刷新出现任一目标 {p_any:.0%}｜D 到 {self.keep_gold} 金全部凑齐 {self._sim_rate:.0%}{reused}")
