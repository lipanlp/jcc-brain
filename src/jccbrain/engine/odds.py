"""牌库与刷新概率计算。

模型：
- 每个费用的公共卡池 = 该费用英雄数 × 每张卡的数量（rules.pool_size）。
- 商店每格独立：先按等级概率抽费用，再在该费用剩余卡中按张数均匀抽。
- 「被拿走」的张数 = 自己持有 + 已知对手持有 + 手动修正。看不到的对手持有量无法得知，
  概率因此偏乐观，可在 UI 里手动补「别人拿了几张」。
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from jccbrain.core.rules import SetRules
from jccbrain.data.models import Hero


@dataclass
class PoolState:
    """卡池快照：英雄表 + 已被拿走的张数。"""

    rules: SetRules
    heroes: dict[str, Hero]                       # 名字 → 英雄（只含进商店的）
    taken: Counter[str] = field(default_factory=Counter)

    basis: str = "按自己持有与手动登记估算"
    opponents: list = field(default_factory=list)

    @classmethod
    def from_db(cls, db, rules: SetRules, taken: Counter[str] | None = None) -> "PoolState":
        return cls(rules, {h.name: h for h in db.pool_heroes()}, Counter(taken or {}))

    def copy(self) -> "PoolState":
        return PoolState(self.rules, self.heroes, Counter(self.taken), self.basis, list(self.opponents))

    def cost_of(self, name: str) -> int | None:
        h = self.heroes.get(name)
        return h.cost if h else None

    def remaining(self, name: str) -> int:
        h = self.heroes.get(name)
        if not h:
            return 0
        return max(self.rules.pool_size.get(h.cost, 0) - self.taken[name], 0)

    def tier_total(self, cost: int) -> int:
        """该费用卡池剩余总张数。"""
        size = self.rules.pool_size.get(cost, 0)
        return sum(max(size - self.taken[h.name], 0) for h in self.heroes.values() if h.cost == cost)

    def slot_prob(self, name: str, level: int, odds: list[float] | None = None) -> float:
        """商店某一格刷出该英雄的概率。"""
        cost = self.cost_of(name)
        if cost is None:
            return 0.0
        tier_odds = (odds or self.rules.odds(level))[cost - 1]
        total = self.tier_total(cost)
        return tier_odds * self.remaining(name) / total if total else 0.0


@dataclass(frozen=True)
class HitOdds:
    name: str
    cost: int
    remaining: int
    slot_prob: float           # 单格概率
    per_roll: float            # 一次刷新至少出现 1 张
    expected_per_roll: float   # 一次刷新期望张数
    rolls_for_50: int | None   # 有 50% 把握凑齐所需的刷新次数
    rolls_for_80: int | None
    gold_for_50: int | None    # 对应花费（刷新 + 买卡）
    gold_for_80: int | None


def _binom_pmf(n: int, p: float) -> list[float]:
    from math import comb
    return [comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


def hit_curve(pool: PoolState, name: str, need: int, level: int, max_rolls: int = 60,
              odds: list[float] | None = None) -> list[float]:
    """curve[r] = 刷新 r 次内累计拿到 need 张的概率（每次刷新后把看到的目标卡全买下）。

    状态 j = 已拿到的张数；每次刷新 5 格，每格命中概率随 j 增加而下降（卡池被自己拿走）。
    """
    slots = pool.rules.shop_slots
    cost = pool.cost_of(name)
    if cost is None or need <= 0:
        return [1.0] * (max_rolls + 1)
    tier_odds = (odds or pool.rules.odds(level))[cost - 1]
    rem0, total0 = pool.remaining(name), pool.tier_total(cost)
    if rem0 < need:
        return [0.0] * (max_rolls + 1)

    dist = [0.0] * (need + 1)
    dist[0] = 1.0
    curve = [0.0]
    trans = []
    for j in range(need):
        p = tier_odds * (rem0 - j) / (total0 - j) if total0 - j > 0 else 0.0
        trans.append(_binom_pmf(slots, p))
    for _ in range(max_rolls):
        nxt = [0.0] * (need + 1)
        nxt[need] = dist[need]
        for j in range(need):
            if dist[j] == 0:
                continue
            for k, pk in enumerate(trans[j]):
                nxt[min(need, j + k)] += dist[j] * pk
        dist = nxt
        curve.append(dist[need])
    return curve


def hit_odds(pool: PoolState, name: str, level: int, need: int = 1,
             odds: list[float] | None = None) -> HitOdds | None:
    cost = pool.cost_of(name)
    if cost is None:
        return None
    p = pool.slot_prob(name, level, odds)
    slots = pool.rules.shop_slots
    curve = hit_curve(pool, name, need, level, max_rolls=150, odds=odds)

    def rolls_for(target: float) -> int | None:
        return next((r for r, v in enumerate(curve) if v >= target), None)

    r50, r80 = rolls_for(0.5), rolls_for(0.8)

    def gold(r: int | None) -> int | None:
        return None if r is None else r * pool.rules.refresh_cost + need * cost

    return HitOdds(
        name=name, cost=cost, remaining=pool.remaining(name), slot_prob=p,
        per_roll=1 - (1 - p) ** slots, expected_per_roll=p * slots,
        rolls_for_50=r50, rolls_for_80=r80, gold_for_50=gold(r50), gold_for_80=gold(r80),
    )


def any_hit_per_roll(pool: PoolState, names: list[str], level: int, odds: list[float] | None = None) -> float:
    """一次刷新至少出现其中任意一张的概率。"""
    p_slot = sum(pool.slot_prob(n, level, odds) for n in set(names))
    return 1 - (1 - min(p_slot, 1.0)) ** pool.rules.shop_slots


def prob_with_gold(pool: PoolState, name: str, need: int, level: int, gold: int,
                   odds: list[float] | None = None) -> float:
    """预算内凑齐目标的概率；与 HitOdds 花费口径一致，先扣所需买卡钱。"""
    if need <= 0:
        return 1.0
    cost = pool.cost_of(name)
    if cost is None or pool.remaining(name) < need:
        return 0.0
    rolls = (gold - need * cost) // pool.rules.refresh_cost
    if rolls <= 0:
        return 0.0
    return hit_curve(pool, name, need, level, max_rolls=rolls, odds=odds)[-1]


# ------------------------------------------------------------------ 蒙特卡洛 D 牌模拟
@dataclass
class RolldownResult:
    sims: int
    hit_rate: dict[str, float]          # 每个目标凑齐的概率
    all_hit_rate: float
    avg_rolls: float
    avg_gold_left: float


def simulate_rolldown(pool: PoolState, targets: dict[str, int], gold: int, level: int,
                      keep_gold: int = 0, sims: int = 2000, seed: int | None = None,
                      odds: list[float] | None = None) -> RolldownResult:
    """模拟「从当前金币 D 到 keep_gold」：每次刷新买下所有还需要的目标卡。

    targets: 名字 → 还需要几张。
    """
    rng = random.Random(seed)
    rules = pool.rules
    odds = odds or rules.odds(level)
    by_cost: dict[int, list[str]] = {}
    for h in pool.heroes.values():
        by_cost.setdefault(h.cost, []).append(h.name)

    hits = Counter()
    all_hits = 0
    total_rolls = 0
    total_left = 0
    for _ in range(sims):
        taken = Counter(pool.taken)
        need = {n: k for n, k in targets.items() if k > 0 and n in pool.heroes}
        g = gold
        rolls = 0
        while need and g - rules.refresh_cost >= keep_gold:
            g -= rules.refresh_cost
            rolls += 1
            for _slot in range(rules.shop_slots):
                cost = _pick_cost(rng, odds)
                names = by_cost.get(cost)
                if not names:
                    continue
                size = rules.pool_size.get(cost, 0)
                weights = [max(size - taken[n], 0) for n in names]
                tot = sum(weights)
                if tot == 0:
                    continue
                n = rng.choices(names, weights)[0]
                if n in need and g >= cost:
                    g -= cost
                    taken[n] += 1
                    need[n] -= 1
                    if need[n] == 0:
                        del need[n]
        for n in targets:
            if n not in need:
                hits[n] += 1
        all_hits += not need
        total_rolls += rolls
        total_left += g
    return RolldownResult(
        sims=sims, hit_rate={n: hits[n] / sims for n in targets}, all_hit_rate=all_hits / sims,
        avg_rolls=total_rolls / sims, avg_gold_left=total_left / sims,
    )


def _pick_cost(rng: random.Random, odds: list[float]) -> int:
    r = rng.random() * sum(odds)
    acc = 0.0
    for i, p in enumerate(odds):
        acc += p
        if r < acc:
            return i + 1
    return len(odds)


def copies_needed(owned_copies: int, target_star: int) -> int:
    """升到目标星级还差几张 1 星卡。"""
    from jccbrain.core.models import COPIES_PER_STAR
    return max(COPIES_PER_STAR.get(target_star, 1) - owned_copies, 0)
