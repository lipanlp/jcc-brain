"""阵容匹配：根据当前持有的棋子、羁绊栏和商店，给推荐阵容打分。

打分 = 契合度 × 强度折扣 × 权重 + 阵容强度 × (1 - 权重) + 商店加分。
- 契合度 = 棋子契合（按重要程度加权：主 C > 副 C > 核心 > 其他）与羁绊契合（左侧羁绊栏里阵容羁绊的数量）的组合。
  羁绊栏读得比单位稳定，单位没认出来（「?」）时也能反映局势。
- 权重随场上/备战席单位数增加：前期主要看阵容强度，棋子越多越看契合度。
- 商店里正好有阵容卡（尤其主 C）时加分，帮助在前期顺着商店走。
- 持续推荐的目标与候选顺序由 CompTracker 按回合维护。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from jccbrain.core.models import GameState
from jccbrain.data.models import Comp, CompHero
from jccbrain.engine.shopstats import ShopStats
from jccbrain.engine.odds import PoolState
from jccbrain.engine import items

TRAIT_FULL = 8          # 阵容主羁绊合计达到这么多个单位视为羁绊契合 100%
SHOP_BONUS = {"carry": 0.06, "other": 0.025, "max": 0.12}


@dataclass
class CompMatch:
    comp: Comp
    score: float
    fit: float                                   # 0~1，棋子与羁绊综合契合度
    owned: list[CompHero] = field(default_factory=list)
    missing: list[CompHero] = field(default_factory=list)
    in_shop: list[str] = field(default_factory=list)
    why: list[str] = field(default_factory=list)  # 推荐理由（界面与大模型用）
    seen: int = 0
    seen_carry: int = 0
    owned_copies: int = 0

    @property
    def name(self) -> str:
        return self.comp.name


class CompMatcher:
    def __init__(self, comps: list[Comp], db=None):
        self.comps = comps
        self.db = db

    def match(self, state: GameState, top: int | None = 5,
              locked_id: str | None = None, shop_stats: ShopStats | None = None,
              pool: PoolState | None = None, augments_taken: list[str] | None = None) -> list[CompMatch]:
        results = self._score_comps(state, self.comps, shop_stats, pool, augments_taken)
        for m in results:
            if locked_id and m.comp.id == locked_id:
                m.score += 10  # 一次性建议中，用户锁定的阵容永远排第一
                m.why.insert(0, "本局锁定")
        results.sort(key=lambda m: (-m.score, m.comp.id))
        return results if top is None else results[:top]

    def match_ids(self, state: GameState, ids: list[str], shop_stats: ShopStats | None = None,
                  pool: PoolState | None = None, augments_taken: list[str] | None = None) -> list[CompMatch]:
        """只刷新指定阵容，保持传入顺序，不重新排名。"""
        by_id = {c.id: c for c in self.comps}
        return self._score_comps(state, [by_id[cid] for cid in ids if cid in by_id], shop_stats, pool, augments_taken)

    def _score_comps(self, state: GameState, comps: list[Comp], shop_stats: ShopStats | None = None,
                     pool: PoolState | None = None, augments_taken: list[str] | None = None) -> list[CompMatch]:
        owned = state.owned_copies()
        shop = {n for n in state.shop_names if n} if state.shop_open else set()
        units = len(state.units)
        if state.board_count:
            units = max(units, state.board_count[0] + len(state.bench))
        fit_weight = min(1.0, units / 6) * 0.8
        traits = {t.name: t.count or 0 for t in state.traits if t.name}
        inventory = items.held(state, self.db) if self.db else None

        results = []
        for comp in comps:
            if not comp.heroes:
                continue
            m = self._score(comp, owned, shop, traits, fit_weight, pool)
            if inventory is not None:
                fit, matched = items.item_fit(comp, inventory, self.db)
                m.score += (.12 if (state.stage_tuple or (1, 1)) < (3, 1) else .06) * fit
                if matched:
                    m.why.append('装备契合：' + '、'.join(matched))
                emblems = [name for name, trait in inventory.emblems if trait in comp.traits[:3]]
                m.score += min(len(emblems) * .04, .08)
                m.why.extend('纹章 ' + name for name in dict.fromkeys(emblems))
            augment_stats = {a.name: a for a in comp.augments}
            for name in dict.fromkeys(augments_taken or []):
                aug = augment_stats.get(name)
                if aug and aug.placement_delta is not None and aug.placement_delta < 0:
                    m.score += min(-aug.placement_delta * .1, .05)
                    m.why.append(f'海克斯 {name} 适配（均名 {aug.placement_delta:+.2f}）')
            names = {h.name for h in comp.heroes}
            m.owned_copies = sum(owned[n] for n in names)
            if shop_stats is not None:
                m.seen = sum(shop_stats.counts[n] for n in names)
                m.seen_carry = sum(shop_stats.counts[n] for n in {h.name for h in comp.heroes if h.is_carry})
            results.append(m)
        return results

    @staticmethod
    def _score(comp: Comp, owned: Counter[str], shop: set[str], traits: dict[str, int],
               fit_weight: float, pool: PoolState | None = None) -> CompMatch:
        total_w = sum(h.weight for h in comp.heroes)
        got_w = 0.0
        owned_h, missing = [], []
        for h in comp.heroes:
            if owned.get(h.name):
                k = 0.5 if h.is_carry else 0.3 if h.is_subcarry else 0.1
                bonus = 1.0 + k * min(owned[h.name] // 3, 2)
                got_w += h.weight * bonus
                owned_h.append(h)
            else:
                missing.append(h)
        hero_fit = min(got_w / total_w, 1.0) if total_w else 0.0
        main_traits = list(comp.traits[:3])
        trait_hits = {t: traits[t] for t in main_traits if traits.get(t)}
        trait_fit = min(sum(trait_hits.values()) / TRAIT_FULL, 1.0)
        fit = 0.65 * hero_fit + 0.35 * trait_fit if trait_hits else hero_fit

        in_shop = [h for h in comp.heroes if h.name in shop]
        shop_bonus = min(sum(SHOP_BONUS["carry"] if h.is_carry else SHOP_BONUS["other"] for h in in_shop),
                         SHOP_BONUS["max"])
        strength = _strength(comp)
        # 均名统计先验影响契合折扣，不读取资料评级。
        score = fit_weight * fit * (0.4 + 0.6 * strength) + (1 - fit_weight) * strength + shop_bonus
        missing.sort(key=lambda h: -h.weight)

        why = []
        if owned_h:
            why.append(f"已有 {len(owned_h)}/{len(comp.heroes)}")
        if trait_hits:
            why.append("羁绊 " + "、".join(f"{t}{n}" for t, n in trait_hits.items()))
        if in_shop:
            why.append("商店有 " + "、".join(h.name for h in in_shop))
        why.append(f"均名 {comp.avg_placement}" if comp.avg_placement is not None else '暂无均名统计，使用中性先验')
        if pool is not None:
            for h in comp.heroes:
                if h.is_carry and h.name in pool.heroes:
                    remaining = pool.remaining(h.name)
                    if remaining <= (4 if h.star_target == 3 else 2):
                        score -= .20 if remaining == 0 else .10
                        why.append(f'主C {h.name} 剩 {remaining} 张（{pool.basis}）')
        if pool is not None:
            opponents = [row for row in pool.opponents if row.hp is None or row.hp > 0]
            for carry in {h.name for h in comp.heroes if h.is_carry}:
                k = sum(sum(u.copies for u in row.units if u.name == carry) >= 2 for row in opponents)
                if k:
                    why.append(f'主C {carry} 被 {k} 家拿着')
            k = sum(any(sum(u.copies for u in row.units if u.name == h.name) >= 2
                        for h in comp.heroes if h.is_carry) for row in opponents)
            score -= min(.08 * k, .20)
            m = sum(bool(row.comp_guess and row.comp_guess['id'] == comp.id and row.comp_guess['fit'] >= .4)
                    for row in opponents)
            if m:
                score -= min(.05 * m, .15)
                why.append(f'{m} 家在玩 {comp.name}')
        return CompMatch(comp, score, fit, owned_h, missing, [h.name for h in in_shop], why)


def _strength(comp: Comp) -> float:
    """仅按均名：3.5→1、5.5→0，缺统计统一中性0.5。"""
    if comp.avg_placement is None:
        return 0.5
    return max(0.0, min(1.0, (5.5 - comp.avg_placement) / 2.0))
