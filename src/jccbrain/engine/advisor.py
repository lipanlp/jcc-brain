"""规则引擎：GameState → Advice。

完全离线、毫秒级，三种模式共用：
- 辅助大脑直接展示 Advice；
- 大模型拿 Advice 当「草稿」再综合判断；
- 全自动模式由 planner 把 Advice 翻译成操作。
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from jccbrain.core.models import GameState, Phase
from jccbrain.core.rules import SetRules, parse_stage
from jccbrain.data.db import GameDB
from jccbrain.engine import economy
from jccbrain.engine.comps import CompMatch, CompMatcher
from jccbrain.core.buy_source import BuySource
from jccbrain.engine.board import BoardPlan, board_for
from jccbrain.engine.odds import HitOdds, PoolState, copies_needed, hit_odds, prob_with_gold
from jccbrain.engine.tracker import CompTracker
from jccbrain.engine.shopstats import ShopStats
from jccbrain.engine import items, augments
from jccbrain.engine.traits import traits_view


@dataclass
class BuyAdvice:
    slot: int
    name: str
    cost: int
    reason: str
    priority: float


@dataclass(frozen=True)
class BuyProjection:
    source: BuySource
    buys: tuple[BuyAdvice, ...] = ()
    valid: bool = False


@dataclass
class SellAdvice:
    slot: int          # 备战席格号
    name: str
    reason: str


@dataclass
class AugmentAdvice:
    index: int
    name: str
    score: float       # 越高越好
    reason: str
    category: list[str] = field(default_factory=list)
    desc: str = ''
    base_score: float = .5
    situation_score: float = .5
    stats_weight: float = .5


@dataclass
class TargetOdds:
    name: str
    have: int          # 已有张数（折合 1 星）
    need: int          # 到目标星级还差
    target_star: int
    odds: HitOdds | None


@dataclass
class Advice:
    stage: str | None
    phase: Phase
    headline: str = ""
    comps: list[CompMatch] = field(default_factory=list)
    econ: economy.EconDecision | None = None
    buys: list[BuyAdvice] = field(default_factory=list)
    sells: list[SellAdvice] = field(default_factory=list)
    augments: list[AugmentAdvice] = field(default_factory=list)
    targets: list[TargetOdds] = field(default_factory=list)
    items: dict[str, list[str]] = field(default_factory=dict)   # 主 C → 推荐装备
    notes: list[str] = field(default_factory=list)
    llm_text: str | None = None                                  # 大模型补充意见
    locked: bool = False
    comp_status: str | None = None
    switch_reason: str | None = None
    board: BoardPlan | None = None
    held_items: dict = field(default_factory=lambda: {'components': {}, 'completed': [], 'emblems': []})
    augments_taken: list[str] = field(default_factory=list)
    item_plan: list[dict] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    buy_source: BuySource | None = None

    augment_situation: dict = field(default_factory=dict)
    ai_augment: str | None = None
    traits_view: dict = field(default_factory=lambda: {'final': [], 'current': []})

    opponents: list[dict] = field(default_factory=list)

    @property
    def target(self) -> CompMatch | None:
        return self.comps[0] if self.comps else None

    def lines(self) -> list[str]:
        """纯文本版（命令行 / 日志用）。"""
        out = [f"【{self.stage or '?'} {self.phase.value}】{self.headline}"]
        if self.target:
            t = self.target
            out.append(f"目标阵容：{t.name}（均名 {t.comp.avg_placement if t.comp.avg_placement is not None else '暂无统计'}，契合 {t.fit:.0%}）")
            if t.missing:
                out.append("  还缺：" + "、".join(h.name for h in t.missing[:6]))
        if self.econ:
            out.append(f"运营：{self.econ.action.value} — {self.econ.reason}")
        for b in self.buys:
            out.append(f"买 第{b.slot + 1}格 {b.name}（{b.reason}）")
        for s in self.sells:
            out.append(f"卖 备战席{s.slot + 1} {s.name}（{s.reason}）")
        for a in self.augments:
            out.append(f"海克斯 {a.index + 1}. {a.name} 评分 {a.score:.2f}（{a.reason}）")
        for t in self.targets:
            if t.odds and t.need:
                g = t.odds.gold_for_50
                budget = f"50%把握约 {g} 金" if g is not None else (
                    "按当前已知卡池不足以凑齐（卡池不足）" if t.odds.remaining < t.need else
                    "当前出现概率为 0" if t.odds.slot_prob == 0 else
                    "计算范围内暂无50%预算估值，不代表不能刷到")
                out.append(f"  {t.name} 差{t.need}张 单次刷新出现 {t.odds.per_roll:.0%}，"
                           + budget)
        out += [f"· {n}" for n in self.notes]
        out += [f"· {p['reason']}" for p in self.item_plan]
        if self.llm_text:
            out.append(f"AI：{self.llm_text}")
        return out


class HeuristicAdvisor:
    def __init__(self, db: GameDB, rules: SetRules, comps_getter=None, scout=None, economy_preference_getter=None):
        self.db = db
        self.scout = scout
        self.rules = rules
        self._economy_preference_getter = economy_preference_getter or (lambda: 'adaptive')
        self._comps_getter = comps_getter or db.comps_ranked
        self.matcher = CompMatcher(self._comps_getter(), db)
        self._selection_key = self._selection_matcher = None
        self._aug_norm: dict[str, str] | None = None

    def reload(self) -> None:
        self.matcher = CompMatcher(self._comps_getter(), self.db)
        self._selection_key = self._selection_matcher = None
        self._aug_norm = None

    def pool(self, state: GameState, extra_taken: dict[str, int] | None = None) -> PoolState:
        taken = state.owned_copies()
        scouted = self.scout.taken() if self.scout else {}
        for k in scouted.keys() | (extra_taken or {}).keys():
            taken[k] += max(scouted.get(k, 0), (extra_taken or {}).get(k, 0))
        pool = PoolState.from_db(self.db, self.rules, taken)
        if self.scout and self.scout.latest():
            pool.basis = self.scout.pool_basis()
            pool.opponents = [row for row in self.scout.latest() if row.slot is not None or row.name]
        return pool

    def advise(self, state: GameState, locked_comp: str | None = None,
               extra_taken: dict[str, int] | None = None, tracker: CompTracker | None = None,
               watch_ids: list[str] | None = None, shop_stats: ShopStats | None = None,
               augments_taken: list[str] | None = None, economy_preference: str | None = None,
               selection: dict | None = None) -> Advice:
        adv = Advice(stage=state.stage, phase=state.phase)
        adv.selection = selection or {}
        # 必须在侦察、阵容匹配、装备与概率计算之前返回，不能只改标题。
        if state.phase == Phase.ENDED:
            adv.stage = None
            adv.headline = f"对局结束，第 {state.placement} 名" if state.placement else "对局结束"
            return adv
        if self.scout:
            self.scout.update(state)
            adv.opponents = self.scout.view()
        inventory = items.held(state, self.db)
        adv.held_items = {'components': dict(inventory.components), 'completed': inventory.completed, 'emblems': inventory.emblems}
        names = state.augments_taken if augments_taken is None else augments_taken
        augments_taken = list(dict.fromkeys(self.resolve_augment(name) for name in names if name and name.strip()))
        adv.augments_taken = augments_taken
        if self.db.is_empty:
            adv.headline = "数据库为空，请先在「数据」菜单同步数据"
            return adv

        pool = self.pool(state, extra_taken)
        matcher = self.matcher
        if selection is not None:
            key = tuple(selection['candidate_ids'])
            if key != self._selection_key:
                self._selection_key = key
                self._selection_matcher = CompMatcher([c for c in self.matcher.comps if c.id in key], self.db)
            matcher = self._selection_matcher
        adv.comps = (tracker.select(state, matcher, locked_comp, watch_ids, shop_stats, pool, augments_taken) if tracker
                     else matcher.match(state, top=5, locked_id=locked_comp, shop_stats=shop_stats, pool=pool,
                                             augments_taken=augments_taken))
        if selection is not None and not adv.comps:
            adv.headline = selection['notice'] or '没有可用候选阵容，自动比较已暂停'
            adv.notes.append(adv.headline)
            return adv
        if self.scout:
            adv.opponents = self.scout.view(adv.target.comp.id if adv.target else None)
        adv.locked = bool(locked_comp and adv.target and adv.target.comp.id == locked_comp)
        if adv.target:
            adv.board = board_for(adv.target.comp, None, state, self.db)
            adv.item_plan = items.plan(adv.target.comp, inventory, state, self.db, self._comps_getter())
        adv.traits_view = traits_view(adv.target.comp if adv.target else None, state, self.db)
        if tracker:
            adv.comp_status = tracker.status
            adv.switch_reason = tracker.switch_reason
            if tracker.hint:
                adv.notes.append(tracker.hint)
        if not state.stage and state.phase == Phase.UNKNOWN:
            adv.headline = "等待对局画面"
            return adv
        target = adv.target
        level = state.level or 1
        odds = [p / 100 for p in state.odds] if state.odds and sum(state.odds) > 0 else None

        # 阵容目标卡的概率
        if target:
            owned = state.owned_copies()
            downgraded = set()
            for h in target.comp.heroes:
                star = self._target_star(state, h, target.comp)
                if h.star_target == 3 and star == 2:
                    downgraded.add(h.name)
                need = copies_needed(owned.get(h.name, 0), star)
                # 读不到等级（选秀、遮挡）时不给概率，避免按 1 级算出误导数字
                o = hit_odds(pool, h.name, level, max(need, 1), odds) if state.level else None
                adv.targets.append(TargetOdds(h.name, owned.get(h.name, 0), need, star, o))
            if downgraded:
                adv.notes.append('本轮按两星补齐，不继续囤牌：' + '、'.join(sorted(downgraded)))
                if adv.board:
                    adv.board.cells = [replace(cell, star_target=2)
                                       if cell.name in downgraded and cell.star_target == 3 else cell
                                       for cell in adv.board.cells]
            adv.items = {h.name: list(h.items) for h in target.comp.heroes if h.items and (h.is_carry or h.is_subcarry)}

        # 海克斯
        if state.phase == Phase.AUGMENT and state.augments:
            adv.augment_situation = augments.situation(state, adv, self.db)
            adv.augments = self._rate_augments(state, target, adv.augment_situation)
            best = max(adv.augments, key=lambda a: a.score)
            adv.headline = f"海克斯选 {best.index + 1}. {best.name}"
            return adv

        # 运营
        # 打法按主 C 的费用判断（副 C 常是高费卡，不能决定 D 牌等级）
        carry_cost = 4
        if target:
            main = [h for h in target.comp.heroes if h.is_carry] or target.comp.carries
            carry_cost = max((h.cost for h in main), default=4)
        style = target.comp.style if target else "fast8"
        adv.econ = economy.decide(state, self.rules, style, carry_cost,
                                  economy_preference if economy_preference is not None else self._economy_preference_getter())
        self._roll_odds(state, adv, pool, odds)
        self._finish_econ(state, adv, style)

        # 商店
        if state.shop_open:
            adv.buys = self._buys(state, target, adv.econ)
            if selection is not None:
                allowed_heroes = {h.name for c in matcher.comps for h in c.heroes}
                adv.buys = [buy for buy in adv.buys if buy.name in allowed_heroes]
        adv.sells = self._sells(state, target, adv.buys, watch_ids)
        adv.notes += self._notes(state, target)
        adv.headline = self._headline(adv, state, adv.locked)
        return adv

    # ------------------------------------------------------------ 细节
    @staticmethod
    def _target_star(state: GameState, hero, comp) -> int:
        """只计算本次建议目标，不修改源攻略；概率、买卡与站位共用。"""
        star = hero.star_target if hero.star_target in (1, 2, 3) else 2
        late = (state.stage_tuple is not None and state.stage_tuple >= (5, 1)
                or state.level is not None and state.level >= 8)
        recommended = hero.items or (comp.guide and any(
            h.items for board in comp.guide.boards.values() for h in board if h.name == hero.name))
        equipped = any(u.name == hero.name and any(
            isinstance(i.get('name'), str) and i['name'].strip() and not i['name'].strip().startswith('?')
            for i in u.items) for u in state.units)
        if (star == 3 and 1 <= hero.cost <= 3 and late and not hero.is_carry and not hero.is_subcarry
                and not recommended and not equipped and state.owned_copies().get(hero.name, 0) < 6):
            return 2
        return star

    def _buys(self, state: GameState, target: CompMatch | None, econ: economy.EconDecision) -> list[BuyAdvice]:
        """买卡建议。只推荐有理由的卡：目标阵容卡、能直接升星的卡；前期再加上和羁绊相关的对子、补场上空位的过渡卡。"""
        owned = state.owned_copies()
        wanted = {}
        for h in target.comp.heroes if target else []:
            if h.name not in wanted or h.weight > wanted[h.name].weight:
                wanted[h.name] = h
        goals = {name: self._target_star(state, h, target.comp) for name, h in wanted.items()}
        early = (state.stage_tuple or (1, 1)) < (3, 1)
        synergy = {t.name for t in state.traits if t.count}
        if target:
            synergy |= set(target.comp.traits[:3])
        board_n = state.board_count[0] if state.board_count else len(state.board)
        cap = state.board_count[1] if state.board_count else (state.level or 1)
        board_free = max(cap - board_n, 0)

        out, fillers = [], []
        for i, s in enumerate(state.shop):
            hero = self.db.hero(s.name) if s else None
            if hero is None:
                continue                      # 空格，或商店里的非英雄商品（装备、道具）
            cost = s.cost or hero.cost
            have = owned.get(s.name, 0)
            fits = sorted(synergy & set(hero.traits))
            if s.name in wanted:
                h = wanted[s.name]
                if copies_needed(have, goals[s.name]) == 0:
                    continue
                pr = 3 + h.weight + (1 if have in (2, 5, 8) else 0)
                tag = "主C" if h.is_carry else "副C" if h.is_subcarry else "核心" if h.is_core else "阵容卡"
                reason = f"{target.name} {tag}"
                out.append(BuyAdvice(i, s.name, cost, reason + ("，可合成" if have in (2, 8) else ""), pr))
            elif have in (2, 8):
                out.append(BuyAdvice(i, s.name, cost, "买了直接升星", 2.5))
            elif early and have >= 1 and fits:
                out.append(BuyAdvice(i, s.name, cost, f"凑对子（{'、'.join(fits)}）", 1.2))
            elif early and board_free and cost <= 3:
                fillers.append(BuyAdvice(i, s.name, cost, "补场上空位" + (f"（{'、'.join(fits)}）" if fits else ""),
                                         0.6 + 0.2 * len(fits) + 0.05 * cost))
        # 补空位只补差几个就推荐几张，挑和羁绊吻合、费用高的
        fillers.sort(key=lambda b: -b.priority)
        out += fillers[:board_free]
        out.sort(key=lambda b: -b.priority)

        # 金币约束：在保留线之上才买低优先级的；备战席至少留 1 格给目标卡
        gold = state.gold or 0
        keep = 0 if econ.action in (economy.EconAction.ROLL, economy.EconAction.ALL_IN) else econ.keep_gold
        picked, spent = [], 0
        free = state.bench_free
        # 使用真实的星级/棋盘或备战席分布模拟合成，不能仅按折算份数假装腾出格子。
        planned = {name: Counter((u.star, u.zone) for u in state.units if u.name == name)
                   for name in {b.name for b in out}}
        for b in out:
            if b.name in goals and copies_needed(owned.get(b.name, 0), goals[b.name]) == 0:
                continue
            must = b.priority >= 3
            limit = gold if must else gold - keep
            if spent + b.cost > limit:
                continue
            units = planned[b.name].copy()
            bench_before = sum(n for (_, zone), n in units.items() if zone == 'bench')
            count_before = sum(units.values())
            units[1, 'bench'] += 1
            for star in (1, 2):
                while units[star, 'board'] + units[star, 'bench'] >= 3:
                    on_board = min(units[star, 'board'], 3)
                    units[star, 'board'] -= on_board
                    units[star, 'bench'] -= 3 - on_board
                    units[star + 1, 'board' if on_board else 'bench'] += 1
            upgrades = sum(units.values()) < count_before + 1
            if b.name not in goals and b.reason == '买了直接升星' and not upgrades:
                continue  # 前一格已完成合成，后一格不能继续冒用同一即时升星理由。
            bench_after = sum(n for (_, zone), n in units.items() if zone == 'bench')
            next_free = free + bench_before - bench_after
            if next_free < 0:
                continue
            if not upgrades and free <= (0 if must else 1):
                continue
            picked.append(b)
            spent += b.cost
            owned[b.name] += 1
            planned[b.name], free = units, next_free
        return picked

    def _roll_odds(self, state: GameState, adv: Advice, pool: PoolState, odds) -> None:
        econ, target = adv.econ, adv.target
        if not econ or not target or not state.level or econ.action not in (economy.EconAction.ROLL, economy.EconAction.ALL_IN):
            return
        carries = [h.name for h in target.comp.heroes if h.is_carry] or [h.name for h in target.comp.heroes if h.is_subcarry]
        row = next((t for name in carries for t in adv.targets if t.name == name and t.need > 0 and t.odds), None)
        if row is None:
            return
        o = row.odds
        if o.remaining < row.need:
            econ.reason += f'；按当前已知卡池，{row.name} 剩 {o.remaining} 张、还需 {row.need} 张，卡池不足以凑齐'
            return
        if o.slot_prob == 0:
            econ.reason += f'；{row.name} 当前出现概率为 0'
            if econ.action == economy.EconAction.ALL_IN:
                return
            # 当前实时概率优先；下一等级只有规则估计，不能把clamp出的等级当支持。
            current = odds if odds is not None else (
                self.rules.odds(state.level) if state.level in self.rules.level_odds else None)
            next_row = self.rules.level_odds.get(state.level + 1)
            if (current is None or next_row is None or not 1 <= o.cost <= min(len(current), len(next_row))
                    or next_row[o.cost - 1] / 100 <= max(current[o.cost - 1], 0)):
                return
            xp_cur, xp_need = state.xp or (0, self.rules.xp_to_next.get(state.level))
            if xp_need is not None and xp_need <= xp_cur:
                xp_cur, xp_need = 0, self.rules.xp_to_next.get(state.level)
            if xp_need is None or state.gold is None:
                econ.reason += '；升级经验或金币未知，暂不建议立即升级'
                return
            buys = -(-(xp_need - xp_cur) // self.rules.xp_per_buy)
            cost = buys * (state.xp_cost or self.rules.xp_cost)
            basis = f'下一等级按规则估计该费用概率 {next_row[o.cost - 1]}%'
            if state.gold < cost:
                adv.econ = economy.EconDecision(economy.EconAction.SAVE,
                    f'{row.name} 当前出现概率为 0；{basis}，升级需 {cost} 金，先攒够升级费用',
                    keep_gold=state.gold)
            else:
                adv.econ = economy.EconDecision(economy.EconAction.LEVEL,
                    f'当前 {state.level} 级刷不到 {row.name}（{o.cost} 费）；{basis}，先升到 {state.level + 1} 级再 D',
                    keep_gold=state.gold - cost, xp_buys=buys)
            return
        gold50 = o.gold_for_50
        if gold50 is None or econ.roll_budget < gold50:
            if gold50 is None:
                econ.reason += '；计算范围内暂无50%预算估值'
            chance = prob_with_gold(pool, row.name, row.need, state.level, econ.roll_budget, odds)
            econ.reason += f'；按 {econ.roll_budget} 金预算约 {int(chance * 10 + .5)} 成把握凑齐 {row.name}'

    def _sells(self, state: GameState, target: CompMatch | None, buys: list[BuyAdvice] | None = None,
               watch_ids: list[str] | None = None) -> list[SellAdvice]:
        if state.bench_free > 2 or not target:
            return []
        wanted = {h.name for h in target.comp.heroes}
        for cid in watch_ids or []:
            comp = next((c for c in self.matcher.comps if c.id == cid), None)
            if comp:
                wanted.update(h.name for h in comp.heroes)
        wanted.update(b.name for b in buys or [])
        synergy = {t.name for t in state.traits if t.active} | set(target.comp.traits[:3])
        owned = state.owned_copies()
        out = []
        for u in state.bench:
            hero = self.db.hero(u.name)
            if (u.known and hero and u.star == 1 and u.name not in wanted and owned[u.name] == 1
                    and u.slot is not None and not synergy.intersection(hero.traits)):
                out.append(SellAdvice(u.slot, u.name, '与阵容和羁绊无关，腾备战席'))
        return out[: 2]

    def _finish_econ(self, state: GameState, adv: Advice, style: str) -> None:
        """已成型时停止无目的 D 牌；统一保护包括概率后处理产生的升级。"""
        econ = adv.econ
        if (econ is None or econ.action in (economy.EconAction.ALL_IN, economy.EconAction.NONE)
                or state.level is None or state.gold is None):
            return
        xp_cur, xp_need = state.xp or (0, self.rules.xp_to_next.get(state.level, 99))
        if xp_need <= xp_cur:
            xp_cur, xp_need = 0, self.rules.xp_to_next.get(state.level, 99)
        short = xp_need - xp_cur
        buys = -(-short // self.rules.xp_per_buy)
        xp_cost = state.xp_cost or self.rules.xp_cost
        formed = state.level >= 8 and bool(adv.target and adv.targets) and all(t.need == 0 for t in adv.targets)
        if formed:
            reason = ('阵容已成型：存钱升人口' if style == 'reroll' else
                      '阵容已成型：存钱上 9 找 5 费' if state.level < 9 else
                      '阵容已成型：保持经济，找 5 费或追三星')
            if style != 'reroll' and state.level < 9 and state.gold >= buys * xp_cost:
                econ = economy.EconDecision(economy.EconAction.LEVEL, reason, xp_buys=buys,
                                           keep_gold=max(state.gold - buys * xp_cost, 0))
            else:
                econ = economy.EconDecision(economy.EconAction.SAVE, reason, keep_gold=min(state.gold // 10 * 10, 50))
        levels = self.rules.fast8_levels if style in ('fast8', 'fast9') else self.rules.standard_levels
        critical = {parse_stage(stage) for stage in levels}
        if style == 'fast9':
            critical.add((5, 1))
        if (econ.action == economy.EconAction.LEVEL and state.stage_tuple not in critical
                and state.hp is not None and state.hp >= 50
                and self.rules.interest(state.gold) > self.rules.interest(state.gold - econ.xp_buys * xp_cost)):
            prefix = '阵容已成型：' if formed else ''
            econ = economy.EconDecision(economy.EconAction.SAVE,
                f'{prefix}差 {short} 经验，非关键回合，不为升级掉利息', keep_gold=min(state.gold // 10 * 10, 50))
        adv.econ = econ

    def _rate_augments(self, state: GameState, target: CompMatch | None, profile: dict) -> list[AugmentAdvice]:
        stats = {a.name: a for a in target.comp.augments} if target else {}
        out = []
        for i, opt in enumerate(state.augments):
            name = self.resolve_augment(opt.name, opt.desc)
            info = augments.classify(name, self.db, opt.desc)
            rating = augments.rate(name, stats.get(name), info, profile)
            out.append(AugmentAdvice(i, name or f"第 {i + 1} 个（没读到名字）", **rating))
        return out

    def resolve_augment(self, name: str, desc: str = "") -> str:
        """把 OCR 读到的海克斯名对上数据库：忽略空格、去掉尾部杂字；名字没读到时用描述开头匹配。"""
        if name and self.db.augment(name):
            return name
        if self._aug_norm is None:
            self._aug_norm = {_norm(a.name): a.name for a in self.db.augments.values()}
        table = self._aug_norm
        for text in (name, desc):
            t = _norm(text)
            if not t:
                continue
            if t in table:
                return table[t]
            # 最长的、作为前缀出现的数据库名（「资本利得I3」→「资本利得 I」，「秘法帮派获得…」→「秘法帮派」）
            hit = max((k for k in table if t.startswith(k)), key=len, default=None)
            if hit:
                return table[hit]
            # 前缀里有错字（「轻量魔法授掷投掷…」→「轻量魔法投掷」）
            best, ratio = None, 0.0
            for k in table:
                r = difflib.SequenceMatcher(None, t[:len(k)], k).ratio()
                if r > ratio:
                    best, ratio = k, r
            if best and ratio >= 0.75 and len(best) >= 3:
                return table[best]
        return name

    def _notes(self, state: GameState, target: CompMatch | None) -> list[str]:
        notes = []
        if state.bench_free == 0:
            notes.append("备战席已满，买卡前先卖")
        if state.board_count and state.level and state.board_count[0] < state.board_count[1]:
            notes.append(f"场上 {state.board_count[0]}/{state.board_count[1]}，还能再上 1 个")
        nxt = self.rules.next_augment_round(state.stage_tuple) if state.stage_tuple else None
        if nxt:
            notes.append(f"下一个海克斯回合 {nxt}")
        if state.stage and self.rules.round_kind(state.stage) == "carousel":
            notes.append(items.carousel_hint(target.comp, items.held(state, self.db), self.db) if target else '选秀回合')
        unknown = [u for u in state.units if not u.known]
        if unknown:
            notes.append(f"{len(unknown)} 个单位未识别（工具 → 标注单位，或买卡时自动学习）")
        return notes

    @staticmethod
    def _headline(adv: Advice, state: GameState, locked: bool) -> str:
        parts = []
        if adv.econ and adv.econ.action != economy.EconAction.NONE:
            parts.append({"save": "存钱", "level": "升级", "roll": "D 牌", "all_in": "梭哈"}[adv.econ.action.value])
        if adv.buys:
            parts.append("买 " + "、".join(b.name for b in adv.buys))
        t = adv.target
        if t:
            early = (state.stage_tuple or (1, 1)) < (2, 1)
            if adv.comp_status == "探索":
                parts.append(f"前期按强度过渡（参考 {t.name}）")
            elif adv.comp_status == "过渡":
                parts.append(f"倾向 {t.name}")
            elif adv.comp_status == "关注":
                parts.append(f"关注比较中（当前 {t.name}）")
            elif adv.comp_status in ("定型", "锁定") or locked or (not early and t.fit >= 0.3):
                parts.append(f"走 {t.name}")
            elif early:
                parts.append(f"前期按强度过渡（参考 {t.name}）")
            else:
                parts.append(f"倾向 {t.name}")
        return "；".join(parts) or "观望"


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")
