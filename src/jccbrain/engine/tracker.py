"""按回合收窄阵容候选，并用连续领先与定型约束目标切换。"""
from __future__ import annotations

import logging

from jccbrain.core.models import GameState
from jccbrain.engine.comps import CompMatch, CompMatcher
from jccbrain.engine.shopstats import ShopStats
from jccbrain.engine.odds import PoolState

log = logging.getLogger("jccbrain.rounds")

LEAD_MARGIN = 0.08
LEAD_ROUNDS = 2
SETTLE_STAGE = (3, 2)
SETTLE_FIT = 0.5


class CompTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.round: str | None = None
        self.target_id: str | None = None
        self.candidate_ids: list[str] = []
        self.status: str | None = None
        self.settled = False
        self.lead: dict[str, int] = {}
        self.hint: str | None = None
        self.switch_reason: str | None = None
        self._lock_id: str | None = None
        self._watch_ids: tuple[str, ...] = ()
        self._watch_matcher: CompMatcher | None = None
        self._stage: tuple[int, int] = (1, 1)

    def select(self, state: GameState, matcher: CompMatcher,
               lock_id: str | None, watch_ids: list[str] | None = None,
               shop_stats: ShopStats | None = None, pool: PoolState | None = None,
               augments_taken: list[str] | None = None) -> list[CompMatch]:
        watched = tuple(dict.fromkeys(watch_ids or []))
        watch_changed = watched != self._watch_ids
        stats = {'shop_stats': shop_stats} if shop_stats is not None else {}
        if pool is not None:
            stats['pool'] = pool
        if augments_taken is not None:
            stats['augments_taken'] = augments_taken
        lock_changed = lock_id != self._lock_id
        new_round = bool(state.stage and state.stage != self.round)
        if not new_round and not lock_changed and not watch_changed and self.target_id:
            matches = matcher.match_ids(state, self.candidate_ids, **stats)
            if self.status == '关注' and any(m.comp.id == self.target_id for m in matches):
                return self._watch_candidates(matches, matcher, advance=False)
            if matches and matches[0].comp.id == self.target_id:
                return self._mark_lock(matches, lock_id)
            # 数据重载后目标已消失：按首次评估重来。

        matches = matcher.match(state, top=None, **stats)
        by_id = {m.comp.id: m for m in matches}
        if self.target_id is not None and self.target_id not in by_id:
            self.reset()
        if lock_changed or watch_changed:
            self.lead.clear()
        self._lock_id = lock_id
        self._watch_ids = watched
        if state.stage:
            self.round = state.stage
        self.switch_reason = None
        self.hint = None
        if not matches:
            return []

        target = by_id.get(self.target_id) or matches[0]
        stage = state.stage_tuple or self._stage
        self._stage = stage
        if lock_id in by_id:
            target = by_id[lock_id]
            self.status = "锁定"
            candidates = [target]
        elif any(cid in by_id for cid in watched):
            candidates = [m for m in matches if m.comp.id in watched]
            candidates = self._watch_candidates(candidates, matcher,
                                                advance=new_round and not lock_changed and not watch_changed)
            target = candidates[0]
            self.status = '关注'
        else:
            challenger = next((m for m in matches if m.comp.id != target.comp.id), None)
            if not self.settled and stage < (2, 1):
                target = matches[0]
                self.status = "探索"
                self.lead.clear()
                candidates = matches[:3]
            else:
                self.status = "定型" if self.settled else "过渡"
                if challenger and challenger.score - target.score >= LEAD_MARGIN:
                    n = self.lead.get(challenger.comp.id, 0) + 1
                    self.lead = {challenger.comp.id: n}
                    delta = challenger.score - target.score
                    if n >= LEAD_ROUNDS:
                        reason = f"{challenger.name} 连续 {n} 回合领先 {delta:.2f}"
                        if self.settled:
                            self.hint = f"可考虑转 {challenger.name}（连续 {n} 回合领先 {delta:.2f}）"
                        else:
                            target = challenger
                            self.switch_reason = reason
                            self.lead.clear()
                            challenger = next((m for m in matches if m.comp.id != target.comp.id), None)
                else:
                    self.lead.clear()
                candidates = [target] + ([challenger] if challenger else [])

            copies = state.owned_copies()
            if (stage >= SETTLE_STAGE or target.fit >= SETTLE_FIT
                    or any(h.is_carry and copies[h.name] >= 3 for h in target.comp.heroes)):
                self.settled = True
            if self.settled:
                self.status = "定型"
                candidates = [target]

        self.target_id = target.comp.id
        self.candidate_ids = [m.comp.id for m in candidates]
        watch_note = '｜关注 ' + '、'.join(by_id[cid].name for cid in watched if cid in by_id) if watched else ''
        log.info("回合 %s｜状态 %s｜目标 %s｜候选 %s｜切换：%s｜提示：%s%s",
                 self.round or "?", self.status, target.name, "、".join(m.name for m in candidates),
                 self.switch_reason or "无", self.hint or "无", watch_note)
        return self._mark_lock(candidates, lock_id)

    def _watch_candidates(self, matches: list[CompMatch], matcher: CompMatcher,
                          *, advance: bool) -> list[CompMatch]:
        """主目标置首，其余仍比较；只在未改变配置的回合边界累计领先。"""
        ranked = sorted(matches, key=lambda m: (-m.score, m.comp.id))
        by_id = {m.comp.id: m for m in ranked}
        target = by_id.get(self.target_id) or ranked[0]
        # reload 会重建 matcher；有效目标保留，旧数据上的领先记录不带过去。
        if self._watch_matcher is not matcher or target.comp.id != self.target_id:
            self.lead.clear()
            advance = False
        self._watch_matcher = matcher
        if any(cid not in by_id for cid in self.lead):
            self.lead.clear()
        if advance:
            challenger = next((m for m in ranked if m.comp.id != target.comp.id), None)
            if challenger and challenger.score - target.score >= LEAD_MARGIN:
                n = self.lead.get(challenger.comp.id, 0) + 1
                self.lead = {challenger.comp.id: n}
                if n >= LEAD_ROUNDS:
                    self.switch_reason = f"{challenger.name} 连续 {n} 回合领先 {challenger.score - target.score:.2f}"
                    target = challenger
                    self.lead.clear()
            else:
                self.lead.clear()
        candidates = [target] + [m for m in ranked if m.comp.id != target.comp.id]
        self.target_id = target.comp.id
        self.candidate_ids = [m.comp.id for m in candidates]
        return candidates

    @staticmethod
    def _mark_lock(matches: list[CompMatch], lock_id: str | None) -> list[CompMatch]:
        if matches and matches[0].comp.id == lock_id:
            matches[0].why.insert(0, "本局锁定")
        return matches
