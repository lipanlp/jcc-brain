"""赛季规则：卡池数量、等级出卡概率、经验表、回合类型。data/rules/<官方赛季id>.json。

赛季（core/seasons.py）通过 rules_set 指向自己的规则表；没有规则表的赛季沿用默认规则表（borrowed_from 记录来源），
并在日志里提示。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from jccbrain.core import paths
from jccbrain.core.seasons import DEFAULT_SEASON_ID, LEGACY_SET_IDS, Season, season

log = logging.getLogger(__name__)


def stage_key(stage: tuple[int, int]) -> int:
    """回合排序用：(3, 2) → 302。"""
    return stage[0] * 100 + stage[1]


def parse_stage(s: str) -> tuple[int, int]:
    a, b = s.split("-", 1)
    return int(a), int(b)


@dataclass(frozen=True)
class SetRules:
    set_id: int
    pool_size: dict[int, int]             # 费用 → 每张卡全场多少张
    level_odds: dict[int, list[int]]      # 等级 → 1~5 费出卡百分比
    xp_to_next: dict[int, int]
    shop_slots: int
    bench_slots: int
    refresh_cost: int
    xp_cost: int
    xp_per_buy: int
    max_interest: int
    streak_gold: list[tuple[int, int]]    # (连胜/败场数下限, 额外金币)
    augment_rounds: list[str]
    carousel_rounds: list[str]
    pve_rounds: list[str]
    standard_levels: dict[str, int]       # 标准运营：到该回合应达到的等级
    fast8_levels: dict[str, int]
    borrowed_from: int | None = None      # 沿用别的赛季的规则表时，记录来源的 set<N>
    season_id: str = DEFAULT_SEASON_ID
    borrowed_from_season: str | None = None
    sources: dict = field(default_factory=dict)
    note: str = ""

    @property
    def borrowed(self) -> bool:
        """是否沿用了别的赛季的规则表。"""
        return self.borrowed_from is not None

    @classmethod
    def from_json(cls, d: dict) -> "SetRules":
        return cls(
            set_id=int(d["set_id"]),
            pool_size={int(k): int(v) for k, v in d["pool_size"].items()},
            level_odds={int(k): list(v) for k, v in d["level_odds"].items()},
            xp_to_next={int(k): int(v) for k, v in d["xp_to_next"].items()},
            shop_slots=d.get("shop_slots", 5),
            bench_slots=d.get("bench_slots", 9),
            refresh_cost=d.get("refresh_cost", 2),
            xp_cost=d.get("xp_cost", 4),
            xp_per_buy=d.get("xp_per_buy", 4),
            max_interest=d.get("max_interest", 5),
            streak_gold=[(int(a), int(b)) for a, b in d.get("streak_gold", [])],
            augment_rounds=list(d.get("augment_rounds", [])),
            carousel_rounds=list(d.get("carousel_rounds", [])),
            pve_rounds=list(d.get("pve_rounds", [])),
            standard_levels=dict(d.get("standard_levels", {})),
            fast8_levels=dict(d.get("fast8_levels", {})),
            season_id=d.get("season_id", DEFAULT_SEASON_ID),
            sources=dict(d.get("sources", {})), note=d.get("note", ""),
        )

    # ---- 查询 ----
    def odds(self, level: int) -> list[float]:
        """等级对应的 1~5 费概率（0~1）。"""
        lv = max(min(level, max(self.level_odds)), min(self.level_odds))
        return [p / 100 for p in self.level_odds[lv]]

    def interest(self, gold: int) -> int:
        return min(max(gold, 0) // 10, self.max_interest)

    def streak_bonus(self, n: int) -> int:
        bonus = 0
        for need, g in self.streak_gold:
            if n >= need:
                bonus = g
        return bonus

    def round_kind(self, stage: str) -> str:
        if stage in self.carousel_rounds:
            return "carousel"
        if stage in self.pve_rounds:
            return "pve"
        return "pvp"

    def target_level(self, stage: tuple[int, int], plan: str = "standard") -> int | None:
        """按运营曲线，当前回合应该达到的等级。"""
        table = self.fast8_levels if plan == "fast8" else self.standard_levels
        best = None
        for k, lv in table.items():
            if stage_key(parse_stage(k)) <= stage_key(stage):
                best = lv if best is None else max(best, lv)
        return best

    def next_augment_round(self, stage: tuple[int, int]) -> str | None:
        for r in self.augment_rounds:
            if stage_key(parse_stage(r)) > stage_key(stage):
                return r
        return None


def available_rule_sets() -> list[str]:
    """有规则表的赛季。"""
    out = []
    for p in paths.RULES_DIR.glob("*.json"):
        if re.fullmatch(r"(?:Mode)?\d+_S\d+", p.stem):
            out.append(p.stem)
    return sorted(out)


@lru_cache(maxsize=8)
def load_rules(set_id: int | str = DEFAULT_SEASON_ID) -> SetRules:
    """按官方赛季 id 读取；兼容 load_rules(18)，缺失时明确借用默认表，不猜最近赛季。"""
    requested = LEGACY_SET_IDS.get(set_id, f"{set_id}_unknown") if isinstance(set_id, int) else set_id
    have = available_rule_sets()
    if not have:
        raise FileNotFoundError(f"{paths.RULES_DIR} 下没有任何规则表")
    src = requested if requested in have else DEFAULT_SEASON_ID
    path = paths.RULES_DIR / f"{src}.json"
    rules = SetRules.from_json(json.loads(path.read_text(encoding="utf-8")))
    if src != requested:
        log.warning("%s 没有规则表，沿用 %s 的未核实规则（卡池数量、经验表等可能不准）", requested, src)
        target = season(requested)
        rules = replace(rules, set_id=set_id if isinstance(set_id, int) else int(target.mode),
                        season_id=requested, borrowed_from=rules.set_id, borrowed_from_season=src,
                        note=f"{requested} 沿用 {src}；本赛季数值未核实。" + rules.note)
    return rules


DEFAULT_RULES_SET = DEFAULT_SEASON_ID


def rules_for_season(s: Season) -> SetRules:
    """赛季对应的规则表；没有自己的规则表时沿用默认的（S18），borrowed 为真。"""
    return load_rules(s.rules_set or s.id)
