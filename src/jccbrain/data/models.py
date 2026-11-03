"""数据库实体（只读视图）。"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class HeroStats:
    star: int
    hp: float | None = None
    attack_damage: float | None = None
    attack_speed: float | None = None
    armor: float | None = None
    magic_resist: float | None = None
    init_mana: float | None = None
    max_mana: float | None = None
    crit_chance: float | None = None  # 官方百分数，25 表示 25%


@dataclass(frozen=True)
class SkillValue:
    name: str
    scaling: str
    values: tuple[float, ...] = ()
    unit: str = ""
    formula: str = ""                 # 没有逐星表时 values 为空，不推算
    raw: str = ""
    parse_error: bool = False


@dataclass(frozen=True)
class UnlockCondition:
    id: str
    description: str
    task_tips: str = ""
    condition_id: str | None = None
    visible: bool = True


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    cost: int
    traits: tuple[str, ...]
    in_pool: bool = True  # 赛季内静态的普通卡池候选，不含当前对局解锁状态
    role_tag: str | None = None
    attack_range: int | None = None
    skill_name: str | None = None
    skill_desc: str | None = None
    picture: str | None = None
    skill_values: tuple[SkillValue, ...] = ()
    sell_price: int | None = None
    skill_icon: str | None = None
    star_stats: tuple[HeroStats, ...] = ()
    skill_brief_values: tuple[str, ...] = ()
    variant_of: str | None = None
    unit_kind: str = "other"
    unlock_conditions: tuple[UnlockCondition, ...] = ()

    @property
    def pool_id(self) -> str:
        """同名形态的归组键；只有 in_pool 的主形态可建立普通池。"""
        return self.variant_of or self.id

    def stats(self, star: int) -> HeroStats | None:
        """查询官方提供的星级；缺失返回 None，不按倍率猜测。"""
        return next((s for s in self.star_stats if s.star == star), None)


@dataclass(frozen=True)
class TraitLevel:
    count: int | None
    text: str
    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class Trait:
    id: str
    name: str
    kind: str
    breakpoints: tuple[int, ...]
    description: str = ""
    level_desc: tuple[str, ...] = ()
    levels: tuple[TraitLevel, ...] = ()
    activation: str = "standard"     # standard / unique / special（0 档位，不能按人数判断）

    picture: str | None = None

    @property
    def unique(self) -> bool:
        return self.activation == "unique"

    def tier_of(self, count: int) -> int:
        """count 个单位激活到第几档（0 = 未激活）。"""
        return 0 if self.activation == "special" else sum(1 for b in self.breakpoints if count >= b)

    def next_breakpoint(self, count: int) -> int | None:
        return next((b for b in self.breakpoints if b > count), None)


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    kind: str | None
    description: str = ""
    recipe: tuple[str, ...] = ()
    fetter_id: str | None = None      # 官方原始字段，空值不伪装成官方关联
    trait_id: str | None = None       # 原始 id 或精确纹章名推导
    trait_name: str | None = None
    effect_type: str | None = None
    raw_kind: str | None = None
    picture: str | None = None


@dataclass(frozen=True)
class Augment:
    id: str
    name: str
    tier: int | None
    description: str = ""
    fetter_id: str | None = None
    fetter_type: str | None = None
    hero_enhancement_type: str | None = None
    is_legend: bool | None = None
    trait_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompHero:
    hero_id: str
    name: str
    cost: int
    is_carry: bool = False
    is_subcarry: bool = False
    is_core: bool = False
    star_target: int | None = None
    position: tuple[int, int] | None = None     # (行, 列)，行 1~4 从前排到后排
    items: tuple[str, ...] = ()

    @property
    def weight(self) -> float:
        """阵容匹配时这张卡的重要程度。"""
        if self.is_carry:
            return 3.0
        if self.is_subcarry:
            return 2.2
        if self.is_core:
            return 1.6
        return 1.0


@dataclass(frozen=True)
class CompAugment:
    augment_id: str
    name: str
    tier: int | None
    rating: str | None
    play_rate: float | None
    avg_placement: float | None
    top4_rate: float | None
    placement_delta: float | None


@dataclass
class Comp:
    id: str
    name: str
    tier: str | None
    avg_placement: float | None
    top4_rate: float | None
    top_rate: float | None
    pick_rate: float | None
    sample_count: int | None
    game_version: str | None
    traits: list[str] = field(default_factory=list)
    heroes: list[CompHero] = field(default_factory=list)
    augments: list[CompAugment] = field(default_factory=list)
    game_code: str | None = None
    source: str = "dataj"                       # 提供方：dataj（胜率统计）| official（官网/微社区攻略）
    guide_id: str | None = None                 # source = official 时对应的 Guide
    category: str | None = None                 # 官网分类：新手推荐 / 高手进阶 / 趣味娱乐
    guide: Guide | None = field(default=None, repr=False, compare=False)
    guides: list[Guide] = field(default_factory=list, repr=False, compare=False)

    @property
    def channels(self) -> tuple[str, ...]:
        """官方原始攻略的渠道；dataj 关联攻略不改变自身来源。"""
        return (self.guide.channels if self.guide else ("web",)) if self.source == "official" else ()

    @property
    def carries(self) -> list[CompHero]:
        return [h for h in self.heroes if h.is_carry or h.is_subcarry]

    @property
    def style(self) -> str:
        """阵容打法：reroll（低费追三星）/ fast8 / fast9。"""
        if self.guide:
            g = self.guide
            costs = {h.name: h.cost for h in g.heroes}
            if any(0 < costs.get(name, 0) <= 3 for name in g.three_star):
                return "reroll"
            if re.search(r"九五|95", g.name):
                return "fast9"
            # 先拉八成型后再升九不应被当成速九；取文字中第一个明确的 8/9 成型目标。
            target = re.search(r"(?:拉|上|升|到|冲)\s*([89八九])(?:级|人口)?|([89八九])\s*(?:级|人口)", g.d_time)
            if target:
                return "fast9" if (target[1] or target[2]) in ("9", "九") else "fast8"
            if re.search(r"(?:[567五六七](?:级|人口).*?(?:慢[dD]|追三星)|慢[dD].*?三星)", g.d_time):
                return "reroll"
            return "fast8"
        three_star = [h for h in self.heroes if h.star_target == 3]
        if any(h.cost <= 3 for h in three_star):
            return "reroll"
        if sum(1 for h in self.heroes if h.cost == 5) >= 3:
            return "fast9"
        return "fast8"

    @property
    def tier_score(self) -> float:
        return {"S": 1.0, "A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}.get(self.tier or "", 0.65)


@dataclass(frozen=True)
class Mechanic:
    """赛季机制事件（S18：自然仙灵）。rounds 为出现区间 [(起, 止或 None)]，upgrades 为各级升级版描述。"""
    id: str
    name: str
    category: str | None
    price: int | None
    description: str = ""
    rounds: tuple[tuple[str, str | None], ...] = ()
    upgrades: tuple[str, ...] = ()


@dataclass(frozen=True)
class Encounter:
    """开局奇遇：每局开始时全场生效的规则。"""
    id: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class GuideHero:
    hero_id: str
    name: str
    star: int | None = None
    unit_type: str = "hero"
    position: str | None = None              # 官方原样 "行,列"
    is_carry: bool = False
    items: tuple[str, ...] = ()              # 装备名
    cost: int = 0


@dataclass
class Guide:
    """官方阵容攻略：文字思路 + 成型/前期/中期/各人口站位。"""
    id: str
    name: str
    quality: str | None
    game_version: str | None
    early_info: str = ""
    d_time: str = ""
    location_info: str = ""
    enemy_info: str = ""
    hex_info: str = ""
    equipment_info: str = ""
    adventure_info: str = ""
    recommended_augments: list[str] = field(default_factory=list)   # 海克斯名
    alternative_augments: list[str] = field(default_factory=list)
    equipment_order: list[str] = field(default_factory=list)        # 散件优先顺序（装备名）
    carry_items: list[str] = field(default_factory=list)            # 主 C 首选装备名
    carry_items_backup: list[str] = field(default_factory=list)
    three_star: list[str] = field(default_factory=list)             # 追三星的英雄名
    adventures: dict[str, list[str]] = field(default_factory=dict)  # 阶段 → 推荐自然仙灵名
    boards: dict[str, list[GuideHero]] = field(default_factory=dict)  # final/early/mid/L<人口> → 站位
    category: str | None = None                                     # 新手推荐 / 高手进阶 / 趣味娱乐
    author: str | None = None
    channels: tuple[str, ...] = ("web",)
    ingame_sort: int | None = None
    use_count: int | None = None
    week_use: int | None = None
    collect_count: int | None = None
    comment_count: int | None = None
    stat_date: str | None = None
    comp: Comp | None = field(default=None, repr=False, compare=False)
    match_score: float | None = None

    @property
    def heroes(self) -> list[GuideHero]:
        """成型阵容（不含召唤物）。"""
        return [h for h in self.boards.get("final", []) if h.unit_type == "hero"]

    @property
    def carries(self) -> list[GuideHero]:
        return [h for h in self.heroes if h.is_carry]

    def text(self) -> str:
        """攻略全文（给界面和大模型用）。"""
        parts = [("前期过渡", self.early_info), ("D 牌节奏", self.d_time), ("站位", self.location_info),
                 ("克制应对", self.enemy_info), ("海克斯", self.hex_info), ("装备", self.equipment_info),
                 ("自然仙灵", self.adventure_info)]
        return "\n".join(f"【{k}】{v}" for k, v in parts if v)
