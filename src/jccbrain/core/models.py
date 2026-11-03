"""对局状态数据模型。

字段与 game-eye 输出的 JSON 一一对应（见 game-eye/README.md「输出字段」），
视觉后端换成别的实现时，只要能产出同样结构的 dict，就能用 GameState.from_dict 解析。
hp_rank 是辅助大脑稳定器补充的名次/存活人数，保留原始 players 供界面和大模型参考。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from jccbrain.core.buy_source import BuySource

# 每个星级相当于几张 1 星卡
COPIES_PER_STAR = {1: 1, 2: 3, 3: 9, 4: 27}


class Phase(str, Enum):
    PLANNING = "planning"     # 准备阶段：可以买卖、站位
    COMBAT = "combat"         # 战斗中：棋盘锁定
    AUGMENT = "augment"       # 海克斯三选一
    CAROUSEL = "carousel"     # 选秀（game-eye 暂未识别）
    PVE = "pve"
    SPECTATE = "spectate"
    DIALOG = "dialog"
    RECONNECT = "reconnect"
    ENDED = "ended"           # 结算
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, v: str | None) -> "Phase":
        try:
            return cls(v or "unknown")
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ShopSlot:
    name: str
    cost: int | None = None
    traits: list[str] = field(default_factory=list)
    unsure: bool = False
    item: bool = False         # 道具卡（铜汤匙、边境村庄…），不是英雄
    upgrade: int | None = None  # 商店显示的可升星标记，不等于当前英雄星级


@dataclass
class Unit:
    name: str                  # "?3" 表示还没认出的第 3 组
    star: int = 1
    zone: str = "board"        # board | bench
    slot: int | None = None    # 备战席格号 0~8
    x: float | None = None     # 棋盘单位血条中心，相对游戏窗口 0~1
    y: float | None = None
    items: list[dict[str, Any]] = field(default_factory=list)  # 装备名字/类别/相对坐标，按装备位排序

    @property
    def known(self) -> bool:
        return bool(self.name) and not self.name.startswith("?")

    @property
    def copies(self) -> int:
        return COPIES_PER_STAR.get(self.star, 1)


@dataclass
class TraitStatus:
    name: str
    count: int | None = None
    active: bool = False
    next: int | None = None


@dataclass
class Player:
    name: str | None
    hp: int | None
    is_self: bool = False
    slot: int | None = None
    y: float | None = None


@dataclass
class AugmentOption:
    name: str
    desc: str = ""
    refresh: bool | None = None


@dataclass
class GameState:
    stage: str | None = None
    timer: int | None = None
    phase: Phase = Phase.UNKNOWN
    banner: str | None = None
    gold: int | None = None
    interest: int | None = None
    level: int | None = None
    xp: tuple[int, int] | None = None
    hp: int | None = None
    hp_rank: tuple[int, int] | None = None  # 稳定器提供的（血量名次，存活人数），不是逐帧视觉读数
    streak: dict[str, Any] | None = None
    xp_cost: int | None = None
    refresh_cost: int | None = None
    odds: list[int] | None = None
    shop_open: bool = False
    shop: list[ShopSlot | None] = field(default_factory=lambda: [None] * 5)
    traits: list[TraitStatus] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    augments: list[AugmentOption] = field(default_factory=list)
    placement: int | None = None
    bench: list[Unit] = field(default_factory=list)
    board: list[Unit] = field(default_factory=list)
    board_count: tuple[int, int] | None = None
    legend: tuple[float, float] | None = None
    units_stale: bool = False
    players_stale: bool = False
    frame: int | None = None
    buy_source: BuySource | None = field(default=None, repr=False, compare=False)  # display provenance, never OCR
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    item_bench: list[dict[str, Any]] = field(default_factory=list)

    next_round: str = "unknown"  # pvp / pve / carousel / augment / unknown
    round_type: str = "unknown"  # 指针所在格类型，取值同next_round；不改变phase/动作策略
    viewing: dict[str, Any] | None = None

    augments_taken: list[str] = field(default_factory=list)  # 视觉提供的本局已选海克斯，按选择顺序

    opponent: dict[str, Any] | None = None  # 可选的敌方半场；不能并入自己的board。

    # ---- 解析 ----
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GameState":
        shop = d.get("shop") or {}
        slots: list[ShopSlot | None] = []
        for s in (shop.get("slots") or [None] * 5):
            slots.append(None if not s or not s.get("name") else ShopSlot(
                name=s["name"], cost=s.get("cost"), traits=list(s.get("traits") or []), unsure=bool(s.get("unsure")),
                item=bool(s.get("item")), upgrade=s.get("upgrade")))
        while len(slots) < 5:
            slots.append(None)

        xp = d.get("xp")
        rank = d.get("hp_rank")
        hp_rank = None
        if isinstance(rank, (list, tuple)) and len(rank) == 2 \
                and all(isinstance(n, int) for n in rank) and 1 <= rank[0] <= rank[1] <= 8:
            hp_rank = (rank[0], rank[1])
        bc = d.get("board_count")
        lg = d.get("legend")
        return cls(
            stage=d.get("stage"),
            timer=d.get("timer"),
            phase=Phase.parse(d.get("phase")),
            next_round=d.get("next_round") if d.get("next_round") in ("pvp", "pve", "carousel", "augment") else "unknown",
            round_type=d.get("round_type") if d.get("round_type") in ("pvp", "pve", "carousel", "augment") else "unknown",
            banner=d.get("banner"),
            gold=d.get("gold"),
            interest=d.get("interest"),
            level=d.get("level"),
            xp=(int(xp[0]), int(xp[1])) if isinstance(xp, list) and len(xp) == 2 else None,
            hp=d.get("hp"),
            hp_rank=hp_rank,
            streak=d.get("streak"),
            xp_cost=d.get("xp_cost"),
            refresh_cost=d.get("refresh_cost"),
            odds=d.get("odds"),
            shop_open=bool(shop.get("open", bool(shop))),
            shop=slots[:5],
            traits=[TraitStatus(t.get("name", ""), t.get("count"), bool(t.get("active")), t.get("next"))
                    for t in d.get("traits") or []],
            players=[Player(p.get("name"), p.get("hp"), bool(p.get("self")), p.get("slot"), p.get("y")) for p in d.get("players") or []],
            viewing=dict(d['viewing']) if isinstance(d.get('viewing'), dict) else None,
            augments=[AugmentOption(a.get("name", ""), a.get("desc", ""), a.get("refresh"))
                      for a in d.get("augments") or []],
            augments_taken=[name for name in d.get("augments_taken", []) if isinstance(name, str) and name.strip()]
                           if isinstance(d.get("augments_taken"), list) else [],
            opponent=dict(d["opponent"]) if isinstance(d.get("opponent"), dict) else None,
            placement=d.get("placement"),
            bench=[Unit(u.get("name") or "?", int(u.get("star") or 1), "bench", slot=u.get("slot"),
                        items=[dict(i) for i in u.get("items") or []])
                   for u in d.get("bench") or []],
            board=[Unit(u.get("name") or "?", int(u.get("star") or 1), "board", x=u.get("x"), y=u.get("y"),
                        items=[dict(i) for i in u.get("items") or []])
                   for u in d.get("board") or []],
            board_count=(int(bc[0]), int(bc[1])) if isinstance(bc, list) and len(bc) == 2 else None,
            legend=(lg["x"], lg["y"]) if isinstance(lg, dict) and "x" in lg else None,
            units_stale=bool(d.get("units_stale")),
            players_stale=bool(d.get("players_stale")),
            frame=d.get("_frame"),
            raw=d,
            item_bench=[dict(i) for i in d.get("item_bench") or []],
        )

    # ---- 派生 ----
    @property
    def stage_tuple(self) -> tuple[int, int] | None:
        """"3-2" → (3, 2)；读不到时 None。"""
        if not self.stage or "-" not in self.stage:
            return None
        try:
            a, b = self.stage.split("-", 1)
            return int(a), int(b)
        except ValueError:
            return None

    @property
    def units(self) -> list[Unit]:
        return self.board + self.bench

    def owned_copies(self) -> Counter[str]:
        """自己持有的每个英雄折合几张 1 星卡（棋盘 + 备战席）。"""
        c: Counter[str] = Counter()
        for u in self.units:
            if u.known:
                c[u.name] += u.copies
        return c

    @property
    def bench_free(self) -> int:
        return 9 - len(self.bench)

    @property
    def shop_names(self) -> list[str | None]:
        return [s.name if s else None for s in self.shop]

    @property
    def can_act(self) -> bool:
        """是否处于可以买卖/刷新的阶段。战斗中商店也能用，但棋盘不能动。"""
        return self.phase in (Phase.PLANNING, Phase.COMBAT) and self.shop_open

    def summary(self) -> str:
        parts = [f"{self.stage or '?'}", f"{self.phase.value}"]
        if self.gold is not None:
            parts.append(f"金{self.gold}")
        if self.level is not None:
            parts.append(f"Lv{self.level}")
        if self.hp is not None:
            parts.append(f"血{self.hp}")
        return " ".join(parts)
