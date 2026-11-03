"""操作指令（纯数据）。决策层只产出这些对象，由 actuator 执行。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class BuyShop:
    slot: int                      # 商店 0~4

    def describe(self) -> str:
        return f"买商店第 {self.slot + 1} 格"


@dataclass(frozen=True)
class Reroll:
    def describe(self) -> str:
        return "刷新商店"


@dataclass(frozen=True)
class BuyXP:
    def describe(self) -> str:
        return "购买经验"


@dataclass(frozen=True)
class ToggleLock:
    def describe(self) -> str:
        return "锁定/解锁商店"


@dataclass(frozen=True)
class SellBench:
    slot: int                      # 备战席 0~8

    def describe(self) -> str:
        return f"出售备战席第 {self.slot + 1} 格"


@dataclass(frozen=True)
class BenchToBoard:
    slot: int
    row: int                       # 1~4，1 = 最前排
    col: int                       # 1~7

    def describe(self) -> str:
        return f"备战席 {self.slot + 1} → 棋盘 {self.row} 行 {self.col} 列"


@dataclass(frozen=True)
class BoardToBench:
    row: int
    col: int
    slot: int

    def describe(self) -> str:
        return f"棋盘 {self.row} 行 {self.col} 列 → 备战席 {self.slot + 1}"


@dataclass(frozen=True)
class MoveBoard:
    src: tuple[int, int]
    dst: tuple[int, int]

    def describe(self) -> str:
        return f"棋盘 {self.src} → {self.dst}"


@dataclass(frozen=True)
class PickAugment:
    index: int                     # 0~2

    def describe(self) -> str:
        return f"选择第 {self.index + 1} 个海克斯"


@dataclass(frozen=True)
class Wait:
    seconds: float

    def describe(self) -> str:
        return f"等待 {self.seconds:.1f}s"


@dataclass(frozen=True)
class ClickPoint:
    point: tuple[float, float]
    label: str

    def describe(self) -> str:
        return self.label


@dataclass(frozen=True)
class RestoreMouse:
    screen_point: tuple[float, float]

    def describe(self) -> str:
        return f"恢复鼠标到 {self.screen_point}"


Action = Union[BuyShop, Reroll, BuyXP, ToggleLock, SellBench, BenchToBoard, BoardToBench, MoveBoard, PickAugment, Wait, ClickPoint, RestoreMouse]
