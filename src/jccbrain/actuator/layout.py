"""界面坐标布局：所有可点击位置，坐标都是相对游戏窗口（含标题栏）的 0~1，与 game-eye 一致。

默认棋盘按 1774×1136 的 PlayCover 录像拟合（稀疏样本，前排高度为先验）；可标定（`jcc calibrate board`），
标定结果写入 var/layout.json，覆盖默认值。
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jccbrain.core import paths

log = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass
class Layout:
    shop_slots: list[Point] = field(default_factory=lambda: [(0.313 + i * 0.1252, 0.90) for i in range(5)])
    refresh_button: Point = (0.186, 0.949)
    xp_button: Point = (0.186, 0.867)
    lock_button: Point = (0.855, 0.797)
    sell_zone: Point = (0.56, 0.90)                 # 把单位拖到商店区域即出售
    bench_x0: float = 0.256
    bench_step: float = 0.0617
    bench_y: float = 0.70
    augment_options: list[Point] = field(default_factory=lambda: [(0.279, 0.52), (0.499, 0.52), (0.719, 0.52)])
    # 棋盘四角格（1 行 1 列、1 行 7 列、4 行 1 列、4 行 7 列）的中心，1 行 = 最前排
    board_corners: dict[str, Point] = field(default_factory=lambda: {
        "r1c1": (0.3265, 0.405), "r1c7": (0.6715, 0.405),
        "r4c1": (0.3301, 0.604071428571), "r4c7": (0.7339, 0.604071428571),
    })
    board_row_offset: float = 0.490341753343        # 偶数行错开的列宽比例：正向右，负向左
    bar_to_hex_dy: float = 0.058928571429           # 血条到格中心的估计距离；英雄模型可带来偏差
    board_calibrated: bool = False

    # scout-measure/spectate-sample.jpg（493×277，内容截图）头像中心x=477，y=35..211。
    # 默认按归一化坐标外推；窗口标题栏/画幅差异须用玩家栏两点标定覆盖。
    player_slots: list[Point] = field(default_factory=lambda: [(477 / 493, (35 + i * 176 / 7) / 277) for i in range(8)])
    player_slots_calibrated: bool = False

    # ---------------------------------------------------------------- 坐标
    def bench_slot(self, slot: int) -> Point:
        return (self.bench_x0 + slot * self.bench_step, self.bench_y)

    def hex_center(self, row: int, col: int) -> Point:
        """棋盘 row(1~4) 行 col(1~7) 列的格中心。四角双线性插值 + 六边形交错。"""
        c = self.board_corners
        # 六边形交错：偶数行按有符号 offset 错位。第 4 行两角为实际位置，先去错位再插值
        off4 = self.board_row_offset * (c["r4c7"][0] - c["r4c1"][0]) / 6
        r4c1 = (c["r4c1"][0] - off4, c["r4c1"][1])
        r4c7 = (c["r4c7"][0] - off4, c["r4c7"][1])
        t = (row - 1) / 3
        left = _lerp(c["r1c1"], r4c1, t)
        right = _lerp(c["r1c7"], r4c7, t)
        step = (right[0] - left[0]) / 6
        shift = self.board_row_offset * step if row % 2 == 0 else 0.0
        s = (col - 1) / 6
        x = left[0] + (right[0] - left[0]) * s + shift
        y = left[1] + (right[1] - left[1]) * s
        return (x, y)

    def all_hexes(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(1, 5) for c in range(1, 8)]

    def nearest_hex(self, bar_x: float, bar_y: float) -> tuple[int, int]:
        """game-eye 给的棋盘单位血条坐标 → 所在格。"""
        x, y = bar_x, bar_y + self.bar_to_hex_dy
        return min(self.all_hexes(), key=lambda rc: _dist2(self.hex_center(*rc), (x, y)))

    def free_hex(self, occupied: set[tuple[int, int]], prefer: tuple[int, int] | None = None,
                 back: bool = False) -> tuple[int, int] | None:
        """找空格：优先阵容推荐位，否则近战从前排、远程从后排，居中优先。"""
        if prefer and prefer not in occupied and 1 <= prefer[0] <= 4 and 1 <= prefer[1] <= 7:
            return prefer
        rows = [4, 3, 2, 1] if back else [1, 2, 3, 4]
        cols = [4, 3, 5, 2, 6, 1, 7]
        for r in rows:
            for c in cols:
                if (r, c) not in occupied:
                    return (r, c)
        return None

    # ---------------------------------------------------------------- 持久化
    @classmethod
    def load(cls, path: Path | None = None) -> "Layout":
        path = path or paths.var_dir() / "layout.json"
        lay = cls()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if hasattr(lay, k):
                        setattr(lay, k, _tuplify(v))
            except (OSError, ValueError) as e:
                log.warning("layout.json 读取失败，使用默认布局：%s", e)
        return lay

    def save(self, path: Path | None = None) -> None:
        path = path or paths.var_dir() / "layout.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _tuplify(v):
    if isinstance(v, list):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return (float(v[0]), float(v[1]))
        return [_tuplify(x) for x in v]
    if isinstance(v, dict):
        return {k: _tuplify(x) for k, x in v.items()}
    return v
