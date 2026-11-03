"""阵容与攻略共用的站位坐标：第 1 行为前排，行列均从 1 开始。"""
from __future__ import annotations


def parse_pos(value) -> tuple[int, int] | None:
    if isinstance(value, str):
        parts = value.split(',')
        if len(parts) != 2:
            return None
        try:
            row, col = (int(part.strip()) for part in parts)
        except ValueError:
            return None
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        row, col = value
        if type(row) is not int or type(col) is not int:
            return None
    else:
        return None
    return (row, col) if 1 <= row <= 4 and 1 <= col <= 7 else None
