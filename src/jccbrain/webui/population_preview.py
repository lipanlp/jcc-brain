"""实时页的只读展示副本；不调用 advise，也不改实际建议或 HUD。"""
from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jccbrain.core.models import GameState
    from jccbrain.data.db import GameDB
    from jccbrain.engine.advisor import Advice
    from jccbrain.engine.odds import PoolState

from jccbrain.engine.board import board_for
from jccbrain.engine.odds import hit_odds
from jccbrain.webui import serialize


def population_preview(state: GameState, advice: Advice, population: int | None,
                       db: GameDB, pool: PoolState) -> dict:
    preview = copy(advice)
    if population is not None:
        preview.board = board_for(advice.target.comp, population, state, db)
        # 保留本轮有效目标星级，不用预览人口改变追星决策。
        downgraded = {row.name for row in advice.targets if row.target_star == 2
                      and any(h.name == row.name and h.star_target == 3 for h in advice.target.comp.heroes)}
        if preview.board:
            preview.board.cells = [replace(cell, star_target=2)
                                   if cell.name in downgraded and cell.star_target == 3 else cell
                                   for cell in preview.board.cells]
        preview.targets = [replace(row, odds=hit_odds(pool, row.name, population, max(row.need, 1)))
                           for row in advice.targets]
    payload = serialize.advice(preview)
    return {'population': population if population is not None else
            (state.board_count[1] if state.board_count else state.level),
            'board': payload['board'], 'targets': payload['targets']}
