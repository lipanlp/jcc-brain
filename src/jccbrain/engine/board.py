"""从已有攻略/阵容中提取当前人口站位，不操作棋盘或推测未知位置。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import hypot, isfinite

from jccbrain.actuator.layout import Layout
from jccbrain.core.models import GameState, Unit
from jccbrain.data.db import GameDB
from jccbrain.data.models import Comp
from jccbrain.data.positions import parse_pos


@dataclass
class BoardCell:
    row: int
    col: int
    name: str
    cost: int
    is_carry: bool
    item: str | None
    star_target: int | None
    owned_star: int | None
    on_board: bool
    items: list[str] = field(default_factory=list)
    role: str | None = None
    traits: list[str] = field(default_factory=list)
    item_details: list[dict] = field(default_factory=list)


@dataclass
class BoardPlan:
    source: str
    population: int
    cells: list[BoardCell]
    next_hint: str | None = None


def hex_width(layout: Layout, row: int) -> float:
    """同行相邻格的横向间距，留出一成描边间隙。"""
    return abs(layout.hex_center(row, 2)[0] - layout.hex_center(row, 1)[0]) * .9


def unit_hex(unit: Unit, layout: Layout) -> tuple[int, int] | None:
    """血条位置映射到棋盘格；缺坐标或偏离棋盘过远时保留未知。"""
    if unit.zone != 'board' or unit.x is None or unit.y is None:
        return None
    x, y = unit.x, unit.y + layout.bar_to_hex_dy
    if not isfinite(x) or not isfinite(y):
        return None
    def distance(rc):
        cx, cy = layout.hex_center(*rc)
        return hypot(x - cx, y - cy)
    nearest = min(layout.all_hexes(), key=distance)
    return nearest if distance(nearest) <= hex_width(layout, nearest[0]) * .6 else None


def _item_details(items: list[str], state: GameState, db: GameDB) -> list[dict]:
    inventory = Counter(i.get('name') for i in state.item_bench if i.get('name'))
    by_name = {i.name: i for i in db.items.values()}
    result = []
    for name in items:
        item = by_name.get(name)
        components = [db.item_name(id) for id in item.recipe] if item else []
        available = inventory.copy()
        held = []
        completed = bool(item and item.kind not in (None, '-1', '基础装备')) or any(
            i.get('name') == name and i.get('kind') in ('成型装备', '成装', '转职纹章', 'completed', 'emblem')
            for i in state.item_bench)
        if available[name] and completed:
            held = [name]
            ready = True
        else:
            for component in components:
                if available[component]:
                    held.append(component)
                    available[component] -= 1
            ready = len(components) == 2 and len(held) == 2
        result.append({'name': name, 'components': components, 'held': held, 'ready': ready})
    return result


def _cells_for(comp: Comp, population: int, state: GameState, db: GameDB) -> tuple[str, list[BoardCell]]:
    guide = comp.guide
    if guide:
        key = f'L{population}'
        if key not in guide.boards:
            levels = [(int(k[1:]), k) for k in guide.boards if k.startswith('L') and k[1:].isdigit()
                      and int(k[1:]) <= population]
            key = max(levels)[1] if levels else 'early' if population <= 5 else 'mid' if population <= 7 else 'final'
        heroes = [h for h in guide.boards.get(key, []) if h.unit_type == 'hero']
        source = f'攻略 {key}'
    else:
        heroes = sorted((h for h in comp.heroes if parse_pos(h.position)),
                        key=lambda h: (not h.is_carry, not h.is_subcarry, not h.is_core, -h.cost))
        source = '成型站位'

    owned: dict[str, int] = {}
    for unit in state.units:
        if unit.known:
            owned[unit.name] = max(owned.get(unit.name, 0), unit.star)
    on_board = {u.name for u in state.board if u.known}
    comp_heroes = {h.name: h for h in comp.heroes}
    seen_names: set[str] = set()
    seen_positions: set[tuple[int, int]] = set()
    cells = []
    selected = []
    for hero in heroes:
        if hero.name not in seen_names:
            seen_names.add(hero.name)
            selected.append(hero)
    for hero in selected[:population]:
        pos = parse_pos(hero.position)
        if pos is None or pos in seen_positions:
            continue
        seen_positions.add(pos)
        ch = comp_heroes.get(hero.name)
        carry = hero.is_carry or bool(ch and ch.is_carry)
        items = list(hero.items or (ch.items if ch else ())
                     or (guide.carry_items if carry and guide else ()))[:3]
        target_star = (hero.star if hero.star is not None else ch.star_target if ch else None) if guide else hero.star_target
        db_hero = db.hero(hero.name)
        cells.append(BoardCell(*pos, hero.name, hero.cost or (db_hero.cost if db_hero else 0), carry,
                               items[0] if items else None, target_star, owned.get(hero.name), hero.name in on_board,
                               items, 'carry' if carry else None, list(db_hero.traits) if db_hero else [],
                               _item_details(items, state, db)))
    tank = max((c for c in cells if c.row <= 2 and not c.is_carry and c.items),
               key=lambda c: (len(c.items), c.cost), default=None)
    if tank:
        tank.role = 'tank'
    return source, cells


def board_for(comp: Comp, population: int | None, state: GameState, db: GameDB) -> BoardPlan | None:
    """population 传已解析的人口；None 时按画面容量、等级依次读取。"""
    if population is None:
        population = state.board_count[1] if state.board_count else state.level
    if not population or population < 1:
        return None
    source, cells = _cells_for(comp, population, state, db)
    if not cells:
        return None
    hint = None
    if population < 10:
        _, following = _cells_for(comp, population + 1, state, db)
        names = {c.name for c in cells}
        added = [c for c in following if c.name not in names][:2]
        if added:
            details = '；'.join(f'上 {c.name}（第{c.row}行第{c.col}列，{"已有" if c.owned_star is not None else "还缺"}）'
                               for c in added)
            hint = f'升 {population + 1} 级：{details}'
    return BoardPlan(source, population, cells, hint)
