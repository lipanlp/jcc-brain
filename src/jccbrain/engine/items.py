"""只消费已有装备识别和库中配方；散件按实有数量分配。"""
from collections import Counter
from dataclasses import dataclass, field

from jccbrain.core.models import GameState
from jccbrain.data.db import GameDB
from jccbrain.data.models import Comp


@dataclass
class HeldItems:
    components: Counter = field(default_factory=Counter)
    completed: list[str] = field(default_factory=list)
    emblems: list[tuple[str, str]] = field(default_factory=list)


def held(state: GameState, db: GameDB) -> HeldItems:
    result = HeldItems()
    by_name = {i.name: i for i in db.items.values()}
    entries = list(state.item_bench) + [i for u in state.units for i in u.items]
    for entry in entries:
        name = entry.get('name')
        if not name or name.startswith('?'):
            continue
        item = by_name.get(name)
        kind = entry.get('kind') or (item.kind if item else '') or ''
        if kind in ('基础', '基础装备', '散件', 'basic', 'component'):
            result.components[name] += 1
        elif '纹章' in kind or kind in ('emblem', '转职') or (item and item.trait_name):
            if item and item.trait_name:
                result.emblems.append((name, item.trait_name))
        elif kind and kind != '-1':
            result.completed.append(name)
    return result


def wishlist(comp: Comp) -> list[str]:
    names = [i for h in comp.carries for i in h.items]
    if comp.guide:
        names += list(comp.guide.carry_items)
        carries = {h.name for h in comp.carries}
        names += [i for h in comp.guide.heroes if h.is_carry or h.name in carries for i in h.items]
    return list(dict.fromkeys(names))[:6]


def recipe(name: str, db: GameDB) -> list[str]:
    item = next((i for i in db.items.values() if i.name == name), None)
    return [db.item_name(i) for i in item.recipe] if item else []


def item_fit(comp: Comp, inventory: HeldItems, db: GameDB) -> tuple[float, list[str]]:
    wishes = wishlist(comp)
    completed, components = Counter(inventory.completed), inventory.components.copy()
    matched = []
    remaining = []
    for name in wishes:
        if completed[name]:
            completed[name] -= 1
            matched.append(name)
        else:
            remaining.append(name)
    for name in remaining:
        parts = recipe(name, db)
        need = Counter(parts)
        if len(parts) == 2 and not need - components:
            components.subtract(need)
            matched.append(name)
    return min(len(matched) / min(len(wishes), 3), 1.0) if wishes else 0.0, matched


def recipient(comp: Comp, name: str) -> str:
    own = next((h.name for h in comp.carries if name in h.items), None)
    if own:
        return own
    if comp.guide:
        own = next((h.name for h in comp.guide.carries if name in h.items), None)
        if own:
            return own
    fallback = comp.carries or (comp.guide.carries if comp.guide else [])
    return next((h.name for h in comp.heroes if h.is_carry), fallback[0].name if fallback else '主C')


def plan(comp: Comp, inventory: HeldItems, state: GameState, db: GameDB, evidence_comps=None) -> list[dict]:
    # inventory保留全局持有/阵容契合语义；分配必须独立读取装备栏，不能借已穿散件。
    from jccbrain.engine.item_plan import plan as allocate
    return allocate(comp, state, db, evidence_comps)


def carousel_hint(comp: Comp, inventory: HeldItems, db: GameDB) -> str:
    completed, components = Counter(inventory.completed), inventory.components.copy()
    needs = []
    seen = set()
    for name in wishlist(comp)[:3]:
        if completed[name]:
            completed[name] -= 1
            continue
        parts = recipe(name, db)
        if len(parts) != 2:
            continue
        need = Counter(parts)
        have = need & components
        components.subtract(have)
        for part in (need - have).elements():
            if part not in seen:
                needs.append(f'{part}（补出 {name}）')
                seen.add(part)
    return '选秀优先拿 ' + '、'.join(needs[:2]) if needs else '选秀优先拿主C或主C所需散件'
