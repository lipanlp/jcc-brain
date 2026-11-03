"""静态阵容详情的只读装配；不读取对局状态，不推断持有/合成就绪。"""
from __future__ import annotations

from collections import Counter
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jccbrain.data.db import GameDB
    from jccbrain.data.models import Comp, Guide, GuideHero

from jccbrain.data.positions import parse_pos


SECTIONS = [('early_info', '前期过渡'), ('d_time', 'D 牌节奏'), ('location_info', '站位思路'),
            ('enemy_info', '克制应对'), ('hex_info', '海克斯思路'), ('equipment_info', '装备思路'),
            ('adventure_info', '自然仙灵思路')]


def _traits(units: list[dict], db: GameDB) -> list[dict]:
    counts = Counter()
    for name in {u['name'] for u in units if u['is_hero']}:
        hero = db.hero(name)
        if hero:
            counts.update(set(hero.traits))
    out = []
    for name, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        trait = db.trait(name)
        standard = trait is not None and trait.activation != 'special'
        out.append({'name': name, 'count': count, 'known': trait is not None,
                    'activation': trait.activation if trait else 'unknown',
                    'tier': trait.tier_of(count) if standard else None,
                    'breakpoints': list(trait.breakpoints) if trait else [],
                    'description': trait.description if trait else '',
                    'levels': [{'count': level.count, 'text': level.text} for level in trait.levels] if trait else [],
                    'level_desc': list(trait.level_desc) if trait else [], 'picture_key': 'trait:' + name})
    return out


def _stage(key: str, label: str, units: list[dict], db: GameDB, *,
           origin: str, source_name: str, population: int | None = None) -> dict:
    positions = [parse_pos(unit['position']) for unit in units]
    counts = Counter(pos for pos in positions if pos)
    for index, (unit, pos) in enumerate(zip(units, positions)):
        raw = unit['position']
        unit['key'] = f'{key}:{index}'
        unit['position_raw'] = ','.join(map(str, raw)) if isinstance(raw, (tuple, list)) else str(raw or '')
        unit['position_note'] = ('坐标冲突，原位 ' + unit['position_raw'] if pos and counts[pos] > 1 else
                                 '未提供站位' if raw is None or raw == '' else '坐标无效或越界') if not pos or counts[pos] > 1 else ''
        unit['position'] = list(pos) if pos and counts[pos] == 1 else None
    return {'id': key, 'label': label, 'origin': origin, 'source_name': source_name, 'population': population,
            'hero_count': sum(u['is_hero'] for u in units), 'other_count': sum(not u['is_hero'] for u in units),
            'units': units, 'traits': _traits(units, db)}


def _guide_units(guide: Guide, heroes: list[GuideHero], roles: dict[str, str] | None = None) -> list[dict]:
    return [{'name': h.name, 'hero_id': h.hero_id, 'unit_type': h.unit_type, 'is_hero': h.unit_type == 'hero',
             'cost': h.cost, 'role': (roles or {}).get(h.name) or ('主C' if h.is_carry else ''),
             'target_star': 3 if h.unit_type == 'hero' and h.name in guide.three_star else h.star,
             'source_star': h.star,
             'position': h.position, 'items': list(h.items)} for h in heroes]


def static_detail(comp: Comp, guide: Guide | None, db: GameDB) -> dict:
    """guide 必须取自经过来源过滤的 comp，不沿原始库反查其他关联攻略。"""
    stages = []
    if comp.source == 'official' and guide and guide.boards.get('final'):
        roles = {h.name: '主C' if h.is_carry else '副C' if h.is_subcarry else '核心' if h.is_core else '' for h in comp.heroes}
        stages.append(_stage('final', '成型', _guide_units(guide, guide.boards['final'], roles), db,
                             origin='comp', source_name=comp.name))
    else:
        units = [{'name': h.name, 'hero_id': h.hero_id, 'unit_type': 'hero', 'is_hero': True, 'cost': h.cost,
                  'role': '主C' if h.is_carry else '副C' if h.is_subcarry else '核心' if h.is_core else '',
                  'target_star': h.star_target, 'source_star': None, 'position': h.position, 'items': list(h.items)} for h in comp.heroes]
        stages.append(_stage('comp', '成型', units, db, origin='comp', source_name=comp.name))
    if guide:
        def order(phase):
            return (0 if phase == 'early' else 1 if phase == 'mid' else 2 if phase.startswith('L') else 3,
                    int(phase[1:]) if re.fullmatch(r'L\d+', phase) else 0, phase)
        for phase in sorted(guide.boards, key=order):
            if phase == 'final' and comp.source == 'official' or not guide.boards[phase]:
                continue
            population = int(phase[1:]) if re.fullmatch(r'L[1-9]\d*', phase) else None
            label = {'early': '前期', 'mid': '中期', 'final': '成型'}.get(phase, f'{population} 人口' if population else phase)
            linked = comp.source != 'official'
            stages.append(_stage(('guide:' if linked else '') + phase, ('攻略·' if linked else '') + label,
                                 _guide_units(guide, guide.boards[phase]), db, origin='guide' if linked else 'comp',
                                 source_name=guide.name, population=population))

    names = {u['name'] for stage in stages for u in stage['units'] if u['is_hero']}
    heroes = {}
    for name in sorted(names):
        h = db.hero(name)
        heroes[name] = {'name': name, 'known': h is not None, 'id': h.id if h else None, 'cost': h.cost if h else None,
                        'traits': list(h.traits) if h else [], 'role_tag': h.role_tag if h else None,
                        'attack_range': h.attack_range if h else None, 'skill_name': h.skill_name if h else None,
                        'skill_desc': h.skill_desc if h else None}
    item_names = {name for stage in stages for u in stage['units'] for name in u['items']}
    if guide:
        item_names.update(guide.carry_items + guide.carry_items_backup + guide.equipment_order)
    by_name = {item.name: item for item in db.items.values()}
    items = {}
    todo = list(item_names)
    while todo:
        name = todo.pop()
        if name in items:
            continue
        item = by_name.get(name) or db.items.get(name)
        components = [{'id': iid, 'name': db.item_name(iid), 'known': iid in db.items} for iid in item.recipe] if item else []
        items[name] = {'name': name, 'known': item is not None, 'id': item.id if item else None,
                       'kind': item.kind if item else None, 'description': item.description if item else '',
                       'components': components}
        todo.extend(part['name'] for part in components if part['name'] not in items)
    augment_names = {a.name for a in comp.augments}
    if guide:
        augment_names.update(guide.recommended_augments + guide.alternative_augments)
    augments = {}
    for name in sorted(augment_names):
        a = db.augment(name)
        augments[name] = {'name': name, 'known': a is not None, 'tier': a.tier if a else None,
                          'description': a.description if a else ''}
    mechanics = {}
    if guide:
        for name in {name for group in guide.adventures.values() for name in group}:
            m = db.mechanics.get(name)
            mechanics[name] = {'name': name, 'known': m is not None, 'description': m.description if m else '',
                               'price': m.price if m else None, 'upgrades': list(m.upgrades) if m else []}
    return {'stages': stages, 'heroes': heroes, 'items': items, 'augments': augments, 'mechanics': mechanics,
            'guide': {'name': guide.name, 'sections': [{'key': key, 'title': title, 'text': getattr(guide, key)}
                                                     for key, title in SECTIONS if getattr(guide, key)],
                      'carry_items': list(guide.carry_items), 'carry_items_backup': list(guide.carry_items_backup),
                      'alternative_augments': list(guide.alternative_augments),
                      'adventures': [{'stage': stage, 'names': list(names)} for stage, names in guide.adventures.items()]}
            if guide else None}
