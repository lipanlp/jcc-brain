"""只读装备分配：装备栏资产与已穿装备分开记账，不执行任何动作。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from jccbrain.core.models import GameState, Unit
from jccbrain.data.db import GameDB
from jccbrain.data.models import Comp, Hero, Item

BASIC = {'基础', '基础装备', '散件', 'basic', 'component'}


@dataclass(frozen=True)
class Demand:
    key: str
    hero: str
    item: str
    role: str
    priority: int


def _item_role(item: Item | None) -> str | None:
    if not item or not item.description:
        return None
    # 只读成装面板的基础属性；不把正文里对敌人的效果误当佩戴者属性。
    stats = item.description.split('\n', 1)[0]
    ad, ap = '物理加成' in stats, '法术加成' in stats
    if ad != ap:
        return 'AD' if ad else 'AP'
    if not ad and not ap and re.search('生命|护甲|魔抗|魔法抗性', stats) and not re.search('攻速|攻击速度|暴击', stats):
        return '坦克'
    return None


def _preferences(comps: Iterable[Comp]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for comp in comps:
        for hero in comp.heroes:
            for name in hero.items:
                result[hero.name].setdefault(name, f'真实阵容「{comp.name}」为{hero.name}推荐{name}')
        if comp.guide:
            for heroes in comp.guide.boards.values():
                for hero in heroes:
                    if hero.unit_type == 'hero':
                        for name in hero.items:
                            result[hero.name].setdefault(name, f'真实攻略「{comp.guide.name}」为{hero.name}推荐{name}')
    return result


def _profile(hero: Hero | None, db: GameDB, preferences: dict, by_name: dict) -> tuple[str | None, str, bool]:
    if hero is None:
        return None, '缺少英雄资料', False
    variants = {h.role_tag.upper() for h in db.variants(hero.name) if h.role_tag and h.role_tag.upper() in ('AD', 'AP')}
    if len(variants) > 1:
        return None, '同名存在AD/AP形态，当前输入未区分形态', True
    tag = (hero.role_tag or '').upper()
    if tag in ('AD', 'AP', 'TANK', '坦克'):
        return ('坦克' if tag in ('TANK', '坦克') else tag), f'英雄资料角色为{hero.role_tag}', False
    types = Counter(_item_role(by_name.get(name)) for name in preferences.get(hero.name, {}))
    ranked = sorted(((types[r], r) for r in ('AD', 'AP', '坦克')), reverse=True)
    if ranked[0][0] >= 2 and ranked[0][0] > ranked[1][0]:
        role = ranked[0][1]
        return role, f'真实装备偏好包含{types[role]}种{role}装备', False
    damage = [v for v in hero.skill_values if '伤害' in v.name and not v.parse_error]
    if any('物理加成' in v.scaling for v in damage):
        return 'AD', '技能伤害明确含物理加成', False
    defensive = [v for v in hero.skill_values if re.search('护盾|治疗|生命|伤害', v.name) and not v.parse_error]
    if any(re.search('生命上限|护甲|魔抗|魔法抗性', v.scaling) for v in defensive):
        return '坦克', '技能明确按生命或双抗缩放', False
    if damage and all('法术加成' in v.scaling for v in damage):
        # 自带护盾/治疗但无明确定位时，不把AP缩放直接当输出位。
        if not any(re.search('护盾|治疗', v.name) for v in hero.skill_values):
            return 'AP', '技能伤害明确按法术加成缩放', False
    return None, '缺少明确角色或伤害缩放依据', False


def _demands(comp: Comp, by_name: dict) -> list[Demand]:
    rows = {h.name: (list(h.items), h.is_carry, h.is_subcarry) for h in comp.heroes}
    if comp.guide:
        for hero in comp.guide.heroes:
            if hero.name not in rows or not rows[hero.name][0]:
                original = rows.get(hero.name)
                rows[hero.name] = (list(hero.items), (original[1] or hero.is_carry) if original else hero.is_carry,
                                   original[2] if original else False)
        if comp.guide.carry_items:
            main = next((name for name, (_, carry, _) in rows.items() if carry), None)
            if main and not rows[main][0]:
                rows[main] = (list(comp.guide.carry_items), True, False)
    demands = []
    for hero, (names, carry, subcarry) in rows.items():
        for index, item in enumerate(names[:3]):
            role = '主C' if carry else '副C' if subcarry else '前排' if _item_role(by_name.get(item)) == '坦克' else '目标英雄'
            priority = 0 if carry else 1 if subcarry else 2 if role == '前排' else 3
            demands.append(Demand(f'{comp.id}:{hero}:{index}', hero, item, role, priority))
    return sorted(demands, key=lambda row: row.priority)


def plan(comp: Comp, state: GameState, db: GameDB, evidence_comps: Iterable[Comp] | None = None) -> list[dict]:
    by_name = {item.name: item for item in db.items.values()}
    demands = _demands(comp, by_name)
    preferences = _preferences([comp, *(evidence_comps if evidence_comps is not None else db.comps_catalog())])
    names = {u.name for u in state.board if u.known} | {d.hero for d in demands}
    profiles = {name: _profile(db.hero(name), db, preferences, by_name) for name in names}
    duplicate_names = Counter(u.name for u in state.board)
    units = {f'board:{index}': unit for index, unit in enumerate(state.board)
             if unit.known and db.hero(unit.name) and not state.units_stale}
    free_slots = {key: max(3 - len(unit.items), 0) for key, unit in units.items()}

    def describe(key: str, unit: Unit) -> dict:
        index = int(key.split(':')[1])
        star = {1: '一星', 2: '二星', 3: '三星', 4: '四星'}.get(unit.star, f'{unit.star}星')
        label = star + unit.name + (f'（棋盘第{index + 1}个）' if duplicate_names[unit.name] > 1 else '')
        return {'key': key, 'index': index, 'name': unit.name, 'star': unit.star, 'label': label}

    def choose(demand: Demand) -> tuple[dict | None, str]:
        required = _item_role(by_name.get(demand.item)) or profiles.get(demand.hero, (None, '', False))[0]
        candidates = []
        for key, unit in units.items():
            if not free_slots[key]:
                continue
            role, why, ambiguous = profiles[unit.name]
            final = unit.name == demand.hero
            exact = preferences.get(unit.name, {}).get(demand.item)
            if not final and (ambiguous or (required and role and required != role)):
                continue
            fits = bool(required and role == required)
            if not final and not fits and not exact:
                continue
            evidence = f'目标阵容明确为{demand.hero}配置{demand.item}' if final else why if fits else exact
            if not final and exact:
                evidence += '；' + exact if fits else ''
            candidates.append(((final, fits, unit.star, db.hero(unit.name).cost, -int(key.split(':')[1])),
                               key, unit, evidence))
        if not candidates:
            why = '棋盘信息陈旧' if state.units_stale else '当前没有已知上场单位' if not units else \
                  '已知上场单位均无空格或缺少可信装备适配依据'
            return None, why
        _, key, unit, evidence = max(candidates, key=lambda row: row[0])
        free_slots[key] -= 1
        return describe(key, unit), evidence

    # 每个装备栏条目是独立资产；未知条目不能当材料。
    components: dict[str, list[dict]] = defaultdict(list)
    completed: dict[str, list[dict]] = defaultdict(list)
    for index, entry in enumerate(state.item_bench):
        name = entry.get('name')
        item = by_name.get(name)
        if not item or not name or name.startswith('?'):
            continue
        asset = {'source': 'item_bench', 'index': index, 'name': name}
        if (entry.get('kind') or item.kind) in BASIC and not item.recipe:
            components[name].append(asset)
        elif item.recipe or (item.kind and item.kind not in ('-1', *BASIC)):
            completed[name].append(asset)

    worn = []
    if not state.units_stale:
        for zone, group in [('board', state.board), ('bench', state.bench)]:
            for index, unit in enumerate(group):
                for item_index, item in enumerate(unit.items):
                    if item.get('name') in by_name:
                        worn.append({'unit': unit, 'key': f'{zone}:{index}', 'zone': zone,
                                     'index': index, 'item_index': item_index, 'name': item['name']})
    claimed: dict[str, dict] = {}
    used_worn = set()
    # 先满足各最终英雄本人已有的装备，避免把副C已穿的同件错误划给主C。
    for exact_owner in (True, False):
        for demand in demands:
            if demand.key in claimed:
                continue
            match = next((i for i, entry in enumerate(worn) if i not in used_worn and entry['name'] == demand.item
                          and (not exact_owner or entry['unit'].known and entry['unit'].name == demand.hero)), None)
            if match is not None:
                claimed[demand.key] = worn[match]
                used_worn.add(match)

    reserved: dict[str, tuple[str, list[dict], list[str]]] = {}
    parts_by_demand = {}
    for demand in demands:
        item = by_name.get(demand.item)
        parts = [db.item_name(i) for i in item.recipe] if item else []
        parts_by_demand[demand.key] = parts if len(parts) == 2 and all(by_name.get(p) for p in parts) else []
        if demand.key not in claimed and completed[demand.item]:
            reserved[demand.key] = ('佩戴', [completed[demand.item].pop(0)], [])
    # 完整合成优先，备料不得抢走能合成另一件的材料。
    for demand in demands:
        if demand.key in claimed or demand.key in reserved:
            continue
        parts = parts_by_demand[demand.key]
        if parts and all(len(components[name]) >= count for name, count in Counter(parts).items()):
            assets = [components[name].pop(0) for name in parts]
            reserved[demand.key] = ('合成', assets, [])
    for demand in demands:
        if demand.key in claimed or demand.key in reserved:
            continue
        parts = parts_by_demand[demand.key]
        assets, missing = [], []
        for name in parts:
            if components[name]:
                assets.append(components[name].pop(0))
            else:
                missing.append(name)
        if assets:
            reserved[demand.key] = ('备料', assets, missing)

    final_members = {h.name for h in comp.heroes if h.is_core or h.is_carry or h.is_subcarry}
    if comp.guide:
        final_members |= {h.name for h in comp.guide.heroes}
    result = []
    for demand in demands:
        parts = parts_by_demand[demand.key]
        base = {'key': demand.key, 'item': demand.item, 'components': parts, 'final_to': demand.hero,
                'give_to': '', 'current_unit': None, 'ready': False, 'assets': [], 'missing': [], 'existing_item': None}
        existing = claimed.get(demand.key)
        if existing:
            unit, key = existing['unit'], existing['key']
            if unit.known and unit.name == demand.hero and existing['zone'] == 'board':
                continue
            current = describe(key, unit) if key in units else None
            holder = current['label'] if current else f'备战席{unit.name}' if unit.known else '未识别单位'
            reason = f'{holder}已穿{demand.item}，不是装备栏空闲装备；最终给{demand.hero}，暂不重复合成'
            if current and unit.name != demand.hero:
                reason += f'；{demand.hero}到位且有空格后，可用拆卸器转交'
                if unit.name not in final_members:
                    reason += '；该单位确需下场时才可出售回收'
            result.append({**base, 'action': '暂留', 'current_unit': current,
                           'give_to': current['name'] if current else '',
                           'existing_item': {**{k: v for k, v in existing.items() if k != 'unit'},
                                             'holder': holder}, 'reason': reason})
            continue
        allocation = reserved.get(demand.key)
        if not allocation:
            continue
        action, assets, missing = allocation
        if action == '备料':
            reason = f'装备栏保留{"、".join(a["name"] for a in assets)}，还差 {"、".join(missing)} 可合成{demand.item}，最终给{demand.hero}'
            result.append({**base, 'action': action, 'assets': assets, 'missing': missing, 'reason': reason})
            continue
        current, why = choose(demand)
        if current:
            reason = f'{action} {demand.item}' + (f'（{"+".join(parts)}）' if action == '合成' else '（来自装备栏）')
            reason += f'给{current["label"]}；{why}'
            if current['name'] != demand.hero:
                reason += f'；先过渡，{demand.role} {demand.hero} 到位且有空格后拆卸转交'
                if current['name'] not in final_members:
                    reason += '；该单位确需下场时才可出售回收'
        else:
            reason = f'暂留{demand.item}' + ('的合成材料' if action == '合成' else '')
            reason += f'在装备栏，最终给{demand.hero}；{why}，先确认后再合成或佩戴'
        result.append({**base, 'action': action if current else '暂留', 'assets': assets,
                       'current_unit': current, 'give_to': current['name'] if current else '',
                       'ready': current is not None, 'reason': reason})
    if not result:
        reason = ('主目标缺少明确装备需求' if not demands else '目标装备已由最终英雄穿戴，无需重复合成'
                  if len(claimed) == len(demands) else '装备栏暂无可分配的已知成装或配方材料，已穿装备不能跨单位借用')
        result.append({'key': f'{comp.id}:hold', 'item': '', 'components': [], 'final_to': '', 'give_to': '',
                       'current_unit': None, 'action': '暂留', 'ready': False, 'assets': [], 'missing': [],
                       'existing_item': None, 'reason': reason})
    available_components = sum(1 for entry in state.item_bench if entry.get('name') in by_name
                               and (entry.get('kind') or by_name[entry['name']].kind) in BASIC)
    if ((state.stage_tuple or (0, 0))[0] == 2 and available_components >= 2
            and ((state.hp is not None and state.hp < 70) or (state.streak or {}).get('kind') == 'loss'
                 and ((state.streak or {}).get('n') or 0) >= 2)):
        result[0]['reason'] += '；2 阶段别囤装备，有可靠适配时优先合成保血，未知或满格时暂留'
    return result
