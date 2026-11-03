"""海克斯情境启发式；关键词与加减分集中维护，不声称是胜率预测。"""
from collections import Counter
import re

from jccbrain.core.models import GameState
from jccbrain.data.structure import relation_ids
from jccbrain.engine import items


# 名字与库中描述优先，未知选项可用视觉描述。允许多类；词命中不代表有阵容统计。
# tier 只参与通用强度估计，不能凭白银/黄金/棱彩推断其功能。
KEYWORDS = {
    'item': ('装备', '散件', '成装', '锻造器', '纹章', '潘多拉', '重铸器'),
    'econ': ('金币', '利息', '经验', '刷新', '免费刷新', '给钱', '经济'),
    'trait': ('羁绊',),
    'hero': ('弈子', '英雄', '星级', '三星', '两星', '二星'),
    'combat': ('攻击', '攻速', '法术', '暴击', '生命', '护甲', '魔抗', '魔法抗性',
               '伤害', '增伤', '护盾', '治疗', '吸血', '战斗加成', '属性', '法力'),
}

# 局势分以0.5为中性，命中的项可叠加，最终限制在[0,1]；均为可解释的偏好。
# 低血优先即时战力，避免单纯因便宜/追三星就推荐继续贪经济。
SITUATION_BONUSES = {
    'item_shortage': .30,     # 缺至少2件目标装备、手上至多2个散件
    'econ_tempo': .20,        # 前3阶段且低于30金，或追三星打法
    'econ_danger': -.30,      # 血量低于40仍投入经济的风险
    'trait_match': .25,       # 主羁绊/已有纹章对应羁绊
    'trait_mismatch': -.15,   # 已知关联但与目标不符
    'combat_danger': .25,     # 低血需要即时战力
    'hero_match': .20,        # 描述指定的英雄在目标阵容里
}


def classify(name, db, desc='') -> dict:
    augment = db.augment(name)
    description = (augment.description if augment else '') or desc
    text = name + ' ' + description
    categories = {kind for kind, words in KEYWORDS.items() if any(w in text for w in words)}
    traits = {t.name for t in db.traits.values() if t.name in text}
    # 直接发放具名装备也算装备类；避免把普通词中的单字英雄误判为指定英雄。
    if any(i.name and i.name in text for i in db.items.values()):
        categories.add('item')
    heroes = {h.name for h in db.heroes.values() if h.name and
              (h.name in text if len(h.name) > 1 else f'【{h.name}】' in text)}
    if augment:
        traits.update(augment.trait_names)
        ids = relation_ids(augment.fetter_id)
        traits.update(t.name for t in db.traits.values() if t.id in ids)
        # 空/0/-1是无关联；非零元数据只提供类别，不臆测枚举的具体数值含义。
        if ids or relation_ids(augment.fetter_type):
            categories.add('trait')
        if relation_ids(augment.hero_enhancement_type):
            categories.add('hero')
    if traits:
        categories.add('trait')
    if heroes:
        categories.add('hero')
    return {'category': [k for k in KEYWORDS if k in categories], 'traits': sorted(traits),
            'heroes': sorted(heroes), 'description': description, 'tier': augment.tier if augment else None}


def situation(state, adv, db) -> dict:
    comp = adv.target.comp if adv.target else None
    inventory = items.held(state, db)
    bench = items.held(GameState(item_bench=state.item_bench), db)
    wishlist = items.wishlist(comp) if comp else []
    missing = list((Counter(wishlist) - Counter(inventory.completed)).elements())
    return {'missing_items': missing, 'missing_item_count': len(missing),
            'components': dict(bench.components), 'component_count': sum(bench.components.values()),
            'gold': state.gold, 'low_gold': state.gold < 30 if state.gold is not None else None,
            'streak': state.streak, 'hp': state.hp,
            'danger': state.hp < 40 if state.hp is not None else None,
            'stage': state.stage, 'stage_major': state.stage_tuple[0] if state.stage_tuple else None,
            'style': comp.style if comp else None,
            'target_traits': list(comp.traits[:3]) if comp else [],
            'target_heroes': list(dict.fromkeys(h.name for h in comp.heroes)) if comp else [],
            'missing_heroes': [h.name for h in adv.target.missing] if adv.target else [],
            'held_emblems': [{'name': name, 'trait': trait} for name, trait in inventory.emblems]}


def situation_score(info, profile) -> tuple[float, list[str]]:
    categories = info['category']
    score, reasons = .5, []

    def add(key, reason):
        nonlocal score
        delta = SITUATION_BONUSES[key]
        score += delta
        reasons.append(f'{reason} {delta:+.2f}')

    if 'item' in categories and profile['missing_item_count'] >= 2 and profile['component_count'] <= 2:
        add('item_shortage', f"缺 {profile['missing_item_count']} 件装备、散件 {profile['component_count']} 件 → 装备类")
    if 'econ' in categories:
        early_poor = (profile['low_gold'] and profile['stage_major'] is not None and profile['stage_major'] <= 3)
        if early_poor or profile['style'] == 'reroll':
            add('econ_tempo', '前期不足30金或追三星 → 经济类')
        if profile['danger']:
            add('econ_danger', '血量低于40，先保血 → 经济类')
    if 'trait' in categories:
        wanted = set(profile['target_traits']) | {e['trait'] for e in profile['held_emblems']}
        matched = wanted & set(info['traits'])
        if matched:
            add('trait_match', '契合主羁绊/持有纹章 ' + '、'.join(sorted(matched)))
        elif wanted and info['traits']:
            add('trait_mismatch', '关联羁绊与目标不符')
    if 'combat' in categories and profile['danger']:
        add('combat_danger', '血量低于40，需要战力 → 战斗类')
    if 'hero' in categories and (heroes := set(profile['target_heroes']) & set(info['heroes'])):
        add('hero_match', '指定英雄在目标阵容 ' + '、'.join(sorted(heroes)))
    return min(1., max(0., score)), reasons or ['没有明确的情境加减分']


def rate(name, stats, info, profile) -> dict:
    if stats and stats.placement_delta is not None:
        base = .5 - stats.placement_delta / 2
        source = f'阵容数据：均名 {stats.placement_delta:+.2f}'
        weight = .6
    elif stats and stats.avg_placement is not None:
        base = (8 - stats.avg_placement) / 7
        source = f'阵容数据：平均名次 {stats.avg_placement:.2f}'
        weight = .6
    else:
        base = .3 + .05 * (info['tier'] or 1)
        source = '无该阵容统计，按通用强度估计' + (f"（档位 {info['tier']}）" if info['tier'] else '（档位未知）')
        weight = .5
    base = min(1., max(0., base))
    situ, reasons = situation_score(info, profile)
    return {'score': round(weight * base + (1-weight) * situ, 3),
            'reason': source + '；局势：' + '；'.join(reasons),
            'category': info['category'], 'desc': info['description'][:80],
            'base_score': base, 'situation_score': situ, 'stats_weight': weight}


def recommended_augment(headline: str, options: list[str]) -> str | None:
    """只标记headline里唯一明确出现的当前选项；不从理由/分数猜AI选择。"""
    clean = lambda text: re.sub(r'\s+', '', text)
    text = clean(headline)
    matches = [name for name in options if name and clean(name) in text
               and not re.search(r'(?:不选|别选|不要(?:选)?|不推荐|排除)' + re.escape(clean(name)), text)
               and not re.search(re.escape(clean(name)) + r'(?:不推荐|不选|不合适)', text)]
    # 潘多拉的装备 与 潘多拉的装备 II 这类包含关系，取完整的较长名字。
    matches = [name for name in matches if not any(name != other and clean(name) in clean(other) for other in matches)]
    return matches[0] if len(matches) == 1 else None
