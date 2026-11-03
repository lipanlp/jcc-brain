"""成型羁绊按不同英雄计数；只展示已有库的档位、纹章与配方。"""
from collections import Counter

from jccbrain.data.structure import trait_levels


def traits_view(comp, state, db) -> dict:
    heroes = (comp.guide.heroes if comp.guide and comp.guide.heroes else comp.heroes) if comp else []
    counts = Counter()
    for name in {h.name for h in heroes}:
        hero = db.hero(name)
        if hero:
            counts.update(set(hero.traits))
    bench = Counter(i.get('name') for i in state.item_bench if i.get('name'))

    def row(name, count):
        trait = db.trait(name)
        tier = trait.tier_of(count or 0) if trait else 0
        emblem = next((i for i in db.items.values() if i.kind == '转职纹章'
                       and (i.trait_name == name or (trait and i.trait_id == trait.id))), None)
        levels = ({v.count: v.text for v in trait.levels} if trait and trait.levels else
                  {v['count']: v['text'] for v in trait_levels(list(trait.breakpoints), list(trait.level_desc))} if trait else {})
        detail = None
        if emblem:
            parts = [db.item_name(i) for i in emblem.recipe]
            detail = {'name': emblem.name, 'components': parts,
                      'held': list((Counter(parts) & bench).elements()), 'owned': bench[emblem.name] > 0}
        return {'name': name, 'count': count, 'tier_index': tier - 1,
                'next_breakpoint': trait.next_breakpoint(count or 0) if trait else None,
                'picture_key': f'trait:{name}', 'breakpoints': list(trait.breakpoints) if trait else [],
                'description': trait.description if trait else '',
                'level_desc': [levels.get(b, '') for b in trait.breakpoints] if trait else [], 'emblem': detail}

    final = [row(name, count) for name, count in counts.items()
             if (trait := db.trait(name)) and trait.tier_of(count) > 0]
    final.sort(key=lambda t: (-t['count'], t['name']))
    final_names = {t['name'] for t in final}
    current = [{**row(t.name, t.count), 'active': t.active, 'in_comp': t.name in final_names} for t in state.traits]
    return {'final': final, 'current': current}
