"""模型到 JSON 数据的纯函数；不依赖窗口或 GUI 框架。"""
from __future__ import annotations

from dataclasses import asdict
from math import hypot, sqrt

from jccbrain.core.models import Phase
from jccbrain.engine.board import hex_width, unit_hex


def unit(u, db) -> dict:
    hero = db.hero(u.name)
    return {**{key: getattr(u, key) for key in ('name', 'star', 'zone', 'slot', 'x', 'y', 'known')},
            'cost': hero.cost if hero else None, 'traits': list(hero.traits) if hero else [],
            'items': [dict(item) for item in u.items]}


def buy_projection(projection) -> dict:
    return {'source': asdict(projection.source), 'valid': projection.valid,
            'buys': [asdict(b) for b in projection.buys] if projection.valid else []}


def ai_projection(projection) -> dict:
    return {'source': asdict(projection.source), 'valid': projection.valid,
            'text': projection.text if projection.valid else '',
            'augment': projection.augment if projection.valid else None,
            'options': list(projection.options) if projection.valid else [], 'buy_frame': projection.buy_frame}


def state(st, db) -> dict:
    return {
        **{key: getattr(st, key) for key in ('stage', 'timer', 'gold', 'interest', 'level', 'hp', 'shop_open', 'units_stale')},
        **({'buy_source': asdict(st.buy_source)} if st.buy_source else {}),
        'phase': st.phase.value, 'xp': list(st.xp) if st.xp is not None else None,
        'streak': {key: st.streak.get(key) for key in ('kind', 'n')} if st.streak else None,
        'shop': [asdict(slot) if slot else None for slot in st.shop],
        'traits': [asdict(trait) for trait in st.traits],
        'board': [unit(u, db) for u in st.board], 'bench': [unit(u, db) for u in st.bench],
        'item_bench': [dict(item) for item in st.item_bench],
        'augments_taken': list(st.augments_taken),
        'summary': st.summary(),
    }


def _hud_positions(st, plan, layout) -> tuple[list[dict], list[dict]]:
    positions, arrows = {}, []
    used_board, used_bench = set(), set()
    has_space = st.board_count is not None and st.board_count[0] < st.board_count[1]
    # 主C、主T先分配，保留原站位列表顺序，单位按索引消费（同名也不能复用）。
    ordered = sorted(enumerate(plan.cells), key=lambda pair:
                     0 if pair[1].role == 'carry' or pair[1].is_carry else 1 if pair[1].role == 'tank' else 2)
    for index, cell in ordered:
        x, y = layout.hex_center(cell.row, cell.col)
        w = hex_width(layout, cell.row)
        key = f'{cell.row}:{cell.col}:{cell.name}'
        status, origin, kind = 'missing', None, None
        board = [(i, u) for i, u in enumerate(st.board)
                 if i not in used_board and u.known and u.name == cell.name]
        bench = [(i, u) for i, u in enumerate(st.bench)
                 if i not in used_bench and u.known and u.name == cell.name]
        if board:
            def distance(pair):
                u = pair[1]
                return hypot(u.x - x, u.y + layout.bar_to_hex_dy - y) if u.x is not None and u.y is not None else float('inf')
            i, u = min(board, key=distance)
            used_board.add(i)
            current = unit_hex(u, layout)
            status = 'ok' if current == (cell.row, cell.col) else 'misplaced'
            if status == 'misplaced' and current is not None:
                origin, kind = layout.hex_center(*current), 'move'
        elif bench:
            def distance(pair):
                slot = pair[1].slot
                return hypot(layout.bench_slot(slot)[0] - x, layout.bench_slot(slot)[1] - y) if slot is not None and 0 <= slot <= 8 else float('inf')
            i, u = min(bench, key=distance)
            used_bench.add(i)
            status = 'bench'
            if has_space and u.slot is not None and 0 <= u.slot <= 8:
                origin, kind = layout.bench_slot(u.slot), 'place'
        positions[index] = {'key': key, 'x': x, 'y': y, 'w': w, 'h': w * 2 / sqrt(3),
                            'name': cell.name, 'state': status, 'role': cell.role}
        if origin is not None:
            arrows.append({'key': key, 'x1': origin[0], 'y1': origin[1], 'x2': x, 'y2': y, 'kind': kind})
    return [positions[i] for i in range(len(plan.cells))], arrows


def hud(st, adv, db, layout, cfg, *, projection=None) -> dict:
    """HUD 的定位规则，归一化到目标窗口；不访问窗口或修改模型。"""
    result = {'mode': 'idle' if st is None else 'board', 'opacity': cfg.opacity,
              'scout_notice': None, 'font_size': cfg.font_size, 'units': [], 'shop': [], 'augments': [], 'positions': [], 'arrows': []}
    if projection is not None:
        result['buy_projection'] = buy_projection(projection)
    if st is None:
        return result
    if st.buy_source:
        result['buy_source'] = asdict(st.buy_source)
    if st.phase == Phase.PLANNING and cfg.show_positions and adv and adv.board:
        result['positions'], result['arrows'] = _hud_positions(st, adv.board, layout)
    if st.phase == Phase.AUGMENT and adv and adv.augments:
        result['mode'] = 'augments'
        best = max(adv.augments, key=lambda a: a.score)
        for aug in adv.augments:
            if not 0 <= aug.index < len(layout.augment_options):
                continue
            x, y = layout.augment_options[aug.index]
            result['augments'].append({'key': aug.index, 'x': x, 'y': y - .17,
                                      'text': f"{'规则推荐 · ' if aug is best else ''}{'AI 推荐 · ' if aug.name == adv.ai_augment else ''}评分 {aug.score:.3f}",
                                      'rule_text': f"{'规则推荐 · ' if aug is best else ''}评分 {aug.score:.3f}",
                                      'best': aug is best, 'ai_best': aug.name == adv.ai_augment, 'name': aug.name})
        return result

    wanted = {hero.name for hero in adv.target.comp.heroes} if adv and adv.target else set()
    if cfg.show_names:
        for zone, units in (('board', st.board), ('bench', st.bench)):
            for i, u in enumerate(units):
                if zone == 'board':
                    if u.x is None or u.y is None:
                        continue
                    x, bottom = u.x, u.y - .022
                else:
                    if u.slot is None:
                        continue
                    x, y = layout.bench_slot(u.slot)
                    bottom = y - .075
                hero = db.hero(u.name)
                result['units'].append({
                    'key': f'{zone}:{u.slot if u.slot is not None else i}', 'x': x, 'y_bottom': bottom,
                    'title': f"{'★' * u.star} {u.name if u.known else u.name + ' 未识别'}",
                    'traits': ' · '.join(hero.traits) if hero and cfg.show_traits else '',
                    'cost': hero.cost if hero else 0, 'target': u.name in wanted, 'unknown': not u.known,
                })
    from jccbrain.core.buy_source import buy_eligible, source_matches_state
    current = projection.source if projection else getattr(adv, 'buy_source', None)
    valid = (projection.valid and source_matches_state(current, st)) if projection else (
        source_matches_state(current, st) if current else not st.buy_source)
    eligible = buy_eligible(st) if current or st.buy_source else (
        st.shop_open and st.phase in (Phase.PLANNING, Phase.COMBAT)
        and not st.units_stale and not (st.raw.get('shop') or {}).get('stale'))
    if adv and eligible and valid:
        candidates = projection.buys if projection else adv.buys
        buys = {buy.slot: buy for buy in candidates if 0 <= buy.slot < len(st.shop)
                and st.shop[buy.slot] and st.shop[buy.slot].name == buy.name
                and (not st.shop[buy.slot].cost or st.shop[buy.slot].cost == buy.cost)}
        for i, slot in enumerate(st.shop):
            if slot is None or i >= len(layout.shop_slots):
                continue
            if i in buys:
                kind, text = 'buy', '买 · ' + buys[i].reason
            elif slot.name in wanted:
                kind, text = 'target', '目标卡'
            else:
                continue
            x, _ = layout.shop_slots[i]
            result['shop'].append({'key': i, 'x': x - .059, 'y': .838, 'w': .118, 'h': .15,
                                   'kind': kind, 'text': text})
    return result


def advice(adv) -> dict:
    target = adv.target
    costs = {hero.name: hero.cost for hero in target.comp.heroes
             if type(hero.cost) is int and 1 <= hero.cost <= 5} if target else {}
    best = max(range(len(adv.augments)), key=lambda i: adv.augments[i].score, default=None)
    return {
        'stage': adv.stage, 'phase': adv.phase.value, 'headline': adv.headline, 'locked': adv.locked,
        # Display categories follow the advisor's explicit clause grammar. Original
        # strategy fields stay intact; the client supplies current purchases.
        'headline_summary': '；'.join(part for part in adv.headline.split('；')
                                     if not part.startswith('买 ')) or '观望',
        'selection': adv.selection,
        **({'buy_source': asdict(adv.buy_source)} if adv.buy_source else {}),
        'comp_status': adv.comp_status, 'switch_reason': adv.switch_reason,
        'opponents': adv.opponents, 'board': asdict(adv.board) if adv.board else None, 'traits_view': adv.traits_view,
        'held_items': adv.held_items, 'augments_taken': list(adv.augments_taken), 'item_plan': adv.item_plan,
        'target': {'id': target.comp.id, 'name': target.name, 'tier': target.comp.tier} if target else None,
        'comps': [{'id': match.comp.id, 'name': match.name, 'tier': match.comp.tier,
                   'avg_placement': match.comp.avg_placement, 'fit': match.fit,
                   'seen': match.seen, 'seen_carry': match.seen_carry, 'owned_copies': match.owned_copies,
                   'why': [why for why in match.why if '均名' not in why],
                   'why_parts': [{'kind': 'shop' if why.startswith('商店有 ') else 'strategy', 'text': why}
                                 for why in match.why if '均名' not in why],
                   'missing': [hero.name for hero in match.missing[:6]]} for match in adv.comps],
        'econ': {**{key: getattr(adv.econ, key) for key in ('reason', 'keep_gold', 'xp_buys', 'roll_budget')},
                 'action': adv.econ.action.value} if adv.econ else None,
        'buys': [{key: getattr(buy, key) for key in ('slot', 'name', 'cost', 'reason')} for buy in adv.buys],
        'sells': [asdict(sell) for sell in adv.sells],
        'augments': [{**asdict(aug), 'best': i == best, 'ai_best': aug.name == adv.ai_augment} for i, aug in enumerate(adv.augments)],
        'augment_situation': adv.augment_situation, 'ai_augment': adv.ai_augment,
        'targets': [{'name': row.name, 'have': row.have, 'need': row.need, 'target_star': row.target_star,
                     'cost': row.odds.cost if row.odds else costs.get(row.name),
                     **{key: getattr(row.odds, key) if row.odds else None
                        for key in ('remaining', 'per_roll', 'gold_for_50')}} for row in adv.targets],
        'items': [{'hero': hero, 'items': list(items)} for hero, items in adv.items.items()],
        'notes': list(adv.notes), 'llm_text': adv.llm_text,
    }


def advice_with_ai(adv, source=None) -> dict:
    result = advice(adv)
    if source is not None:
        result['ai_source'] = asdict(source)
    return result


def comp_row(c) -> dict:
    channels = list(c.channels)
    source_label = ' / '.join(label for channel, label in (('web', '官网'), ('ingame', '微社区'))
                              if channel in channels) if c.source == 'official' else 'dataj'
    if c.source == 'official' and c.category:
        source_label += '·' + c.category
    return {**{key: getattr(c, key) for key in ('id', 'name', 'tier', 'source', 'style', 'avg_placement',
                                              'top4_rate', 'top_rate', 'pick_rate', 'sample_count')},
            'channels': channels, 'source_label': source_label,
            'style_label': {'reroll': '追三星', 'fast8': '速8', 'fast9': '速9'}[c.style],
            'core': [{'name': hero.name, 'carry': hero.is_carry} for hero in c.heroes if hero.is_core or hero.is_carry]}


def comp_detail(c, db) -> dict:
    from jccbrain.webui.comp_detail import static_detail

    guide = c.guide
    return {
        'visualization': static_detail(c, guide, db), 'game_version': c.game_version,
        **comp_row(c), 'traits': list(c.traits),
        'heroes': [{'name': hero.name, 'cost': hero.cost,
                    'role': '主C' if hero.is_carry else '副C' if hero.is_subcarry else '核心' if hero.is_core else '',
                    'star_target': hero.star_target, 'position': list(hero.position) if hero.position else None,
                    'items': list(hero.items)} for hero in c.heroes],
        'augments': [{key: getattr(aug, key) for key in ('name', 'rating', 'avg_placement', 'placement_delta', 'play_rate')}
                     for aug in c.augments[:40]],
        'game_code': c.game_code,
        'guide': {'id': guide.id, 'name': guide.name, 'channels': list(guide.channels),
                  'text': guide.text(), 'author': guide.author, 'recommended_augments': list(guide.recommended_augments),
                  **{key: getattr(guide, key) for key in ('use_count', 'week_use', 'collect_count', 'comment_count', 'stat_date')},
                  'equipment_order': list(guide.equipment_order)} if guide else None,
    }


def db_rows(db, kind: str, query: str = '') -> list[dict]:
    query = query.strip()

    def hit(*values):
        return not query or any(query in (value or '') for value in values)

    if kind == 'heroes':
        return [{'name': h.name, 'cost': h.cost, 'traits': list(h.traits), 'attack_range': h.attack_range,
                 'skill_name': h.skill_name, 'skill_desc': h.skill_desc}
                for h in sorted(db.pool_heroes(), key=lambda h: (h.cost, h.name))
                if hit(h.name, '、'.join(h.traits), h.skill_desc)]
    if kind == 'traits':
        return [{'name': t.name, 'kind': '特殊机制' if t.activation == 'special' else '独有羁绊' if t.unique else
                 '职业' if t.kind == 'class' else '特质', 'breakpoints': list(t.breakpoints),
                 'levels': [{'count': v.count, 'text': v.text, 'values': list(v.values)} for v in t.levels],
                 'description': t.description, 'level_desc': list(t.level_desc)}
                for t in db.traits.values() if hit(t.name, t.description, ' '.join(t.level_desc))]
    if kind == 'items':
        return [{'name': i.name, 'kind': i.kind, 'recipe': [db.item_name(r) for r in i.recipe],
                 'trait_name': i.trait_name, 'description': i.description} for i in db.items.values()
                if i.kind not in (None, '-1') and hit(i.name, i.kind, i.description)]
    if kind == 'augments':
        return [{'name': a.name, 'tier': a.tier, 'tier_label': {1: '白银', 2: '黄金', 3: '棱彩'}.get(a.tier, ''),
                 'description': a.description} for a in db.augments.values() if hit(a.name, a.description)]
    if kind == 'mechanics':
        return [{'name': m.name, 'category': m.category, 'price': m.price, 'rounds': [list(r) for r in m.rounds],
                 'description': m.description, 'upgrades': list(m.upgrades)} for m in db.mechanics.values()
                if hit(m.name, m.category, m.description)]
    if kind == 'encounters':
        return [{'name': e.name, 'description': e.description} for e in db.encounters.values()
                if hit(e.name, e.description)]
    if kind == 'guides':
        return [{'name': g.name, 'quality': g.quality, 'heroes': [h.name for h in g.heroes],
                 'three_star': list(g.three_star), 'text': g.text()}
                for g in db.guides.values() if hit(g.name, g.text(), '、'.join(h.name for h in g.heroes))]
    raise ValueError(f'未知的数据类别：{kind}')
