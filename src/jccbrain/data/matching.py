"""官方攻略与统计阵容的保守关联（赛季内、主 C 加权，允许不匹配）。"""
from __future__ import annotations

from jccbrain.data.models import Comp, Guide, Hero


def similarity(guide: Guide, comp: Comp, heroes: dict[str, Hero]) -> tuple[float, bool, int]:
    def canonical(hid):
        h = heroes.get(hid)
        return h.pool_id if h else hid

    gw, cw = {}, {}
    for h in guide.heroes:
        hid = canonical(h.hero_id)
        gw[hid] = max(gw.get(hid, 0), 3.0 if h.is_carry else 1.0)
    for h in comp.heroes:
        hid = canonical(h.hero_id)
        cw[hid] = max(cw.get(hid, 0), h.weight)
    common, union = set(gw) & set(cw), set(gw) | set(cw)
    if not union:
        return 0.0, False, 0
    # 双边最大权重确保缺失任一侧主 C 都显著降低分数。
    score = sum(max(gw[i], cw[i]) for i in common) / sum(max(gw.get(i, 0), cw.get(i, 0)) for i in union)
    gc = {canonical(h.hero_id) for h in guide.carries}
    cc = {canonical(h.hero_id) for h in comp.heroes if h.is_carry}
    carry_ok = bool(gc & cc) if gc and cc else set(gw) == set(cw)
    return score, carry_ok, len(common)


def candidates(guide: Guide, comps: dict[str, Comp], heroes: dict[str, Hero]) -> list[tuple[float, Comp]]:
    rows = []
    for c in comps.values():
        score, carry_ok, common = similarity(guide, c, heroes)
        if carry_ok and common >= min(4, len(guide.heroes), len(c.heroes)) and common:
            rows.append((score, c))
    return sorted(rows, key=lambda r: (-r[0], r[1].id))


# 18.18.5 的 63 攻略/38 统计阵容实测后设定；低分与模糊候选保持独立。
MIN_SCORE = 0.72
MIN_MARGIN = 0.08


def link_guides(guides: dict[str, Guide], comps: dict[str, Comp], heroes: dict[str, Hero]) -> None:
    for c in comps.values():
        c.guide, c.guides, c.guide_id, c.category = None, [], None, None
    for g in guides.values():
        g.comp, g.match_score = None, None
        rows = candidates(g, comps, heroes)
        if not rows or rows[0][0] < MIN_SCORE:
            continue
        score, comp = rows[0]
        if len(rows) > 1 and score - rows[1][0] < MIN_MARGIN:
            continue
        g.comp, g.match_score = comp, score
        comp.guides.append(g)
    for c in comps.values():
        if c.guides:
            c.guides.sort(key=lambda g: -(g.match_score or 0))  # 同分沿用官方排序
            c.guide = c.guides[0]
            c.guide_id, c.category = c.guide.id, c.guide.category
