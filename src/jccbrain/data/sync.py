"""同步游戏数据到本地数据库。

数据源可多选（设置 data.source_official / data.source_dataj），官方资料库是权威：
- 官方资料库（official.py）：英雄、羁绊、装备、海克斯、自然仙灵、开局奇遇、官方阵容攻略，全部以官方为准；
  官方取不到时（且选了 dataj）退回 dataj 的基础数据。
- dataj.cc：阵容统计（排行、名次、阵容内海克斯表现、过渡站位），并补官方没有的搜索别名、英雄 AD/AP 标签、
  自然仙灵出现回合和分类。阵容里的英雄名、费用、装备名一律按官方数据改写。dataj 只覆盖部分赛季（Season.dataj_set）。

    jcc db sync                      全量同步当前赛季
    jcc db sync --quick              不拉每个阵容的详情
    jcc db sync --season 16_S19      同步指定赛季
    jcc db sync --sources official   只用官方资料库

每次只替换这一个赛季的数据，其他赛季不受影响。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

from jccbrain.core.seasons import Season
from jccbrain.data import official as off
from jccbrain.data.dataj import DatajClient, DatajError
from jccbrain.data.db import GameDB, _j
from jccbrain.data.hero_details import hero_key, parse_skill_values, star_row
from jccbrain.data.units import classify_units
from jccbrain.data.structure import activation, flag, item_kind, linked_trait, trait_levels

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


def _split_ids(v: str | None) -> list[str]:
    return [x for x in (v or "").split("|") if x and x != "-1"]


# ------------------------------------------------------------------ 基础数据转换
def convert_gamedata(data: dict) -> dict[str, list[dict]]:
    """dataj /gamedata 原始结构 → 各表行。纯函数，便于测试。"""
    # 羁绊：同一个 checkId 每档一条记录，取第一条
    traits: dict[str, dict] = {}
    for t in data.get("trait", []):
        cid = t["checkId"]
        if cid in traits:
            continue
        traits[cid] = {
            "id": cid, "name": t["name"], "kind": "class" if str(t.get("type")) == "1" else "origin",
            "breakpoints": _j([int(x) for x in (t.get("numList") or "").split("|") if x.isdigit()]),
            "description": t.get("prefix") or "",
            "level_desc": _j([x for x in (t.get("desc2Text") or "").split("|") if x]),
            "picture": t.get("picture"),
        }
        row = traits[cid]
        points, descriptions = json.loads(row["breakpoints"]), json.loads(row["level_desc"])
        row.update(levels=_j(trait_levels(points, descriptions)), activation=activation(points))

    # 元数据取最低星级记录，所有星级（包括召唤物）独立入表。
    heroes: list[dict] = []
    hero_stars = []
    seen_ids: set[str] = set()
    # 按星级、id 排序，单位元数据取最低星级。
    for h in sorted(data.get("hero", []), key=lambda h: str(h["id"])):
        key = hero_key(h)
        if key is None:
            continue
        hid, star = key
        hero_stars.append(star_row(h))
        if hid in seen_ids:
            continue
        seen_ids.add(hid)
        cost = int(h.get("price") or 0)
        trait_ids = _split_ids(h.get("raceId")) + _split_ids(h.get("jobId"))
        names = [traits[t]["name"] for t in trait_ids if t in traits]
        heroes.append({
            "id": hid, "name": h["name"], "cost": cost, "traits": _j(names), "trait_ids": _j(trait_ids),
            "in_pool": 0, "role_tag": h.get("tag"), "attack_range": h.get("attackRange"),
            "hp": h.get("initHp") if star == 1 else None, "armor": h.get("armor") if star == 1 else None,
            "magic_resist": h.get("magicResist") if star == 1 else None,
            "attack_damage": h.get("initAttackDamage") if star == 1 else None,
            "attack_speed": float(h.get("attackSpeed") or 0) if star == 1 else None,
            "init_mana": h.get("initMp") if star == 1 else None, "max_mana": h.get("maxMp") if star == 1 else None,
            "skill_name": h.get("skillName"), "skill_desc": h.get("skillDesc"), "picture": h.get("picture"),
            "skill_values": _j(parse_skill_values(h.get("skillValueDesc"), h.get("skillBriefValue"))),
            "skill_brief_values": _j((h.get("skillBriefValue") or "").split("|") if h.get("skillBriefValue") else []),
            "sell_price": h.get("sellPrice"), "skill_icon": h.get("skillIcon"), "raw": _j(h),
        })
    classify_units(heroes)

    items = [{
        "id": str(e["id"]), "name": e["name"], "kind": item_kind(e.get("type"), e["name"]),
        "raw_kind": e.get("type"), "fetter_id": e.get("fetterId"),
        "trait_id": linked_trait(e.get("fetterId"), e["name"], e.get("type"), traits),
        "effect_type": e.get("effectType"),
        "description": "\n".join(dict.fromkeys(x for x in (e.get("basicDesc"), e.get("descText")) if x)),
        "recipe": _j([x for x in (e.get("synthesis1"), e.get("synthesis2")) if x and x != "0"]),
        "picture": e.get("picture"),
    } for e in data.get("equip", [])]

    augments = [{
        "id": str(a["id"]), "name": a["name"], "tier": a.get("level"),
        "description": a.get("descText") or "", "icon": a.get("icon"),
        "fetter_id": a.get("fetterId"), "fetter_type": a.get("fetterType"),
        "hero_enhancement_type": a.get("heroEnhancementType"), "is_legend": flag(a.get("isLegend")),
    } for a in data.get("hex", [])]

    mechanics = [{
        "id": m["id"], "name": m["name"], "category": m.get("categoryLabel"), "price": m.get("price"),
        "description": m.get("descText") or "", "rounds": _j(m.get("roundSpans") or []),
        "upgrades": _j(m.get("upgrades") or []),
    } for m in data.get("adventure", [])]

    aliases: dict[tuple[str, str], dict] = {}
    for a in data.get("searchAlias", []):
        for kw in a.get("keywords") or []:
            aliases[(a["entityType"], kw)] = {"entity_type": a["entityType"], "name": a["name"], "keyword": kw}

    return {"traits": list(traits.values()), "heroes": heroes, "hero_stars": hero_stars, "items": items, "augments": augments,
            "mechanics": mechanics, "aliases": list(aliases.values())}


def convert_comp(c: dict, items: dict[str, str], game_version: str | None,
                 detail: dict | None = None) -> tuple[dict, list[dict]]:
    """阵容排行条目（可附详情）→ comps 行 + comp_heroes 行。"""
    src = detail or c
    positions = {h["heroId"]: h.get("position") for h in (detail or {}).get("heroes", [])}
    equips: dict[str, list[str]] = {}
    for e in src.get("equips", []):
        equips.setdefault(e["heroId"], []).append(items.get(e["equipId"], e["equipId"]))
    comp = {
        "id": str(c["compId"]), "name": c["name"], "tier": c.get("tier"),
        "avg_placement": c.get("avgPlacement"), "top4_rate": c.get("top4Rate"), "top_rate": c.get("topRate"),
        "pick_rate": c.get("pickRate"), "sample_count": c.get("sampleCount"), "game_version": game_version,
        "game_code": (detail or {}).get("gameCode"),
        "traits": _j([t["name"] for t in src.get("traits", [])]),
        "conditions": _j(src.get("conditions", [])),
        "placements": _j((detail or {}).get("placementDistribution", [])),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    heroes = [{
        "comp_id": comp["id"], "hero_id": h["heroId"], "hero_name": h["heroName"], "cost": h.get("price"),
        "is_carry": int(bool(h.get("isCarry"))), "is_subcarry": int(bool(h.get("isSubCarry"))),
        "is_core": int(bool(h.get("isCore"))), "star_target": h.get("heroStarNum"),
        "position": positions.get(h["heroId"]) or h.get("position"),
        "items": _j(equips.get(h["heroId"], [])), "sort": i,
    } for i, h in enumerate(src.get("heroes", []))]
    return comp, heroes


def convert_comp_hexes(comp_id: str, hexes: list[dict]) -> list[dict]:
    return [{
        "comp_id": comp_id, "augment_id": str(h["hexId"]), "name": h.get("name"), "tier": h.get("level"),
        "rating": h.get("tier"), "play_rate": h.get("playRate"), "avg_placement": h.get("avgPlacement"),
        "top4_rate": h.get("top4Rate"), "top_rate": h.get("topRate"),
        "placement_delta": h.get("placementDelta"), "rounds": _j(h.get("rounds") or []),
    } for h in hexes]


def convert_comp_builds(comp_id: str, groups: list[dict], hero_names: dict[str, str]) -> list[dict]:
    rows = []
    for g in groups:
        for b in g.get("builds", [])[:3]:
            rows.append({
                "comp_id": comp_id, "people": g.get("peopleNum"), "rank": b.get("rank") or len(rows),
                "win_rate": b.get("winRate"), "sample_count": b.get("sampleCount"),
                "heroes": _j([{"hero_id": str(h.get("heroBaseId")), "name": hero_names.get(str(h.get("heroBaseId"))),
                               "star": h.get("heroStarNum"), "position": h.get("position")}
                              for h in b.get("heroes", [])]),
            })
    return rows


# ------------------------------------------------------------------ 同步入口
class SyncError(RuntimeError):
    pass


def sync_season(db: GameDB, season: Season, *, official: off.OfficialClient | None = None,
                dataj: DatajClient | None = None, official_version: str | None = None,
                dataj_version: str | None = None, details: bool = True, progress: Progress | None = None) -> dict:
    """同步一个赛季。official / dataj 为 None 表示不用该数据源，至少要有一个。

    official_version：官方资料库版本（默认最新）；dataj_version：dataj 统计版本（默认最新）。
    """
    say = progress or (lambda m: log.info(m))
    sid = season.id
    if official is None and dataj is None:
        raise SyncError("至少要选一个数据源")
    if dataj is not None and season.dataj_set is None:
        say(f"dataj.cc 没有「{season.label}」的数据，只用官方资料库")
        dataj = None
        if official is None:
            raise SyncError(f"「{season.label}」只有官方资料库有数据，请在数据源里勾选官方资料库")

    # 1. 官方资料库：基础数据 + 攻略 + 赛季列表
    off_files: dict[str, list[dict]] | None = None
    off_version = None
    lineups: list[dict] = []
    web_lineups: list[dict] = []
    ingame: list[dict] = []
    stats: dict[str, dict] = {}
    if official is not None:
        try:
            db.save_catalog(official.season_tabs())
        except off.OfficialError as e:
            log.warning("官网赛季列表获取失败：%s", e)
        try:
            entry = official.find_version(season, official_version)
            off_version = entry.get("version")
            say(f"{season.label} 官方资料库版本 {off_version}，拉取基础数据…")
            off_files = official.gamedata(entry)
            say("拉取官方阵容攻略…")
            try:
                lineups = official.lineups(entry, sid)
            except off.OfficialError as e:
                log.warning("官方阵容攻略获取失败：%s", e)
                say(f"官方阵容攻略获取失败：{e}")
            if hasattr(official, "ingame_lineups"):
                say("拉取微社区阵容…")
                try:
                    ingame = official.ingame_lineups(sid)
                except off.OfficialError as e:
                    log.warning("微社区阵容获取失败：%s", e)
                    say(f"微社区阵容获取失败：{e}")
            web_lineups = lineups
            lineups = off.merge_lineups(web_lineups, ingame)
        except off.OfficialError as e:
            if dataj is None:
                raise SyncError(f"官方资料库获取失败：{e}") from e
            log.warning("官方资料库获取失败，改用 dataj 基础数据：%s", e)
            say(f"官方资料库获取失败，改用 dataj 基础数据：{e}")

    # 2. dataj：统计版本 + 基础数据（官方没拿到时作为基础数据；拿到时只用来补官方没有的字段）
    version, dj_data = None, None
    if dataj is not None:
        try:
            version = dataj_version or dataj.latest_version()
            dj_data = dataj.gamedata()
        except DatajError as e:
            if off_files is None:
                raise SyncError(f"dataj 获取失败：{e}") from e
            log.warning("dataj 获取失败，本次只更新官方数据：%s", e)
            say(f"dataj 获取失败，本次只更新官方数据（阵容统计保持不变）：{e}")
            version, dj_data = None, None
    source = "official" if off_files is not None else "dataj"
    tables = convert_gamedata(off.to_gamedata(off_files, dj_data) if off_files is not None else dj_data)
    encounters = off.convert_encounters(off_files.get("galaxy", [])) if off_files else []
    lineup_tags = official.lineup_tags() if lineups else None
    guides, guide_heroes = off.convert_guides(lineups, off_files.get("adventure", []) if off_files else [], lineup_tags)
    if ingame and hasattr(official, "lineup_stats"):
        try:
            stats = official.lineup_stats([g["id"] for g in guides])
        except off.OfficialError as e:
            log.warning("微社区热度获取失败：%s", e)
            say(f"微社区热度获取失败：{e}")
    hit = off.apply_stats(guides, stats)
    ingame_count = sum("ingame" in json.loads(g["channels"]) for g in guides)
    say(f"英雄 {sum(1 for h in tables['heroes'] if h['in_pool'])} / 羁绊 {len(tables['traits'])} / "
        f"装备 {len(tables['items'])} / 海克斯 {len(tables['augments'])} / 自然仙灵 {len(tables['mechanics'])} / "
        f"开局奇遇 {len(encounters)} / 官方攻略 {len(guides)}（微社区 {ingame_count} / 带热度 {hit}）"
        f"（基础数据来源 {source}）")

    # 3. dataj 阵容统计；英雄名、费用、装备名以基础数据（官方）为准
    comps, comp_heroes, comp_augs, comp_builds = [], [], [], []
    if dj_data is not None:
        say(f"拉取阵容排行（dataj {version}）…")
        ranked = dataj.comp_rank(version)
        item_names = {r["id"]: r["name"] for r in tables["items"]}
        hero_rows = {r["id"]: r for r in tables["heroes"]}
        hero_names = {i: r["name"] for i, r in hero_rows.items()}
        aug_names = {r["id"]: r["name"] for r in tables["augments"]}
        for i, c in enumerate(ranked, 1):
            detail = None
            if details:
                say(f"[{i}/{len(ranked)}] {c['name']}")
                try:
                    detail = dataj.comp_detail(c["compId"], version)
                    hexes = convert_comp_hexes(str(c["compId"]), dataj.comp_hexes(c["compId"], version))
                    for h in hexes:
                        h["name"] = aug_names.get(h["augment_id"], h["name"])
                    comp_augs += hexes
                    comp_builds += convert_comp_builds(str(c["compId"]), dataj.comp_builds(c["compId"], version),
                                                       hero_names)
                except DatajError as e:
                    log.warning("阵容 %s 详情获取失败：%s", c["name"], e)
            row, heroes = convert_comp(c, item_names, version, detail)
            for h in heroes:
                if ref := hero_rows.get(str(h["hero_id"])):
                    h["hero_name"], h["cost"] = ref["name"], ref["cost"]
            comps.append(row)
            comp_heroes += heroes

    if off_files is not None and not getattr(official, "offline", False):
        from jccbrain.data.archive import save_archive

        try:
            archive = save_archive(db.path.parent / "raw", sid, entry, off_files, web_lineups, lineup_tags,
                                   getattr(official, "raw_responses", {}), ingame_lineups=ingame,
                                   lineup_stats=stats, posts=getattr(official, "raw_posts", []))
        except OSError as e:
            raise SyncError(f"原始数据存档失败，本次未写入游戏数据：{e}") from e
        say(f"官方原始数据已存档：{archive}")

    with db.transaction() as conn:
        for table in ("traits", "heroes", "hero_stars", "items", "augments", "mechanics", "aliases"):
            db.replace_table(conn, table, tables[table], sid)
        if off_files is not None:
            db.replace_table(conn, "unlock_conditions", off.convert_unlocks(off_files.get("mission", [])), sid)
            db.replace_table(conn, "encounters", encounters, sid)
            if lineups:
                db.replace_table(conn, "guides", guides, sid)
                db.replace_table(conn, "guide_heroes", guide_heroes, sid)
        if dj_data is not None:
            db.replace_table(conn, "comps", comps, sid)
            db.replace_table(conn, "comp_heroes", comp_heroes, sid)
            if details:
                db.replace_table(conn, "comp_augments", comp_augs, sid)
                db.replace_table(conn, "comp_builds", comp_builds, sid)
        else:  # 统计没更新，沿用上次的版本号
            row = conn.execute("SELECT stats_version FROM seasons WHERE season_id = ?", (sid,)).fetchone()
            version = row[0] if row else None
        db.mark_synced(conn, season, off_version or version, datetime.now().isoformat(timespec="seconds"),
                       stats_version=version, data_source=source)
    if db.season_id != sid:
        db.use_season(sid)
    else:
        db.reload()
    stats = db.stats()
    say(f"同步完成：{stats}")
    return stats
