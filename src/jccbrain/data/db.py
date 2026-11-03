"""游戏数据库（SQLite）。

写：只有 sync.py 在同步时写入，整批事务提交；每次只替换一个赛季（season_id，官方编号如 18_S19）的行。
读：GameDB 打开时把「当前赛季」的全部数据载入内存（总共几百条），之后所有查询走内存，多线程可直接共享。
切换赛季用 use_season()，在原对象上重新载入，持有 GameDB 引用的模块不用重建。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

from jccbrain.core import paths
from jccbrain.core.seasons import DEFAULT_SEASON_ID, LEGACY_SET_IDS, SEASONS, Season, make_season, season
from jccbrain.data.models import (Augment, Comp, CompAugment, CompHero, Encounter, Guide, GuideHero, Hero, Item,
                                  HeroStats, Mechanic, SkillValue, Trait, TraitLevel, UnlockCondition)
from jccbrain.data.structure import activation, item_kind, linked_trait, relation_ids, trait_levels
from jccbrain.data.positions import parse_pos
from jccbrain.data.official import INGAME_SEASONS

log = logging.getLogger(__name__)

SCHEMA_VERSION = 9
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# 按赛季区分的游戏数据表
SET_TABLES = ("heroes", "hero_stars", "unlock_conditions", "traits", "items", "augments", "mechanics", "aliases",
              "comps", "comp_heroes", "comp_augments", "comp_builds", "encounters", "guides", "guide_heroes")

_CACHE_FIELDS = ('game_version', 'stats_version', 'data_source', 'synced_at', 'heroes', 'heroes_by_name',
                 'traits', 'items', 'augments', 'augments_by_name', 'aliases', 'comps', 'mechanics',
                 'encounters', 'guides', 'official_comps')


def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _pos(s: str | None) -> tuple[int, int] | None:
    return parse_pos(s)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _create_schema(conn: sqlite3.Connection) -> None:
    """逐条执行 schema.sql（不用 executescript：它会先隐式提交，迁移就不是原子的了）。"""
    sql = "\n".join(line for line in SCHEMA_FILE.read_text(encoding="utf-8").splitlines()
                    if not line.lstrip().startswith("--"))
    for stmt in sql.split(";"):
        if stmt.strip():
            conn.execute(stmt)


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    """旧库 → v3（赛季用官方 id 区分）。

    v1：游戏数据表没有赛季列，数据归到旧 meta.set_id（默认 18）；v2：按 dataj setId（set_id 整数列）区分。
    setId 按 LEGACY_SET_IDS 换成官方 id，换不了的（如 dataj 的「星神」，官网已不在列）丢弃，需要时重新同步。
    """
    tables = [t for t in (*SET_TABLES, "seasons") if _columns(conn, t) and "season_id" not in _columns(conn, t)]
    if not tables:
        return
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall()) if _columns(conn, "meta") else {}
    v1_set = int(meta.get("set_id") or 18)
    conn.create_function("legacy_season", 1, lambda sid: LEGACY_SET_IDS.get(int(sid)) if sid is not None else None)
    if not conn.in_transaction:
        conn.execute("BEGIN")
    for t in tables:
        conn.execute(f"ALTER TABLE {t} RENAME TO _old_{t}")
    # 旧索引跟着改名后的表走，名字会和新表的冲突，先删掉
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' "
                                "AND substr(tbl_name, 1, 5) = '_old_' AND sql IS NOT NULL").fetchall():
        conn.execute(f"DROP INDEX {name}")
    _create_schema(conn)
    for t in tables:
        old_cols = _columns(conn, f"_old_{t}")
        key = "legacy_season(set_id)" if "set_id" in old_cols else "?"
        params = () if "set_id" in old_cols else (LEGACY_SET_IDS.get(v1_set),)
        cols = [c for c in old_cols if c in _columns(conn, t) and c != "set_id"]
        if t == "seasons":
            cols = [c for c in cols if c != "name"]
            conn.execute(f"INSERT OR REPLACE INTO seasons (season_id, name, {', '.join(cols)}) "
                         f"SELECT {key} AS sid, name, {', '.join(cols)} FROM _old_seasons WHERE sid IS NOT NULL", params)
        else:
            conn.execute(f"INSERT OR REPLACE INTO {t} (season_id, {', '.join(cols)}) "
                         f"SELECT {key} AS sid, {', '.join(cols)} FROM _old_{t} WHERE sid IS NOT NULL", params)
        conn.execute(f"DROP TABLE _old_{t}")
    if meta.get("synced_at") and LEGACY_SET_IDS.get(v1_set):   # v1：同步信息在 meta 里
        sid = LEGACY_SET_IDS[v1_set]
        conn.execute("INSERT OR REPLACE INTO seasons(season_id, name, game_version, stats_version, data_source, synced_at)"
                     " VALUES (?, ?, ?, ?, 'dataj', ?)",
                     (sid, season(sid).name, meta.get("game_version"), meta.get("game_version"), meta["synced_at"]))
    # 旧库的基础数据都来自 dataj
    conn.execute("UPDATE seasons SET data_source = 'dataj', stats_version = COALESCE(stats_version, game_version) "
                 "WHERE data_source IS NULL")
    # 名称以官方赛季表为准
    for row in conn.execute("SELECT season_id FROM seasons").fetchall():
        conn.execute("UPDATE seasons SET name = ? WHERE season_id = ?", (season(row[0]).name, row[0]))
    conn.execute("DELETE FROM meta WHERE key IN ('set_id', 'game_version', 'synced_at')")
    log.info("数据库已迁移到按官方赛季存储（%s）", "、".join(tables))


def enabled_comp_sources(data_settings) -> frozenset[str]:
    """阵容消费与设置选项共用的来源开关；不影响基础英雄/装备资料。"""
    return frozenset(source for source in ('official', 'ingame', 'dataj') if getattr(data_settings, f'source_{source}'))


class GameDB:
    def __init__(self, path: Path | None = None, season_id: str = DEFAULT_SEASON_ID):
        self.path = Path(path or paths.db_path())
        self.season_id = str(season_id)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            with self.transaction() as c:
                c.execute("BEGIN")
                version = (c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
                           if _columns(c, "meta") else None)
                _migrate_legacy(c)
                _create_schema(c)
                for name, declaration in {"skill_values": "TEXT NOT NULL DEFAULT '[]'",
                                          "skill_brief_values": "TEXT NOT NULL DEFAULT '[]'",
                                          "sell_price": "INTEGER", "skill_icon": "TEXT",
                                          "variant_of": "TEXT", "unit_kind": "TEXT NOT NULL DEFAULT 'other'"}.items():
                    if name not in _columns(c, "heroes"):
                        c.execute(f"ALTER TABLE heroes ADD COLUMN {name} {declaration}")
                for name in ("category", "author"):            # v6：官方阵容分类、作者
                    if name not in _columns(c, "guides"):
                        c.execute(f"ALTER TABLE guides ADD COLUMN {name} TEXT")
                # v9：微社区来源与热度
                for name, declaration in {"channels": "TEXT NOT NULL DEFAULT '[\"web\"]'",
                                          "ingame_sort": "INTEGER", "use_count": "INTEGER", "week_use": "INTEGER",
                                          "collect_count": "INTEGER", "comment_count": "INTEGER",
                                          "stat_date": "TEXT"}.items():
                    if name not in _columns(c, "guides"):
                        c.execute(f"ALTER TABLE guides ADD COLUMN {name} {declaration}")
                additions = {
                    "traits": {"levels": "TEXT NOT NULL DEFAULT '[]'", "activation": "TEXT NOT NULL DEFAULT 'standard'"},
                    "items": {"fetter_id": "TEXT", "trait_id": "TEXT", "effect_type": "TEXT", "raw_kind": "TEXT"},
                    "augments": {"fetter_id": "TEXT", "fetter_type": "TEXT", "hero_enhancement_type": "TEXT",
                                 "is_legend": "INTEGER"},
                }
                for table, cols in additions.items():
                    for name, declaration in cols.items():
                        if name not in _columns(c, table):
                            c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
                if version is None or int(version[0]) < 7:
                    by_season = defaultdict(dict)
                    for r in c.execute("SELECT * FROM traits").fetchall():
                        points = json.loads(r["breakpoints"])
                        c.execute("UPDATE traits SET levels = ?, activation = ? WHERE season_id = ? AND id = ?",
                                  (_j(trait_levels(points, json.loads(r["level_desc"] or "[]"))), activation(points),
                                   r["season_id"], r["id"]))
                        by_season[r["season_id"]][r["id"]] = dict(r)
                    for r in c.execute("SELECT * FROM items").fetchall():
                        tid = linked_trait(None, r["name"], r["kind"], by_season[r["season_id"]])
                        c.execute("UPDATE items SET raw_kind = ?, kind = ?, trait_id = ? WHERE season_id = ? AND id = ?",
                                  (r["kind"], item_kind(r["kind"], r["name"]), tid, r["season_id"], r["id"]))
                if version is None or int(version[0]) < 4:
                    # 旧库只有一星属性。保留已知值，不能虚构其余星级和暴击率。
                    c.execute("INSERT OR IGNORE INTO hero_stars "
                              "(season_id, hero_id, star, hp, attack_damage, attack_speed, armor, magic_resist, "
                              "init_mana, max_mana) SELECT season_id, id, 1, hp, attack_damage, attack_speed, armor, "
                              "magic_resist, init_mana, max_mana FROM heroes")
                if version is None or int(version[0]) < 5:
                    # 旧库保持已有卡池标记；有原始记录的库可还原变体，没有的等下次同步补齐。
                    from jccbrain.data.units import classify_units

                    old_rows = [dict(r) for r in c.execute("SELECT * FROM heroes")]
                    grouped = defaultdict(list)
                    for r in old_rows:
                        try:
                            source = json.loads(r["raw"] or "null")
                        except ValueError:
                            source = None
                        if isinstance(source, dict) and {"id", "heroType"} <= source.keys():
                            grouped[r["season_id"]].append(r)
                        elif r["in_pool"]:
                            c.execute("UPDATE heroes SET unit_kind = 'champion' WHERE season_id = ? AND id = ?",
                                      (r["season_id"], r["id"]))
                    for rows in grouped.values():
                        classify_units(rows)
                        for r in rows:
                            c.execute("UPDATE heroes SET variant_of = ?, unit_kind = ?, in_pool = ? "
                                      "WHERE season_id = ? AND id = ?", (r["variant_of"], r["unit_kind"], r["in_pool"],
                                                                         r["season_id"], r["id"]))
                if version is None or int(version[0]) != SCHEMA_VERSION:
                    c.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            self.reload()
        except BaseException:
            self._conn.close()
            raise

    @property
    def season(self) -> Season:
        return season(self.season_id, self.catalog())

    def use_season(self, season_id: str) -> None:
        """切换到另一个赛季并重新载入（该赛季没同步过时数据为空）。"""
        self.season_id = str(season_id)
        self.reload()

    def catalog(self) -> list[Season]:
        """赛季列表：官网刷新过就用官网的，否则用内置默认列表。"""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM season_catalog ORDER BY sort").fetchall()
        return [make_season(r["season_id"], r["name"], r["mode"], r["season"]) for r in rows] or list(SEASONS)

    def catalog_name(self, season_id: str) -> str:
        return season(season_id, self.catalog()).label

    def save_catalog(self, seasons: list[Season]) -> None:
        if not seasons:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self.transaction() as c:
            c.execute("DELETE FROM season_catalog")
            c.executemany("INSERT INTO season_catalog VALUES (?, ?, ?, ?, ?, ?)",
                          [(s.id, s.name, s.mode, s.season, i, now) for i, s in enumerate(seasons)])

    # ---------------------------------------------------------------- 写入
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def replace_table(self, conn: sqlite3.Connection, table: str, rows: list[dict],
                      season_id: str | None = None) -> None:
        """清空某个赛季（默认当前赛季）在该表的行并整批写入（同步用）。"""
        sid = self.season_id if season_id is None else str(season_id)
        conn.execute(f"DELETE FROM {table} WHERE season_id = ?", (sid,))
        if not rows:
            return
        cols = [c for c in rows[0].keys() if c != "season_id"]
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} (season_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
            [(sid, *(r[c] for c in cols)) for r in rows],
        )

    def mark_synced(self, conn: sqlite3.Connection, s: Season, game_version: str | None, synced_at: str,
                    stats_version: str | None = None, data_source: str | None = None) -> None:
        conn.execute("INSERT OR REPLACE INTO seasons(season_id, name, game_version, stats_version, data_source, "
                     "synced_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (s.id, s.name, game_version, stats_version, data_source, synced_at))

    def synced_seasons(self) -> dict[str, dict]:
        """已同步过的赛季：season_id → {name, game_version, stats_version, data_source, synced_at}。"""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM seasons").fetchall()
        return {r["season_id"]: {k: r[k] for k in ("name", "game_version", "stats_version", "data_source", "synced_at")}
                for r in rows}

    # ---------------------------------------------------------------- 载入内存
    def reload(self) -> None:
        with self._lock:
            self._load()

    def reload_from(self, loaded: GameDB) -> None:
        """发布后台已加载完成的同库同赛季缓存；保留本连接及其他模块持有的 GameDB 引用。

        loaded 发布后不得再改动其缓存。解析、攻略关联等工作已经在后台完成，
        界面锁内只需一次字典更新，不再次读取 SQLite 或解析游戏数据。
        """
        if self.path.resolve() != loaded.path.resolve() or self.season_id != loaded.season_id:
            raise ValueError('只能接收同一数据库、同一赛季的已加载数据')
        with loaded._lock:
            snapshot = {name: getattr(loaded, name) for name in _CACHE_FIELDS}
        with self._lock:
            self.__dict__.update(snapshot)

    def _load(self) -> None:
        sid = self.season_id

        def q(sql: str):
            # 所有游戏数据表都只取当前赛季
            head, _, tail = sql.partition(" ORDER BY ")
            return self._conn.execute(f"{head} WHERE season_id = ?" + (f" ORDER BY {tail}" if tail else ""), (sid,))

        season_row = self._conn.execute("SELECT * FROM seasons WHERE season_id = ?", (sid,)).fetchone()
        self.game_version: str | None = season_row["game_version"] if season_row else None
        self.stats_version: str | None = season_row["stats_version"] if season_row else None
        self.data_source: str | None = season_row["data_source"] if season_row else None
        self.synced_at: str | None = season_row["synced_at"] if season_row else None
        self.heroes: dict[str, Hero] = {}
        self.heroes_by_name: dict[str, Hero] = {}
        stars: dict[str, list[HeroStats]] = defaultdict(list)
        unlocks: dict[str, list[UnlockCondition]] = defaultdict(list)
        for r in q("SELECT * FROM unlock_conditions ORDER BY id"):
            unlocks[r["hero_id"]].append(UnlockCondition(r["id"], r["description"] or "", r["task_tips"] or "",
                                                       r["condition_id"], bool(r["visible"])))
        for r in q("SELECT * FROM hero_stars ORDER BY star"):
            stars[r["hero_id"]].append(HeroStats(**{k: r[k] for k in HeroStats.__dataclass_fields__}))
        for r in q("SELECT * FROM heroes ORDER BY id"):
            values = tuple(SkillValue(**{**v, "values": tuple(v["values"])})
                           for v in json.loads(r["skill_values"]))
            h = Hero(r["id"], r["name"], r["cost"], tuple(json.loads(r["traits"])), bool(r["in_pool"]),
                     r["role_tag"], r["attack_range"], r["skill_name"], r["skill_desc"], r["picture"],
                     values, r["sell_price"], r["skill_icon"], tuple(stars[r["id"]]),
                     tuple(json.loads(r["skill_brief_values"])), r["variant_of"], r["unit_kind"], tuple(unlocks[r["id"]]))
            self.heroes[h.id] = h
            # 同名变体：优先保留进商店的那个
            if h.name not in self.heroes_by_name or (h.in_pool and not self.heroes_by_name[h.name].in_pool):
                self.heroes_by_name[h.name] = h

        self.traits: dict[str, Trait] = {}
        for r in q("SELECT * FROM traits"):
            t = Trait(r["id"], r["name"], r["kind"], tuple(json.loads(r["breakpoints"])),
                      r["description"] or "", tuple(json.loads(r["level_desc"] or "[]")),
                      tuple(TraitLevel(v["count"], v["text"], tuple(v["values"])) for v in json.loads(r["levels"])),
                      r["activation"], r["picture"])
            self.traits[t.name] = t

        self.items: dict[str, Item] = {}
        trait_names = {t.id: t.name for t in self.traits.values()}
        for r in q("SELECT * FROM items"):
            self.items[r["id"]] = Item(r["id"], r["name"], r["kind"], r["description"] or "",
                                       tuple(json.loads(r["recipe"] or "[]")), r["fetter_id"], r["trait_id"],
                                       trait_names.get(r["trait_id"]), r["effect_type"], r["raw_kind"], r["picture"])

        self.augments: dict[str, Augment] = {}
        self.augments_by_name: dict[str, Augment] = {}
        for r in q("SELECT * FROM augments"):
            a = Augment(r["id"], r["name"], r["tier"], r["description"] or "", r["fetter_id"], r["fetter_type"],
                        r["hero_enhancement_type"], None if r["is_legend"] is None else bool(r["is_legend"]),
                        tuple(trait_names[i] for i in relation_ids(r["fetter_id"]) if i in trait_names))
            self.augments[a.id] = a
            self.augments_by_name.setdefault(a.name, a)

        self.aliases: dict[str, str] = {r["keyword"]: r["name"] for r in q("SELECT * FROM aliases")}

        comp_heroes: dict[str, list[CompHero]] = defaultdict(list)
        for r in q("SELECT * FROM comp_heroes ORDER BY comp_id, sort"):
            comp_heroes[r["comp_id"]].append(CompHero(
                r["hero_id"], r["hero_name"], r["cost"] or 0, bool(r["is_carry"]), bool(r["is_subcarry"]),
                bool(r["is_core"]), r["star_target"], _pos(r["position"]), tuple(json.loads(r["items"] or "[]"))))
        comp_augs: dict[str, list[CompAugment]] = defaultdict(list)
        for r in q("SELECT * FROM comp_augments ORDER BY comp_id, avg_placement"):
            comp_augs[r["comp_id"]].append(CompAugment(
                r["augment_id"], r["name"], r["tier"], r["rating"], r["play_rate"], r["avg_placement"],
                r["top4_rate"], r["placement_delta"]))
        self.comps: dict[str, Comp] = {}
        for r in q("SELECT * FROM comps ORDER BY avg_placement"):
            self.comps[r["id"]] = Comp(
                r["id"], r["name"], r["tier"], r["avg_placement"], r["top4_rate"], r["top_rate"], r["pick_rate"],
                r["sample_count"], r["game_version"], json.loads(r["traits"] or "[]"),
                comp_heroes.get(r["id"], []), comp_augs.get(r["id"], []), r["game_code"])

        self.mechanics: dict[str, Mechanic] = {}
        for r in q("SELECT * FROM mechanics"):
            rounds = tuple((x.get("start"), x.get("end")) for x in json.loads(r["rounds"] or "[]"))
            ups = tuple(u.get("desc", "") for u in json.loads(r["upgrades"] or "[]"))
            price = int(r["price"]) if str(r["price"] or "").isdigit() else None
            self.mechanics[r["name"]] = Mechanic(r["id"], r["name"], r["category"], price, r["description"] or "",
                                                 rounds, ups)
        self.encounters: dict[str, Encounter] = {
            r["name"]: Encounter(r["id"], r["name"], r["description"] or "") for r in q("SELECT * FROM encounters")}
        self._load_guides(q)

    def _load_guides(self, q) -> None:
        hero_name = {h.id: h.name for h in self.heroes.values()}
        hero_cost = {h.id: h.cost for h in self.heroes.values()}
        aug_name = {a.id: a.name for a in self.augments.values()}
        boards: dict[str, dict[str, list[GuideHero]]] = defaultdict(lambda: defaultdict(list))
        for r in q("SELECT * FROM guide_heroes ORDER BY guide_id, phase, sort"):
            boards[r["guide_id"]][r["phase"]].append(GuideHero(
                r["hero_id"], hero_name.get(r["hero_id"], r["hero_id"]), r["star"], r["unit_type"] or "hero",
                r["position"], bool(r["is_carry"]), tuple(self.item_name(i) for i in json.loads(r["items"] or "[]")),
                hero_cost.get(r["hero_id"], 0)))
        self.guides: dict[str, Guide] = {}
        for r in q("SELECT * FROM guides ORDER BY sort DESC"):
            hexes = json.loads(r["hexes"] or "{}")
            carry = json.loads(r["carry_items"] or "{}")
            self.guides[r["id"]] = Guide(
                r["id"], r["name"], r["quality"], r["game_version"], r["early_info"] or "", r["d_time"] or "",
                r["location_info"] or "", r["enemy_info"] or "", r["hex_info"] or "", r["equipment_info"] or "",
                r["adventure_info"] or "",
                [aug_name.get(i, i) for i in hexes.get("recommend", [])],
                [aug_name.get(i, i) for i in hexes.get("replace", [])],
                [self.item_name(i) for i in json.loads(r["equipment_order"] or "[]")],
                [self.item_name(i) for i in carry.get("main", [])],
                [self.item_name(i) for i in carry.get("backup", [])],
                [hero_name.get(i, i) for i in json.loads(r["three_star"] or "[]")],
                json.loads(r["adventures"] or "{}"),
                {k: list(v) for k, v in boards.get(r["id"], {}).items()},
                category=r["category"], author=r["author"],
                channels=tuple(json.loads(r["channels"] or '["web"]')), ingame_sort=r["ingame_sort"],
                use_count=r["use_count"], week_use=r["week_use"], collect_count=r["collect_count"],
                comment_count=r["comment_count"], stat_date=r["stat_date"],
            )
        self.official_comps: dict[str, Comp] = {
            c.id: c for c in (self._guide_to_comp(g) for g in self.guides.values()) if c.heroes}
        from jccbrain.data.matching import link_guides

        link_guides(self.guides, self.comps, self.heroes)
        for c in self.official_comps.values():
            g = self.guides[c.guide_id]
            c.guide, c.guides = g, [g]
            if g.comp is None:
                g.comp = c

    def _guide_to_comp(self, g: Guide) -> Comp:
        """官网推荐阵容 → Comp（没有胜率统计）：成型站位里的英雄，追三星的标 star_target=3，羁绊按人数算激活。"""
        by_id = {h.id: h for h in self.heroes.values()}
        three = set(g.three_star)
        heroes: list[CompHero] = []
        seen: set[str] = set()
        for gh in g.heroes:
            if gh.name in seen:
                continue
            seen.add(gh.name)
            h = by_id.get(gh.hero_id)
            heroes.append(CompHero(
                gh.hero_id, gh.name, h.cost if h else 0, is_carry=gh.is_carry, is_subcarry=False,
                is_core=gh.is_carry or gh.name in three, star_target=3 if gh.name in three else None,
                position=_pos(gh.position), items=gh.items))
        counts: dict[str, int] = defaultdict(int)
        for ch in heroes:
            if h := by_id.get(ch.hero_id):
                for t in h.traits:
                    counts[t] += 1
        traits = [t for t, n in sorted(counts.items(), key=lambda kv: -kv[1])
                  if (tr := self.traits.get(t)) and tr.tier_of(n) > 0]
        return Comp(f"official:{g.id}", g.name, g.quality, None, None, None, None, None, g.game_version, traits,
                    heroes, [], None, source="official", guide_id=g.id, category=g.category)

    # ---------------------------------------------------------------- 查询
    @property
    def is_empty(self) -> bool:
        return not self.heroes

    def hero(self, name: str) -> Hero | None:
        return self.heroes_by_name.get(name) or self.heroes_by_name.get(self.aliases.get(name, ""))

    def pool_heroes(self) -> list[Hero]:
        """赛季内静态的普通卡池候选（每个名字一个），不含对局内解锁状态。"""
        return [h for h in self.heroes_by_name.values() if h.in_pool and 1 <= h.cost <= 5]

    def variants(self, name_or_id: str) -> list[Hero]:
        """主形态和全部变体（含传入的变体自身），按 id 排序；支持名字/别名/基础 id。"""
        hero = self.heroes.get(name_or_id) or self.hero(name_or_id)
        if hero is None:
            return []
        return sorted((h for h in self.heroes.values() if h.pool_id == hero.pool_id), key=lambda h: h.id)

    def pool_count_by_cost(self) -> dict[int, int]:
        out: dict[int, int] = defaultdict(int)
        for h in self.pool_heroes():
            out[h.cost] += 1
        return dict(out)

    def trait(self, name: str) -> Trait | None:
        return self.traits.get(name)

    def augment(self, name: str) -> Augment | None:
        return self.augments_by_name.get(name) or self.augments_by_name.get(self.aliases.get(name, ""))

    def item_name(self, item_id: str) -> str:
        it = self.items.get(item_id)
        return it.name if it else item_id

    def recipes_from(self, item: Item | str) -> list[Item]:
        """按基础装备对象、id 或名称反查所有合成成装（同一散件需两个也只返回一次）。"""
        iid = item.id if isinstance(item, Item) else item
        if iid not in self.items:
            iid = next((i.id for i in self.items.values() if i.name == iid), iid)
        return [i for i in self.items.values() if iid in i.recipe]

    def _enabled_sources(self, enabled_sources) -> frozenset[str]:
        enabled = frozenset(('official', 'ingame', 'dataj') if enabled_sources is None else enabled_sources)
        return enabled if self.season_id in INGAME_SEASONS else enabled - {'ingame'}

    @staticmethod
    def _guide_available(guide: Guide, enabled) -> bool:
        return ('web' in guide.channels and 'official' in enabled
                or 'ingame' in guide.channels and 'ingame' in enabled)

    def _available_comp(self, c: Comp, enabled) -> Comp | None:
        if c.source == 'official':
            return c if (c.guide and self._guide_available(c.guide, enabled)) else None
        if c.source not in enabled:
            return None
        # 只裁剪消费视图，不改共享缓存；dataj 仍可用时也不能附带已关闭渠道的攻略。
        linked_guides = c.guides or ([c.guide] if c.guide else [])
        guides = [g for g in linked_guides if self._guide_available(g, enabled)]
        guide = next(iter(guides), None)
        if len(guides) == len(linked_guides) and guide is c.guide:
            return c
        return replace(c, guides=guides, guide=guide, guide_id=guide.id if guide else None,
                       category=guide.category if guide else None)

    def comps_catalog(self, source: str = "all", *, enabled_sources=None) -> list[Comp]:
        """浏览目录：保留每条原始攻略；双渠道攻略按稳定 id 只出现一次。"""
        if source not in ('all', 'official', 'ingame', 'dataj'):
            raise ValueError(f'未知的阵容来源：{source}')
        enabled = self._enabled_sources(enabled_sources)
        selected = enabled if source == 'all' else enabled & {source}
        dataj = sorted(self.comps.values(), key=lambda c: (c.avg_placement or 9))
        rank = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        order = {gid: i for i, gid in enumerate(self.guides)}        # guides 已按官方排序值从高到低载入
        official = sorted(self.official_comps.values(),
                          key=lambda c: (rank.get(c.tier or "", 9), order.get(c.guide_id, 1 << 30)))
        return ([view for c in dataj if 'dataj' in selected
                 if (view := self._available_comp(c, enabled)) is not None]
                + [c for c in official if self._available_comp(c, selected) is not None])

    def comps_ranked(self, source: str = "all", *, enabled_sources=None) -> list[Comp]:
        """推荐集合：all 保持原有关联去重，单渠道可读取该渠道的所有攻略。"""
        rows = self.comps_catalog(source, enabled_sources=enabled_sources)
        if source != 'all':
            return rows
        dataj_ids = {c.id for c in rows if c.source == 'dataj'}
        return [c for c in rows if c.source != 'official' or not c.guide.comp
                or c.guide.comp.id not in dataj_ids]

    def comp(self, comp_id: str, *, enabled_sources=None, raw: bool = False) -> Comp | None:
        """按原始 id 校验来源；默认保留锁定/推荐的关联兼容，raw 返回原始攻略。"""
        c = self.comps.get(comp_id) or self.official_comps.get(comp_id)
        enabled = self._enabled_sources(enabled_sources)
        if c is None or (c := self._available_comp(c, enabled)) is None:
            return None
        # 关联对象可能来自已消失的dataj缓存；必须重新按当前库解析，不能返回幽灵身份。
        linked = self.comps.get(c.guide.comp.id) if (not raw and c.source == "official"
                                                   and c.guide and c.guide.comp) else None
        return (self._available_comp(linked, enabled) or c) if linked else c

    def stats(self) -> dict[str, int | str | None]:
        return {
            "season_id": self.season_id, "season": self.season.name,
            "heroes": len(self.pool_heroes()), "traits": len(self.traits), "items": len(self.items),
            "hero_units": len(self.heroes), "hero_stars": sum(len(h.star_stats) for h in self.heroes.values()),
            "skill_parse_errors": sum(any(v.parse_error for v in h.skill_values) for h in self.heroes.values()),
            "augments": len(self.augments), "mechanics": len(self.mechanics), "encounters": len(self.encounters),
            "comps": len(self.comps), "guides": len(self.guides),
            "web_guides": sum("web" in g.channels for g in self.guides.values()),
            "ingame_guides": sum("ingame" in g.channels for g in self.guides.values()),
            "dual_channel_guides": sum("web" in g.channels and "ingame" in g.channels for g in self.guides.values()),
            "comps_catalog_total": len(self.comps_catalog()),
            "guides_with_stats": sum(g.use_count is not None for g in self.guides.values()),
            "guide_stat_date": max((g.stat_date for g in self.guides.values() if g.stat_date), default=None),
            "linked_guides": sum(g.comp is not None and g.comp.source == "dataj" for g in self.guides.values()),
            "linked_comps": sum(c.guide is not None for c in self.comps.values()),
            "comps_total": len(self.comps_ranked()),
            "unlock_conditions": sum(len(h.unlock_conditions) for h in self.heroes.values()),
            "game_version": self.game_version, "stats_version": self.stats_version, "data_source": self.data_source,
            "synced_at": self.synced_at,
        }

    def close(self) -> None:
        self._conn.close()
