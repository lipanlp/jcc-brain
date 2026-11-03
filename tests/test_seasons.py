"""赛季（官方 id）、多数据源同步（官方为准）、旧库迁移、切换赛季。"""
import json
import sqlite3

import pytest

from jccbrain.core.rules import rules_for_season
from jccbrain.core.seasons import SEASONS, season
from jccbrain.core.settings import Settings
from jccbrain.data import official as off
from jccbrain.data.db import SCHEMA_VERSION, GameDB
from jccbrain.data.sync import SyncError, convert_gamedata, sync_season

from conftest import COMP, COMP_DETAIL, COMP_HEXES, GAMEDATA

S18 = season("18_S19")


class FakeDataj:
    def __init__(self, gamedata: dict, comps: list[dict], version: str = "18.4"):
        self._gamedata, self._comps, self._version = gamedata, comps, version
        self.calls = 0

    def latest_version(self):
        return self._version

    def gamedata(self):
        self.calls += 1
        return self._gamedata

    def comp_rank(self, version):
        return self._comps

    def comp_detail(self, comp_id, version):
        return COMP_DETAIL

    def comp_hexes(self, comp_id, version):
        return COMP_HEXES

    def comp_builds(self, comp_id, version):
        return []


def _official_files() -> dict:
    """把 dataj 夹具改成官方的键名，并改掉几处数值，用来验证「以官方为准」。"""
    chess = []
    for h in GAMEDATA["hero"]:
        o = {k: v for k, v in h.items() if k not in ("raceId", "jobId", "isVisible", "tag")}
        o.update(species=h["raceId"], **{"class": h["jobId"]}, heroType=str(h["heroType"]), price=str(h["price"]))
        if h["name"] == "绯红树怪":
            o["name"] = "绯红树怪·官方"
        chess.append(o)
    traits = [{**{k: v for k, v in t.items() if k != "desc2Text"}, "desc2": t["desc2Text"]} for t in GAMEDATA["trait"]]
    equip = [{**e, "desc": e["basicDesc"]} for e in GAMEDATA["equip"]]
    hexes = [{"id": x["id"], "name": x["name"], "level": str(x["level"]), "desc": "官方：" + x["descText"]}
             for x in GAMEDATA["hex"]]
    adventure = [{"adventureId": "181326", "title": "不倒翁", "desc": "18秒后刷新", "price": "3", "category": ""},
                 {"adventureId": "181327", "title": "不倒翁", "desc": "15秒后刷新", "price": "3", "category": ""}]
    galaxy = [{"id": "18001", "name": "金金金之旅", "desc": "所有强化符文都变为黄金阶。", "logo": "x.png"}]
    return {"chess": chess, "trait": traits, "equip": equip, "hex": hexes, "adventure": adventure, "galaxy": galaxy}


LINEUP = {
    "id": "4932", "quality": "S", "lineupauthor_data": {"name": "掌盟阵容推荐小助手"}, "sortID": "98304", "simulator_edition": "18.18.4", "rel_time": "2026-10-19",
    "detail": json.dumps({
        "line_name": "【野怪小红】", "line_tag": "2", "early_info": "前期连败", "d_time": "3-2 拉 6 慢 D",
        "hexbuff": {"recomm": "2021", "replace": "1002"}, "equipment_order": "1001",
        "carry_hero_equip_replace": {"main": "2010", "backup": ""}, "level_3_heros": "11508",
        "hero_location": [
            {"chess_type": "hero", "equipment_id": "2010", "hero_id": "11508", "is_carry_hero": True, "location": "4,7"},
            {"chess_type": "pet", "equipment_id": "", "hero_id": "18704", "is_carry_hero": False, "location": "1,1"}],
        "y21_early_heros": [{"chess_type": "hero", "equipment_id": "", "hero_id": "11511", "location": "1,3"}],
        "levelMap": {"4": [], "6": [{"chess_type": "hero", "hero_id": "11508", "location": "4,6"}]},
        "adventure": [{"stage_num": 2, "adventure": ["181327"]}],
    }, ensure_ascii=False),
}


class FakeOfficial:
    def __init__(self, files=None, lineups=None, fail=False):
        self.files = files or _official_files()
        self._lineups = [LINEUP] if lineups is None else lineups
        self.fail = fail

    def season_tabs(self):
        return list(SEASONS)

    def find_version(self, s, version=None):
        if self.fail:
            raise off.OfficialError("断网")
        return {"mode": s.mode, "season": s.season, "version": version or "18.18.5"}

    def gamedata(self, entry):
        return self.files

    def lineups(self, entry, season_id=None):
        return self._lineups

    def lineup_tags(self):
        return {"1": "新手推荐", "2": "高手进阶", "3": "趣味娱乐"}

    def newest_version(self, s):
        return "18.18.5"

    def close(self):
        pass


def test_seasons_follow_official():
    assert [s.name for s in SEASONS] == ["画之灵", "英雄联盟传奇·恭喜发财", "自然之力", "英雄联盟传奇", "时空裂痕"]
    assert S18.mode == "18" and S18.season == "S19" and S18.dataj_set == 18
    assert season("16_S19").dataj_set == 16 and season("Mode16_S100024").dataj_set is None
    unknown = season("Mode20_S200001")
    assert unknown.mode == "20" and unknown.season == "S200001"


def test_official_is_authoritative(tmp_path):
    db = GameDB(tmp_path / "jcc.sqlite3")
    stats = sync_season(db, S18, official=FakeOfficial(), dataj=FakeDataj(GAMEDATA, [COMP]))
    assert stats["data_source"] == "official" and stats["game_version"] == "18.18.5" and stats["stats_version"] == "18.4"
    # 英雄名、海克斯描述以官方为准；dataj 阵容里的英雄名也被改写
    assert db.hero("绯红树怪·官方") and not db.hero("绯红树怪")
    assert db.augment("飞升").description.startswith("官方：")
    assert db.comps["104"].heroes[0].name == "绯红树怪·官方"
    # dataj 补官方没有的：别名（指向官方名字）
    assert db.aliases["红buff"] == "绯红树怪·官方" and db.hero("红buff").name == "绯红树怪·官方"
    # 仙灵：官方逐级记录合成一条，后面的是升级版
    m = db.mechanics["不倒翁"]
    assert m.price == 3 and m.description == "18秒后刷新" and m.upgrades == ("15秒后刷新",)
    assert db.encounters["金金金之旅"].description.startswith("所有强化符文")
    g = db.guides["4932"]
    assert g.name == "【野怪小红】" and g.d_time == "3-2 拉 6 慢 D"
    assert g.recommended_augments == ["飞升"] and g.alternative_augments == ["存心失利"]
    assert g.carry_items == ["无尽之刃"] and g.equipment_order == ["暴风大剑"]
    assert g.three_star == ["绯红树怪·官方"]
    assert [h.name for h in g.heroes] == ["绯红树怪·官方"] and g.carries[0].items == ("无尽之刃",)
    assert set(g.boards) == {"final", "early", "L6"}
    assert g.adventures == {"2": ["不倒翁"]}
    assert "【D 牌节奏】3-2 拉 6 慢 D" in g.text()
    assert [s.id for s in db.catalog()][2] == "18_S19"
    db.close()


def test_sources_selection(tmp_path):
    db = GameDB(tmp_path / "jcc.sqlite3")
    with pytest.raises(SyncError):
        sync_season(db, S18)

    # 只选官方：没有 dataj 阵容统计，也没有别名
    st = sync_season(db, S18, official=FakeOfficial())
    assert st["data_source"] == "official" and st["comps"] == 0 and st["guides"] == 1 and not db.aliases

    # 只选 dataj：基础数据来自 dataj
    st = sync_season(db, S18, dataj=FakeDataj(GAMEDATA, [COMP]))
    assert st["data_source"] == "dataj" and db.hero("绯红树怪") and st["comps"] == 1

    # 官方失败时退回 dataj
    st = sync_season(db, S18, official=FakeOfficial(fail=True), dataj=FakeDataj(GAMEDATA, [COMP]))
    assert st["data_source"] == "dataj"

    # dataj 没有这个赛季：只用官方
    dj = FakeDataj(GAMEDATA, [COMP])
    st = sync_season(db, season("Mode11_S100030"), official=FakeOfficial(), dataj=dj)
    assert dj.calls == 0 and st["season_id"] == "Mode11_S100030" and st["data_source"] == "official"
    with pytest.raises(SyncError):
        sync_season(db, season("Mode11_S100030"), dataj=dj)
    db.close()


def test_seasons_are_isolated(tmp_path):
    db = GameDB(tmp_path / "jcc.sqlite3", season_id="18_S19")
    sync_season(db, S18, official=FakeOfficial(), dataj=FakeDataj(GAMEDATA, [COMP]))
    files16 = _official_files()
    for h in files16["chess"]:
        h["name"] = "传奇" + h["name"]
    sync_season(db, season("16_S19"), official=FakeOfficial(files16, lineups=[]))
    assert db.season_id == "16_S19" and db.hero("传奇奥恩") and not db.guides and not db.comps
    assert set(db.synced_seasons()) == {"18_S19", "16_S19"}
    db.use_season("18_S19")
    assert db.hero("奥恩") and "104" in db.comps and db.guides
    db.use_season("1_S1")
    assert db.is_empty and db.game_version is None
    db.close()


# ------------------------------------------------------------------ 迁移
def _insert(conn, table, rows):
    # 旧版本迁移夹具只写当时已有的列。
    old_columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    cols = [c for c in rows[0] if c in old_columns]
    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [tuple(r[c] for c in cols) for r in rows])


HERO_COLS = ("id TEXT, name TEXT NOT NULL, cost INTEGER NOT NULL, traits TEXT NOT NULL, trait_ids TEXT NOT NULL, "
             "in_pool INTEGER NOT NULL, role_tag TEXT, attack_range INTEGER, hp INTEGER, armor INTEGER, "
             "magic_resist INTEGER, attack_damage INTEGER, attack_speed REAL, init_mana INTEGER, max_mana INTEGER, "
             "skill_name TEXT, skill_desc TEXT, picture TEXT, raw TEXT")


def test_migrate_v1(tmp_path):
    """v1：没有赛季列，同步信息在 meta 里。"""
    path = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(f"CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
                       f"CREATE TABLE heroes ({HERO_COLS}, PRIMARY KEY (id));"
                       f"CREATE INDEX idx_heroes_name ON heroes(name);")
    _insert(conn, "heroes", convert_gamedata(GAMEDATA)["heroes"])
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [("schema_version", "1"), ("set_id", "18"),
                                                        ("game_version", "18.4"), ("synced_at", "2026-10-22T00:21")])
    conn.commit()
    conn.close()

    db = GameDB(path)
    assert db.meta("schema_version") == str(SCHEMA_VERSION)
    assert db.hero("绯红树怪") and db.game_version == "18.4" and db.synced_seasons()["18_S19"]["name"] == "自然之力"
    db.close()
    db = GameDB(path)            # 再次打开不会重复迁移
    tables = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert db.hero("绯红树怪") and not any(t.startswith("_old_") for t in tables)
    db.close()


def test_migrate_v2(tmp_path):
    """v2：按 dataj setId 区分；18 → 18_S19，16 → 16_S19，17（星神，官网已下架）丢弃。"""
    path = tmp_path / "v2.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(f"CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
                       f"CREATE TABLE seasons (set_id INTEGER PRIMARY KEY, name TEXT NOT NULL, game_version TEXT,"
                       f" synced_at TEXT);"
                       f"CREATE TABLE heroes (set_id INTEGER NOT NULL, {HERO_COLS}, PRIMARY KEY (set_id, id));"
                       f"CREATE INDEX idx_heroes_name ON heroes(set_id, name);")
    rows = convert_gamedata(GAMEDATA)["heroes"]
    for sid in (18, 17, 16):
        _insert(conn, "heroes", [{"set_id": sid, **r} for r in rows])
    _insert(conn, "seasons", [{"set_id": 18, "name": "自然之力", "game_version": "18.4", "synced_at": "t1"},
                              {"set_id": 17, "name": "星神", "game_version": "17.8", "synced_at": "t2"}])
    conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    conn.commit()
    conn.close()

    db = GameDB(path)
    assert db.hero("奥恩") and set(db.synced_seasons()) == {"18_S19"}
    assert db.synced_seasons()["18_S19"]["game_version"] == "18.4"
    db.use_season("16_S19")
    assert db.hero("奥恩")
    n = db._conn.execute("SELECT COUNT(DISTINCT season_id) FROM heroes").fetchone()[0]
    assert n == 2
    db.close()


# ------------------------------------------------------------------ 规则、设置、切换
def test_rules_for_season():
    r = rules_for_season(S18)
    assert not r.borrowed and r.set_id == 18
    r16 = rules_for_season(season("16_S19"))
    assert r16.borrowed and r16.borrowed_from == 18 and r16.pool_size == r.pool_size


def test_settings_legacy_set_id(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"data": {"set_id": 16, "use_official": False}}), encoding="utf-8")
    s = Settings.load(p)
    assert s.data.season_id == "16_S19" and s.data.source_official is False and s.data.source_dataj is True
    p.write_text(json.dumps({"data": {"set_id": 17}}), encoding="utf-8")
    assert Settings.load(p).data.season_id == "18_S19"      # 星神已下架，回到默认赛季


def test_official_season_tabs_parse(monkeypatch):
    text = """
    season_tabs: [
      /** 说明 */
      {set_id: "Mode11", mode: "11", season: "S100030", version: "", id: "Mode11_S100030", name: "画之灵", set_name: "画之灵"},
      {set_id: "Mode16S100024", mode: "16", season: "S100024", version: "", id: "Mode16_S100024", name: "英雄联盟传奇\\n恭喜发财"},
      {set_id: "18", mode: "18", season: "S19", version: "", id: "18_S19", name: "自然之力", set_name: "自然之力"},
      // {set_id: "17", mode: "17", season: "S18", version: "", id: "17_S18", name: "星神"},
    ],
    lineup_url:{ '18_S19':`https://x/m19/${channel}/18/lineup_detail_total.json`, },
    var channel = '11';
    """
    c = off.OfficialClient()
    monkeypatch.setattr(c, "_get_text", lambda url, fresh=False: text)
    tabs = c.season_tabs()
    assert [s.id for s in tabs] == ["Mode11_S100030", "Mode16_S100024", "18_S19"]
    assert tabs[1].name == "英雄联盟传奇·恭喜发财" and tabs[2].dataj_set == 18
    assert c.lineup_url({"mode": "18", "season": "S19"}, "18_S19") == "https://x/m19/11/18/lineup_detail_total.json"
    c.close()


def test_context_switch_season(tmp_path, monkeypatch):
    monkeypatch.setenv("JCC_HOME", str(tmp_path))
    from jccbrain.context import AppContext

    s = Settings()
    s.assistant.default_lock_comp_id = "104"
    s.data.game_version = "18.4"
    ctx = AppContext(s)
    ctx.game_lock_id = "104"
    ctx.game_watch_ids = ["104"]
    ctx.game_augments = ["飞升"]
    sync_season(ctx.db, S18, official=FakeOfficial(), dataj=FakeDataj(GAMEDATA, [COMP]))
    assert ctx.rules.set_id == 18 and ctx.advisor.db is ctx.db

    ctx.switch_season("16_S19")
    assert ctx.db.season_id == "16_S19" and ctx.db.is_empty and ctx.rules.borrowed
    assert s.assistant.default_lock_comp_id == "" and ctx.game_lock_id is None and s.data.game_version == ""
    assert ctx.game_watch_ids == []
    assert ctx.game_augments == []
    assert Settings.load().data.season_id == "16_S19"        # 已保存

    ctx.switch_season("18_S19")
    assert ctx.db.hero("绯红树怪·官方") and ctx.advisor.matcher.comps
    ctx.close()


def test_official_lineups_become_comps(tmp_path):
    db = GameDB(tmp_path / "jcc.sqlite3")
    sync_season(db, S18, official=FakeOfficial(), dataj=FakeDataj(GAMEDATA, [COMP]))
    g = db.guides["4932"]
    assert g.category == "高手进阶" and g.author == "掌盟阵容推荐小助手"
    c = db.comp("official:4932")
    assert c.source == "official" and c.guide_id == "4932" and c.category == "高手进阶" and c.tier == "S"
    assert [h.name for h in c.heroes] == ["绯红树怪·官方"]          # 召唤物（pet）不算
    carry = c.heroes[0]
    assert carry.is_carry and carry.star_target == 3 and carry.position == (4, 7) and carry.items == ("无尽之刃",)
    assert c.style == "reroll" and c.avg_placement is None
    assert [x.source for x in db.comps_ranked()] == ["dataj", "official"]
    assert [x.id for x in db.comps_ranked("official")] == ["official:4932"]
    assert [x.id for x in db.comps_ranked("dataj")] == ["104"]
    db.close()


def test_sync_all_seasons(tmp_path, monkeypatch):
    monkeypatch.setenv("JCC_HOME", str(tmp_path))
    import jccbrain.data.dataj as dataj_mod
    import jccbrain.data.official as off_mod
    from jccbrain.context import AppContext

    monkeypatch.setattr(off_mod, "OfficialClient", lambda *a, **k: FakeOfficial())

    class Dj(FakeDataj):
        def __init__(self, *a, **k):
            super().__init__(GAMEDATA, [COMP])

        def close(self):
            pass

    monkeypatch.setattr(dataj_mod, "DatajClient", Dj)
    ctx = AppContext(Settings())
    assert {sid for sid, _, _ in ctx.seasons_needing_sync()} == {s.id for s in SEASONS}
    done = ctx.sync_all_seasons(details=False)
    assert set(done) == {s.id for s in SEASONS}
    assert ctx.db.season_id == "18_S19"                     # 同步完回到当前赛季
    synced = ctx.db.synced_seasons()
    assert synced["Mode11_S100030"]["stats_version"] is None   # dataj 没有这个赛季
    assert synced["18_S19"]["stats_version"] == "18.4"
    assert ctx.seasons_needing_sync() == []                  # 都是最新的
    assert ctx.sync_all_seasons() == []
    ctx.close()
