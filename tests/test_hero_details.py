"""逐星属性、技能解析、旧库无损迁移；仅用截取的官方样本。"""
import json
import sqlite3
from pathlib import Path

import pytest

from jccbrain.data import official as off
from jccbrain.data.db import GameDB, SCHEMA_VERSION
from jccbrain.data.hero_details import parse_skill_values
from jccbrain.data.sync import sync_season
from jccbrain.core.seasons import season
from test_seasons import FakeOfficial, HERO_COLS


@pytest.fixture
def official_sample():
    return json.loads((Path(__file__).parent / "fixtures" / "official_hero_stars.json").read_text())


def test_skill_values_percent_decimal_and_formula():
    result = parse_skill_values("护盾:400(【法术加成】) 400/460/550|攻速：90%(【法术加成】) 90%/100%/130%")
    assert result[0]["name"] == "护盾" and result[0]["values"] == [400, 460, 550]
    assert result[1]["unit"] == "%" and result[1]["scaling"] == "法术加成"
    assert parse_skill_values("伤害：1.5/2.7/4.05")[0]["values"] == [1.5, 2.7, 4.05]
    assert parse_skill_values("回复：10(【生命上限】【法术加成】) 10/20/30")[0]["scaling"] == "生命上限、法术加成"
    formula = parse_skill_values("伤害：59=170的35%", "10/15/25|160/240/370")[0]
    assert formula["values"] == [] and formula["formula"] == "59=170的35%"
    assert parse_skill_values("伤害：32(【法术加成】)", "32/48/72")[0]["values"] == [32, 48, 72]
    assert parse_skill_values("") == []
    assert parse_skill_values("0", "0") == []
    assert parse_skill_values("永久生命值 18/20/50")[0]["values"] == [18, 20, 50]
    assert parse_skill_values("未知格式")[0]["parse_error"]


def test_official_stars_sync_and_season_isolation(tmp_path, official_sample):
    db = GameDB(tmp_path / "game.db")
    client = FakeOfficial(files=official_sample, lineups=[])
    sync_season(db, season("18_S19"), official=client)
    h = db.hero("奥恩")
    assert [h.stats(i).hp for i in (1, 2, 3, 4)] == [700, 1260, 2268, 3276]
    assert h.stats(1).crit_chance == 25 and h.stats(1).attack_speed == 0.6
    assert h.stats(5) is None and h.sell_price == 1 and h.skill_icon.endswith("s18_ornn.png")
    assert h.skill_values[0].values == (400, 460, 550) and h.skill_brief_values[0] == "400/460/550"
    assert db.heroes["8704"].stats(3) is not None       # heroType != 0 的召唤物
    assert db.heroes["8101"].stats(1) is None          # 官方没有一星记录
    assert db.heroes["8101"].stats(2) is not None
    assert db.stats()["skill_parse_errors"] == 0
    sync_season(db, season("16_S19"), official=client)
    db.use_season("18_S19")
    assert db.hero("奥恩").stats(4).hp == 3276
    # 同步替换当前赛季，去掉的星级不能遗留；另一个赛季的属性仍保留。
    shorter = {**official_sample, "chess": [h for h in official_sample["chess"] if h["id"][0] == "1"]}
    sync_season(db, season("18_S19"), official=FakeOfficial(files=shorter, lineups=[]))
    assert db.hero("奥恩").stats(2) is None
    db.use_season("16_S19")
    assert db.hero("奥恩").stats(4).hp == 3276
    db.close()


def test_v3_migration_preserves_known_stats_and_reopens(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("INSERT INTO meta VALUES ('schema_version', '3')")
        c.execute("INSERT INTO meta VALUES ('custom', '保留')")
        c.execute(f"CREATE TABLE heroes (season_id TEXT, {HERO_COLS}, PRIMARY KEY (season_id, id))")
        for sid, hp in (("18_S19", 700), ("16_S19", 600)):
            c.execute("INSERT INTO heroes (season_id, id, name, cost, traits, trait_ids, in_pool, hp, raw) "
                      "VALUES (?, '1500', '奥恩', 1, '[]', '[]', 1, ?, '原始数据')", (sid, hp))
    for _ in range(2):
        db = GameDB(path)
        assert db.meta("schema_version") == str(SCHEMA_VERSION) and db.meta("custom") == "保留"
        assert db.hero("奥恩").stats(1).hp == 700
        assert db.hero("奥恩").stats(2) is None and db.hero("奥恩").skill_values == ()
        assert db._conn.execute("SELECT raw FROM heroes LIMIT 1").fetchone()[0] == "原始数据"
        db.use_season("16_S19")
        assert db.hero("奥恩").stats(1).hp == 600
        db.close()
