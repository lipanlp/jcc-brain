import json
import sqlite3
from pathlib import Path

from jccbrain.core.seasons import season
from jccbrain.data.db import GameDB
from jccbrain.data.sync import sync_season, convert_gamedata
from jccbrain.data.structure import trait_levels
from test_seasons import FakeOfficial


def test_structured_fields_and_official_link_priority(tmp_path):
    files = json.loads((Path(__file__).parent / "fixtures" / "official_structure.json").read_text())
    files["equip"].append({"id": "test", "name": "永恒之森纹章", "type": "转职纹章", "fetterID": "348"})
    files["hex"] = [{"id": "linked", "name": "英雄强化", "fetterId": "348|450", "fetterType": "2",
                     "hero_enhancement_type": "1500", "is_legend": 1}]
    db = GameDB(tmp_path / "game.db")
    sync_season(db, season("18_S19"), official=FakeOfficial(files=files, lineups=[]))
    trait = db.trait("魔战士")
    assert [(v.count, v.values) for v in trait.levels] == [(2, (20,)), (3, (30,)), (4, (50,))]
    assert db.trait("大元素使").unique
    assert [v.count for v in db.trait("宿敌").levels] == [1, 2]
    assert [v.values for v in db.trait("神谕").levels] == [(1, 3), (1, 4), (2, 6), (2, 8)]
    special = db.trait("日月双蚀")
    assert special.activation == "special" and special.tier_of(0) == special.tier_of(9) == 0
    assert special.levels[0].count == 0 and special.levels[0].values == (8, 3.5)
    emblem = db.items["41801"]
    assert emblem.fetter_id == "" and emblem.trait_id == "450" and emblem.trait_name == "永恒之森"
    assert db.items["test"].trait_name == "魔战士"  # 显式字段优先于名称
    assert db.items["41805"].trait_name == "约德尔人"
    assert "携带者获得" in emblem.description and "攻击速度" in emblem.description
    assert db.items["9436"].kind == "补剂" and db.items["9436"].raw_kind == "-1"
    assert db.items["9453"].kind == "赐福" and db.items["9436"].effect_type == "50"
    aug = db.augment("英雄强化")
    assert aug.fetter_id == "348|450" and aug.trait_names == ("魔战士", "永恒之森")
    assert aug.hero_enhancement_type == "1500" and aug.is_legend is True and aug.fetter_type == "2"
    db.close()


def test_recipes_from(db):
    assert [i.name for i in db.recipes_from("1001")] == ["无尽之刃"]
    assert db.recipes_from(db.items["1001"]) == db.recipes_from("暴风大剑")
    assert db.recipes_from("不存在") == []


def test_trait_levels_preserve_text_without_guessed_tiers():
    assert trait_levels([2], ["(2)20%【物理加成】或【法术加成】"])[0]["values"] == [20]
    assert trait_levels([], ["特殊效果 {0}"])[0] == {"count": None, "text": "特殊效果 {0}", "values": []}


def test_v6_structure_migration_preserves_old_data(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO meta VALUES ('schema_version', '6');
            CREATE TABLE traits (season_id TEXT, id TEXT, name TEXT, kind TEXT, breakpoints TEXT,
                description TEXT, level_desc TEXT, picture TEXT, PRIMARY KEY (season_id,id));
            INSERT INTO traits VALUES ('18_S19','450','永恒之森','origin','[3,5,7]','保留描述','["(3)10%生命值"]',NULL);
            CREATE TABLE items (season_id TEXT,id TEXT,name TEXT,kind TEXT,description TEXT,recipe TEXT,picture TEXT,
                PRIMARY KEY(season_id,id));
            INSERT INTO items VALUES ('18_S19','a','生命值补剂','-1','保留描述','[]',NULL);
            INSERT INTO items VALUES ('18_S19','b','永恒之森纹章','转职纹章','原文','[]',NULL);
            CREATE TABLE augments (season_id TEXT,id TEXT,name TEXT,tier INTEGER,description TEXT,icon TEXT,
                PRIMARY KEY(season_id,id));
            INSERT INTO augments VALUES ('18_S19','x','强化',2,'原文',NULL);
        """)
    for _ in range(2):
        db = GameDB(path)
        assert db.trait("永恒之森").levels[0].values == (10,)
        assert db.items["a"].kind == "补剂" and db.items["a"].description == "保留描述"
        assert db.items["b"].trait_name == "永恒之森" and db.items["b"].description == "原文"
        assert db.augment("强化").is_legend is None
        db.close()
