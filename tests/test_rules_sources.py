import json
import sqlite3
from pathlib import Path

from jccbrain.core import paths
from jccbrain.core.rules import available_rule_sets, load_rules, rules_for_season
from jccbrain.core.seasons import season
from jccbrain.data.db import GameDB
from jccbrain.data.sync import sync_season
from test_seasons import FakeOfficial, _official_files


def test_rules_by_official_id_with_explicit_provenance():
    assert season("18_S19").rules_set == "18_S19"
    assert "18_S19" in available_rule_sets()
    rules = rules_for_season(season("18_S19"))
    assert load_rules(18) == rules and not rules.borrowed
    assert rules.season_id == "18_S19" and "未核实" in rules.note
    for key in rules.pool_size:
        assert rules.sources[f"pool_size.{key}"]["status"] == "未核实"
    for key, odds in rules.level_odds.items():
        assert len(odds) == 5 and sum(odds) == 100
        assert rules.sources[f"level_odds.{key}"]["status"] == "未核实"
    raw = json.loads((paths.RULES_DIR / "18_S19.json").read_text())
    for field in ("xp_to_next", "augment_rounds", "pve_rounds", "carousel_rounds", "streak_gold"):
        assert field in rules.sources and rules.sources[field]["source"] is None
    borrowed = rules_for_season(season("Mode16_S100024"))
    assert borrowed.borrowed and borrowed.borrowed_from_season == "18_S19"
    assert borrowed.season_id == "Mode16_S100024" and borrowed.set_id == 16
    assert borrowed.pool_size == rules.pool_size and "未核实" in borrowed.note


def test_official_unlock_conditions_and_migration(tmp_path):
    sample = json.loads((Path(__file__).parent / "fixtures" / "official_unlocks.json").read_text())
    files = _official_files()
    # 假库里加入加里奥，条件正文为官方 17.17.8b 原文，不联网。
    files["chess"].append({"id": "15285", "name": "加里奥", "price": "5", "heroType": "7", "showHeroTag": "1",
                           "species": "341", "class": "-1"})
    files["mission"] = sample
    path = tmp_path / "db.sqlite"
    db = GameDB(path)
    sync_season(db, season("16_S19"), official=FakeOfficial(files=files, lineups=[]))
    h = db.hero("加里奥")
    assert len(h.unlock_conditions) == 1
    assert h.unlock_conditions[0].description == "在战斗中登场：8独特星级总数的【德玛西亚】;需求达到8级"
    assert h.unlock_conditions[0].visible and not h.in_pool  # 存条件不猜当前可用状态
    db.use_season("18_S19")
    assert db.is_empty
    db.close()
    # v7 没有解锁表；升级保留英雄及所有星级，新增表等待同步填充。
    with sqlite3.connect(path) as c:
        before = c.execute("SELECT COUNT(*) FROM hero_stars").fetchone()[0]
        c.execute("DROP TABLE unlock_conditions")
        c.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
    db = GameDB(path, season_id="16_S19")
    assert db.hero("加里奥") and db.hero("加里奥").unlock_conditions == ()
    assert db._conn.execute("SELECT COUNT(*) FROM hero_stars").fetchone()[0] == before
    db.close()
