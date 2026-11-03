"""官网两个赛季的单位分类与形态归组。夹具只截取分类需要的字段，不联网。"""
import copy
import json
import sqlite3
from pathlib import Path

from jccbrain.core.seasons import season
from jccbrain.data.db import GameDB
from jccbrain.data.sync import convert_gamedata, sync_season
from jccbrain.data.official import to_gamedata
from test_seasons import FakeOfficial, HERO_COLS, _insert


SAMPLES = json.loads((Path(__file__).parent / "fixtures" / "official_units.json").read_text())


def test_official_variants_and_special_units(tmp_path):
    db = GameDB(tmp_path / "game.db")
    sync_season(db, season("18_S19"), official=FakeOfficial(files=SAMPLES["18_S19"], lineups=[]))
    assert db.pool_count_by_cost() == {1: 14, 2: 13, 3: 14, 4: 14, 5: 10}
    lux = db.variants("拉克丝")
    assert len(lux) == 10 and sum(h.in_pool for h in lux) == 1
    assert {h.pool_id for h in lux} == {"5459"}
    assert all(h.unit_kind == "variant" and h.variant_of == "5459" for h in lux[1:])
    assert len({h.traits for h in lux}) == 10
    assert db.variants("5461") == lux and db.variants("不存在") == []
    assert db.heroes["8704"].unit_kind == "summon" and not db.heroes["8704"].in_pool
    monsters = [h for h in db.pool_heroes() if "峡谷野怪" in h.traits]
    assert len(monsters) == 10 and all(h.unit_kind == "champion" for h in monsters)

    sync_season(db, season("16_S19"), official=FakeOfficial(files=SAMPLES["16_S19"], lineups=[]))
    # 最小 id 的卡莎是隐藏形态，必须优先采用官方展示标记。
    assert db.hero("卡莎").id == "4360"
    assert db.heroes["4348"].variant_of == "4360"
    assert db.hero("凯隐").id == "4368" and db.heroes["4365"].variant_of == "4368"
    # 未核实类型没有被猜成普通池卡。
    assert db.hero("加里奥").unit_kind == "other" and not db.hero("加里奥").in_pool
    assert db.hero("盛宴塔姆").unit_kind == "other" and not db.hero("盛宴塔姆").in_pool
    assert db.hero("提伯斯").unit_kind == "summon" and not db.hero("提伯斯").in_pool
    assert db.pool_count_by_cost() == {1: 14, 2: 20, 3: 23, 4: 28, 5: 28}
    db.close()


def test_official_marker_overrides_fallback_and_order():
    sample = copy.deepcopy(SAMPLES["18_S19"])
    normal = to_gamedata(sample)
    expected = convert_gamedata(normal)["heroes"]
    reverse = {**normal, "hero": list(reversed(normal["hero"]))}
    assert convert_gamedata(reverse)["heroes"] == expected
    # 官方明确隐藏的单位，即使无同名主形态，亦不能因 isVisible=True 而进池。
    hidden = next(h for h in normal["hero"] if h["id"] == "15461")
    rows = convert_gamedata({**normal, "hero": [hidden]})["heroes"]
    assert rows[0]["in_pool"] == 0 and rows[0]["unit_kind"] == "other"
    # dataj 缺少 showHeroTag 时保留可见性降级路径。
    del hidden["showHeroTag"]
    rows = convert_gamedata({**normal, "hero": [hidden]})["heroes"]
    assert rows[0]["in_pool"] == 1


def test_v4_migration_recovers_variants_without_losing_rows(tmp_path):
    path = tmp_path / "v4.db"
    rows = convert_gamedata(to_gamedata(SAMPLES["16_S19"]))["heroes"]
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("INSERT INTO meta VALUES ('schema_version', '4')")
        c.execute(f"CREATE TABLE heroes (season_id TEXT, {HERO_COLS}, "
                  "skill_values TEXT NOT NULL DEFAULT '[]', skill_brief_values TEXT NOT NULL DEFAULT '[]', "
                  "sell_price INTEGER, skill_icon TEXT, PRIMARY KEY (season_id, id))")
        _insert(c, "heroes", [{"season_id": "16_S19", **r} for r in rows])
    for _ in range(2):
        db = GameDB(path, season_id="16_S19")
        assert len(db.heroes) == len(rows)
        assert db.hero("卡莎").id == "4360" and db.heroes["4348"].variant_of == "4360"
        assert db.hero("加里奥").unit_kind == "other"
        db.close()
