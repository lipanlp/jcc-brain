import json

from jccbrain.data.sync import convert_comp_builds, convert_gamedata

from conftest import GAMEDATA


def test_convert_gamedata_heroes():
    t = convert_gamedata(GAMEDATA)
    heroes = {h["id"]: h for h in t["heroes"]}
    assert "1500" in heroes and len([h for h in t["heroes"] if h["name"] == "奥恩"]) == 1   # 只取 1 星
    assert json.loads(heroes["1500"]["traits"]) == ["永恒之森", "护卫"]
    assert heroes["8702"]["in_pool"] == 0          # 召唤物
    assert heroes["1513"]["in_pool"] == 0          # 同名变体
    assert heroes["1508"]["in_pool"] == 1


def test_convert_gamedata_traits_dedup_and_kind():
    t = convert_gamedata(GAMEDATA)
    traits = {r["name"]: r for r in t["traits"]}
    assert len([r for r in t["traits"] if r["name"] == "永恒之森"]) == 1
    assert traits["护卫"]["kind"] == "class" and traits["永恒之森"]["kind"] == "origin"
    assert json.loads(traits["猎人"]["breakpoints"]) == [2, 3, 4, 5]


def test_db_queries(db):
    assert db.hero("绯红树怪").cost == 1
    assert db.hero("红buff").name == "绯红树怪"          # 别名
    assert db.pool_count_by_cost() == {1: 3, 2: 1, 3: 1, 4: 1, 5: 1}
    assert db.trait("护卫").tier_of(4) == 2 and db.trait("护卫").next_breakpoint(3) == 4
    assert db.item_name("2010") == "无尽之刃"
    assert db.items["2010"].recipe == ("1001", "1001")
    assert db.augment("飞升").tier == 2


def test_db_comp(db):
    comp = db.comps["104"]
    assert comp.style == "reroll"
    carry = comp.heroes[0]
    assert carry.name == "绯红树怪" and carry.is_carry and carry.position == (4, 7)
    assert carry.items == ("无尽之刃",)
    assert comp.game_code == "#野怪小红#abc"
    assert [a.name for a in comp.augments] == ["飞升", "存心失利"]   # 按均名排序


def test_convert_comp_builds():
    rows = convert_comp_builds("95", [{"peopleNum": 4, "builds": [
        {"rank": 1, "winRate": 20.7, "sampleCount": 3459,
         "heroes": [{"heroBaseId": 1505, "heroStarNum": 1, "position": "1,7"}]}]}], {"1505": "洛"})
    assert rows[0]["people"] == 4 and json.loads(rows[0]["heroes"])[0]["name"] == "洛"
