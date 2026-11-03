"""测试夹具：用合成的小型 dataj 数据建一个临时数据库，不依赖网络。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jccbrain.core.rules import load_rules
from jccbrain.data.db import GameDB
from jccbrain.data.models import Guide, GuideHero
from jccbrain.data.sync import convert_comp, convert_comp_hexes, convert_gamedata

FIXTURES = Path(__file__).parent / "fixtures"


def _hero(hid: str, name: str, cost: int, race: str, job: str, rng: int = 1, visible: bool = True) -> dict:
    return {"id": hid, "name": name, "price": cost, "heroType": 0, "isVisible": visible, "raceId": race,
            "jobId": job, "attackRange": rng, "tag": None, "skillName": "技能", "skillDesc": "描述"}


def _trait(tid: str, check: str, name: str, kind: str, nums: str) -> dict:
    return {"id": tid, "checkId": check, "name": name, "type": kind, "numList": nums, "prefix": f"{name}效果",
            "desc2Text": "|".join(f"({n})加成" for n in nums.split("|"))}


GAMEDATA = {
    "trait": [
        _trait("1", "450", "永恒之森", "0", "3|5|7"), _trait("2", "450", "永恒之森", "0", "3|5|7"),
        _trait("3", "341", "护卫", "1", "2|4|6"), _trait("4", "345", "猎人", "1", "2|3|4|5"),
        _trait("5", "466", "峡谷野怪", "0", "3|5|7|10"),
    ],
    "hero": [
        _hero("11500", "奥恩", 1, "450", "341"),
        _hero("21500", "奥恩", 1, "450", "341"),                 # 2 星记录，应被忽略
        _hero("11508", "绯红树怪", 1, "466", "345", rng=4),
        _hero("11511", "苍蓝哨戒", 1, "466", "341"),
        _hero("12510", "峡谷迅捷蟹", 2, "466", "341"),
        _hero("13503", "远古石甲虫", 3, "466", "341"),
        _hero("14509", "厄斐琉斯", 4, "450", "345", rng=4),
        _hero("15458", "远古巨龙", 5, "466", "345"),
        _hero("18702", "小型石甲虫", 3, "-1", "-1"),             # 召唤物：没有羁绊，不进卡池
        _hero("11513", "绯红树怪", 1, "466", "345"),             # 同名变体，不重复进卡池
    ],
    "equip": [
        {"id": "1001", "name": "暴风大剑", "type": "基础装备", "basicDesc": "+攻击力", "synthesis1": "0", "synthesis2": "0"},
        {"id": "2010", "name": "无尽之刃", "type": "成型装备", "basicDesc": "暴击", "synthesis1": "1001", "synthesis2": "1001"},
    ],
    "hex": [
        {"id": "2021", "name": "飞升", "level": 2, "descText": "战斗后段增伤"},
        {"id": "1002", "name": "存心失利", "level": 1, "descText": "输了给钱"},
    ],
    "adventure": [],
    "searchAlias": [{"entityType": "hero", "name": "绯红树怪", "keywords": ["红buff"]}],
}

COMP = {
    "compId": "104", "name": "野怪小红", "tier": "S", "avgPlacement": 3.5, "top4Rate": 68.9, "topRate": 22.2,
    "pickRate": 0.3, "sampleCount": 5888,
    "heroes": [
        {"heroId": "1508", "heroName": "绯红树怪", "price": 1, "isCarry": True, "isSubCarry": False, "isCore": True, "heroStarNum": 3},
        {"heroId": "1511", "heroName": "苍蓝哨戒", "price": 1, "isCarry": False, "isSubCarry": False, "isCore": True, "heroStarNum": 3},
        {"heroId": "3503", "heroName": "远古石甲虫", "price": 3, "isCarry": False, "isSubCarry": False, "isCore": True, "heroStarNum": None},
        {"heroId": "5458", "heroName": "远古巨龙", "price": 5, "isCarry": False, "isSubCarry": True, "isCore": True, "heroStarNum": None},
        {"heroId": "2510", "heroName": "峡谷迅捷蟹", "price": 2, "isCarry": False, "isSubCarry": False, "isCore": False, "heroStarNum": None},
    ],
    "traits": [{"id": "x", "name": "峡谷野怪"}],
    "equips": [{"heroId": "1508", "equipId": "2010"}],
    "conditions": [],
}
COMP_DETAIL = {**COMP, "gameCode": "#野怪小红#abc",
               "heroes": [{**h, "position": p} for h, p in zip(COMP["heroes"], ["4,7", "4,5", "1,5", "4,3", "1,2"])]}
COMP_HEXES = [
    {"hexId": 2021, "name": "飞升", "level": 2, "tier": "B", "playRate": 13, "avgPlacement": 3.2,
     "top4Rate": 70, "topRate": 20, "placementDelta": -0.3, "rounds": ["3-2"]},
    {"hexId": 1002, "name": "存心失利", "level": 1, "tier": "D", "playRate": 2, "avgPlacement": 4.4,
     "top4Rate": 50, "topRate": 8, "placementDelta": 0.9, "rounds": ["2-1"]},
]


@pytest.fixture
def db(tmp_path) -> GameDB:
    db = GameDB(tmp_path / "test.sqlite3")
    tables = convert_gamedata(GAMEDATA)
    items = {r["id"]: r["name"] for r in tables["items"]}
    comp, heroes = convert_comp(COMP, items, "18.4", COMP_DETAIL)
    with db.transaction() as conn:
        for t in ("traits", "heroes", "items", "augments", "mechanics", "aliases"):
            db.replace_table(conn, t, tables[t])
        db.replace_table(conn, "comps", [comp])
        db.replace_table(conn, "comp_heroes", heroes)
        db.replace_table(conn, "comp_augments", convert_comp_hexes("104", COMP_HEXES))
    db.reload()
    yield db
    db.close()


@pytest.fixture
def rules():
    return load_rules(18)


@pytest.fixture
def fixture_states() -> list[dict]:
    return [json.loads(l) for l in (FIXTURES / "game_eye_states.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def board_comp(db):
    """带多人口站位的合成攻略，只用于站位测试，不读取运行目录。"""
    def hero(name, pos, **kw):
        h = db.hero(name)
        return GuideHero(h.id, name, position=pos, cost=h.cost, **kw)

    early = [hero('奥恩', '1,1'), hero('绯红树怪', '4,7', is_carry=True, items=('无尽之刃',))]
    mid = early + [hero('苍蓝哨戒', '1,3'), hero('峡谷迅捷蟹', '1,5')]
    final = mid + [hero('远古巨龙', '4,3'), hero('厄斐琉斯', '4,5')]
    guide = Guide('board-test', '合成站位攻略', 'S', None,
                  carry_items=['无尽之刃'], boards={'early': early, 'mid': mid, 'final': final,
                                                  'L6': mid, 'L8': final})
    comp = db.comp('104')
    comp.guide = guide
    return comp


def make_state(**kw) -> dict:
    """构造一个 game-eye 风格的状态 dict。"""
    base = {"stage": "3-2", "phase": "planning", "gold": 50, "level": 6, "xp": [0, 36], "hp": 80,
            "refresh_cost": 2, "xp_cost": 4, "odds": [30, 40, 25, 5, 0],
            "shop": {"open": True, "slots": [None] * 5}, "bench": [], "board": [], "traits": [], "players": []}
    base.update(kw)
    return base
