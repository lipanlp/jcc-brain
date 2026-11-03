import copy
import json
import pytest

from jccbrain.data.models import Comp, CompHero, Guide, GuideHero, Hero
from jccbrain.data.matching import link_guides, similarity
from jccbrain.data.db import GameDB
from jccbrain.data.sync import sync_season
from test_seasons import FakeOfficial, FakeDataj, S18, LINEUP
from conftest import GAMEDATA, COMP, COMP_DETAIL


def comp(cid="c", primary="a", ids="abcdefgh"):
    return Comp(cid, cid, "S", 3.5, 60, 20, 1, 100, "test",
                heroes=[CompHero(i, i, 2 if i == "a" else 4, is_carry=i == primary) for i in ids])


def guide(gid="g", primary="a", ids="abcdefgh"):
    return Guide(gid, gid, "S", "test", boards={"final": [
        GuideHero(i, i, is_carry=i == primary, cost=2 if i == "a" else 4) for i in ids]})


def test_reject_wrong_carry_and_ambiguous_matches():
    g = guide()
    correct, wrong = comp(), comp("wrong", primary="b")
    assert similarity(g, wrong, {})[1] is False
    link_guides({g.id: g}, {correct.id: correct, wrong.id: wrong}, {})
    assert g.comp is correct and correct.guide is g and wrong.guide is None
    # 同样好的两个统计候选不强行选择，重算会清除旧链接。
    twin = comp("twin")
    link_guides({g.id: g}, {correct.id: correct, twin.id: twin}, {})
    assert g.comp is None and correct.guide is None
    distant = comp(ids="axyz")
    link_guides({g.id: g}, {distant.id: distant}, {})
    assert g.comp is None


def test_variant_identity_and_multiple_guides():
    g, c = guide(ids="vbcdefgh", primary="v"), comp()
    heroes = {"v": Hero("v", "a", 2, (), variant_of="a", unit_kind="variant")}
    assert similarity(g, c, heroes)[0] == 1
    second = guide("g2")
    link_guides({g.id: g, second.id: second}, {c.id: c}, heroes)
    assert len(c.guides) == 2 and c.guide is g


@pytest.mark.parametrize("chase,timing,name,expected", [
    (["a"], "上9", "九五", "reroll"),
    ([], "4-2升8级，全二后再上9", "阵容", "fast8"),
    ([], "直接升9人口", "阵容", "fast9"),
    ([], "7人口慢D追三星", "阵容", "reroll"),
    ([], "", "巨龙九五", "fast9"),
])
def test_style_prefers_guide(chase, timing, name, expected):
    c, g = comp(), guide()
    g.three_star, g.d_time, g.name = chase, timing, name
    c.guide = g
    assert c.style == expected


def test_database_links_dedup_and_legacy_id_resolution(tmp_path):
    row = copy.deepcopy(LINEUP)
    detail = json.loads(row["detail"])
    detail["hero_location"] = [{"hero_id": "1" + h["heroId"], "chess_type": "hero",
                                "is_carry_hero": h["isCarry"]} for h in COMP_DETAIL["heroes"]]
    row["detail"] = json.dumps(detail)
    path = tmp_path / "game.db"
    db = GameDB(path)
    sync_season(db, S18, official=FakeOfficial(lineups=[row]), dataj=FakeDataj(GAMEDATA, [COMP]))
    for _ in range(2):
        c, g = db.comps["104"], db.guides["4932"]
        assert c.guide is g and g.comp is c and c.guide_id == g.id
        assert c.avg_placement == 3.5 and c.source == "dataj"
        assert db.comp("official:4932") is c
        assert db.comps_ranked() == [c] and len(db.comps_ranked("official")) == 1
        db.use_season("16_S19")
        assert not db.guides and not db.comps_ranked()
        db.use_season("18_S19")
        db.close()
        db = GameDB(path)
    db.close()
