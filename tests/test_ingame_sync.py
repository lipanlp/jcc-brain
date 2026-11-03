import copy
import gzip
import json

import httpx
import pytest

from jccbrain.data import official as off
from jccbrain.data.archive import ArchivedOfficial, load_archive, rebuild_version
from jccbrain.data.db import GameDB
from jccbrain.data.sync import sync_season
from test_seasons import FakeOfficial, LINEUP, S18


STAT = {"use": 10, "week_use": 3, "collect": 2, "comment": 1, "stat_date": "20261022"}


class IngameOfficial(FakeOfficial):
    def ingame_lineups(self, sid):
        shared = copy.deepcopy(LINEUP)
        shared["sortID"] = "42"
        shared["detail"] = json.loads(shared["detail"])
        extra = copy.deepcopy(shared)
        extra["id"], extra["sortID"] = "new", "43"
        extra["detail"]["line_name"] = "微社区独有攻略"
        return [shared, extra]

    def lineup_stats(self, ids):
        assert set(ids) == {LINEUP["id"], "new"}
        return {LINEUP["id"]: dict(STAT)}


def test_merge_keeps_web_content_order_and_ingame_sort():
    a = {**LINEUP, "id": "A", "sortID": "10"}
    b = {**LINEUP, "id": "B", "sortID": "20"}
    ingame_b = {**b, "sortID": "99"}
    c = {**LINEUP, "id": "C", "sortID": "30", "detail": json.loads(LINEUP["detail"])}
    merged = off.merge_lineups([a, b], [ingame_b, c])
    assert [ln["id"] for ln in merged] == ["A", "B", "C"]
    assert [ln["_channels"] for ln in merged] == [["web"], ["web", "ingame"], ["ingame"]]
    assert [ln["_ingame_sort"] for ln in merged] == [None, 99, 30]
    assert merged[1]["sortID"] == "20"
    assert isinstance(merged[2]["detail"], str)
    assert isinstance(c["detail"], dict)
    assert "_channels" not in a and "_channels" not in ingame_b
    guides, _ = off.convert_guides(merged)
    new_keys = {"channels", "ingame_sort", "use_count", "week_use", "collect_count", "comment_count", "stat_date"}
    assert all(set(g) == set(guides[0]) and new_keys <= g.keys() for g in guides)
    assert [g["sort"] for g in guides] == [10, 20, 30]
    assert off.apply_stats(guides, {"B": STAT}) == 1
    assert guides[1]["use_count"] == 10 and guides[1]["week_use"] == 3
    assert guides[0]["use_count"] is None and guides[2]["stat_date"] is None


@pytest.mark.parametrize("sort", [None, "invalid"])
def test_merge_invalid_ingame_sort_is_unknown(sort):
    merged = off.merge_lineups([], [{**LINEUP, "sortID": sort}])
    assert merged[0]["_ingame_sort"] is None


def test_sync_merges_channels_and_stats(tmp_path):
    db = GameDB(tmp_path / "game.db", S18.id)
    try:
        progress = []
        stats = sync_season(db, S18, official=IngameOfficial(), progress=progress.append)
        assert len(db.guides) == 2
        shared, extra = db.guides[LINEUP["id"]], db.guides["new"]
        assert shared.channels == ("web", "ingame") and shared.ingame_sort == 42
        assert (shared.use_count, shared.week_use, shared.collect_count, shared.comment_count) == (10, 3, 2, 1)
        assert shared.stat_date == "20261022"
        assert extra.channels == ("ingame",) and extra.ingame_sort == 43 and extra.use_count is None
        assert stats["ingame_guides"] == 2 and stats["guides_with_stats"] == 1
        assert any("官方攻略 2（微社区 2 / 带热度 1）" in m for m in progress)
        assert db._conn.execute("SELECT sort FROM guides WHERE id = ?", (LINEUP["id"],)).fetchone()[0] == 98304
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["ingame", "stats"])
def test_sync_ingame_failures_degrade(tmp_path, caplog, failure):
    class Failed(IngameOfficial):
        def ingame_lineups(self, sid):
            if failure == "ingame":
                raise off.OfficialError("微社区断网")
            return super().ingame_lineups(sid)

        def lineup_stats(self, ids):
            raise off.OfficialError("热度断网")

    db = GameDB(tmp_path / "game.db", S18.id)
    try:
        progress = []
        stats = sync_season(db, S18, official=Failed(), progress=progress.append)
        assert stats["guides"] == (1 if failure == "ingame" else 2)
        assert all(g.use_count is None and g.stat_date is None for g in db.guides.values())
        if failure == "ingame":
            assert db.guides[LINEUP["id"]].channels == ("web",)
        message = "微社区阵容获取失败" if failure == "ingame" else "微社区热度获取失败"
        assert message in caplog.text and any(message in m for m in progress)
    finally:
        db.close()


@pytest.mark.parametrize("web_fails", [False, True])
def test_ingame_sync_works_when_web_has_no_lineups(tmp_path, web_fails):
    class NoWeb(IngameOfficial):
        def lineups(self, entry, sid=None):
            if web_fails:
                raise off.OfficialError("官网攻略暂不可用")
            return []

    db = GameDB(tmp_path / "game.db", S18.id)
    try:
        result = sync_season(db, S18, official=NoWeb())
        assert result["guides"] == 2
        assert all(g.channels == ("ingame",) for g in db.guides.values())
    finally:
        db.close()


class RawIngameOfficial(IngameOfficial):
    def __init__(self):
        super().__init__()
        self.raw_responses = {
            off.INGAME_HOME: b'<script>ZMProjectConfig = {"lineup_way_id":"2"}</script>',
            off.INGAME_LINEUP.format(season_num="19", way="2", mode="18"):
                json.dumps({"lineup_list": self.ingame_lineups(S18.id)}).encode(),
        }
        self.raw_posts = [{"url": off.INGAME_STAT_URL,
                           "request": {"mode": 6, "lineup_ids": [LINEUP["id"], "new"]},
                           "body": json.dumps({"result": 0, "data": [
                               {"lineup_id": LINEUP["id"], "use": 10, "week_use_num": 3,
                                "collect": 2, "comment": 1, "stat_date": "20261022"}]}).encode()}]


def guide_summary(db):
    return {key: (g.channels, g.ingame_sort, g.use_count, g.week_use, g.collect_count, g.comment_count, g.stat_date)
            for key, g in db.guides.items()}


def test_archive_keeps_raw_ingame_evidence_and_rebuilds(tmp_path, monkeypatch):
    official = RawIngameOfficial()
    db = GameDB(tmp_path / "online.db", S18.id)
    db2 = GameDB(tmp_path / "rebuilt.db", S18.id)
    try:
        online = sync_season(db, S18, official=official)
        folder = tmp_path / "raw" / S18.id / online["game_version"]
        manifest_path = folder / "manifest.json"
        manifest_before = manifest_path.read_bytes()
        manifest = json.loads(manifest_before)
        files = manifest["files"]
        assert any(ref["url"] == off.INGAME_HOME for ref in files)
        assert any(ref["url"].endswith("lineup_detail_total_new.json") for ref in files)
        post = next(ref for ref in files if ref["file"].startswith("lineup_stat.0.json."))
        assert post["url"] == off.INGAME_STAT_URL
        assert post["request"] == official.raw_posts[0]["request"]
        assert gzip.decompress((folder / post["file"]).read_bytes()) == official.raw_posts[0]["body"]
        snapshot = load_archive(tmp_path / "raw", S18.id, online["game_version"])
        assert snapshot["ingame_lineups"] and snapshot["lineup_stats"] == {LINEUP["id"]: STAT}
        assert snapshot["lineups"] == [LINEUP]  # 官网原始记录，不能保存合并后的两条
        assert all("_channels" not in ln for ln in snapshot["ingame_lineups"])

        def no_network(*args, **kwargs):
            pytest.fail("离线重建不可联网")

        monkeypatch.setattr(httpx.Client, "get", no_network)
        monkeypatch.setattr(httpx.Client, "post", no_network)
        rebuilt = rebuild_version(db2, S18.id, online["game_version"])
        for key in ("guides", "ingame_guides", "guides_with_stats", "guide_stat_date"):
            assert rebuilt[key] == online[key]
        assert guide_summary(db2) == guide_summary(db)
        assert manifest_path.read_bytes() == manifest_before
        assert ArchivedOfficial(snapshot).lineup_stats([LINEUP["id"], "absent"]) == {LINEUP["id"]: STAT}
    finally:
        db.close()
        db2.close()


def test_legacy_snapshot_rebuilds_web_only_without_stats(tmp_path, monkeypatch):
    db = GameDB(tmp_path / "game.db", S18.id)
    try:
        official = RawIngameOfficial()
        online = sync_season(db, S18, official=official)
        snapshot = load_archive(tmp_path / "raw", S18.id, online["game_version"])
        del snapshot["ingame_lineups"]
        del snapshot["lineup_stats"]
        archived = ArchivedOfficial(snapshot)
        assert archived.ingame_lineups(S18.id) == []
        assert archived.lineup_stats([LINEUP["id"]]) == {}
        monkeypatch.setattr(httpx.Client, "request", lambda *a, **kw: pytest.fail("旧快照重建不可联网"))
        rebuilt = sync_season(db, S18, official=archived)
        assert rebuilt["guides"] == 1
        assert rebuilt["ingame_guides"] == 0 and rebuilt["guides_with_stats"] == 0
        assert db.guides[LINEUP["id"]].channels == ("web",)
        assert db.guides[LINEUP["id"]].use_count is None
    finally:
        db.close()
