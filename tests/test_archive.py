import copy
import gzip
import json
from types import SimpleNamespace

import httpx
import pytest

from jccbrain import cli
from jccbrain.core.seasons import season
from jccbrain.data import archive, official as off
from jccbrain.data.archive import ArchiveError, diff_versions, load_archive, rebuild_version, save_archive
from jccbrain.data.db import GameDB
from jccbrain.data.sync import SyncError, sync_season
from test_seasons import FakeOfficial, _official_files, S18


def entry(version):
    return {"version": version, "mode": "18", "season": "S19", "herourl": f"/18/{version}-S19/chess.js"}


def test_original_response_gzip_and_offline_rebuild(tmp_path, monkeypatch):
    raw = b'{ "data": [], "note": "keep whitespace" }\n'
    client = off.OfficialClient(interval_s=0)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw)))
    client.gamedata(entry("v1"))
    files = _official_files()
    save_archive(tmp_path / "raw", S18.id, entry("v1"), files, [], None, client.raw_responses)
    client.close()
    manifest = json.loads((tmp_path / "raw" / S18.id / "v1" / "manifest.json").read_text())
    ref = manifest["files"][0]
    assert gzip.decompress((tmp_path / "raw" / S18.id / "v1" / ref["file"]).read_bytes()) == raw
    monkeypatch.setattr(httpx.Client, "get", lambda *a, **kw: pytest.fail("离线重建不能联网"))
    db = GameDB(tmp_path / "game.db")
    stats = rebuild_version(db, S18.id, "v1")
    assert stats["game_version"] == "v1" and db.hero("奥恩")
    assert json.loads((tmp_path / "raw" / S18.id / "v1" / "manifest.json").read_text()) == manifest
    db.close()


def test_diff_stats_and_text_without_invented_direction(tmp_path):
    old = _official_files()
    new = copy.deepcopy(old)
    old["chess"][0]["initAttackDamage"] = "60"
    new["chess"][0]["initAttackDamage"] = "63"
    new["equip"][0]["desc"] = "新效果，不直接判断强弱"
    for version, files in (("v1", old), ("v2", new)):
        save_archive(tmp_path, S18.id, entry(version), files, [], None, {})
    result = diff_versions(tmp_path, S18.id, "v1", "v2")
    stat = next(c for c in result["changes"] if c["field"] == "attack_damage")
    assert (stat["old"], stat["new"], stat["direction"], stat["star"]) == (60, 63, "buff", 1)
    assert next(c for c in result["changes"] if c["entity"] == "items")["direction"] == "changed"
    assert diff_versions(tmp_path, S18.id, "v2", "v1")["summary"]["nerf"] == 1
    assert diff_versions(tmp_path, S18.id, "v1", "v1")["changes"] == []


def test_atomic_manifest_and_corruption_detection(tmp_path, monkeypatch):
    old, new = _official_files(), _official_files()
    new["chess"][0]["name"] = "改名"
    save_archive(tmp_path, S18.id, entry("v1"), old, [], None, {})
    original = archive._atomic

    def fail_manifest(path, body):
        if path.name == "manifest.json":
            raise OSError("磁盘故障")
        original(path, body)

    monkeypatch.setattr(archive, "_atomic", fail_manifest)
    with pytest.raises(OSError):
        save_archive(tmp_path, S18.id, entry("v1"), new, [], None, {})
    assert load_archive(tmp_path, S18.id, "v1")["gamedata"] == old
    manifest_path = tmp_path / S18.id / "v1" / "manifest.json"
    ref = json.loads(manifest_path.read_text())["snapshot"]
    (manifest_path.parent / ref["file"]).write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ArchiveError, match="校验失败"):
        load_archive(tmp_path, S18.id, "v1")
    with pytest.raises(ArchiveError, match="非法"):
        load_archive(tmp_path, "../outside", "v1")


def test_sync_archive_failure_keeps_previous_game_data(tmp_path, monkeypatch):
    db = GameDB(tmp_path / "game.db")
    sync_season(db, S18, official=FakeOfficial())
    changed = _official_files()
    changed["chess"][0]["name"] = "改名"
    monkeypatch.setattr(archive, "save_archive", lambda *a, **kw: (_ for _ in ()).throw(OSError("写入失败")))
    with pytest.raises(SyncError, match="未写入游戏数据"):
        sync_season(db, S18, official=FakeOfficial(files=changed))
    assert db.hero("奥恩") and not db.hero("改名")
    db.close()


def test_cli_check_reports_unknown_and_uses_no_data_files(tmp_path, monkeypatch, capsys):
    db = GameDB(tmp_path / "game.db")
    db.save_catalog([S18, season("16_S19")])
    ctx = SimpleNamespace(db=db, seasons_needing_sync=lambda: [(S18.id, None, "v1")], close=lambda: None)
    monkeypatch.setattr(cli, "_ctx", lambda args: ctx)

    class VersionsOnly:
        def find_version(self, s):
            if s.id == "16_S19":
                raise off.OfficialError("无法确认")
            return {"version": "v1"}

        def close(self):
            pass

    monkeypatch.setattr(off, "OfficialClient", VersionsOnly)
    assert cli.cmd_db_check(SimpleNamespace(json=True)) == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["latest"] == "v1" and rows[0]["status"] == "需要同步"
    assert rows[1]["latest"] is None and "无法确认" in rows[1]["status"]
    assert not (tmp_path / "raw").exists()
    db.close()
