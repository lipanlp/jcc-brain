"""官方原始响应 gzip 存档、离线重建与版本差异。文件内容寻址，manifest 最后原子替换。"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from jccbrain.core.seasons import make_season
from jccbrain.data import official as off


class ArchiveError(ValueError):
    pass


def archive_dir(root: Path, season_id: str, version: str) -> Path:
    for value in (season_id, version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ArchiveError(f"非法赛季或版本：{value}")
    return Path(root) / season_id / version


def _atomic(path: Path, body: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".archive-", delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _blob(folder: Path, name: str, body: bytes) -> dict:
    digest = hashlib.sha256(body).hexdigest()
    filename = f"{name}.{digest[:16]}.gz"
    _atomic(folder / filename, gzip.compress(body, mtime=0))
    return {"file": filename, "sha256": digest, "bytes": len(body)}


def save_archive(root: Path, sid: str, entry: dict, files: dict, lineups: list,
                 tags: dict | None, responses: dict[str, bytes], *, ingame_lineups: list | None = None,
                 lineup_stats: dict | None = None, posts: list | None = None) -> Path:
    folder = archive_dir(root, sid, str(entry["version"]))
    folder.mkdir(parents=True, exist_ok=True)
    expected = {off.GAMEDATA_BASE + entry[k] for k in off.FILES if entry.get(k)}
    raw = []
    for url, body in responses.items():
        if url not in expected and url not in (off.BASIC_CONFIG, off.VERSION_INDEX, off.LINEUP_TAGS_URL,
                                               off.INGAME_HOME) and \
                not url.endswith(("lineup_detail_total.json", "lineup_detail_total_new.json")):
            continue
        raw.append({"url": url, **_blob(folder, Path(urlsplit(url).path).name, body)})
    for i, post in enumerate(posts or []):
        raw.append({"url": post["url"], "request": post["request"],
                    **_blob(folder, f"lineup_stat.{i}.json", post["body"])})
    snapshot = {"entry": entry, "gamedata": files, "lineups": lineups, "tags": tags,
                "ingame_lineups": ingame_lineups or [], "lineup_stats": lineup_stats or {}}
    blob = _blob(folder, "snapshot.json", json.dumps(snapshot, ensure_ascii=False).encode())
    manifest = {"season_id": sid, "version": str(entry["version"]),
                "captured_at": datetime.now(timezone.utc).isoformat(), "files": raw, "snapshot": blob,
                "missing_urls": sorted(expected - responses.keys()),
                "note": "files 是原始响应；snapshot 是供离线重建使用的解码快照。"}
    path = folder / "manifest.json"
    _atomic(path, json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    return path


def _read(folder: Path, ref: dict) -> bytes:
    path = (folder / ref["file"]).resolve()
    if path.parent != folder.resolve():
        raise ArchiveError("存档文件路径超出版本目录")
    try:
        body = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, zlib.error) as e:
        raise ArchiveError(f"无法读取存档 {path.name}：{e}") from e
    if hashlib.sha256(body).hexdigest() != ref["sha256"]:
        raise ArchiveError(f"存档校验失败：{path.name}")
    return body


def load_archive(root: Path, sid: str, version: str) -> dict:
    folder = archive_dir(root, sid, version)
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["season_id"] != sid or manifest["version"] != version:
            raise ArchiveError("存档赛季或版本不一致")
        # 所有原始文件校验通过才允许重建/比较，避免用损坏的证据生成结果。
        for ref in manifest["files"]:
            _read(folder, ref)
        result = json.loads(_read(folder, manifest["snapshot"]))
        if str(result["entry"]["version"]) != version:
            raise ArchiveError("存档快照版本不一致")
        return result
    except (OSError, ValueError, KeyError) as e:
        if isinstance(e, ArchiveError):
            raise
        raise ArchiveError(f"无法加载 {sid}/{version} 存档：{e}") from e


class ArchivedOfficial:
    """sync_season 的离线客户端，不创建 HTTP 客户端、不覆盖已存原始证据。"""
    offline = True

    def __init__(self, snapshot: dict):
        self.snapshot = snapshot

    def season_tabs(self):
        return []  # 不用单版本快照覆盖现有赛季目录

    def find_version(self, season, version=None):
        entry = self.snapshot["entry"]
        if entry["mode"] != season.mode or entry["season"] != season.season or \
                (version and entry["version"] != version):
            raise off.OfficialError("快照与请求赛季/版本不符")
        return entry

    def gamedata(self, entry):
        return self.snapshot["gamedata"]

    def lineups(self, entry, sid=None):
        return self.snapshot["lineups"]

    def ingame_lineups(self, sid):
        return self.snapshot.get("ingame_lineups") or []

    def lineup_stats(self, ids):
        stats = self.snapshot.get("lineup_stats") or {}
        return {str(i): stats[str(i)] for i in ids if str(i) in stats}

    def lineup_tags(self):
        return self.snapshot["tags"] or off.LINEUP_TAGS


def rebuild_version(db, sid: str, version: str) -> dict:
    """从 db.path.parent/raw 重建指定赛季的官方部分；既有 dataj 统计不变。"""
    from jccbrain.data.sync import sync_season

    snapshot = load_archive(db.path.parent / "raw", sid, version)
    entry = snapshot["entry"]
    s = make_season(sid, entry.get("name") or sid, entry["mode"], entry["season"])
    return sync_season(db, s, official=ArchivedOfficial(snapshot))


FIELDS = {
    "heroes": ("cost", "skill_name", "skill_desc", "skill_values", "skill_brief_values", "sell_price"),
    "hero_stars": ("hp", "attack_damage", "attack_speed", "armor", "magic_resist", "init_mana", "max_mana", "crit_chance"),
    "traits": ("breakpoints", "description", "levels"),
    "items": ("kind", "description", "recipe", "trait_id", "effect_type"),
    "augments": ("tier", "description", "fetter_id", "fetter_type", "hero_enhancement_type", "is_legend"),
}
JSON_FIELDS = {"skill_values", "skill_brief_values", "breakpoints", "levels", "recipe"}


def diff_versions(root: Path, sid: str, old_version: str, new_version: str) -> dict:
    """只读离线存档。英雄属性可判强弱，文字/机制变化仅报 changed，避免误判语义。"""
    from jccbrain.data.sync import convert_gamedata

    old, new = [convert_gamedata(off.to_gamedata(load_archive(root, sid, v)["gamedata"]))
                for v in (old_version, new_version)]
    names = {r["id"]: r["name"] for t in (old, new) for r in t["heroes"]}
    changes = []
    for table, fields in FIELDS.items():
        def rows(data):
            return {(r["hero_id"], r["star"]) if table == "hero_stars" else r["id"]: r for r in data[table]}
        a, b = rows(old), rows(new)
        for key in sorted(a.keys() | b.keys()):
            before, after = a.get(key), b.get(key)
            row = after or before
            identity = {"entity": table, "id": row.get("id", row.get("hero_id")),
                        "name": row.get("name", names.get(row.get("hero_id"), ""))}
            if table == "hero_stars":
                identity["star"] = row["star"]
            if before is None or after is None:
                changes.append({**identity, "field": "record", "old": before, "new": after,
                                "direction": "added" if before is None else "removed"})
                continue
            for field in fields:
                x, y = before[field], after[field]
                if field in JSON_FIELDS:
                    x, y = json.loads(x), json.loads(y)
                if x == y:
                    continue
                direction = "changed"
                if table == "hero_stars" and isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    beneficial = y < x if field == "max_mana" else y > x
                    direction = "buff" if beneficial else "nerf"
                changes.append({**identity, "field": field, "old": x, "new": y, "direction": direction})
    return {"season_id": sid, "old_version": old_version, "new_version": new_version,
            "summary": dict(Counter(c["direction"] for c in changes)), "changes": changes,
            "note": "仅英雄基础属性自动标注加强/削弱；技能、羁绊、装备、强化的文字与机制变化需人工判断。"}
