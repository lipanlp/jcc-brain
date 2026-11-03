"""dataj.cc（金铲铲大数据）接口客户端。

接口来自 https://www.dataj.cc 前端代码（/api/web 前缀），返回格式统一为
{"code": 200, "success": true, "message": "...", "data": ...}。
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


class DatajError(RuntimeError):
    pass


class DatajClient:
    def __init__(self, base_url: str = "https://www.dataj.cc/api/web", set_id: int = 18,
                 interval_s: float = 0.3, timeout_s: float = 30.0):
        self.set_id = set_id
        self.interval_s = interval_s
        self._last = 0.0
        self._http = httpx.Client(
            base_url=base_url, timeout=timeout_s, follow_redirects=True,
            headers={"User-Agent": "jccbrain/0.1 (+local desktop tool)", "Referer": "https://www.dataj.cc/comp"},
        )

    def _get(self, path: str, **query: Any) -> Any:
        wait = self.interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        q = {k: v for k, v in query.items() if v is not None}
        for attempt in range(3):
            try:
                r = self._http.get(path, params=q)
                self._last = time.monotonic()
                r.raise_for_status()
                body = r.json()
                break
            except (httpx.HTTPError, ValueError) as e:
                if attempt == 2:
                    raise DatajError(f"{path} 请求失败：{e}") from e
                time.sleep(1.0 * (attempt + 1))
        if not body.get("success"):
            raise DatajError(f"{path} 返回错误：{body.get('message')}")
        return body.get("data")

    # ---- 基础数据 ----
    def game_versions(self) -> list[dict]:
        """所有版本（新→旧），含 gameVersion / startTime / battleCount。"""
        data = self._get("/gamedata/gameVersion", setId=self.set_id)
        if isinstance(data, dict):
            data = data.get("gameVersions") or data.get("list") or []
        return list(data or [])

    def latest_version(self) -> str | None:
        vs = self.game_versions()
        return vs[0]["gameVersion"] if vs else None

    def gamedata(self) -> dict:
        """英雄 / 羁绊 / 装备 / 海克斯 / 赛季机制 / 搜索别名。"""
        return self._get("/gamedata", setId=str(self.set_id))

    # ---- 阵容 ----
    def comp_rank(self, game_version: str | None = None) -> list[dict]:
        return self._get("/comp/rank", setId=self.set_id, gameVersion=game_version) or []

    def comp_recommended(self) -> list[dict]:
        return self._get("/comp/recommended", setId=self.set_id) or []

    def comp_detail(self, comp_id: str, game_version: str | None = None) -> dict:
        return self._get(f"/comp/{comp_id}", setId=self.set_id, gameVersion=game_version) or {}

    def comp_hexes(self, comp_id: str, game_version: str | None = None) -> list[dict]:
        data = self._get(f"/comp/{comp_id}/hexes", setId=self.set_id, gameVersion=game_version) or {}
        return data.get("hexes", []) if isinstance(data, dict) else data

    def comp_builds(self, comp_id: str, game_version: str | None = None) -> list[dict]:
        return self._get(f"/comp/{comp_id}/battle-builds", setId=self.set_id, gameVersion=game_version) or []

    def close(self) -> None:
        self._http.close()
