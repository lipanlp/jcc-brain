"""赛季表，以金铲铲官方资料库为准。

赛季 id 用官方编号 "<mode>_<season>"（如自然之力 = "18_S19"，官网 basicConfig.js 的 season_tabs）。
同一个 mode 可能有多个赛季（英雄联盟传奇 16_S19 与「恭喜发财」Mode16_S100024），所以不能只用数字区分。
dataj.cc 只覆盖部分赛季，dataj_set 是它的 setId（没有就为 None）。

赛季列表会在同步时从官网刷新并存进数据库；这里的 SEASONS 是离线时的默认列表（2026-10 官网的 5 个赛季）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Season:
    id: str                         # 官方 id，如 "18_S19"
    name: str                       # 如「自然之力」
    mode: str                       # 官方 mode，如 "18"
    season: str                     # 官方 season，如 "S19"
    dataj_set: int | None = None    # dataj.cc 的 setId，没有对应时为 None
    rules_set: str | None = None    # data/rules/<官方赛季id>.json（文件名不含 .json）

    @property
    def label(self) -> str:
        return self.name


# 官网顺序
SEASONS: tuple[Season, ...] = (
    Season("Mode11_S100030", "画之灵", "11", "S100030"),
    Season("Mode16_S100024", "英雄联盟传奇·恭喜发财", "16", "S100024"),
    Season("18_S19", "自然之力", "18", "S19", dataj_set=18, rules_set="18_S19"),
    Season("16_S19", "英雄联盟传奇", "16", "S19", dataj_set=16),
    Season("1_S1", "时空裂痕", "1", "S1"),
)

DEFAULT_SEASON_ID = "18_S19"

# 已知赛季的 dataj / 规则表对应关系（官网刷新出来的新赛季没有这些信息，按 id 补上）
_EXTRA = {s.id: s for s in SEASONS}

# 旧版数据库/设置用 dataj setId 区分赛季，迁移时换成官方 id；没有对应的（如 dataj 的「星神」，官网已下架）丢弃
LEGACY_SET_IDS = {18: "18_S19", 16: "16_S19"}


def make_season(season_id: str, name: str, mode: str, season_code: str) -> Season:
    """官网列表里的一条 → Season，补上已知的 dataj / 规则表对应。"""
    known = _EXTRA.get(season_id)
    return Season(season_id, name, mode, season_code,
                  known.dataj_set if known else None, known.rules_set if known else None)


def season(season_id: str, catalog: tuple[Season, ...] | list[Season] = SEASONS) -> Season:
    """按 id 找赛季；不认识的 id 也能用（名称就是 id，mode/season 从 id 拆出）。"""
    hit = next((s for s in catalog if s.id == season_id), None) or _EXTRA.get(season_id)
    if hit:
        return hit
    mode, _, code = season_id.removeprefix("Mode").partition("_")
    return Season(season_id, season_id, mode, code)
