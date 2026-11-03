"""金铲铲之战官方资料库（jcc.qq.com 背后的静态 JSON，game.gtimg.cn 上公开可取）。

- 版本索引 `jkzlk/js/config/versiondataconfig.js`：每个模式/赛季/版本一条，带各数据文件的相对路径。
- 每个版本的数据文件：chess（英雄）、trait（羁绊）、equip、hex、adventure（自然仙灵等每回合商店事件）、
  galaxy（开局奇遇）……字段与 dataj /gamedata 基本相同，只是部分键名不同（dataj 就是从这里转来的）。
- 官方阵容攻略 `lineup_detail_total.json`：地址写在官网的 `data-js/basicConfig.js` 里。

赛季以官方为准：id 为 "<mode>_<season>"（自然之力 = 18_S19），列表来自官网 basicConfig.js 的 season_tabs。
没有胜率统计：阵容排行、阵容内海克斯表现从 dataj 拿（只覆盖部分赛季）。
CDN 会缓存版本索引等「会变的」文件，请求时带时间戳参数绕过缓存（不然会拿到旧版本）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from jccbrain.core.seasons import Season, make_season

log = logging.getLogger(__name__)

GAMEDATA_BASE = "https://game.gtimg.cn/images/lol/act/jkzlk/js"
VERSION_INDEX = f"{GAMEDATA_BASE}/config/versiondataconfig.js"
BASIC_CONFIG = "https://jcc.qq.com/data-js/basicConfig.js"
LINEUP_FALLBACK = ("https://game.gtimg.cn/images/lol/act/jkzlkauto/json/lineupJson/"
                   "m{season_num}/{channel}/{mode}/lineup_detail_total.json")
LINEUP_CHANNEL = "11"           # 官网「阵容列表正式服务渠道」
LINEUP_TAGS_URL = "https://game.gtimg.cn/images/lol/act/jkzlkauto/json/tagJson/tag.json"
# 官网阵容页的分类（tag.json 取不到时用）
LINEUP_TAGS = {"1": "新手推荐", "2": "高手进阶", "3": "趣味娱乐"}

INGAME_HOME = "https://jcc.qq.com/ingame/"
INGAME_WAY_ID = "2"
INGAME_LINEUP = ("https://game.gtimg.cn/images/lol/act/jkzlkauto/json/lineupJson/"
                 "m{season_num}/{way}/{mode}/lineup_detail_total_new.json")
INGAME_SEASONS = {"18_S19": "19"}
INGAME_STAT_URL = "https://mlol.qt.qq.com/go/jgame/lineup_stat"
INGAME_STAT_BATCH = 50

# 同步用到的数据文件：versiondataconfig 里的键名 → 本地名
FILES = {"herourl": "chess", "traiturl": "trait", "equipurl": "equip", "hexurl": "hex",
         "adventureurl": "adventure", "galaxyurl": "galaxy", "missionurl": "mission"}


class OfficialError(RuntimeError):
    pass


class OfficialClient:
    def __init__(self, timeout_s: float = 30.0, interval_s: float = 0.2):
        self.interval_s = interval_s
        self._last = 0.0
        self._config: str | None = None
        self._ingame_way_id: str | None = None
        self._versions: list[dict] | None = None
        self.raw_responses: dict[str, bytes] = {}
        self.raw_posts: list[dict] = []
        self._http = httpx.Client(timeout=timeout_s, follow_redirects=True,
                                  headers={"User-Agent": "jccbrain/0.1 (+local desktop tool)",
                                           "Referer": "https://jcc.qq.com/"})

    def _get_text(self, url: str, fresh: bool = False) -> str:
        """fresh=True：会随更新变化的文件（版本索引、官网配置、阵容），带时间戳绕过 CDN 缓存。"""
        wait = self.interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"t": str(int(time.time()))} if fresh else None
        for attempt in range(3):
            try:
                r = self._http.get(url, params=params)
                self._last = time.monotonic()
                r.raise_for_status()
                self.raw_responses[url] = r.content
                return r.text
            except httpx.HTTPError as e:
                if attempt == 2:
                    raise OfficialError(f"{url} 请求失败：{e}") from e
                time.sleep(1.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def _get_json(self, url: str, fresh: bool = False) -> Any:
        try:
            return json.loads(self._get_text(url, fresh))
        except ValueError as e:
            raise OfficialError(f"{url} 不是 JSON：{e}") from e

    def _post_json(self, url: str, body: dict) -> Any:
        for attempt in range(3):
            wait = self.interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._http.post(url, json=body)
                self._last = time.monotonic()
                r.raise_for_status()
                self.raw_posts.append({"url": url, "request": body, "body": r.content})
                return r.json()
            except (httpx.HTTPError, ValueError) as e:
                if attempt == 2:
                    raise OfficialError(f"{url} 请求失败或不是 JSON：{e}") from e
                time.sleep(1.0 * (attempt + 1))
        raise AssertionError("unreachable")

    # ---- 赛季与版本 ----
    def _basic_config(self) -> str:
        if self._config is None:
            self._config = self._get_text(BASIC_CONFIG, fresh=True)
        return self._config

    def season_tabs(self) -> list[Season]:
        """官网当前列出的赛季（basicConfig.js 的 season_tabs，跳过注释掉的）。"""
        text = self._basic_config()
        m = re.search(r"season_tabs\s*:\s*\[(.*?)\n\s*\]", text, re.S)
        if not m:
            raise OfficialError("官网配置里没有 season_tabs")
        out = []
        for line in m.group(1).splitlines():
            if line.lstrip().startswith(("//", "/*", "*")):
                continue
            g = re.search(r'mode:\s*"(\w+)",\s*season:\s*"(\w+)",.*?id:\s*"([^"]+)",\s*name:\s*"([^"]+)"', line)
            if g:
                mode, code, sid, name = g.groups()
                out.append(make_season(sid, name.replace("\\n", "·"), mode, code))
        if not out:
            raise OfficialError("官网配置的 season_tabs 解析不出赛季")
        return out

    def versions(self) -> list[dict]:
        if self._versions is None:
            self._versions = list(self._get_json(VERSION_INDEX, fresh=True) or [])
        return list(self._versions)

    def find_version(self, s: Season, version: str | None = None) -> dict:
        """该赛季的版本条目：指定版本号，或官方标记的最新版本（没有标记时取开始时间最晚的）。"""
        rows = [v for v in self.versions()
                if v.get("mode") == s.mode and v.get("season") == s.season and v.get("lang", "CN") == "CN"]
        if not rows:
            raise OfficialError(f"官方资料库里找不到「{s.label}」（mode {s.mode} / {s.season}）")
        if version:
            hit = [v for v in rows if v.get("version") == version]
            if not hit:
                raise OfficialError(f"{s.label} 没有版本 {version}，可选：{', '.join(v['version'] for v in rows)}")
            return hit[0]
        newest = [v for v in rows if str(v.get("is_newest_version")) == "1"]
        return newest[-1] if newest else max(rows, key=lambda v: v.get("version_start_time") or "")

    # ---- 数据文件 ----
    def gamedata(self, entry: dict) -> dict[str, list[dict]]:
        """一个版本的全部数据文件：本地名 → 记录列表。"""
        out: dict[str, list[dict]] = {}
        for key, name in FILES.items():
            path = entry.get(key)
            if not path:
                out[name] = []
                continue
            body = self._get_json(f"{GAMEDATA_BASE}{path}")
            data = body.get("data", body) if isinstance(body, dict) else body
            out[name] = list(data.values()) if isinstance(data, dict) else list(data or [])
        return out

    def lineup_url(self, entry: dict, season_id: str | None = None) -> str:
        """官方阵容地址。官网配置里的规则并不统一（有的赛季 m 后面的数字和赛季号对不上），优先按配置取。"""
        key = season_id or f"{entry['mode']}_{entry['season']}"
        try:
            text = self._basic_config()
            channel = (re.search(r"var\s+channel\s*=\s*'(\d+)'", text) or [None, LINEUP_CHANNEL])[1]
            m = re.search(rf"'{re.escape(key)}'\s*:\s*`([^`]+lineup_detail_total\.json)`", text)
            if m:
                return m.group(1).replace("${channel}", channel)
        except OfficialError as e:
            log.warning("官网配置读取失败，按默认规则拼阵容地址：%s", e)
        return LINEUP_FALLBACK.format(season_num=entry["season"].lstrip("S"), channel=LINEUP_CHANNEL,
                                      mode=entry["mode"])

    def lineup_tags(self) -> dict[str, str]:
        """阵容分类：line_tag → 名称（新手推荐 / 高手进阶 / 趣味娱乐）。"""
        try:
            body = self._get_json(LINEUP_TAGS_URL, fresh=True) or {}
            tags = {str(k): v.get("title") for k, v in body.items() if isinstance(v, dict) and v.get("title")}
            return tags or dict(LINEUP_TAGS)
        except OfficialError as e:
            log.warning("阵容分类获取失败，用内置分类：%s", e)
            return dict(LINEUP_TAGS)

    def newest_version(self, s: Season) -> str | None:
        """该赛季官方最新版本号（只读版本索引，不下载数据文件）。"""
        try:
            return self.find_version(s).get("version")
        except OfficialError:
            return None

    def lineups(self, entry: dict, season_id: str | None = None) -> list[dict]:
        body = self._get_json(self.lineup_url(entry, season_id), fresh=True)
        return list((body or {}).get("lineup_list") or [])

    def ingame_way_id(self) -> str:
        """微社区阵容渠道，首页配置不可用时缓存默认值。"""
        if self._ingame_way_id is None:
            try:
                text = self._get_text(INGAME_HOME, fresh=True)
                match = re.search(r"ZMProjectConfig\s*=\s*", text)
                if not match:
                    raise ValueError("微社区首页没有 ZMProjectConfig")
                config, _ = json.JSONDecoder().raw_decode(text, match.end())
                if not isinstance(config, dict):
                    raise ValueError("微社区 ZMProjectConfig 不是对象")
                way = config["lineup_way_id"]
                if way is None or not str(way).strip():
                    raise ValueError("微社区 lineup_way_id 为空")
                self._ingame_way_id = str(way)
            except (OfficialError, ValueError, KeyError, TypeError) as e:
                log.warning("微社区配置读取失败，使用默认阵容渠道 %s：%s", INGAME_WAY_ID, e)
                self._ingame_way_id = INGAME_WAY_ID
        return self._ingame_way_id

    def ingame_lineup_url(self, season_id: str) -> str | None:
        """已接入赛季的微社区阵容地址，其余赛季不请求微社区。"""
        season_num = INGAME_SEASONS.get(season_id)
        if season_num is None:
            return None
        return INGAME_LINEUP.format(season_num=season_num, way=self.ingame_way_id(),
                                    mode=season_id.split("_", 1)[0])

    def ingame_lineups(self, season_id: str) -> list[dict]:
        url = self.ingame_lineup_url(season_id)
        if url is None:
            return []
        body = self._get_json(url, fresh=True)
        return [normalize_lineup(ln) for ln in (body or {}).get("lineup_list") or []]

    def lineup_stats(self, lineup_ids) -> dict[str, dict]:
        """分批读取微社区热度；任一批失败时由调用方统一降级。"""
        ids = list(dict.fromkeys(str(i) for i in lineup_ids if str(i)))
        out = {}
        for start in range(0, len(ids), INGAME_STAT_BATCH):
            body = self._post_json(INGAME_STAT_URL, {"mode": 6, "lineup_ids": ids[start:start + INGAME_STAT_BATCH]})
            if not isinstance(body, dict):
                raise OfficialError("微社区热度响应不是对象")
            try:
                success = int(body.get("result")) == 0
            except (TypeError, ValueError, OverflowError):
                success = False
            if not success:
                raise OfficialError(f"微社区热度获取失败：{body.get('msg') or body.get('err_msg') or body}")
            rows = body.get("data") or []
            if not isinstance(rows, list):
                raise OfficialError("微社区热度 data 不是列表")
            for row in rows:
                if not isinstance(row, dict) or row.get("lineup_id") is None:
                    continue
                stat = {}
                for source, target in (("use", "use"), ("week_use_num", "week_use"),
                                       ("collect", "collect"), ("comment", "comment")):
                    try:
                        value = int(row.get(source))
                    except (TypeError, ValueError, OverflowError):
                        value = -1
                    stat[target] = value if value >= 0 else None
                stat["stat_date"] = str(row.get("stat_date") or "") or None
                if stat["stat_date"] is None and all(stat[key] == 0 for key in ("use", "collect", "comment")):
                    continue
                out[str(row["lineup_id"])] = stat
        return out

    def close(self) -> None:
        self._http.close()


# ------------------------------------------------------------------ 转换
def normalize_lineup(ln: dict) -> dict:
    """统一官网、微社区记录的详情和作者格式，不修改原始记录。"""
    out = dict(ln)
    if isinstance(out.get("detail"), dict):
        out["detail"] = json.dumps(out["detail"], ensure_ascii=False)
    author = out.get("lineupauthor_data")
    if isinstance(author, str):
        try:
            author = json.loads(author)
        except ValueError:
            author = None
    out["lineupauthor_data"] = author if isinstance(author, dict) else None
    return out


def merge_lineups(web: list[dict], ingame: list[dict]) -> list[dict]:
    """官网内容优先，按 id 合并官方渠道，保留两个渠道各自的排序。"""
    web = [normalize_lineup(ln) for ln in web]
    ingame = [normalize_lineup(ln) for ln in ingame]
    web_ids = {str(ln["id"]) for ln in web}
    ingame_by_id = {str(ln["id"]): ln for ln in ingame}
    out = web + [ln for ln in ingame if str(ln["id"]) not in web_ids]
    for ln in out:
        gid = str(ln["id"])
        ln["_channels"] = (["web"] if gid in web_ids else []) + (["ingame"] if gid in ingame_by_id else [])
        try:
            ln["_ingame_sort"] = int(ingame_by_id[gid].get("sortID"))
        except (KeyError, TypeError, ValueError, OverflowError):
            ln["_ingame_sort"] = None
    return out


def apply_stats(guides: list[dict], stats: dict[str, dict]) -> int:
    """把官方微社区热度写入已转换攻略，返回匹配条数。"""
    hit = 0
    for guide in guides:
        stat = stats.get(str(guide["id"]))
        if stat is None:
            continue
        for source, target in (("use", "use_count"), ("week_use", "week_use"), ("collect", "collect_count"),
                               ("comment", "comment_count"), ("stat_date", "stat_date")):
            guide[target] = stat.get(source)
        hit += 1
    return hit


def _rename(d: dict, mapping: dict[str, str]) -> dict:
    out = dict(d)
    for src, dst in mapping.items():
        if src in d:
            out[dst] = d[src]
    return out


def to_gamedata(off: dict[str, list[dict]], dataj: dict | None = None) -> dict:
    """官方数据文件 → dataj /gamedata 结构，交给 sync.convert_gamedata 转成表行。

    官方没有的内容从 dataj 补（dataj 拿不到时留空）：英雄 AD/AP 标签、搜索别名，
    以及自然仙灵的出现回合和分类（官方只有每一级的描述和价格）。
    """
    dj = dataj or {}
    dj_heroes = {str(h.get("id")): h for h in dj.get("hero", [])}
    heroes = []
    for h in off.get("chess", []):
        r = _rename(h, {"species": "raceId", "class": "jobId", "initHP": "initHp", "initMP": "initMp",
                        "maxMP": "maxMp"})
        r["heroType"] = int(h.get("heroType") or 0)
        r["isVisible"] = True                       # 官方只列游戏里有的单位
        r["tag"] = dj_heroes.get(str(h.get("id")), {}).get("tag")
        heroes.append(r)

    traits = [_rename(t, {"desc": "descText", "desc2": "desc2Text", "values": "valuesText"})
              for t in off.get("trait", [])]
    equips = [_rename(e, {"desc": "descText", "fetterID": "fetterId", "EffectType": "effectType"})
              for e in off.get("equip", [])]
    hexes = [_rename(x, {"desc": "descText", "is_legend": "isLegend", "hero_enhancement_type": "heroEnhancementType"})
             for x in off.get("hex", [])]
    for x in hexes:
        x["level"] = int(x["level"]) if str(x.get("level", "")).isdigit() else x.get("level")

    # 自然仙灵：官方每一级单独一条（同名），第一条是基础版，后面是升级版
    dj_adv = {a["name"]: a for a in dj.get("adventure", [])}
    grouped: dict[str, list[dict]] = {}
    for a in off.get("adventure", []):
        grouped.setdefault(a.get("title") or "", []).append(a)
    adventures = []
    for title, rows in grouped.items():
        if not title:
            continue
        base, ups = rows[0], rows[1:]
        ref = dj_adv.get(title, {})
        adventures.append({
            "id": str(base["adventureId"]), "name": title, "descText": base.get("desc") or "",
            "price": base.get("price"), "categoryLabel": ref.get("categoryLabel") or base.get("category") or None,
            "roundSpans": ref.get("roundSpans") or [],
            "upgrades": [{"desc": u.get("desc") or "", "price": u.get("price"), "type": "upgrade"} for u in ups],
            "logo": base.get("logo"),
        })

    # 别名指向的是 dataj 的名字；经由英雄 id 换成官方名字（官方改了名也能搜到）
    off_name = {str(h.get("id")): h.get("name") for h in off.get("chess", [])}
    dj_to_off = {h.get("name"): off_name[i] for i, h in dj_heroes.items() if i in off_name}
    aliases = [{**a, "name": dj_to_off.get(a.get("name"), a.get("name"))} if a.get("entityType") == "hero" else a
               for a in dj.get("searchAlias", [])]

    return {"hero": heroes, "trait": traits, "equip": equips, "hex": hexes, "adventure": adventures,
            "searchAlias": aliases}


def convert_encounters(galaxy: list[dict]) -> list[dict]:
    """开局奇遇（官方叫 galaxy）→ encounters 表行。"""
    return [{"id": str(g["id"]), "name": g.get("name") or "", "description": g.get("desc") or "",
             "picture": g.get("logo")} for g in galaxy if g.get("id")]


def convert_unlocks(missions: list[dict]) -> list[dict]:
    """官方 mission.js 的静态解锁条件，保留原文；不推断某一局是否已解锁或卡池张数。"""
    out = []
    for group in missions:
        hid = _base_id(str(group.get("heroid") or ""))
        if not hid:
            continue
        for m in group.get("mission", []):
            out.append({"hero_id": hid, "id": str(m["id"]), "description": m.get("desc") or "",
                        "task_tips": m.get("tasktips") or "", "condition_id": str(m.get("conditionId", "")),
                        "visible": int(str(m.get("show", 1)) == "1"), "raw": json.dumps(m, ensure_ascii=False)})
    return out


def _base_id(hid: str) -> str:
    """官方英雄 id 的第一位是星级，去掉得到基础 id（与 heroes 表一致）。"""
    hid = str(hid or "")
    return hid[1:] if len(hid) == 5 else hid


def _ids(s: str | None) -> list[str]:
    return [x for x in str(s or "").split(",") if x]


# 攻略里的几套站位：detail 字段 → phase
GUIDE_BOARDS = {"hero_location": "final", "y21_early_heros": "early", "y21_metaphase_heros": "mid"}


def convert_guides(lineups: list[dict], adventures: list[dict] | None = None,
                   tags: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """官方推荐阵容（官网 jcc.qq.com/#/lineup）→ guides 行 + guide_heroes 行（每套站位一组，phase = final/early/mid/L<人口>）。

    攻略里推荐的自然仙灵是逐级的官方 adventureId，这里用官方 adventure 列表换成名称再存。
    """
    adv_name = {str(a.get("adventureId")): a.get("title") for a in adventures or []}
    tags = tags or LINEUP_TAGS
    guides, heroes = [], []
    for ln in lineups:
        try:
            det = json.loads(ln.get("detail") or "{}")
        except ValueError:
            log.warning("官方阵容 %s 详情解析失败，跳过", ln.get("id"))
            continue
        gid = str(ln["id"])
        hexbuff = det.get("hexbuff") or {}
        carry_eq = det.get("carry_hero_equip_replace") or {}
        guides.append({
            "id": gid, "name": det.get("line_name") or "", "quality": ln.get("quality"),
            "sort": int(ln.get("sortID") or 0), "game_version": ln.get("simulator_edition"),
            "released_at": ln.get("rel_time"),
            "category": tags.get(str(det.get("line_tag") or ""), None),
            "author": (ln.get("lineupauthor_data") or {}).get("name"),
            "channels": json.dumps(ln.get("_channels") or ["web"]), "ingame_sort": ln.get("_ingame_sort"),
            "use_count": None, "week_use": None, "collect_count": None, "comment_count": None, "stat_date": None,
            "early_info": det.get("early_info") or "", "d_time": det.get("d_time") or "",
            "location_info": det.get("location_info") or "", "enemy_info": det.get("enemy_info") or "",
            "hex_info": det.get("hex_info") or "", "equipment_info": det.get("equipment_info") or "",
            "adventure_info": det.get("adventure_info") or "",
            "hexes": json.dumps({"recommend": _ids(hexbuff.get("recomm")), "replace": _ids(hexbuff.get("replace"))}),
            "equipment_order": json.dumps(_ids(det.get("equipment_order"))),
            "carry_items": json.dumps({"main": _ids(carry_eq.get("main")), "backup": _ids(carry_eq.get("backup"))}),
            "three_star": json.dumps([_base_id(h) for h in _ids(det.get("level_3_heros"))]),
            "hero_replace": json.dumps([{"hero_id": _base_id(r.get("hero_id")),
                                         "replace": [_base_id(x) for x in _ids(r.get("replace_heros"))]}
                                        for r in det.get("hero_replace") or []]),
            "adventures": json.dumps({str(a.get("stage_num")): [adv_name.get(str(x), str(x)) for x in a.get("adventure") or []]
                                      for a in det.get("adventure") or []}, ensure_ascii=False),
        })
        boards = [(phase, det.get(key)) for key, phase in GUIDE_BOARDS.items()]
        boards += [(f"L{lv}", units) for lv, units in (det.get("levelMap") or {}).items()]
        for phase, units in boards:
            for i, u in enumerate(units or []):
                hid = str(u.get("hero_id") or "")
                if not hid:
                    continue
                heroes.append({
                    "guide_id": gid, "phase": phase, "sort": i, "hero_id": _base_id(hid),
                    "star": int(hid[0]) if len(hid) == 5 and hid[0].isdigit() else None,
                    "unit_type": u.get("chess_type") or "hero", "position": u.get("location"),
                    "is_carry": int(bool(u.get("is_carry_hero"))), "items": json.dumps(_ids(u.get("equipment_id"))),
                })
    return guides, heroes
