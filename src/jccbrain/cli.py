"""命令行入口：`jcc <命令>`。不带参数时打开图形界面。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from jccbrain import __version__


def setup_logging(verbose: bool = False) -> None:
    from logging.handlers import RotatingFileHandler

    from jccbrain.core import paths

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = RotatingFileHandler(paths.log_dir() / "jcc.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(sh)


def _ctx(args):
    from jccbrain.context import AppContext
    from jccbrain.core.settings import Settings

    s = Settings.load()
    if getattr(args, "replay", None):
        s.vision.backend = "replay"
        s.vision.replay_file = args.replay
    if getattr(args, "live", False):
        s.actuator.dry_run = False
    return AppContext(s)


# ---------------------------------------------------------------- 数据
def _source_names(value: str) -> frozenset[str]:
    picked = frozenset(x.strip() for x in value.split(',') if x.strip())
    unknown = picked - {'official', 'ingame', 'dataj'}
    if unknown:
        raise argparse.ArgumentTypeError('未知的数据源：' + '、'.join(sorted(unknown)))
    if not picked:
        raise argparse.ArgumentTypeError('至少要选一个数据源：official,ingame,dataj')
    return picked


def cmd_db_sync(args) -> int:
    ctx = _ctx(args)
    d = ctx.settings.data
    if args.sources is not None:
        picked = args.sources
        d.source_official, d.source_ingame, d.source_dataj = (source in picked for source in ('official', 'ingame', 'dataj'))
    if args.official_version:
        d.official_version = args.official_version
    if args.version:
        d.game_version = args.version
    if args.all:
        ctx.sync_all_seasons(details=not args.quick, progress=print, only_needed=not args.force)
    else:
        ctx.sync(details=not args.quick, progress=print, season_id=args.season)
    return 0


def cmd_db_season(args) -> int:
    ctx = _ctx(args)
    catalog = ctx.db.catalog()
    if args.season_id is not None:
        hit = next((s for s in catalog if args.season_id in (s.id, s.name)), None)
        if not hit:
            print(f"没有这个赛季：{args.season_id}。可选：{'、'.join(f'{s.id}（{s.name}）' for s in catalog)}")
            return 1
        ctx.switch_season(hit.id)
        st = ctx.db.stats()
        hint = f"版本 {st['game_version']}" if not ctx.db.is_empty else "还没同步，先运行 jcc db sync"
        print(f"已切换到 {hit.label}（{hint}）")
        return 0
    synced = ctx.db.synced_seasons()
    for s in catalog + [ctx.db.season] * (ctx.db.season_id not in {x.id for x in catalog}):
        info = synced.get(s.id)
        mark = "*" if s.id == ctx.settings.data.season_id else " "
        state = (f"版本 {info['game_version']}（{info['data_source']}），同步于 {info['synced_at']}" if info else "未同步")
        dataj = "" if s.dataj_set else "  [dataj 无数据]"
        print(f"{mark} {s.id:<16} {s.label:<12} {state}{dataj}")
    return 0


def cmd_db_stats(args) -> int:
    ctx = _ctx(args)
    stats = ctx.db.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["ingame_guides"] > 0:
        print(f"微社区：阵容 {stats['ingame_guides']} 套，带热度 {stats['guides_with_stats']} 套，"
              f"统计日期 {stats['guide_stat_date'] or '—'}")
    print("每个费用的英雄数：", ctx.db.pool_count_by_cost())
    return 0


def cmd_db_check(args) -> int:
    """只读官方目录与版本索引，不拉英雄/阵容等数据文件。"""
    from jccbrain.data.official import OfficialClient, OfficialError

    ctx = _ctx(args)
    official = OfficialClient()
    try:
        # 沿用应用既有检查流程；再读取各赛季最新值，避免把查询失败当作“已最新”。
        ctx.seasons_needing_sync()
        synced = ctx.db.synced_seasons()
        rows = []
        for s in ctx.db.catalog():
            local = (synced.get(s.id) or {}).get("game_version")
            try:
                newest = official.find_version(s).get("version")
                if not newest:
                    raise OfficialError("版本索引缺少版本号")
                state = "需要同步" if newest != local else "已是最新"
            except OfficialError as e:
                newest, state = None, f"无法确认：{e}"
            rows.append({"season_id": s.id, "season": s.label, "local": local, "latest": newest, "status": state})
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(f"{row['season_id']} {row['season']}：本地 {row['local'] or '未同步'} / "
                      f"官方 {row['latest'] or '未知'}，{row['status']}")
        return 1 if any(row["latest"] is None for row in rows) else 0
    finally:
        official.close()
        ctx.close()


def cmd_db_diff(args) -> int:
    from jccbrain.data.archive import ArchiveError, diff_versions

    ctx = _ctx(args)
    try:
        sid = args.season or ctx.db.season_id
        result = diff_versions(ctx.db.path.parent / "raw", sid, args.old_version, args.new_version)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{sid} {args.old_version} → {args.new_version}：{result['summary']}")
            labels = {"buff": "加强", "nerf": "削弱", "changed": "变化", "added": "新增", "removed": "移除"}
            for c in result["changes"]:
                star = f" {c['star']}星" if "star" in c else ""
                print(f"[{labels[c['direction']]}] {c['name']}{star} {c['field']}：{c['old']} → {c['new']}")
            print(result["note"])
        return 0
    except ArchiveError as e:
        print(f"无法比较：{e}。请先用 jcc db sync --season <赛季> --official-version <版本> 保存相应存档。")
        return 1
    finally:
        ctx.close()


def cmd_comps(args) -> int:
    ctx = _ctx(args)
    for c in ctx.enabled_comps(args.source)[: args.top]:
        core = " ".join(("★" if h.is_carry else "") + h.name for h in c.heroes if h.is_core or h.is_carry)
        channel = ' / '.join(label for key, label in (('web', '官网推荐'), ('ingame', '微社区')) if key in c.channels)
        stats = (f"均名 {c.avg_placement!s:<5} 前四 {c.top4_rate}% 登顶 {c.top_rate}%" if c.source == "dataj"
                 else f"{channel}{('·' + c.category) if c.category else ''}")
        print(f"[{c.tier}] {c.name:<10} {stats} 打法 {c.style:<6} 核心：{core}  (id={c.id})")
    return 0


def cmd_hero(args) -> int:
    ctx = _ctx(args)
    h = ctx.db.hero(args.name)
    if not h:
        print("没找到", args.name)
        return 1
    print(f"{h.name}  {h.cost} 费  羁绊：{'、'.join(h.traits)}  射程 {h.attack_range}")
    print(f"技能 {h.skill_name}：{h.skill_desc}")
    for t in h.traits:
        tr = ctx.db.trait(t)
        if tr:
            print(f"  {tr.name}（{tr.kind}）{'/'.join(map(str, tr.breakpoints))}：{' '.join(tr.level_desc)}")
    in_comps = [c.name for c in ctx.db.comps_ranked() if any(x.name == h.name for x in c.heroes)]
    print("出现在阵容：", "、".join(in_comps) or "无")
    return 0


def _parse_taken(items: list[str] | None) -> dict[str, int]:
    out = {}
    for it in items or []:
        name, _, n = it.partition("=")
        out[name] = int(n or 1)
    return out


def cmd_odds(args) -> int:
    from collections import Counter

    from jccbrain.engine.odds import PoolState, hit_odds

    ctx = _ctx(args)
    pool = PoolState.from_db(ctx.db, ctx.rules, Counter(_parse_taken(args.taken)))
    for name in args.names:
        o = hit_odds(pool, name, args.level, args.need)
        if not o:
            print(f"{name}：不在卡池里")
            continue
        print(f"{name}（{o.cost}费）剩 {o.remaining} 张｜单格 {o.slot_prob:.2%}｜单次刷新出现 {o.per_roll:.1%}"
              f"｜期望 {o.expected_per_roll:.2f} 张/次｜凑 {args.need} 张：50% 需 {o.rolls_for_50} 次"
              f"（{o.gold_for_50} 金），80% 需 {o.rolls_for_80} 次（{o.gold_for_80} 金）")
    return 0


def cmd_rolldown(args) -> int:
    from collections import Counter

    from jccbrain.engine.odds import PoolState, simulate_rolldown

    ctx = _ctx(args)
    pool = PoolState.from_db(ctx.db, ctx.rules, Counter(_parse_taken(args.taken)))
    targets = {}
    for t in args.targets:
        name, _, n = t.partition("=")
        targets[name] = int(n or 1)
    r = simulate_rolldown(pool, targets, args.gold, args.level, args.keep, sims=args.sims)
    print(f"{args.level} 级 {args.gold} 金 D 到 {args.keep} 金（模拟 {r.sims} 次）：")
    for n, p in r.hit_rate.items():
        print(f"  {n} 凑齐 {targets[n]} 张：{p:.1%}")
    print(f"  全部凑齐：{r.all_hit_rate:.1%}｜平均刷新 {r.avg_rolls:.1f} 次｜平均剩 {r.avg_gold_left:.1f} 金")
    return 0


# ---------------------------------------------------------------- 模式
class _PrintSink:
    def __init__(self, show_advice: bool):
        self.show_advice = show_advice
        self._last = None

    def on_state(self, state): ...

    def on_advice(self, advice):
        if not self.show_advice:
            return
        text = "\n".join(advice.lines())
        if text != self._last:
            self._last = text
            print("\n" + text, flush=True)

    def on_action(self, action, real):
        print(("▶ " if real else "▷ [演练] ") + action.describe(), flush=True)

    def on_log(self, text):
        print(text, flush=True)

    def on_finished(self, mode, reason):
        print(f"[{mode}] 结束：{reason}", flush=True)


def _run_mode(ctx, mode) -> int:
    from jccbrain.modes import ModeRunner

    runner = ModeRunner(ctx)
    problems = runner.start(mode)
    if problems:
        print("无法启动：\n  " + "\n  ".join(problems))
        return 1
    try:
        while runner.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n停止中…")
        runner.stop()
    ctx.close()
    return 0


def cmd_advise(args) -> int:
    from jccbrain.modes import AssistantMode

    ctx = _ctx(args)
    if args.once:
        st = ctx.vision.read()
        print("\n".join(ctx.advisor.advise(st, ctx.effective_lock_id()).lines()))
        ctx.close()
        return 0
    return _run_mode(ctx, AssistantMode(ctx, _PrintSink(True)))


def cmd_autoroll(args) -> int:
    from jccbrain.modes import AutoRollMode, RollTarget

    ctx = _ctx(args)
    if args.read_interval_ms is not None:
        ctx.settings.autoroll.read_interval_ms = args.read_interval_ms
    targets = None
    if args.target:
        targets = []
        for t in args.target:
            name, _, star = t.partition("=")
            targets.append(RollTarget(name, int(star or 2)))
    mode = AutoRollMode(ctx, _PrintSink(False), targets, args.keep, args.max_rolls)
    return _run_mode(ctx, mode)


def cmd_autopilot(args) -> int:
    from jccbrain.modes import AutopilotMode

    ctx = _ctx(args)
    return _run_mode(ctx, AutopilotMode(ctx, _PrintSink(False)))


# ---------------------------------------------------------------- 标定 / 设置
def cmd_calibrate(args) -> int:
    from jccbrain.actuator.window import WindowLocator, mouse_location

    ctx = _ctx(args)
    win = WindowLocator(ctx.settings.actuator.window_owner).find(0)
    if not win:
        print("找不到游戏窗口")
        return 1
    lay = ctx.layout
    points = {
        "board": [("r1c1", "棋盘最前排最左格"), ("r1c7", "棋盘最前排最右格"),
                  ("r4c1", "棋盘最后排最左格"), ("r4c7", "棋盘最后排最右格")],
        "buttons": [("refresh_button", "刷新按钮"), ("xp_button", "购买经验按钮"), ("lock_button", "锁定商店按钮")],
        "bench": [("bench0", "备战席第 1 格"), ("bench8", "备战席第 9 格")],
    }[args.what]
    print(f"游戏窗口 {win.width:.0f}×{win.height:.0f} @ ({win.x:.0f},{win.y:.0f})。"
          "把鼠标移到提示的位置中心后按回车（不要点击）。")
    got = {}
    for key, label in points:
        input(f"→ {label}：")
        rx, ry = win.to_relative(*mouse_location())
        got[key] = (round(rx, 4), round(ry, 4))
        print(f"   {key} = {got[key]}")
    if args.what == "board":
        lay.board_corners = got
        lay.board_calibrated = True
    elif args.what == "buttons":
        for k, v in got.items():
            setattr(lay, k, v)
    else:
        lay.bench_x0, lay.bench_y = got["bench0"]
        lay.bench_step = round((got["bench8"][0] - got["bench0"][0]) / 8, 4)
    lay.save()
    print("已保存到 var/layout.json")
    return 0


def cmd_config(args) -> int:
    from dataclasses import asdict

    from jccbrain.core.settings import Settings

    s = Settings.load()
    if args.action == "show":
        d = asdict(s)
        if d["llm"]["api_key"]:
            d["llm"]["api_key"] = d["llm"]["api_key"][:4] + "…"
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0
    section, _, key = args.key.partition(".")
    target = getattr(s, section) if key else s
    attr = key or section
    if not hasattr(target, attr):
        print("没有这个设置项：", args.key)
        return 1
    old = getattr(target, attr)
    value = args.value
    if isinstance(old, bool):
        value = value.lower() in ("1", "true", "yes", "on")
    elif isinstance(old, int):
        value = int(value)
    elif isinstance(old, float):
        value = float(value)
    setattr(target, attr, value)
    s.save()
    print(f"{args.key} = {value!r}")
    return 0


def cmd_vision(args) -> int:
    if args.action == "file":
        return _vision_files(args)
    ctx = _ctx(args)
    try:
        ctx.vision.start()
        if args.action == "status":
            print(json.dumps(ctx.vision.status(), ensure_ascii=False, indent=2)[:3000])
        else:
            st = ctx.vision.read()
            print(json.dumps(st.raw, ensure_ascii=False, indent=2))
    finally:
        ctx.close()
    return 0


def _vision_files(args) -> int:
    """逐张识别本地截图，每张一行 JSON（与 game-eye file 输出格式一致，用于回归对比）。"""
    import tempfile
    from pathlib import Path

    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader

    ctx = _ctx(args)
    reader = build_reader(ctx.db, Path(tempfile.mkdtemp(prefix="jcc-gallery-")), ctx.settings.vision.ocr)
    for f in args.files:
        t0 = time.perf_counter()
        d = reader.read(load_image(f), debug=args.debug)
        d["_ms"] = {"read": int((time.perf_counter() - t0) * 1000), "ocr_cells": reader.last_ocr_cells}
        d["_file"] = Path(f).name
        print(json.dumps(d, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def cmd_llm(args) -> int:
    """test：测试连接；ask：读一帧（或回放第一帧）后，打印规则建议和大模型建议。"""
    from jccbrain.engine.llm import LLMClient

    ctx = _ctx(args)
    cfg = ctx.settings.llm
    if not (cfg.api_key or __import__("os").environ.get("JCC_LLM_API_KEY")) or not cfg.model:
        print("还没配置大模型：jcc config set llm.api_key <KEY>；jcc config set llm.model <模型名>")
        return 1
    client = LLMClient(cfg)
    if args.action == "preset":
        from jccbrain.engine.llm import PRESETS, apply_preset

        if args.name not in PRESETS:
            for p in PRESETS.values():
                print(f"{p.key:9} {p.title:8} {p.model:15} 平时 {p.effort:4} 海克斯 {p.effort_augment:4} {p.note}")
            return 0
        p = apply_preset(cfg, args.name)
        ctx.settings.save()
        print(f"已切换到「{p.title}」：{p.model}，平时 {p.effort}，海克斯 {p.effort_augment}")
        return 0
    if args.action == "test":
        print(f"{cfg.provider} {cfg.base_url} {cfg.model}（思考 {cfg.reasoning_effort or '默认'}）→ {client.ping()}")
        return 0
    st = ctx.vision.read()
    adv = ctx.advisor.advise(st, ctx.effective_lock_id(), ctx.extra_taken)
    print("\n".join(adv.lines()))
    reply = client.advise(st, adv)
    print(f"\nAI（{reply.latency_s:.1f}s）：{reply.headline}")
    for a in reply.actions:
        print("  ·", a)
    if reply.reason:
        print("  理由：", reply.reason)
    if reply.switch_comp and not adv.locked:
        print("  建议转：", reply.switch_comp)
    ctx.close()
    return 0


def cmd_gui(args) -> int:
    from jccbrain.webui.app import run_webui

    return run_webui()


# ---------------------------------------------------------------- 参数
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jcc", description="金铲铲大脑：金铲铲之战 AI 分析决策工具")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("gui", help="打开图形界面（默认）").set_defaults(func=cmd_gui)

    db = sub.add_parser("db", help="游戏数据库").add_subparsers(dest="db_cmd", required=True)
    s = db.add_parser("sync", help="同步最新游戏资料和阵容")
    s.add_argument("--quick", action="store_true", help="只同步基础数据与阵容排行，不拉每个阵容的详情")
    s.add_argument("--version", help="dataj 统计版本，如 18.4（默认最新）")
    s.add_argument("--season", help="指定赛季（官方 id，如 16_S19；默认当前赛季）")
    s.add_argument("--all", action="store_true", help="同步官网列出的全部赛季（默认只同步没同步过或官方有新版本的）")
    s.add_argument("--force", action="store_true", help="和 --all 一起用：不管版本，全部重新同步")
    s.add_argument("--sources", type=_source_names,
                   help="本次来源，逗号分隔：official（官网推荐）,ingame（微社区）,dataj（统计）；显式覆盖全部三项。官网/微社区共用完整官方同步，默认按设置")
    s.add_argument("--official-version", help="官方资料库版本，如 18.18.5（默认最新）")
    s.set_defaults(func=cmd_db_sync)
    db.add_parser("stats", help="数据库统计").set_defaults(func=cmd_db_stats)
    s = db.add_parser("check", help="检查各赛季官方最新版本（不下载数据文件）")
    s.add_argument("--json", action="store_true", help="输出 JSON")
    s.set_defaults(func=cmd_db_check)
    s = db.add_parser("diff", help="离线比较两个已存档官方版本")
    s.add_argument("old_version")
    s.add_argument("new_version")
    s.add_argument("--season", help="官方赛季 id（默认当前赛季）")
    s.add_argument("--json", action="store_true", help="输出 JSON")
    s.set_defaults(func=cmd_db_diff)
    s = db.add_parser("season", help="列出赛季；带 id 或名称则切换当前赛季：jcc db season 18_S19")
    s.add_argument("season_id", nargs="?")
    s.set_defaults(func=cmd_db_season)

    s = sub.add_parser("comps", help="推荐阵容排行")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--source", default="all", choices=["all", "official", "ingame", "dataj"],
                   help="all 为关联去重排行；official=官网推荐，ingame=微社区原始攻略，dataj=统计；均遵循设置开关")
    s.set_defaults(func=cmd_comps)

    s = sub.add_parser("hero", help="查看英雄")
    s.add_argument("name")
    s.set_defaults(func=cmd_hero)

    s = sub.add_parser("odds", help="刷新概率：jcc odds 绯红树怪 --level 5 --need 3")
    s.add_argument("names", nargs="+")
    s.add_argument("--level", type=int, required=True)
    s.add_argument("--need", type=int, default=1)
    s.add_argument("--taken", nargs="*", help="已被拿走的张数：名字=张数")
    s.set_defaults(func=cmd_odds)

    s = sub.add_parser("rolldown", help="D 牌模拟：jcc rolldown 绯红树怪=6 苍蓝哨戒=6 --gold 60 --level 5")
    s.add_argument("targets", nargs="+", help="名字=还需要的张数")
    s.add_argument("--gold", type=int, required=True)
    s.add_argument("--level", type=int, required=True)
    s.add_argument("--keep", type=int, default=0)
    s.add_argument("--sims", type=int, default=3000)
    s.add_argument("--taken", nargs="*")
    s.set_defaults(func=cmd_rolldown)

    s = sub.add_parser("advise", help="辅助大脑：实时建议（命令行版）")
    s.add_argument("--replay", help="用回放代替实时画面：jsonl 状态文件或截图目录")
    s.add_argument("--once", action="store_true", help="只读一帧")
    s.set_defaults(func=cmd_advise)

    s = sub.add_parser("autoroll", help="自动 D 牌")
    s.add_argument("--target", nargs="*", help="目标：名字=星级，省略则取目标阵容")
    s.add_argument("--keep", type=int, help="保留金币")
    s.add_argument("--max-rolls", type=int)
    s.add_argument("--read-interval-ms", type=int, help="本次 D 牌读画面间隔 ms（不保存设置）")
    s.add_argument("--live", action="store_true", help="真实操作（否则演练）")
    s.add_argument("--replay")
    s.set_defaults(func=cmd_autoroll)

    s = sub.add_parser("autopilot", help="全自动托管")
    s.add_argument("--live", action="store_true", help="真实操作（否则演练）")
    s.add_argument("--replay")
    s.set_defaults(func=cmd_autopilot)

    s = sub.add_parser("calibrate", help="坐标标定（鼠标指位置 + 回车）")
    s.add_argument("what", choices=["board", "buttons", "bench"])
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("config", help="查看/修改设置：jcc config set llm.model deepseek-chat")
    s.add_argument("action", choices=["show", "set"])
    s.add_argument("key", nargs="?")
    s.add_argument("value", nargs="?")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("llm", help="大模型：test 测试连接 / ask 问一次 / preset 切换档位（fast/balanced/deep）")
    s.add_argument("action", choices=["test", "ask", "preset"])
    s.add_argument("name", nargs="?", help="preset 的档位名，省略则列出全部")
    s.add_argument("--replay", help="ask 时用回放代替实时画面")
    s.set_defaults(func=cmd_llm)

    s = sub.add_parser("vision", help="视觉模块：status / read / file 截图…")
    s.add_argument("action", choices=["status", "read", "file"])
    s.add_argument("files", nargs="*", help="file 模式：要识别的截图")
    s.add_argument("--debug", action="store_true", help="附带各区域原始 OCR 文字")
    s.add_argument("--replay")
    s.set_defaults(func=cmd_vision)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if not getattr(args, "func", None):
        return cmd_gui(args)
    if args.func is cmd_config and args.action == "set" and (not args.key or args.value is None):
        print("用法：jcc config set <分组.项> <值>")
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
