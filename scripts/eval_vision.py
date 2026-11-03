"""视觉识别评测：对样本截图跑 pyeye，与参考答案逐字段比对，输出准确率和分阶段耗时。

    .venv/bin/python scripts/eval_vision.py                       # 默认样本目录 + 参考答案
    .venv/bin/python scripts/eval_vision.py --truth swift         # 改和 Swift 版 game-eye 输出比
    .venv/bin/python scripts/eval_vision.py --json out.json       # 结果另存一份（对比前后用）
    .venv/bin/python scripts/eval_vision.py --diff                # 打印每处不一致
    .venv/bin/python scripts/eval_vision.py --samples /path/to/track --timing-only --repeat 3
    .venv/bin/python scripts/eval_vision.py --warmup               # 单列启动预热及预热后的首帧
    .venv/bin/python scripts/eval_vision.py --db /tmp/jcc-vf-db.sqlite3 --units --gallery /tmp/jcc-vf-gallery

参考答案：
- truth（默认）  tests/fixtures/vision_truth.jsonl：Swift 版输出逐帧人工核对过、改正了 OCR 错误的版本
- swift          tests/fixtures/game_eye_states.jsonl：Swift 版原始输出

逐帧按文件名顺序用同一个 GameReader 识别（回合类型、单位跟踪依赖跨帧记忆，和实战一样）。
--units 保留完整目录帧序列，另按人工 unit_truth 评单位名；不改变既有字段的参考答案。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAIN_DIR = ROOT
DEFAULT_SAMPLES = MAIN_DIR / "game-eye" / "samples"
TRUTH = {"truth": ROOT / "tests" / "fixtures" / "vision_truth.jsonl",
         "swift": ROOT / "tests" / "fixtures" / "game_eye_states.jsonl"}

SCALARS = ["stage", "timer", "phase", "gold", "interest", "level", "xp", "odds", "xp_cost", "refresh_cost",
           "streak", "board_count", "hp", "placement", "right_panel", "players_stale", "units_stale"]
GROUPS = {
    "数字": ["stage", "timer", "gold", "level", "xp", "odds", "xp_cost", "refresh_cost", "board_count", "hp"],
    "商店": ["shop.open", "shop.name", "shop.cost", "shop.traits", "shop.item", "shop.upgrade"],
    "羁绊": ["traits.name", "traits.count"],
    "玩家": ["players.name", "players.hp"],
    "单位": ["bench.n", "board.n", "bench.star", "board.star"],
}


def fields(d: dict) -> dict[str, object]:
    """一帧状态 → {字段: 值}。列表类字段拆成逐元素的 字段[i]，统计时按元素算准确率。"""
    out: dict[str, object] = {}
    for k in SCALARS:
        if k in d:
            out[k] = d[k]
    shop = d.get("shop") or {}
    out["shop.open"] = bool(shop.get("open"))
    for i, s in enumerate(shop.get("slots") or []):
        s = s or {}
        out[f"shop.name[{i}]"] = s.get("name")
        if s.get("name"):
            out[f"shop.cost[{i}]"] = s.get("cost")
            out[f"shop.traits[{i}]"] = s.get("traits")
            out[f"shop.unsure[{i}]"] = bool(s.get("unsure"))
            out[f"shop.item[{i}]"] = bool(s.get("item"))
            out[f"shop.upgrade[{i}]"] = s.get("upgrade")
    for i, t in enumerate(d.get("traits") or []):
        out[f"traits.name[{i}]"] = t.get("name")
        out[f"traits.count[{i}]"] = (t.get("count"), t.get("active"), t.get("next"))
    for i, p in enumerate(d.get("players") or []):     # 按名字对齐：漏一行不会让后面全错
        who = "<self>" if p.get("self") else p.get("name")
        out[f"players.name[{who}]"] = True
        out[f"players.hp[{who}]"] = p.get("hp")
    for zone in ("bench", "board"):
        us = d.get(zone) or []
        out[f"{zone}.n"] = len(us)
        out[f"{zone}.star"] = sorted(u.get("star") or 0 for u in us)
    for i, a in enumerate(d.get("augments") or []):
        out[f"augments.name[{i}]"] = a.get("name")
        out[f"augments.refresh[{i}]"] = a.get("refresh")
    if d.get("legend"):
        out["legend"] = True
    return out


def base(key: str) -> str:
    return key.split("[", 1)[0]


def score(pred: dict, truth: dict, stats: dict, diffs: list, name: str) -> None:
    fp, ft = fields(pred), fields(truth)
    skip = set(truth.get("_ignore") or [])        # 画面上被挡住、人也看不清的字段不计分
    for k in sorted(set(fp) | set(ft)):
        if base(k) in skip:
            continue
        ok = fp.get(k) == ft.get(k)
        st = stats[base(k)]
        st[0] += ok
        st[1] += 1
        if not ok:
            diffs.append((name, k, ft.get(k), fp.get(k)))


def build(db_path: Path | None, gallery: Path, ocr: str):
    from jccbrain.vision.pyeye import build_reader

    db = None
    if db_path:
        from jccbrain.data.db import GameDB
        tmp = Path(tempfile.mkdtemp(prefix="jcc-eval-db-")) / "jcc.sqlite3"
        tmp.write_bytes(db_path.read_bytes())        # 复制一份，不碰原库
        db = GameDB(tmp)
    return build_reader(db, gallery, ocr)


def run(files: list[Path], args) -> tuple[dict[str, dict], dict[str, list[float]]]:
    from eval_units import copied_gallery
    from jccbrain.vision.eye.capture import load_image

    preds: dict[str, dict] = {}
    times: dict[str, list[float]] = defaultdict(list)
    start = time.perf_counter()
    with copied_gallery(args.gallery) as gallery:
        reader = build(args.db, gallery, args.ocr)
        times["init"].append((time.perf_counter() - start) * 1000)
        try:
            start = time.perf_counter()
            if args.warmup and (warmup := getattr(reader.ocr, "warmup", None)):
                warmup()
            times["warmup"].append((time.perf_counter() - start) * 1000)
            for f in files:
                t0 = time.perf_counter()
                img = load_image(f)
                t1 = time.perf_counter()
                d = reader.read(img)
                t2 = time.perf_counter()
                times["load"].append((t1 - t0) * 1000)
                for k, v in reader.last_ms.items():
                    times[k].append(v)
                times["read"].append((t2 - t1) * 1000)
                times["ocr_cells"].append(reader.last_ocr_cells)
                for counter in ('cell_cache_hits', 'cell_cache_checks', 'thumbnail_cache_hits', 'ocr_submitted',
                                'augment_desc_cache_hits'):
                    times[counter].append(getattr(reader, 'last_' + counter, 0))
                preds[f.name] = d
        finally:
            if pool := getattr(reader.ocr, "pool", None):
                pool.shutdown(wait=True)
    times["first"].append(times["total"][0])
    return preds, times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--truth", default="truth", help="truth / swift / jsonl 路径")
    ap.add_argument("--db", type=Path, default=None, help="用游戏数据库做词典（复制后只读）")
    ap.add_argument("--ocr", default="rapid", help="OCR 后端：rapid / vision")
    ap.add_argument("--repeat", type=int, default=1, help="重复跑几遍（看 OCR 随机性和稳定耗时）")
    ap.add_argument("--timing-only", action="store_true", help="目录内全部 jpg 只计时，不用同名参考答案评分")
    ap.add_argument("--warmup", action="store_true", help="识别前预热模型，单独报告预热耗时；默认仍包含冷首帧")
    ap.add_argument("--items", action="store_true", help="另跑全部装备参考帧的像素专项，分别报告有无、名字精度与覆盖率")
    ap.add_argument("--phase", action="store_true", help="另用phase_truth.jsonl评阶段、下一/当前回合类型，逐类报告")
    ap.add_argument("--phase-truth", type=Path, default=ROOT / "tests/fixtures/phase_truth.jsonl")
    ap.add_argument('--augments', action='store_true', help='另评已选强化符文整列表、名字精度与召回')
    ap.add_argument('--augment-truth', type=Path, default=ROOT / 'tests/fixtures/augment_truth.jsonl')
    ap.add_argument('--digits', action='store_true', help='另评人工数字真值，分模板来源帧/验证帧统计')
    ap.add_argument('--digit-truth', type=Path, default=ROOT / 'tests/fixtures/digit_truth.jsonl')
    ap.add_argument('--scout', action='store_true', help='侦察接口专项，按scout_truth读取混合来源帧并清缓存计时')
    ap.add_argument('--scout-truth', type=Path, default=ROOT / 'tests/fixtures/scout_truth.jsonl')
    ap.add_argument('--opponent', action='store_true', help='按人工敌方真值评分颜色候选（产品禁用时也评分候选）')
    ap.add_argument('--opponent-truth', type=Path, default=ROOT / 'tests/fixtures/opponent_truth.jsonl')
    ap.add_argument("--units", action="store_true", help="按目录全部帧顺序回放，另评人工单位名真值")
    ap.add_argument("--gallery", type=Path, help="复制指定图库到临时目录后回放，绝不写回输入图库")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None, help="把识别结果写成 jsonl")
    args = ap.parse_args()
    if args.opponent:
        if args.db is None:
            ap.error('--opponent 需要 --db 指定当前赛季卡池')
        from eval_opponent import run as run_opponent
        return run_opponent(args)
    if args.scout:
        if args.db is None:
            ap.error('--scout 需要 --db 指定当前赛季卡池')
        from eval_scout import run as run_scout
        return run_scout(args)
    if args.units and args.db is None:
        ap.error('--units 需要 --db 指定只读卡池来源')
    if args.gallery is not None and not args.gallery.is_dir():
        ap.error('--gallery 必须是现有图库目录')

    phase_predictions = {}
    augment_predictions = {}
    digit_predictions = {}
    digit_truth = {d['_file']: d for d in map(json.loads, args.digit_truth.read_text().splitlines())} if args.digits else {}
    augment_truth = {d['_file']: d for d in map(json.loads, args.augment_truth.read_text().splitlines())} if args.augments else {}
    phase_truth = {d["_file"]: d for d in map(json.loads, args.phase_truth.read_text().splitlines())} if args.phase else {}
    unit_runs = []
    if args.units:
        from eval_units import combine, evaluate, frame_key, gallery_summary, load_truth, pool_heroes, summary, validate_truth

        heroes = pool_heroes(args.db)
        unit_pool = {hero['name'] for hero in heroes}
        unit_truth = load_truth()
        validate_truth(unit_truth, unit_pool)

    truth_file = TRUTH.get(args.truth, Path(args.truth))
    truth = {} if args.timing_only else {
        d["_file"]: d for d in map(json.loads, truth_file.read_text(encoding="utf-8").splitlines()) if d}
    files = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in ((".jpg", ".jpeg", ".png") if args.phase or args.augments or args.digits else (".jpg",))
                   and (args.units or args.phase or args.augments or args.digits or args.timing_only or p.name in truth))
    if not files:
        print("没有可评测的样本", file=sys.stderr)
        return 1

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    times: dict[str, list[float]] = defaultdict(list)
    diffs: list = []
    runs = []
    for _ in range(args.repeat):
        preds, t = run(files, args)
        for k, v in t.items():
            times[k] += v
        runs.append({"mean": statistics.mean(t["total"]), "first": t["first"][0]})
        if not args.timing_only:
            for f in files:
                if f.name in truth:
                    score(preds[f.name], truth[f.name], stats, diffs, f.name)
        if args.phase:
            from eval_phase import frame_key as phase_frame_key
            phase_predictions.update({f"{_}:{phase_frame_key(f, DEFAULT_SAMPLES)}": preds[f.name] for f in files})
        if args.augments:
            from eval_phase import frame_key
            augment_predictions.update({f'{_}:{frame_key(f, DEFAULT_SAMPLES)}': preds[f.name] for f in files})
        if args.digits:
            from eval_phase import frame_key as digit_frame_key
            digit_predictions.update({f'{_}:{digit_frame_key(f, DEFAULT_SAMPLES)}': preds[f.name] for f in files})
        if args.units:
            unit_runs.append(evaluate({frame_key(f, DEFAULT_SAMPLES): preds[f.name] for f in files}, unit_truth, unit_pool))
        if args.dump:
            with args.dump.open("w", encoding="utf-8") as fh:
                for f in files:
                    fh.write(json.dumps({"_file": f.name, **preds[f.name]}, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"样本 {len(files)} 帧 × {args.repeat} 遍，参考：{'仅计时，无准确率' if args.timing_only else truth_file.name}\n")
    if args.units and not args.timing_only:
        print(f'既有字段评分 {sum(f.name in truth for f in files)} 帧；单位名另用 unit_truth.jsonl。\n')
    print(f"{'字段':<22}{'准确率':>8}{'对/总':>10}")
    total = [0, 0]
    for k in sorted(stats):
        ok, n = stats[k]
        total[0] += ok
        total[1] += n
        print(f"{k:<22}{ok / n:>8.1%}{f'{ok}/{n}':>10}")
    if total[1]:
        print(f"{'合计':<22}{total[0] / total[1]:>8.1%}{f'{total[0]}/{total[1]}':>10}")
    group_acc = {}
    for g, keys in GROUPS.items():
        ok = sum(stats[k][0] for k in keys if k in stats)
        n = sum(stats[k][1] for k in keys if k in stats)
        if n:
            group_acc[g] = ok / n
    print("\n分组：" + "  ".join(f"{g} {a:.1%}" for g, a in group_acc.items()))

    def ms(k: str) -> str:
        v = times.get(k) or [0]
        return f"{statistics.mean(v):7.1f} {statistics.median(v):7.1f} {max(v):7.1f}"
    print(f"\n{'耗时 ms':<10}{'平均':>8}{'中位':>8}{'最大':>8}")
    for k in ("init", "warmup", "first", "load", "ocr", "units", "total", "read"):
        print(f"{k:<10}{ms(k)}")
    print(f"OCR 格数 平均 {statistics.mean(times['ocr_cells']):.1f}")
    print("每轮 total 平均 ms：" + " / ".join(f"{r['mean']:.1f}" for r in runs))

    if args.diff:
        print("\n不一致（帧 字段 参考 → 识别）：")
        for name, k, want, got in diffs:
            print(f"  {name} {k}: {want!r} → {got!r}")
    item_report = None
    if args.items:
        from eval_items import evaluate
        item_report = evaluate(DEFAULT_SAMPLES, ROOT / "tests/fixtures/item_truth.jsonl", args.repeat)
        named = item_report['name_accuracy']
        print(f"\n装备像素专项：有无 {item_report['presence_accuracy']:.2%}，召回 {item_report['presence_recall']:.2%}，"
              f"名字精度 {named:.2%}" if named is not None else "\n装备像素专项：没有输出已知名字")
        print(f"装备名字覆盖率 {item_report['coverage']:.2%}，角标 "
              f"{item_report['counts']['badge_ok']}/{item_report['counts']['badge_total']}")
    units_report = None
    if args.units:
        units_report = combine(unit_runs)
        units_report['pool'] = {'season_id': '18_S19', 'heroes': heroes, 'size': len(heroes),
                                'by_cost': {cost: sum(hero['cost'] == cost for hero in heroes) for cost in range(1, 6)}}
        units_report['gallery'] = gallery_summary(args.gallery, unit_pool)
        print('\n' + summary(units_report))
    phase_report = None
    if args.phase:
        from eval_phase import evaluate as evaluate_phase
        repeated_truth = {key: phase_truth[key.split(':', 1)[1]] for key in phase_predictions
                          if key.split(':', 1)[1] in phase_truth}
        phase_report = evaluate_phase(phase_predictions, repeated_truth)
        print('\n阶段专项：' + json.dumps(phase_report, ensure_ascii=False))
    augment_report = None
    if args.augments:
        from eval_augments import evaluate as evaluate_augments
        selected = {f'{i}:{key}': row for i in range(args.repeat) for key, row in augment_truth.items()
                    if any(frame_key(f, DEFAULT_SAMPLES) == key for f in files)}
        augment_report = evaluate_augments(augment_predictions, selected)
        print('\n已选强化符文：' + json.dumps(augment_report, ensure_ascii=False))
    digit_report = None
    if args.digits:
        from eval_digits import evaluate as evaluate_digits
        selected = {key: digit_truth[key.split(':', 1)[1]] for key in digit_predictions
                    if key.split(':', 1)[1] in digit_truth}
        digit_report = evaluate_digits(digit_predictions, selected)
        print('\n数字专项：' + json.dumps(digit_report, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps({
            "timing_only": args.timing_only, "warmup": args.warmup, "frames": len(files), "runs": runs,
            "acc": {k: [ok, n] for k, (ok, n) in sorted(stats.items())},
            "total": total, "groups": group_acc,
            "ms": {k: {"mean": statistics.mean(v), "median": statistics.median(v), "max": max(v)}
                   for k, v in times.items() if v},
            "diffs": diffs, "items": item_report,
            **({'units_name': units_report} if units_report is not None else {}),
            **({'phase': phase_report} if phase_report is not None else {}),
            **({'digits': digit_report} if digit_report is not None else {}),
            **({'augments_taken': augment_report} if augment_report is not None else {})}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
