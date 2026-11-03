"""分析 record_assistant.py 的录制结果：识别质量、建议闪烁、买卡建议、阵容推荐、大模型。

用法：.venv/bin/python scripts/analyze_session.py var/sessions/<时间> [--timeline]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--timeline", action="store_true", help="逐阶段打印建议变化")
    args = ap.parse_args()
    d = Path(args.dir)
    states = load(d / "states.jsonl")
    events = load(d / "events.jsonl")
    advices = [e for e in events if e["kind"] == "advice"]
    print(f"# 录制 {d.name}：{len(states)} 帧，{states[-1]['_t'] - states[0]['_t']:.0f} 秒，"
          f"建议推送 {len(advices)} 次")

    # ---------------- 识别质量
    ingame = [s for s in states if s.get("stage")]
    print(f"\n## 识别\n读到阶段的帧 {len(ingame)}/{len(states)}")
    for k in ("gold", "level", "hp", "xp"):
        miss = sum(1 for s in ingame if s.get(k) is None)
        print(f"  {k} 缺失 {miss}/{len(ingame)}")
    ph = Counter(s.get("phase") for s in states)
    print("  阶段分布 " + "，".join(f"{k}:{v}" for k, v in ph.most_common()))
    stages = []
    for s in ingame:
        if not stages or stages[-1] != s["stage"]:
            stages.append(s["stage"])
    print("  阶段序列 " + " → ".join(stages))
    # 阶段回跳（OCR 误读）
    back = [(a, b) for a, b in zip(stages, stages[1:]) if tuple(map(int, b.split("-"))) < tuple(map(int, a.split("-")))]
    if back:
        print(f"  阶段回跳 {len(back)} 次：{back[:10]}")
    unk = [sum(1 for u in s.get("board", []) + s.get("bench", []) if str(u.get("name", "")).startswith("?"))
           for s in ingame]
    if unk:
        print(f"  每帧未识别单位 平均 {statistics.mean(unk):.1f}，最多 {max(unk)}")
    ms = [s["_ms"]["capture"] + s["_ms"]["read"] for s in states if "_ms" in s]
    if ms:
        print(f"  单帧耗时 中位 {statistics.median(ms):.0f}ms，p90 {sorted(ms)[int(len(ms) * .9)]}ms")

    # ---------------- 闪烁
    print("\n## 建议稳定性")
    flips = Counter()
    prev = None
    for a in advices:
        if prev and prev["stage"] == a["stage"] and prev["phase"] == a["phase"]:
            if a["headline"] != prev["headline"]:
                flips["headline"] += 1
            if (a["comps"] or [{}])[0].get("name") != (prev["comps"] or [{}])[0].get("name"):
                flips["目标阵容"] += 1
            if [b["name"] for b in a["buys"]] != [b["name"] for b in prev["buys"]]:
                flips["买卡列表"] += 1
            if (a["econ"] or {}).get("action") != (prev["econ"] or {}).get("action"):
                flips["运营动作"] += 1
        prev = a
    for k, v in flips.items():
        print(f"  同一回合同一阶段内 {k} 变化 {v} 次")
    # 目标阵容在整局的切换
    tgt = [(a["stage"], (a["comps"] or [{}])[0].get("name")) for a in advices if a["stage"]]
    seq = []
    for st, n in tgt:
        if not seq or seq[-1][1] != n:
            seq.append((st, n))
    print(f"  整局目标阵容切换 {len(seq) - 1} 次：" + " → ".join(f"{n}@{st}" for st, n in seq[:30]))
    # AI 文字在屏幕上停留的时间
    shown = []
    on = None
    for a in advices:
        if a["llm_text"] and on is None:
            on = a["_t"]
        elif not a["llm_text"] and on is not None:
            shown.append(a["_t"] - on)
            on = None
    if shown:
        print(f"  AI 文字显示 {len(shown)} 次，停留 中位 {statistics.median(shown):.2f}s，最长 {max(shown):.2f}s")

    # ---------------- 买卡
    print("\n## 买卡建议")
    reasons = Counter()
    per_stage = defaultdict(set)
    for a in advices:
        for b in a["buys"]:
            reasons[b["reason"].split("，")[0] if "阵容" not in b["reason"] else "阵容卡"] += 1
            per_stage[a["stage"]].add(b["name"])
    for k, v in reasons.most_common():
        print(f"  {k}：{v} 次推送")
    total = sum(len(v) for v in per_stage.values())
    print(f"  每回合平均推荐不同卡 {total / max(len(per_stage), 1):.1f} 张")

    # ---------------- 大模型
    req = [e for e in events if e["kind"] == "llm_request"]
    rep = [e for e in events if e["kind"] == "llm_reply"]
    err = [e for e in events if e["kind"] == "llm_error"]
    print(f"\n## 大模型\n请求 {len(req)}，成功 {len(rep)}，失败 {len(err)}")
    if rep:
        lat = [r["latency_s"] for r in rep]
        print(f"  耗时 中位 {statistics.median(lat):.1f}s，最长 {max(lat):.1f}s")
        bad = [r for r in req if not r["stage"] or r["phase"] == "unknown"]
        print(f"  在读不到阶段/unknown 时请求 {len(bad)} 次")
        sw = [r for r in rep if r["switch_comp"]]
        print(f"  建议转阵容 {len(sw)} 次：" + "、".join(f"{r['switch_comp']}@{r['stage']}" for r in sw[:10]))

    if args.timeline:
        print("\n## 时间线（每回合每阶段第一条 + AI）")
        last = None
        for e in events:
            if e["kind"] == "advice" and (e["stage"], e["phase"]) != last:
                last = (e["stage"], e["phase"])
                print(f"[{e['_t']:6.0f}s] " + " | ".join(e["lines"][:4]))
            elif e["kind"] == "llm_reply":
                print(f"[{e['_t']:6.0f}s]   AI({e['latency_s']}s) {e['headline']}：{'；'.join(e['actions'])}")


if __name__ == "__main__":
    main()
