"""录制一局「辅助大脑」：跑真实的 AssistantMode，把每帧画面、识别状态、推给界面的建议和大模型往返全部落盘。

用法：
    .venv/bin/python scripts/record_assistant.py [--minutes 60] [--no-llm]
停止：Ctrl+C，或 touch 输出目录下的 STOP 文件。
输出：var/sessions/<时间>/
    frames/00001.jpg …   原始画面（之后可用 --replay 该目录离线复现）
    states.jsonl         每帧识别结果（含 _file、_t）
    events.jsonl         advice / log / llm 事件（含 _t，按时间顺序）
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import cv2

from jccbrain.context import AppContext
from jccbrain.cli import setup_logging
from jccbrain.core import paths
from jccbrain.engine.llm import build_context
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.base import ModeSink
from jccbrain.vision.base import VisionError


def _jsonable(o):
    if is_dataclass(o):
        return {k: _jsonable(v) for k, v in asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "value") and not isinstance(o, (int, float, str)):
        return o.value
    return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)


class RecordingSink(ModeSink):
    def __init__(self, out: Path, ctx: AppContext):
        self.out = out
        self.ctx = ctx
        self.t0 = time.monotonic()
        self.n = 0
        self.lock = threading.Lock()
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self.states = (out / "states.jsonl").open("a", encoding="utf-8")
        self.events = (out / "events.jsonl").open("a", encoding="utf-8")

    def t(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def event(self, kind: str, **kw) -> None:
        with self.lock:
            self.events.write(json.dumps({"_t": self.t(), "kind": kind, **kw}, ensure_ascii=False) + "\n")
            self.events.flush()

    def on_state(self, state) -> None:
        self.n += 1
        name = f"{self.n:05d}.jpg"
        img = getattr(self.ctx.vision, "_last_img", None)
        if img is not None:
            cv2.imwrite(str(self.out / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        d = dict(state.raw)
        d.update(_file=name, _t=self.t())
        with self.lock:
            self.states.write(json.dumps(_jsonable(d), ensure_ascii=False) + "\n")
            self.states.flush()
        if self.n % 20 == 0:
            print(f"[{self.t():7.1f}s] 帧 {self.n} {state.stage} {state.phase.value} 金{state.gold} Lv{state.level}", flush=True)

    def on_advice(self, adv) -> None:
        self.event("advice", frame=self.n, adv_id=id(adv), stage=adv.stage, phase=adv.phase.value,
                   headline=adv.headline, lines=adv.lines(), llm_text=adv.llm_text,
                   comp_status=adv.comp_status, switch_reason=adv.switch_reason,
                   buys=[_jsonable(b) for b in adv.buys], sells=[_jsonable(s) for s in adv.sells],
                   econ=_jsonable(adv.econ) if adv.econ else None,
                   comps=[{"name": m.name, "score": round(m.score, 3), "fit": round(m.fit, 3),
                           "id": m.comp.id, "seen": m.seen, "seen_carry": m.seen_carry,
                           "owned_copies": m.owned_copies,
                           "missing": [h.name for h in m.missing[:6]]} for m in adv.comps])

    def on_log(self, text: str) -> None:
        self.event("log", frame=self.n, text=text)
        print(f"[{self.t():7.1f}s] {text}", flush=True)

    def on_finished(self, mode: str, reason: str) -> None:
        self.event("finished", reason=reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--replay", help="离线复现：录制得到的 states.jsonl（不循环，默认不等待、不开大模型）")
    ap.add_argument("--interval", type=float, default=0, help="--replay 时每帧间隔秒数（按真实节奏复测大模型时用）")
    ap.add_argument("--llm", action="store_true", help="--replay 时也调用大模型")
    ap.add_argument("--out", help="输出目录名（默认按时间）")
    ap.add_argument("--watch", help="本局关注的阵容 id，逗号分隔，最多 4 个")
    args = ap.parse_args()
    setup_logging()

    ctx = AppContext()
    if args.watch:
        ids = list(dict.fromkeys(cid.strip() for cid in args.watch.split(',') if cid.strip()))
        if len(ids) > 4 or any(ctx.enabled_comp(cid) is None for cid in ids):
            ap.error('--watch 需要最多 4 个当前赛季有效阵容 id')
        ctx.game_watch_ids = ids
    ctx.settings.vision.backend = "pyeye"
    if args.replay:
        from jccbrain.vision.replay import ReplayProvider
        ctx.__dict__["vision"] = ReplayProvider(args.replay, args.interval, loop=False)
        args.no_llm = args.no_llm or not args.llm
    if args.no_llm:
        ctx.settings.llm.enabled = False
    out = paths.var_dir() / "sessions" / (args.out or datetime.now().strftime("%Y%m%d-%H%M%S"))
    sink = RecordingSink(out, ctx)
    mode = AssistantMode(ctx, sink)

    # 包一层大模型调用，记录输入输出（不改模式本身的行为）
    llm = ctx.llm
    orig = llm.advise

    def advise(state, adv):
        ctx_json = build_context(state, adv)
        sink.event("llm_request", frame=sink.n, stage=state.stage, phase=state.phase.value, context=ctx_json)
        try:
            r = orig(state, adv)
        except Exception as e:
            sink.event("llm_error", stage=state.stage, error=str(e))
            raise
        sink.event("llm_reply", stage=state.stage, phase=state.phase.value, latency_s=round(r.latency_s, 2),
                   headline=r.headline, actions=r.actions, reason=r.reason, switch_comp=r.switch_comp, raw=r.raw)
        return r

    llm.advise = advise

    deadline = time.monotonic() + args.minutes * 60

    def watchdog():
        while not mode.stopped:
            if time.monotonic() > deadline or (out / "STOP").exists():
                mode.stop_event.set()
            time.sleep(0.5)

    signal.signal(signal.SIGINT, lambda *_: mode.stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: mode.stop_event.set())
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"录制到 {out}（LLM {'开' if ctx.llm.ready else '关'}）", flush=True)
    ctx.vision.start()
    try:
        reason = mode.loop()
    except VisionError as e:
        reason = str(e)
    sink.on_finished(mode.key, reason)
    time.sleep(1)
    print(f"结束：{reason}，共 {sink.n} 帧 → {out}", flush=True)


if __name__ == "__main__":
    main()
