"""从样本截图切数字模板 → src/jccbrain/vision/eye/digits.npz。

    .venv/bin/python scripts/make_digit_templates.py [--samples DIR] [--sheet out.png]

用 tests/fixtures/vision_truth.jsonl 的答案给切出来的字贴标签：一个区域切出的字数（去掉「-」和贴左右边的碎片）
和答案字数一致才收；每类保留互不相似的样本作模板。--sheet 输出全部样本字的拼图，人工检查有没有贴错。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.digits import GLYPH_H, GLYPH_W, TEMPLATE_FILE, normalize, normalize_float, segment  # noqa: E402
from jccbrain.vision.eye.image import crop, load_regions  # noqa: E402

REGIONS = ["stage", "timer", "gold", "level", "xp", "streak", "xp_cost", "refresh_cost", "odds"]


def expected(d: dict, k: str) -> str | None:
    v = d.get(k)
    if v is None:
        return None
    if k == "xp":
        return "%d/%d" % tuple(v)
    if k == "odds":
        return "".join(f"{x}%" for x in v)
    if k == "streak":
        return str(v["n"])
    return str(v)


def canvas(mask: np.ndarray) -> np.ndarray:
    v = normalize(mask)
    return (v.reshape(GLYPH_H, GLYPH_W) > 0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=Path("game-eye/samples"))
    ap.add_argument("--sheet", type=Path, default=None)
    args = ap.parse_args()
    R = load_regions()
    truth = [json.loads(l) for l in (ROOT / "tests/fixtures/vision_truth.jsonl").read_text(encoding="utf-8").splitlines()]
    got: dict[str, list[np.ndarray]] = defaultdict(list)
    for d in truth:
        img = load_image(args.samples / d["_file"])
        for k in REGIONS:
            e = expected(d, k)
            if not e:
                continue
            c = crop(img, R[k])
            gs = [g for g in segment(c, img.shape[0]) if not (g.h < 0.45 * 15 and g.w > g.h)]
            chars = e.replace("-", "")
            # 多出来的先去掉贴左边的（金币图标边），再去贴右边的（头像碎片）
            if len(gs) > len(chars) and gs[0].x0 == 0:
                gs = gs[1:]
            if len(gs) > len(chars) and gs[-1].x1 == c.shape[1]:
                gs = gs[:-1]
            if len(gs) != len(chars):
                print("跳过", d["_file"], k, e, len(gs))
                continue
            for ch, g in zip(chars, gs):
                got[ch].append(canvas(g.mask))
    # 每类保留互不相似（相关 <0.95）的样本作模板：同一个字有几种写法（断笔的「%」、「1」有没有底横）
    chars, glyphs = [], []
    for c in sorted(got):
        kept: list[np.ndarray] = []
        for g in got[c]:
            v = normalize_float(g)
            if all(float(v @ k) < 0.95 for k in kept):
                kept.append(v)
                chars.append(c)
                glyphs.append(g)
    np.savez_compressed(TEMPLATE_FILE, chars=np.array("".join(chars)), glyphs=np.stack(glyphs).astype(np.uint8))
    print("模板", {c: f"{len(got[c])}→{chars.count(c)}" for c in sorted(got)}, "→", TEMPLATE_FILE)
    if args.sheet:
        rows = []
        for c in chars:
            row = np.hstack([np.pad(g, 1, constant_values=0.5) for g in got[c][:40]])
            rows.append(np.pad(row, ((0, 0), (0, 40 * (GLYPH_W + 2) - row.shape[1]))))
        cv2.imwrite(str(args.sheet), cv2.resize((np.vstack(rows) * 255).astype(np.uint8), None, fx=2, fy=2,
                                                interpolation=cv2.INTER_NEAREST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
