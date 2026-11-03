"""经 CoreGraphics 从已标注截图裁出商店图标测试格，不修改参考答案。

    .venv/bin/python scripts/make_shop_fixtures.py [--samples DIR]

锁按钮只有未锁的真实样本；海克斯/结算遮挡时标签为无法判断，闭锁需另用合成测试。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.image import crop, load_regions  # noqa: E402
from jccbrain.vision.eye.shop import LOCK_REGION, upgrade_region  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=Path("game-eye/samples"))
    ap.add_argument("--output", type=Path, default=ROOT / "tests/fixtures")
    args = ap.parse_args()
    truth = [json.loads(line) for line in (ROOT / "tests/fixtures/vision_truth.jsonl").read_text("utf-8").splitlines()]
    arrays, frames = {}, {}
    regions = load_regions()
    for state in truth:
        filename = state["_file"]
        image = load_image(args.samples / filename)
        slots = (state.get("shop") or {}).get("slots") or []
        marker_keys, upgrades = [], []
        for i in range(5):
            key = f"{Path(filename).stem}_slot{i}"
            arrays[key] = crop(image, upgrade_region(regions["shop"].cell(i, 0)))
            marker_keys.append(key)
            upgrades.append((slots[i] or {}).get("upgrade") if i < len(slots) else None)
        key = f"{Path(filename).stem}_lock"
        arrays[key] = crop(image, LOCK_REGION)
        frames[filename] = {"window": [image.shape[1], image.shape[0]], "markers": marker_keys,
                            "upgrade": upgrades, "lock": key, "locked": False if slots else None}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "shop_cells.npz", **arrays)
    (args.output / "shop_cells.json").write_text(json.dumps(
        {"decoder": "CoreGraphics", "frames": frames}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(frames)} 帧，{len(arrays)} 个图标格 → {args.output}")


if __name__ == "__main__":
    main()
