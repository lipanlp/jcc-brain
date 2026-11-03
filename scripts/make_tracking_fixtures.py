"""CoreGraphics 解码 T4 实际复现帧，保存跟踪所用特征/直方图和六个原尺寸单位裁剪。

    .venv/bin/python scripts/make_tracking_fixtures.py [--samples DIR]

不保存整帧；三处 feature_source 记录修复前颜色缓存沿用的向量来源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jccbrain.vision.eye.capture import load_image  # noqa: E402
from jccbrain.vision.eye.units import Gallery, UnitTracker, feature_print, fp_distance  # noqa: E402

CLASSES = {
    "a": ["0021_0", "0025_0", "0026_0", "0027_0", "0051_0", "0076_0"],
    "b": ["0021_1"],
    "c": ["0021_5", "0051_4", "0055_4", "0056_2"],
    "d": ["0021_7"], "e": ["0021_4"], "f": ["0021_9"],
    "g": ["0021_10", "0055_5", "0056_5", "0076_5"],
}
FEATURE_SOURCE = {"0012_0": "0009_0", "0025_0": "0021_0", "0051_4": "0025_4"}
CROPS = {"0009_0", "0021_0", "0026_0", "0027_0", "0055_5", "0056_5"}
POSITIVE = [("0012_0", "0021_0"), ("0025_0", "0026_0"),
            ("0026_0", "0027_0"), ("0055_5", "0056_5")]


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=Path("game-eye/samples"))
    ap.add_argument("--output", type=Path, default=ROOT / "tests/fixtures")
    args = ap.parse_args()
    arrays, metadata, detected = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="jcc-tracking-fixtures-") as tmp:
        tracker = UnitTracker(Gallery(Path(tmp)))

        def get(key):
            stem, index = key.split("_")
            relative = f"{stem}.jpg" if stem in ("0009", "0012") else f"track/{stem}.jpg"
            if relative not in detected:
                image = load_image(args.samples / relative)
                detected[relative] = tracker.detect(image)[1]
            return relative, detected[relative][int(index)]

        keys = sorted({"0009_0", "0012_0"} | {k for group in CLASSES.values() for k in group})
        for key in keys:
            filename, unit = get(key)
            source = FEATURE_SOURCE.get(key, key)
            _, reference = get(source)
            fp = feature_print(reference.crop)
            assert fp is not None
            arrays[f"{key}_fp"], arrays[f"{key}_hist"] = fp, unit.hist
            if key in CROPS:
                arrays[f"{key}_crop"] = unit.crop
            metadata[key] = {"file": filename, "zone": unit.zone, "slot": unit.slot,
                             "x": unit.x, "y": unit.y, "star": unit.star,
                             "feature_source": source,
                             "crop_sha256": hashlib.sha256(unit.crop.tobytes()).hexdigest()}

    negatives = [fp_distance(arrays[f"{a}_fp"], arrays[f"{b}_fp"])
                 for ca, aa in CLASSES.items() for cb, bb in CLASSES.items() if ca < cb
                 for a in aa for b in bb]
    info = {"decoder": "CoreGraphics", "window": [1774, 1136], "samples": metadata,
            "classes": CLASSES, "positive": POSITIVE,
            "positive_distances": [fp_distance(arrays[f"{a}_fp"], arrays[f"{b}_fp"]) for a, b in POSITIVE],
            "negative_count": len(negatives), "negative_min": min(negatives)}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "tracking_units.npz", **arrays)
    (args.output / "tracking_units.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{len(keys)} 个特征、{len(CROPS)} 个原尺寸裁剪；{info['positive_distances']} / 异类最小 {min(negatives):.4f}")


if __name__ == "__main__":
    main()
