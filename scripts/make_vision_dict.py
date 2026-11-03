"""从游戏数据库生成视觉 OCR 纠错词典 src/jccbrain/vision/eye/dict.json（没有数据库时的兜底）。

    .venv/bin/python scripts/make_vision_dict.py [--db var/jcc.sqlite3]

换赛季后先 `jcc db sync`，再跑一遍本脚本。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jccbrain.core import paths  # noqa: E402
from jccbrain.data.db import GameDB  # noqa: E402
from jccbrain.vision.eye.text import DICT_FILE, Dictionary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=paths.var_dir() / "jcc.sqlite3")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp()) / "jcc.sqlite3"
    shutil.copy(args.db, tmp)                 # 只读：复制一份再打开
    db = GameDB(tmp)
    if db.is_empty:
        print("数据库是空的，先 jcc db sync", file=sys.stderr)
        return 1
    d = Dictionary.from_db(db)
    DICT_FILE.write_text(json.dumps(d.to_json(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"英雄 {len(d.champions)}、羁绊 {len(d.traits)}、海克斯 {len(d.augments)} → {DICT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
