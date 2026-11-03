"""项目内的固定路径。运行时数据统一放在 var/（已 gitignore），打包成 .app 后改放到用户目录。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent          # src/jccbrain
PROJECT_DIR = PACKAGE_DIR.parent.parent                        # 仓库根目录
RULES_DIR = PACKAGE_DIR / "data" / "rules"


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def var_dir() -> Path:
    """数据库、日志、用户设置的存放目录。可用环境变量 JCC_HOME 覆盖。"""
    if env := os.environ.get("JCC_HOME"):
        p = Path(env)
    elif _frozen():
        p = Path.home() / "Library" / "Application Support" / "金铲铲大脑"
    else:
        p = PROJECT_DIR / "var"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return var_dir() / "jcc.sqlite3"


def settings_path() -> Path:
    return var_dir() / "settings.json"


def log_dir() -> Path:
    p = var_dir() / "logs"
    p.mkdir(exist_ok=True)
    return p


def legacy_gallery_dir() -> Path | None:
    """旧 game-eye（Swift 版，已弃用）的单位图库，首次运行时导入新图库。打包后从 .app 内读取。"""
    p = Path(getattr(sys, "_MEIPASS", "")) / "game-eye-gallery" if _frozen() else PROJECT_DIR / "game-eye" / "gallery"
    return p if p.is_dir() else None
