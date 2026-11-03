"""打包入口：双击 .app 时打开图形界面；带参数时等同于 jcc 命令行。"""
import sys

from jccbrain.cli import main

if __name__ == "__main__":
    # macOS 启动 .app 时可能带 -psn_xxx 参数，忽略
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    sys.exit(main(args or ["gui"]))
