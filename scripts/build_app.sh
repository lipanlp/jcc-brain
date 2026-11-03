#!/bin/zsh
# 打包成独立运行的 macOS 应用：dist/金铲铲大脑.app
#   ./scripts/build_app.sh
# 视觉为纯 Python（vision/eye）；若本地还有旧 game-eye 的图库，会打包进去，首次运行导入到
# ~/Library/Application Support/金铲铲大脑/gallery。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[[ -x $PY ]] || { echo "先创建虚拟环境：uv venv --python 3.12 .venv && uv pip install -e '.[dev,build]'"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "打包需要 Node.js 和 npm，请先运行 brew install node"; exit 1; }
(cd web && npm ci && npm run build)
$PY -m PyInstaller --version >/dev/null 2>&1 || uv pip install --python $PY pyinstaller

EYE_ARGS=()
if [[ -d game-eye/gallery ]]; then
  EYE_ARGS=("--add-data=game-eye/gallery:game-eye-gallery")
fi

$PY -m PyInstaller --noconfirm --clean --windowed \
  --name "金铲铲大脑" \
  --osx-bundle-identifier cc.jccbrain.app \
  --add-data="src/jccbrain/webui/static:jccbrain/webui/static" \
  --add-data="src/jccbrain/data/schema.sql:jccbrain/data" \
  --add-data="src/jccbrain/data/rules:jccbrain/data/rules" \
  --add-data="src/jccbrain/vision/eye/regions.json:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/dict.json:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/item_catalog.json:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/extra_items.json:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/extra_icons.npz:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/digits.npz:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/phase_icons.npz:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/phase_scenes.npz:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/augment_ui.npz:jccbrain/vision/eye" \
  --add-data="src/jccbrain/vision/eye/unit_refs:jccbrain/vision/eye/unit_refs" \
  "${EYE_ARGS[@]}" \
  --collect-submodules jccbrain \
  --collect-all webview \
  --collect-all rapidocr_onnxruntime \
  --collect-binaries onnxruntime \
  scripts/app_entry.py

# 录屏权限（读取游戏画面）和辅助功能权限（自动操作）需要在首次运行时授予
/usr/libexec/PlistBuddy -c "Add :NSScreenCaptureUsageDescription string 读取游戏画面以识别对局状态" \
  "dist/金铲铲大脑.app/Contents/Info.plist" 2>/dev/null || true

echo "完成：dist/金铲铲大脑.app"
