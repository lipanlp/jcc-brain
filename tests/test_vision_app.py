"""打包视觉回归：先 make app，再 make test；没有本地 .app 时跳过。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dist" / "金铲铲大脑.app" / "Contents"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.mark.skipif(sys.platform != "darwin", reason="需要 macOS .app")
def test_frozen_rapidocr_reads_real_frames(tmp_path):
    executable = APP / "MacOS" / "金铲铲大脑"
    if not executable.is_file():
        pytest.skip("先运行 make app 生成视觉打包产物")

    resources = APP / "Resources"
    assert (resources / "jccbrain/vision/eye/digits.npz").is_file()
    item_catalog = resources / "jccbrain/vision/eye/item_catalog.json"
    assert item_catalog.is_file()
    assert json.loads(item_catalog.read_text('utf-8'))['season_id'] == '18_S19'
    rapid = resources / "rapidocr_onnxruntime"
    assert (rapid / "config.yaml").is_file()
    for model in ("ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx"):
        assert (rapid / "models" / model).is_file()
    capi = APP / "Frameworks" / "onnxruntime" / "capi"
    assert (capi / "onnxruntime_pybind11_state.so").is_file()
    assert list(capi.glob("libonnxruntime*.dylib"))

    # 从仓库外运行，隔离个人设置/数据库；读取包内词典、模型及模板。
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    (runtime_home / "settings.json").write_text(
        json.dumps({"vision": {"ocr": "rapid"}}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["JCC_HOME"] = str(runtime_home)
    frames = sorted((FIXTURES / "frames").glob("*.jpg"))
    result = subprocess.run(
        [str(executable), "vision", "file", *map(str, frames)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    # create_engine 捕获 ImportError 会退回 Apple Vision；即使结果对，也不能算 RapidOCR 打包成功。
    logs = result.stderr + (runtime_home / "logs" / "jcc.log").read_text(encoding="utf-8")
    assert "OCR 改用 Apple Vision" not in logs, logs
    states = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [state["_file"] for state in states] == [frame.name for frame in frames]

    truth = {d["_file"]: d for d in map(json.loads,
             (FIXTURES / "vision_truth.jsonl").read_text(encoding="utf-8").splitlines())}
    # 回合类型、单位组名依赖此前完整帧序列；这里核对六张独立夹具的数字与商店字段。
    numeric = ("stage", "timer", "gold", "interest", "hp", "level", "xp", "odds",
               "xp_cost", "refresh_cost", "streak", "board_count", "placement")
    for state in states:
        want = truth[state["_file"]]
        for key in numeric:
            if key in want and key not in want.get("_ignore", []):
                assert state.get(key) == want[key], (state["_file"], key)
        assert state["shop"]["open"] == want["shop"]["open"]
        slots, expected = state["shop"].get("slots", []), want["shop"].get("slots", [])
        assert len(slots) == len(expected)
        for slot, target in zip(slots, expected):
            for key in ("name", "cost", "traits", "upgrade"):
                assert (slot or {}).get(key) == (target or {}).get(key), (state["_file"], key)
            assert bool((slot or {}).get("item")) == bool((target or {}).get("item"))
