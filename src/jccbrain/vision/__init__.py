"""视觉感知模块：把游戏画面变成 GameState。

所有后端都实现 VisionProvider 接口，上层（决策、模式、UI）只依赖这个接口：
- PyEyeProvider         纯 Python（默认）：Quartz 截图 + Apple Vision OCR + 血条/单位识别，见 vision/eye/
- ImageReplayProvider   用 PyEye 逐张识别一个截图目录（离线调试）
- ReplayProvider        回放 jsonl 文件，用于离线调试和测试
"""
from jccbrain.vision.base import VisionError, VisionProvider, create_provider

__all__ = ["VisionError", "VisionProvider", "create_provider"]
