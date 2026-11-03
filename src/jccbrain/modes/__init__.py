"""工作模式。每个模式是一个在后台线程里跑的循环，通过 ModeSink 回调把结果交给 UI / 命令行。

- AssistantMode  辅助大脑：只看不动，实时给建议
- AutoRollMode   自动 D 牌：按目标卡自动刷新、买卡，到保留金币线或凑齐即停
- AutopilotMode  全自动托管：从开局到结束由 AI 决策并执行
"""
from jccbrain.modes.base import Mode, ModeRunner, ModeSink
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.modes.autopilot import AutopilotMode

__all__ = ["Mode", "ModeRunner", "ModeSink", "AssistantMode", "AutoRollMode", "RollTarget", "AutopilotMode"]
