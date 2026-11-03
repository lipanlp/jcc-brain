"""金铲铲大脑：金铲铲之战 AI 分析决策与自动化工具。

模块划分（详见 docs/ARCHITECTURE.md）：
- core      配置、路径、对局状态数据模型、赛季规则
- data      游戏数据库（SQLite）与 dataj.cc 数据同步
- vision    视觉感知：截图 + Apple Vision OCR + 单位识别（纯 Python），以及回放
- engine    AI 决策：牌库概率、阵容匹配、运营、建议生成、大模型
- actuator  自动操作：窗口定位、坐标布局、鼠标键盘事件、安全保护
- modes     三种工作模式：辅助大脑 / 自动 D 牌 / 全自动托管
- ui        Qt 主窗口、菜单与 HUD 浮层
"""

__version__ = "0.1.0"
