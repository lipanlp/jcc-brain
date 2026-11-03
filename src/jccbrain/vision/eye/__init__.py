"""纯 Python 视觉识别（由 game-eye Swift 版移植，阈值与区域坐标沿用其实测值）。

- regions.json  区域坐标（相对窗口 0~1），来自 game-eye/config.json
- capture       Quartz 截取游戏窗口（被遮挡也能截）
- image         numpy/OpenCV 图像工具、颜色判断
- ocr           Apple Vision 文字识别：小区域放大补边、竖向拼图批量识别
- reader        一帧 → 结构化状态（与 game-eye 输出字段一致）
- units         血条定位、星级、单位识别（Vision 特征向量 + 图库）与跨帧跟踪
"""
