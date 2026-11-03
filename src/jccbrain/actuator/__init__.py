"""自动操作模块。

- actions   操作指令（纯数据）
- layout    界面坐标布局（相对游戏窗口 0~1），可标定
- window    定位游戏窗口在屏幕上的位置
- executor  执行器：MacExecutor（Quartz 鼠标事件）/ DryRunExecutor（只记录）
- safety    安全保护：急停、限速、鼠标甩角落立即停止
"""
