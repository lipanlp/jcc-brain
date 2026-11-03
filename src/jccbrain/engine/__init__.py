"""AI 决策引擎。

- odds      牌库剩余与刷新概率（精确 DP + 蒙特卡洛 D 牌模拟）
- comps     当前棋子 → 推荐阵容匹配
- economy   运营：存钱/升级/D 牌时机
- advisor   规则引擎，产出结构化建议 Advice（离线可用，也是大模型的输入）
- llm       大模型接入（OpenAI 兼容 / Anthropic），在规则建议之上做综合判断
- planner   把建议翻译成可执行的操作序列（全自动模式用）
"""
