# 架构设计

## 分层与依赖方向

```
            ┌────────────────────────────────────────────────────────────┐
  入口       │ cli.py（jcc 命令）  webui/（pywebview）+ web/（Vue 3 前端） │
            └───────────────┬────────────────────────────────────────────┘
                            │ 只通过 AppContext 拿到各模块
            ┌───────────────▼─────────────────────────────┐
  编排       │  context.py（依赖注入）   modes/（三种工作模式）  │
            └──┬──────────────┬──────────────┬────────────┘
               │              │              │
  能力    ┌────▼─────┐  ┌─────▼──────┐  ┌────▼──────┐
          │ vision/  │  │  engine/   │  │ actuator/ │
          │ 视觉感知  │  │  AI 决策    │  │ 自动操作   │
          └────┬─────┘  └─────┬──────┘  └────┬──────┘
               │              │              │
  基础    ┌────▼──────────────▼──────────────▼──────┐
          │ core/（状态模型、规则、设置、路径）  data/（数据库、同步） │
          └─────────────────────────────────────────┘
```

规则：
- 上层依赖下层，下层不 import 上层。`engine` 只依赖 `actuator.actions`（纯数据）与 `actuator.layout`（坐标），不依赖执行器。
- `vision` / `engine` / `actuator` 三个能力模块之间不互相创建对象，统一由 `AppContext` 组装注入，替换实现只改一处。
- UI 与 CLI 是平级的两个入口，所有业务逻辑都在它们下面，界面里只做展示和转发。

## 一次决策的数据流

```
游戏画面 ──game-eye──▶ JSON ──GameState.from_dict──▶ GameState
                                                     │
            GameDB（英雄/羁绊/阵容/海克斯统计）────────▶ HeuristicAdvisor ──▶ Advice
            SetRules（卡池/概率/经验/运营曲线）────────▶      │                 │
                                                     │        ├─▶ UI / HUD 展示
                                       LLMClient ◀───┘        ├─▶ AutoPlanner ──▶ [Action] ──▶ Executor ──▶ 鼠标事件
                                       （可选，异步补充意见）    └─▶ AutoRollMode（自动 D 牌）
```

## 模块说明

| 模块 | 文件 | 职责 | 扩展点 |
|---|---|---|---|
| core | `models.py` | `GameState` 等数据模型，字段对齐 game-eye 输出 | 视觉新增字段时在 `from_dict` 里补 |
| | `seasons.py` | 赛季表，以官方为准：id 为官方编号（`18_S19` = 自然之力），附 dataj setId 与规则表对应 | 官网新赛季会在同步时自动出现；要 dataj 统计/规则表时在 `SEASONS` 补对应 |
| | `rules.py` + `data/rules/set18.json` | 赛季规则表 | 新建 `set<N>.json` 并在赛季的 `rules_set` 指过去；没有时沿用自然之力的 |
| | `settings.py` | 分组设置，JSON 持久化 | 加字段即可，旧设置自动补默认值 |
| data | `db.py` / `schema.sql` | SQLite，每张游戏数据表带 `season_id` 按赛季区分；只把当前赛季载入内存，`use_season()` 切换 | 改表结构要递增 `SCHEMA_VERSION` 并补迁移 |
| | `official.py` | 金铲铲官方资料库（权威）：版本索引、英雄/羁绊/装备/海克斯/仙灵/奇遇、官方阵容攻略、官网赛季列表 | 官方新增数据文件在 `FILES` 里加 |
| | `dataj.py` / `sync.py` | dataj.cc 接口（阵容统计）；`sync_season()` 按勾选的数据源合并，官方为准 | 新数据源：转成同样的表行即可 |
| vision | `base.py` | `VisionProvider` 接口 | 新后端（比如 YOLO 识别）实现 `read()` 并在 `create_provider` 注册 |
| | `pyeye.py` + `eye/` | 纯 Python 识别（默认） | 新增识别字段：reader 里解析，`GameState.from_dict` 里接收 |
| | `replay.py` | 回放 jsonl 状态文件 | |
| engine | `odds.py` | 卡池、概率 DP、D 牌蒙特卡洛 | |
| | `comps.py` / `economy.py` / `advisor.py` | 阵容匹配、运营、规则建议 | 调规则直接改这里，单测覆盖 |
| | `stabilizer.py` | 辅助大脑用：沿用读不到的字段、单帧误读不采用，避免建议闪烁 | 新增易抖字段时加到 `CARRY`/`CONFIRM` |
| | `llm.py` | 大模型（OpenAI 兼容 / Anthropic），支持思考强度、JSON 输出 | 改提示词 `SYSTEM_PROMPT`；海克斯回合可单独设思考强度 |
| | `planner.py` | 建议 → 操作 | 新操作先在 `actions.py` 定义 |
| actuator | `layout.py` | 所有点击坐标（相对窗口 0~1） | 标定写入 `var/layout.json` |
| | `executor.py` | Quartz 鼠标事件 / 演练 | |
| | `safety.py` | 急停、限速、甩角落停止 | |
| modes | `assistant.py` / `autoroll.py` / `autopilot.py` | 三种工作模式 | 新模式继承 `Mode` 实现 `loop()` |
| webui | `app.py` | 主窗口、静态资源/开发服务器入口与关闭清理 | 开发模式由 `JCC_WEB_DEV=1` 选择 |
| | `api.py` | JS 可调用的接口、模式启停、后台同步及上下文锁 | 新接口使用 `@expose`，补离线测试 |
| | `serialize.py` / `settings_schema.py` | 状态、建议、HUD 等纯数据序列化，以及设置元数据与校验 | 与前端 `types.ts` 同步 |
| | `events.py` | EventBus 合并/广播、WebSink、日志转发 | 新事件增加类型和订阅 |
| | `macos.py` / `hud.py` | 原生标题栏、透明/穿透/置顶、HUD 窗口管理与跟随 | AppKit 修改在主线程执行 |
| web | `src/main.ts` / `App.vue` | Vue 3、Pinia、hash 路由及应用外壳 | `/gallery` 为开发用组件展示页 |
| | `src/pages/` | 五个主页面、HUD 浮层及概率小面板 | 通过 bridge 调用 API、订阅 store 数据 |
| | `src/components/ui/` / `components/` | 基础控件，以及布局、检查器、对话框 | 不引入 UI 组件库 |
| | `src/stores/` | 状态/建议、最多 3000 条日志、牌库参数和结果 | 主窗口只初始化一次 |
| | `src/bridge/` / `composables/` | API/事件桥、通知、确认、快捷键等 | 调用失败统一显示 Toast |
| | `src/styles/tokens.css` | 浅深主题、颜色、间距和动画变量 | 遵守减弱动态效果设置 |

## 界面通信

JS → Python：`bridge/api.ts` 等待 `pywebviewready`，再调用 `window.pywebview.api.<方法>(...)`。
Api 公开方法用 `@expose` 返回 `{ok: true, data}` 或 `{ok: false, error}`；失败由前端统一 Toast 并抛出异常。
API 调用运行在后台线程，修改上下文的操作持锁；同步使用独立数据库连接。

Python → JS：EventBus 调用 `window.__jcc_emit({type, payload})`。前端先注册入口，挂载完成前暂存事件。
事件有 `state`、`advice`、`hud`、`action`、`log`、`mode_finished`、`status`、`sync_progress`、`sync_done`、
`data_changed`、`target_changed`。其中 state/advice/hud 每 100ms 合并，只保留各自最新一份；log 同周期批量发送；
其余事件即时发送。

总线按 main / hud / hud_panel 三个 key 广播，attach/detach 管理连接，某一窗口失败不影响其他窗口。
主窗口维护业务 stores；两个子窗口只订阅 hud/advice，不套用主窗口外壳。窗口隐藏时保持连接，销毁前解除连接。

HUD 位置由纯函数序列化成相对游戏窗口的百分比。透明原生窗口忽略鼠标事件、不激活应用，每秒查询目标窗口，
变化时同步位置和尺寸；目标不可见时隐藏，重新找到后恢复。macOS 按主屏高度在 Quartz 左上角与 AppKit 左下角坐标间换算，
支持副屏。概率小面板固定 300×220、置顶，顶部可拖动，内部滚动；两个子窗口随主窗口关闭并停止跟随线程。

构建时 Vite 输出 `webui/static/`，开发时加载本地 Vite 服务器。打包脚本先构建前端，再把 static 和 pywebview 的
JS/原生后端资源一并收集；运行时从 `Path(__file__)` 所在目录寻找 static，开发和打包使用相同的入口。

## 视觉模块

默认后端 `PyEyeProvider`（`vision/pyeye.py` + `vision/eye/`）在本进程内完成识别，由原 Swift 版 game-eye 移植：

| 文件 | 职责 |
|---|---|
| `eye/capture.py` | `CGWindowListCreateImage` 按窗口 id 截图；像素经 DeviceRGB 位图转换（与 Swift 版取值一致，阈值才对得上） |
| `eye/ocr.py` | pyobjc 调 Apple Vision；多个小区域竖向拼图（≤1600px）一次识别再拆回 |
| `eye/reader.py` | 一帧 → 状态 dict（字段与 game-eye 一致）；文字格按「文字像素」缓存；回合类型跨帧判断 |
| `eye/units.py` | numpy 血条检测、星级、Vision 特征向量 + 图库、跨帧跟踪、买卡自动标注 |
| `eye/text.py` | OCR 纠错词典：优先用游戏数据库的完整英雄/羁绊表 |
| `eye/regions.json` | 区域坐标（相对窗口 0~1，1774×1136 下实测） |

回归：`tests/fixtures/frames/` 的真实截图走端到端测试；`jcc vision file` 输出与 game-eye 同格式 JSON，
可和 `tests/fixtures/game_eye_states.jsonl`（Swift 版输出）逐字段对比。

注意：不要用 ScreenCaptureKit 的 SCStream 视频流，本机会报 -3805 并卡死后续截图。
原 Swift 版 game-eye 及其 MCP 接口已弃用并从代码中移除；只在首次运行时导入它的单位图库（`paths.legacy_gallery_dir()`）。
视觉待完善项（数字模板匹配、装备、选秀、棋盘行列等）见 `game-eye/待完善清单.md`。

## 实战录制与复测

`scripts/record_assistant.py` 跑真实的辅助大脑，把每帧画面、识别结果、推给界面的建议、大模型往返存到
`var/sessions/<时间>/`；`--replay 某次的 states.jsonl` 可离线复现（`--interval 0.85 --llm` 按真实节奏带大模型）。
`scripts/analyze_session.py` 统计识别缺失、建议跳变、AI 文字停留时间、买卡理由等。

## 安全设计（自动操作）

1. 默认**演练模式**：只记录动作不点击，确认决策和标定无误后再在设置里关闭。
2. 每步执行前 `SafetyGuard.check()`：急停标志、鼠标甩到主屏左上角、每秒操作上限。
3. 规划器每次只出「一小步」，执行完重新读画面再规划；画面不变化时暂停，防止误操作连点。
4. 真实操作前弹窗确认；找不到游戏窗口或没有辅助功能权限时拒绝启动。

## 运行数据

| 路径 | 内容 |
|---|---|
| `var/jcc.sqlite3` | 游戏数据库，所有赛季在同一个库里（`jcc db sync [--season ID]` 重建某个赛季） |
| `var/settings.json` | 用户设置（含 API Key，不入库） |
| `var/layout.json` | 坐标标定结果 |
| `var/gallery/` | 单位图库（特征向量 .npy + 裁剪图 .jpg），跨对局累积 |
| `var/logs/jcc.log` | 日志（滚动 3 份） |

打包后的 .app 改用 `~/Library/Application Support/金铲铲大脑/`；环境变量 `JCC_HOME` 可覆盖。
