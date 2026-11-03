# 开发规范

## 分支

- `main`：始终可运行、测试全过。
- 新功能 / 修复在分支上做：`feat/xxx`、`fix/xxx`，完成后合并回 `main`。

## 提交信息（Conventional Commits）

```
<类型>(<范围>): <一句话说明>

<可选：为什么这样改、影响了什么>
```

类型：`feat` 新功能 · `fix` 修复 · `refactor` 重构 · `perf` 性能 · `test` 测试 · `docs` 文档 · `build` 打包/依赖 · `chore` 杂项

范围：`core` `data` `vision` `engine` `actuator` `modes` `webui` `web` `cli`

一个提交只做一件事；功能代码和它的测试可以同一个提交，无关的格式调整单独提交。

## 每次提交前

```bash
make test
```

改了决策规则（`engine/`）要补对应单测；改了 `schema.sql` 要递增 `db.py` 的 `SCHEMA_VERSION`。

## 代码风格

- Python ≥ 3.11，类型标注，`from __future__ import annotations`。
- 注释和文档用中文，标识符用英文。
- 模块间依赖方向见 `docs/ARCHITECTURE.md`，不要让下层 import 上层。
- 不提交 `var/`（数据库、设置、API Key、日志）。

## 前端开发

先安装 Node.js（`brew install node`），在 `web/` 执行一次 `npm ci`。在两个终端分别运行：

```bash
cd web && npm run dev
```

```bash
JCC_WEB_DEV=1 .venv/bin/python -m jccbrain gui
# 也可以用 JCC_WEB_DEV=1 make gui，它会先构建一次静态资源
```

Vite 固定监听 5173 端口，pywebview 加载开发服务器，保存前端文件后热更新。普通 `make gui` 先构建静态页面；
`cd web && npm run build` 同时执行 vue-tsc 类型检查和生产构建。`#/gallery` 保留为基础组件展示页。

新增 API 时，在 `src/jccbrain/webui/api.py` 添加方法并使用 `@expose`，保持 snake_case 参数与 `{ok,data,error}` 约定；
补充对应的离线 pytest，更新 `web/src/types.ts`，前端通过 `bridge/api.ts` 的 call 调用。新增推送还需要在 EventBus、
前端事件订阅和类型中同步，避免重复订阅。

可复用的基础组件只放在 `web/src/components/ui/`，不引入 UI 组件库；颜色只使用 `styles/tokens.css` 的变量。
保持行 key 稳定，动画仅改变 transform/opacity，支持减弱动态效果。主窗口与 HUD 路由分开，子窗口只订阅 hud/advice。
GUI 或打包验收使用工作树内的临时 `JCC_HOME`，避免写入真实设置；不提交 static、node_modules、build 或 dist。
