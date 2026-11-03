"""模式基类与运行器。"""
from __future__ import annotations

import logging
import threading
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jccbrain.actuator.actions import Action
from jccbrain.actuator.safety import AbortError
from jccbrain.core.models import GameState
from jccbrain.engine.advisor import Advice, BuyProjection
from jccbrain.engine.session import GameWatcher
from jccbrain.engine.shopstats import ShopStats
from jccbrain.engine.tracker import CompTracker
from jccbrain.vision.base import VisionError

if TYPE_CHECKING:
    from jccbrain.context import AppContext

log = logging.getLogger(__name__)


class ModeSink:
    """模式输出的回调接口。默认实现只写日志，UI 用 Qt 信号覆盖这些方法。"""

    def on_state(self, state: GameState) -> None: ...

    def on_advice(self, advice: Advice) -> None: ...

    def on_ai_projection(self, projection) -> None: ...

    def on_buy_projection(self, projection: BuyProjection) -> None: ...

    def on_lock_changed(self) -> None: ...

    def on_watch_changed(self) -> None: ...

    def on_augments_changed(self) -> None: ...

    def on_scout_changed(self) -> None: ...

    def on_mode_changed(self) -> None: ...

    def scout_toast_capable(self) -> bool:
        """准备能力不等于已经呈现；只有独立提示窗能够确认。"""
        return False

    def scout_hud_visible(self, window: str | None = None) -> bool:
        """仅绑定的独立侦察提示窗身份能确认呈现。"""
        return False

    def on_action(self, action: Action, real: bool) -> None:
        log.info("%s%s", "" if real else "[演练] ", action.describe())

    def on_log(self, text: str) -> None:
        log.info(text)

    def on_finished(self, mode: str, reason: str) -> None:
        log.info("%s 结束：%s", mode, reason)


class Mode(ABC):
    key = "base"
    title = "模式"
    acts = False            # 是否会操作游戏

    def __init__(self, ctx: "AppContext", sink: ModeSink):
        self.ctx = ctx
        self.sink = sink
        self.stop_event = threading.Event()
        self.tracker = CompTracker()
        self.shop_stats = ShopStats()
        self._season_id = ctx.db.season_id
        self._game_watcher = GameWatcher()

    @abstractmethod
    def loop(self) -> str:
        """主循环，返回结束原因。应经常检查 self.stopped。"""

    @property
    def stopped(self) -> bool:
        return self.stop_event.is_set() or (self.acts and self.ctx.guard.stopped)

    def sleep(self, s: float) -> None:
        self.stop_event.wait(s)

    def read(self, fresh: bool = True) -> GameState:
        st = self.ctx.vision.read(fresh)
        self.sink.on_state(st)
        return st

    def _on_new_game(self) -> None:
        """子类可在同一局切换时丢弃异步结果。"""

    def _selection_changed(self, selection: dict) -> None:
        """仅辅助模式使用；其他模式候选范围保持不变。"""

    def compute_advice(self, state: GameState, *, new_game: bool = False) -> Advice:
        if new_game:
            self._game_watcher = GameWatcher()
        new_game = self._game_watcher.update(state) or new_game
        if new_game:
            self._on_new_game()
        if new_game or self._season_id != self.ctx.db.season_id:
            if getattr(self.ctx.advisor, "scout", None):
                self.ctx.advisor.scout.reset()
            self.tracker.reset()
            self.shop_stats.reset()
            self._season_id = self.ctx.db.season_id
        if new_game:
            self.ctx.game_watch_ids = []
            self.sink.on_watch_changed()
            self.ctx.game_augments = []
        if new_game or self.ctx.game_lock_id is None:
            default = self.ctx.settings.assistant.default_lock_comp_id
            self.ctx.game_lock_id = (default if self.key == 'assistant' or self.ctx.enabled_comp(default) else '')
            comp = self.ctx.enabled_comp(self.ctx.game_lock_id) if self.ctx.game_lock_id else None
            prefix = '新的一局：锁定已重置为 ' if new_game else '开始辅助：锁定为 '
            self.sink.on_log(prefix + (comp.name if comp else '来源暂不可用，锁定保留' if self.ctx.game_lock_id else '自动'))
            self.sink.on_lock_changed()
        taken = list(self.ctx.game_augments)
        for name in state.augments_taken:
            name = self.ctx.advisor.resolve_augment(name)
            if name and name not in taken:
                taken.append(name)
        if new_game or taken != self.ctx.game_augments:
            self.ctx.game_augments = taken
            self.sink.on_augments_changed()
        self.shop_stats.update(state)
        selection = self.ctx.assistant_selection() if self.key == 'assistant' and hasattr(self.ctx, 'assistant_selection') else None
        if selection is not None:
            self._selection_changed(selection)
        args = {'selection': selection} if selection is not None else {}
        adv = self.ctx.advisor.advise(state, selection['lock_id'] if selection is not None else self.ctx.effective_lock_id(), self.ctx.extra_taken,
                                      tracker=self.tracker, watch_ids=selection['watch_ids'] if selection is not None else self.ctx.game_watch_ids,
                                      shop_stats=self.shop_stats, augments_taken=self.ctx.game_augments, **args)
        return adv

    def advise(self, state: GameState) -> Advice:
        adv = self.compute_advice(state)
        self.sink.on_advice(adv)
        return adv

    def act(self, actions: list[Action]) -> None:
        ex = self.ctx.executor
        for a in actions:
            if self.stopped:
                return
            ex.execute(a)
            self.sink.on_action(a, not ex.dry_run)

    def preflight(self) -> list[str]:
        return self.ctx.executor.preflight() if self.acts else []


class ModeRunner:
    """在后台线程里跑一个模式；同一时间只允许一个模式。"""

    def __init__(self, ctx: "AppContext"):
        self.ctx = ctx
        self.mode: Mode | None = None
        self._thread: threading.Thread | None = None
        self._control_lock = threading.RLock()
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        return self.running and self._stopping

    def start(self, mode: Mode) -> list[str]:
        """启动模式；有前置问题时不启动并返回问题列表。"""
        with self._control_lock:
            if self.running:
                return ["旧模式仍在停止中，请等待线程退出后再启动" if self.stopping else "已有运行模式，请先停止"]
            problems = mode.preflight()
            if problems:
                return problems
            if mode.acts or getattr(mode, "scout", None) or getattr(mode, "learning", None):
                self.ctx.guard.reset()
            self._stopping = False
            self.mode = mode
            self._thread = threading.Thread(target=self._run, args=(mode,), name=f"mode-{mode.key}", daemon=True)
            self._thread.start()
            return []

    def _run(self, mode: Mode) -> None:
        reason = "完成"
        try:
            self.ctx.vision.start()
            reason = mode.loop()
        except AbortError as e:
            reason = f"已中止：{e}"
        except VisionError as e:
            reason = f"视觉模块错误：{e}"
        except Exception as e:  # noqa: BLE001 - 后台线程兜底，错误交给 UI 显示
            log.error("模式 %s 异常：%s", mode.key, traceback.format_exc())
            reason = f"异常：{e}"
        mode.sink.on_finished(mode.key, reason)

    def stop(self, timeout: float = 3.0) -> bool:
        with self._control_lock:
            self._stopping = True
            if self.mode:
                self.mode.stop_event.set()
                if getattr(self.mode, "learning", None):
                    self.mode.learning.stop()
                if getattr(self.mode, "scout", None):
                    self.mode.scout.stop()
                if self.mode.acts or getattr(self.mode, "learning", None):
                    self.ctx.guard.stop("切换/停止模式")
            if self.running and threading.current_thread() is not self._thread:
                self._thread.join(timeout)
            # join 超时不释放所有权，尤其不能让新模式 reset 共用急停标志。
            if self.running:
                return False
            self._thread = None
            self.mode = None
            self._stopping = False
            return True
