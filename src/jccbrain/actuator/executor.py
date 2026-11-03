"""执行器：把 Action 变成真实的鼠标操作。

MacExecutor 用 Quartz CGEvent 发送鼠标事件，需要在「系统设置 → 隐私与安全性 → 辅助功能」
给运行本程序的终端 / 金铲铲大脑.app 授权。DryRunExecutor 只记录不执行，用来先验证决策和标定。
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Callable

from jccbrain.actuator import actions as A
from jccbrain.actuator.layout import Layout, Point
from jccbrain.actuator.safety import AbortError, SafetyGuard
from jccbrain.actuator.window import WindowLocator, accessibility_trusted
from jccbrain.core.settings import ActuatorSettings

log = logging.getLogger(__name__)

Listener = Callable[[A.Action, bool], None]   # (动作, 是否真实执行)


class Executor(ABC):
    def __init__(self, layout: Layout, guard: SafetyGuard):
        self.layout = layout
        self.guard = guard
        self.listeners: list[Listener] = []

    def execute(self, action: A.Action, *, before_do: Callable[[], None] | None = None, event_guard=None) -> None:
        self.guard.check()
        if before_do is not None:
            before_do()  # 限速等待后再次检查侦察取消/鼠标移动，旧调用行为不变。
        if event_guard is None:
            self._do(action)
        else:
            self._do_guarded(action, event_guard)
        for fn in self.listeners:
            fn(action, not self.dry_run)

    def _do_guarded(self, action, event_guard):
        # 非原生执行器须显式注入测试监听；默认仍只经过 before_do。
        self._do(action)

    def execute_all(self, acts: list[A.Action]) -> int:
        n = 0
        for a in acts:
            self.execute(a)
            n += 1
        return n

    @property
    @abstractmethod
    def dry_run(self) -> bool: ...

    @abstractmethod
    def _do(self, action: A.Action) -> None: ...

    def mouse_position(self) -> tuple[float, float]:
        """演练不读取系统鼠标；平台执行器覆盖这个只读方法。"""
        return (0.0, 0.0)

    def preflight(self) -> list[str]:
        """执行前检查，返回问题列表（空 = 可以执行）。"""
        return []


class DryRunExecutor(Executor):
    dry_run = True

    def __init__(self, layout: Layout, guard: SafetyGuard, delay_s: float = 0.15):
        super().__init__(layout, guard)
        self.delay_s = delay_s
        self.history: list[A.Action] = []

    def _do(self, action: A.Action) -> None:
        log.info("[演练] %s", action.describe())
        self.history.append(action)
        self.guard.sleep(action.seconds if isinstance(action, A.Wait) else self.delay_s)


class MacExecutor(Executor):
    dry_run = False

    def __init__(self, layout: Layout, guard: SafetyGuard, cfg: ActuatorSettings,
                 locator: WindowLocator | None = None):
        super().__init__(layout, guard)
        self.cfg = cfg
        self.locator = locator or WindowLocator(cfg.window_owner)

    def preflight(self) -> list[str]:
        problems = []
        if not accessibility_trusted():
            problems.append("没有「辅助功能」权限：系统设置 → 隐私与安全性 → 辅助功能，勾选当前终端或 金铲铲大脑.app")
        if not self.locator.find(0):
            problems.append(f"找不到游戏窗口（进程名包含「{self.cfg.window_owner}」）")
        return problems

    def mouse_position(self) -> tuple[float, float]:
        from jccbrain.actuator.window import mouse_location
        return mouse_location()

    # ---------------------------------------------------------------- 动作分发
    def _do(self, action: A.Action) -> None:
        lay = self.layout
        match action:
            case A.BuyShop(slot):
                self.click(lay.shop_slots[slot])
            case A.Reroll():
                self.click(lay.refresh_button)
            case A.BuyXP():
                self.click(lay.xp_button)
            case A.ToggleLock():
                self.click(lay.lock_button)
            case A.SellBench(slot):
                self.drag(lay.bench_slot(slot), lay.sell_zone)
            case A.BenchToBoard(slot, row, col):
                self.drag(lay.bench_slot(slot), lay.hex_center(row, col))
            case A.BoardToBench(row, col, slot):
                self.drag(lay.hex_center(row, col), lay.bench_slot(slot))
            case A.MoveBoard(src, dst):
                self.drag(lay.hex_center(*src), lay.hex_center(*dst))
            case A.PickAugment(index):
                self.click(lay.augment_options[index])
            case A.ClickPoint(point, _label):
                self.click(point)
            case A.RestoreMouse(screen_point):
                import Quartz
                self._post(Quartz.kCGEventMouseMoved, Quartz.CGPointMake(*screen_point))
            case A.Wait(seconds):
                self.guard.sleep(seconds)
            case _:
                raise ValueError(f"未知动作 {action}")
        self.guard.sleep(self.cfg.click_delay_ms / 1000)

    def _do_guarded(self, action, event_guard):
        """仅侦察显式传入；不改变其他调用。检查与发送间仍有有限 OS 竞态。"""
        import Quartz
        if isinstance(action, A.ClickPoint):
            point = self._screen(action.point)
            event_guard.begin('click', point, (0., 0.))
            pt = Quartz.CGPointMake(*point)
            down = False
            try:
                event_guard.check()
                self._post(Quartz.kCGEventMouseMoved, pt)
                time.sleep(.02)
                event_guard.check()
                self._post(Quartz.kCGEventLeftMouseDown, pt)
                down = True
                time.sleep(.03 + random.uniform(0, .02))
                event_guard.check()
                self._post(Quartz.kCGEventLeftMouseUp, pt)
                down = False
            finally:
                if down:
                    # 已发出的本程序 down 必须配对释放；不切换目标、不移动鼠标。
                    self._post(Quartz.kCGEventLeftMouseUp, pt)
        elif isinstance(action, A.RestoreMouse):
            point = action.screen_point
            event_guard.begin('restore', point, (0., 0.))
            event_guard.check()
            self._post(Quartz.kCGEventMouseMoved, Quartz.CGPointMake(*point))
        else:
            raise AbortError('侦察事件保护仅允许点击/鼠标恢复')
        event_guard.finish(event_guard.check)
        self.guard.sleep(self.cfg.click_delay_ms / 1000)

    # ---------------------------------------------------------------- 鼠标
    def _screen(self, p: Point) -> tuple[float, float]:
        win = self.locator.find()
        if not win:
            raise AbortError("游戏窗口不见了")
        x, y = win.to_screen(*p)
        j = self.cfg.jitter_px
        return (x + random.uniform(-j, j), y + random.uniform(-j, j))

    def click(self, p: Point) -> None:
        import Quartz

        x, y = self._screen(p)
        pt = Quartz.CGPointMake(x, y)
        self._post(Quartz.kCGEventMouseMoved, pt)
        time.sleep(0.02)
        self._post(Quartz.kCGEventLeftMouseDown, pt)
        time.sleep(0.03 + random.uniform(0, 0.02))
        self._post(Quartz.kCGEventLeftMouseUp, pt)

    def drag(self, a: Point, b: Point) -> None:
        import Quartz

        (x0, y0), (x1, y1) = self._screen(a), self._screen(b)
        self._post(Quartz.kCGEventMouseMoved, Quartz.CGPointMake(x0, y0))
        time.sleep(0.03)
        self._post(Quartz.kCGEventLeftMouseDown, Quartz.CGPointMake(x0, y0))
        time.sleep(0.06)
        steps = max(self.cfg.drag_steps, 2)
        for i in range(1, steps + 1):
            t = i / steps
            self._post(Quartz.kCGEventLeftMouseDragged, Quartz.CGPointMake(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            time.sleep(0.012)
        time.sleep(0.05)
        self._post(Quartz.kCGEventLeftMouseUp, Quartz.CGPointMake(x1, y1))

    @staticmethod
    def _post(kind, pt) -> None:
        import Quartz

        ev = Quartz.CGEventCreateMouseEvent(None, kind, pt, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def create_executor(cfg: ActuatorSettings, layout: Layout, guard: SafetyGuard) -> Executor:
    if cfg.dry_run:
        return DryRunExecutor(layout, guard)
    return MacExecutor(layout, guard, cfg)
