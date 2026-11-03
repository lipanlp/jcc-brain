"""用受控真实后台线程验证超时所有权；不读取屏幕或发送系统输入。"""
import threading
from types import SimpleNamespace

from jccbrain.actuator.actions import BuyShop
from jccbrain.actuator.executor import DryRunExecutor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import SafetyGuard
from jccbrain.modes.base import Mode, ModeRunner, ModeSink
from test_autoroll import snapshot


class BlockedVision:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.starts = 0
    def stop(self):
        pass
    def start(self):
        self.starts += 1
    def read(self, fresh=True):
        self.entered.set()
        assert self.release.wait(4), '测试线程未释放'
        return snapshot()


class BlockedMode(Mode):
    key, title, acts = 'assistant', '假阻塞模式', True
    def preflight(self):
        return []
    def loop(self):
        self.read()
        self.act([BuyShop(0)])
        return '已停止'


def context(db):
    guard = SafetyGuard(1000, failsafe_corner=False)
    return SimpleNamespace(db=db, vision=BlockedVision(), guard=guard,
        executor=DryRunExecutor(Layout(), guard, delay_s=0))


def test_stop_timeout_preserves_owner_and_guard_until_actual_thread_exit(db):
    ctx = context(db)
    runner = ModeRunner(ctx)
    first, second = BlockedMode(ctx, ModeSink()), BlockedMode(ctx, ModeSink())
    try:
        assert runner.start(first) == []
        assert ctx.vision.entered.wait(1)
        old_thread = runner._thread
        assert runner.stop(.01) is False
        assert runner.running and runner.stopping
        assert runner.mode is first and runner._thread is old_thread
        assert first.stop_event.is_set() and ctx.guard.stopped
        assert runner.start(second) and runner.start(first)
        assert ctx.vision.starts == 1 and ctx.guard.stopped
    finally:
        ctx.vision.release.set()
        assert runner.stop(1)
    assert not runner.running and not runner.stopping and runner.mode is None
    assert not ctx.executor.history
    # 退出后可以正常获得下一次所有权，guard 只在此时重置。
    assert runner.start(second) == []
    runner._thread.join(1)
    assert ctx.executor.history == [BuyShop(0)]
    assert runner.stop(1)


def test_concurrent_starts_create_exactly_one_background_thread(db):
    ctx = context(db)
    runner = ModeRunner(ctx)
    barrier = threading.Barrier(9)
    results = []
    def start():
        mode = BlockedMode(ctx, ModeSink())
        barrier.wait()
        results.append(runner.start(mode))
    threads = [threading.Thread(target=start) for _ in range(8)]
    try:
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join(1)
        assert all(not t.is_alive() for t in threads)
        assert sum(not problems for problems in results) == 1
        assert ctx.vision.entered.wait(1) and ctx.vision.starts == 1
    finally:
        ctx.vision.release.set()
        runner.stop(1)
