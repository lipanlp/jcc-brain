"""受控假截图/假执行器验证扫描，不发送任何系统输入。"""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from jccbrain.actuator.actions import ClickPoint, RestoreMouse
from jccbrain.actuator.executor import DryRunExecutor, MacExecutor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import AbortError, SafetyGuard
from jccbrain.actuator.window import WindowInfo
from jccbrain.core.models import GameState, Phase, Player, Unit
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.scout import ScoutBook, ScoutShot
from jccbrain.modes.base import ModeSink
from jccbrain.modes.scout import ScoutRunner
from scout_fakes import ControlledInput

WINDOW = WindowInfo('game', '', 101, 100, 100, 1000, 500, 42)


class Clock:
    t = 0
    def __call__(self): return self.t
    def sleep(self, n): self.t += n


class Ex(DryRunExecutor):
    dry_run = False
    def __init__(self, layout, guard):
        super().__init__(layout, guard, 0)
        self.mouse = (400, 300)
        self.problems = []
    def mouse_position(self): return self.mouse
    def preflight(self): return self.problems.copy()
    def _do_guarded(self, action, event_guard):
        point = WINDOW.to_screen(*action.point) if isinstance(action, ClickPoint) else action.screen_point
        event_guard.begin('click' if isinstance(action, ClickPoint) else 'restore', point, (0.,0.))
        for kind in (('MouseMoved','LeftMouseDown','LeftMouseUp') if isinstance(action, ClickPoint) else ('MouseMoved',)):
            event_guard.check()
            self.monitor.event(kind, own=True, point=point)
        event_guard.finish(event_guard.check)
        self._do(action)
    def _do(self, action):
        self.history.append(action)
        if not self.dry_run:
            if isinstance(action, ClickPoint): self.mouse = WINDOW.to_screen(*action.point)
            if isinstance(action, RestoreMouse): self.mouse = action.screen_point


class Capture:
    def __init__(self, clock, values=None):
        self.clock, self.values = clock, iter(values) if values is not None else None
        self.current = 0
        self.calls = 0
        self.hook = lambda n: None
    def on_click(self, action, real):
        if isinstance(action, ClickPoint) and self.values is None:
            self.current = (int(action.label.split('第')[1].split('位')[0])-1)*100
    def grab(self):
        self.calls += 1
        self.hook(self.calls)
        self.clock.sleep(.02)
        if self.values is not None:
            self.current = next(self.values, self.current)
        return np.full((160, 240, 3), self.current, np.uint8), WINDOW


class Sink(ModeSink):
    def __init__(self): self.logs, self.actions, self.changes = [], [], 0
    def on_log(self, text): self.logs.append(text)
    def on_action(self, action, real): self.actions.append((action, real))
    def on_scout_changed(self): self.changes += 1


@pytest.fixture
def setup(db, rules, tmp_path, monkeypatch):
    monkeypatch.setenv('JCC_HOME', str(tmp_path/'runtime'))
    settings = Settings()
    book = ScoutBook(db, directory=tmp_path/'book')
    guard = SafetyGuard(999, False)
    ctx = SimpleNamespace(settings=settings, scout=book, guard=guard, layout=Layout(), db=db)
    ctx.layout.player_slots_calibrated = True
    ctx.advisor = HeuristicAdvisor(db, rules, scout=book)
    ctx.executor = Ex(ctx.layout, guard)
    clock, sink = Clock(), Sink()
    ctx.locator = SimpleNamespace(find=lambda _: WINDOW)
    ctx.monitor = ControlledInput(clock, WINDOW, ctx.executor.mouse_position)
    ctx.executor.monitor = ctx.monitor
    capture = Capture(clock)
    ctx.executor.listeners.append(capture.on_click)
    reads = []
    def read(image):
        reads.append(int(image[0, 0, 0]))
        return {'board': [{'name': '绯红树怪', 'star': 2}], 'level': 6}
    import inspect
    injected = {'input_monitor':ctx.monitor} if 'input_monitor' in inspect.signature(ScoutRunner).parameters else {}
    runner = ScoutRunner(ctx, sink, capture=capture, reader_factory=lambda: SimpleNamespace(read_scout=read),
                         probe=lambda *_: {}, clock=clock, sleep=clock.sleep, **injected)
    state = GameState(stage='4-1', phase=Phase.PLANNING, timer=20,
                      players=[Player('自己', 80, True, 0), Player('甲', 70, False, 1, .2), Player('乙', 60, False, 2)])
    book.update(state)
    return ctx, runner, capture, state, sink, reads


def finish(runner):
    if runner._thread: runner._thread.join(3)
    assert not runner.status['processing']


def clicks(ctx):
    return [a for a in ctx.executor.history if isinstance(a, ClickPoint)]


def test_cut_stable_capture_return_restore_and_background(setup):
    ctx, runner, capture, state, sink, reads = setup
    runner.run(state, manual=True)
    finish(runner)
    assert reads == [100, 200]
    assert runner.status['captured'] == runner.status['recognized'] == 2
    assert [a.label for a in clicks(ctx)] == ['侦察：玩家栏第2位', '侦察：玩家栏第3位', '侦察：玩家栏第1位']
    assert clicks(ctx)[0].point[1] == .2
    assert isinstance(ctx.executor.history[-1], RestoreMouse) and ctx.executor.mouse == (400, 300)
    assert len(ctx.scout.latest()) == 2 and not runner.scanning
    assert any('后台识别' in line for line in sink.logs)


@pytest.mark.parametrize('change', ['disabled', 'round', 'timer', 'phase', 'viewing'])
def test_trigger_rejects_unready_states(setup, change):
    ctx, runner, cap, state, _, _ = setup
    ctx.settings.scout.enabled = True
    if change == 'disabled': ctx.settings.scout.enabled = False
    if change == 'round': state.stage = '2-2'
    if change == 'timer': state.timer = 14
    if change == 'phase': state.phase = Phase.COMBAT
    if change == 'viewing': state.viewing = {'slot': 1}
    assert not runner.maybe_run(state) and not ctx.executor.history and cap.calls == 0


def test_automatic_only_once_per_round_manual_bypasses_switch(setup):
    ctx, runner, _, state, _, _ = setup
    ctx.settings.scout.enabled = True
    runner.sink.scout_toast_capable = lambda: True
    runner.sink.scout_hud_visible = lambda window=None: window == 'toast'
    assert not runner.maybe_run(state)
    assert runner.notice_shown(runner.status['notice']['id'], 'toast')
    runner.clock.sleep(3)
    assert runner.maybe_run(state)
    finish(runner)
    assert not runner.maybe_run(state)
    ctx.settings.scout.enabled = False
    runner.request()
    assert runner.maybe_run(state)
    finish(runner)
    assert len(clicks(ctx)) == 6


def test_targets_skip_self_dead_unknown_and_sort_priority(setup):
    ctx, runner, _, state, _, _ = setup
    ctx.scout.add(ScoutShot(slot=1, round='3-2', t=1, board=[Unit('奥恩')]))
    ctx.scout.add(ScoutShot(slot=2, round='4-1', t=2, board=[Unit('奥恩')]))
    state.players += [Player('新', 10, False, 3), Player('死人', 0, False, 4), Player('缺槽', 99)]
    ctx.scout.update(state)
    assert [p.slot for p in runner.targets(state)] == [3, 1, 2]
    ctx.settings.scout.max_targets = 2
    assert [p.slot for p in runner.targets(state)] == [3, 1]


def test_skip_recent_uses_hp_at_observation_not_latest_roster(setup):
    ctx, runner, _, state, _, _ = setup
    ctx.scout.add(ScoutShot(slot=1, round='4-1', t=1))
    ctx.settings.scout.skip_recent_rounds = 1
    ctx.scout.update(state)
    assert [p.slot for p in runner.targets(state)] == [2]
    state.players[1].hp -= 10
    ctx.scout.update(state)
    assert {p.slot for p in runner.targets(state)} == {1, 2}


def test_timeout_preserves_unloaded_image_without_trusting_board(setup):
    ctx, runner, cap, state, sink, reads = setup
    cap.values = iter([0] * 100)
    runner._request_retry = lambda _: None  # 此例只验证超时；补截在独立用例中覆盖。
    runner.run(state, manual=True)
    finish(runner)
    assert reads == [0, 0] and len(ctx.scout.latest()) == 2
    assert not ctx.scout.taken() and all(not row.loaded and not row.units for row in ctx.scout.latest())
    assert runner.status['captured'] == 2
    assert sum('加载超时，保留' in line for line in sink.logs) == 2
    assert clicks(ctx)[-1].point == ctx.layout.player_slots[0]


def test_does_not_wait_for_recognition_before_next_click(setup):
    ctx, runner, _, state, _, _ = setup
    entered, release = threading.Event(), threading.Event()
    def read(_):
        entered.set()
        assert release.wait(2)
        return {}
    runner._reader_factory = lambda: SimpleNamespace(read_scout=read)
    runner.run(state, manual=True)
    assert len(clicks(ctx)) == 3 and not runner.scanning
    assert entered.wait(1) and runner.status['processing']
    release.set()
    finish(runner)


def test_read_scout_preferred_and_full_reader_fallback(setup):
    ctx, runner, _, state, _, reads = setup
    runner._reader_factory = lambda: SimpleNamespace(read=lambda im: (reads.append('fallback'), {})[1])
    runner.run(state, manual=True)
    finish(runner)
    assert reads == ['fallback', 'fallback']


@pytest.mark.parametrize('reason', ['mouse', 'stop', 'guard', 'combat', 'augment', 'timer'])
def test_interrupts_and_cleans_up(setup, reason):
    ctx, runner, cap, state, sink, _ = setup
    def hook(n):
        if n != 2: return
        if reason == 'mouse':
            ctx.executor.mouse = (900, 800)
            ctx.monitor.event('MouseMoved')
        if reason == 'stop': runner.stop()
        if reason == 'guard': ctx.guard.stop('急停')
    cap.hook = hook
    if reason in ('combat', 'augment'): runner._probe = lambda *_: {'phase': reason}
    if reason == 'timer': runner._probe = lambda *_: {'timer': 4}
    runner.run(state, manual=True)
    finish(runner)
    assert runner.status['reason'] and not runner.scanning
    assert not ctx.scout.latest()
    assert any('侦察已中断' in line for line in sink.logs)
    if reason == 'mouse': assert ctx.executor.mouse == (900, 800)
    if reason == 'guard': assert len(clicks(ctx)) == 1  # 不绕过急停强点自己


def test_real_scout_requires_player_calibration(setup):
    ctx, runner, cap, state, _, _ = setup
    ctx.layout.player_slots_calibrated = False
    runner.run(state, manual=True)
    assert '玩家栏尚未标定' in runner.status['reason']
    assert not ctx.executor.history and cap.calls == 0 and not runner.busy


def test_dry_run_default_records_only_plan(setup):
    ctx, runner, cap, state, _, reads = setup
    ctx.executor.dry_run = True
    ctx.layout.player_slots_calibrated = False
    runner._capture = None
    runner._simulated_frames = False
    runner.run(state, manual=True)
    assert cap.calls == 0 and not reads
    assert len(clicks(ctx)) == 3 and runner.status['captured'] == 0


def test_preflight_failure_skips_and_missing_own_slot(setup):
    ctx, runner, cap, state, sink, _ = setup
    ctx.executor.problems = ['无辅助功能权限']
    runner.run(state, manual=True)
    assert not ctx.executor.history and cap.calls == 0 and '无辅助功能权限' in sink.logs[-1]
    ctx.executor.problems = []
    state.players[0].slot = None
    runner.run(state, manual=True)
    assert '尚未识别自己' in sink.logs[-1]


def test_late_background_new_game_is_discarded(setup):
    ctx, runner, _, state, _, _ = setup
    entered, release = threading.Event(), threading.Event()
    def read(_):
        entered.set()
        assert release.wait(2)
        return {'board': [{'name': '奥恩'}]}
    runner._reader_factory = lambda: SimpleNamespace(read_scout=read)
    runner.run(state, manual=True)
    assert entered.wait(1)
    ctx.scout.reset()
    release.set()
    finish(runner)
    assert not ctx.scout.latest()


def test_identity_mismatch_is_discarded(setup):
    ctx, runner, _, state, sink, _ = setup
    runner._reader_factory = lambda: SimpleNamespace(read_scout=lambda _: {'viewing': {'slot': 7}, 'board': [{'name':'奥恩'}]})
    runner.run(state, manual=True)
    finish(runner)
    assert not ctx.scout.latest() and any('身份不符' in line for line in sink.logs)


def test_recognition_exception_does_not_break_scan(setup):
    ctx, runner, _, state, sink, _ = setup
    def fail(_): raise RuntimeError('bad frame')
    runner._reader_factory = lambda: SimpleNamespace(read_scout=fail)
    runner.run(state, manual=True)
    finish(runner)
    assert runner.status['processed'] == 2 and runner.status['recognized'] == 0
    assert len(clicks(ctx)) == 3


def test_new_actions_and_existing_dispatch_are_unchanged(monkeypatch):
    import sys
    from jccbrain.actuator import actions as A
    cfg = Settings().actuator
    guard = SimpleNamespace(check=lambda: None, sleep=lambda _: None)
    ex = MacExecutor(Layout(), guard, cfg)
    calls = []
    ex.click = lambda p: calls.append(('click', p))
    ex.drag = lambda a,b: calls.append(('drag', a,b))
    old = [(A.BuyShop(0), ('click', ex.layout.shop_slots[0])), (A.Reroll(), ('click', ex.layout.refresh_button)),
           (A.BuyXP(), ('click', ex.layout.xp_button)), (A.ToggleLock(), ('click', ex.layout.lock_button)),
           (A.SellBench(1), ('drag', ex.layout.bench_slot(1), ex.layout.sell_zone)),
           (A.BenchToBoard(1,2,3), ('drag', ex.layout.bench_slot(1), ex.layout.hex_center(2,3))),
           (A.BoardToBench(2,3,1), ('drag', ex.layout.hex_center(2,3), ex.layout.bench_slot(1))),
           (A.MoveBoard((1,1),(2,2)), ('drag', ex.layout.hex_center(1,1), ex.layout.hex_center(2,2))),
           (A.PickAugment(1), ('click', ex.layout.augment_options[1]))]
    for action, expected in old:
        ex.execute(action)
        assert calls[-1] == expected
    ex.execute(A.Wait(0))
    ex.execute(ClickPoint((.5,.6),'侦察'))
    assert calls[-1] == ('click',(.5,.6))
    monkeypatch.setitem(sys.modules, 'Quartz', SimpleNamespace(kCGEventMouseMoved=7, CGPointMake=lambda x,y:(x,y)))
    ex._post = lambda kind,point: calls.append((kind,point))
    ex.execute(RestoreMouse((10,20)))
    assert calls[-1] == (7,(10,20))
    monkeypatch.setattr('jccbrain.actuator.window.mouse_location', lambda:(20,30))
    assert ex.mouse_position() == (20,30)


def test_guard_cancels_during_rate_limit_and_callback_cancels_after_wait(monkeypatch):
    guard = SafetyGuard(1,False)
    ex = DryRunExecutor(Layout(),guard,0)
    guard._recent.append(__import__('time').monotonic())
    guard.sleep = lambda _: guard.stop('限速期间急停')
    with pytest.raises(AbortError): ex.execute(ClickPoint((.5,.5),'侦察'))
    assert not ex.history
    guard.reset()
    guard._recent.clear()
    def cancel(): raise AbortError('用户移动鼠标')
    with pytest.raises(AbortError): ex.execute(ClickPoint((.5,.5),'侦察'),before_do=cancel)
    assert not ex.history


def test_settings_and_points_roundtrip(tmp_path, db, monkeypatch):
    from jccbrain.webui.settings_schema import _prepare, schema
    settings = Settings()
    assert not settings.scout.enabled and settings.scout.min_timer == 15 and settings.scout.cut_threshold == 25
    staged = _prepare(settings, {'scout': {'rounds':'3-2, 4-1,3-2', 'max_targets':3}}, db)
    assert staged.scout.rounds == ['3-2','4-1']
    path = tmp_path/'settings.json'
    staged.save(path)
    assert Settings.load(path).scout == staged.scout
    assert next(s for s in schema(settings,db,full=True) if s['key']=='scout')['title'] == '侦察'
    with pytest.raises(ValueError): _prepare(settings, {'scout':{'rounds':'bad'}}, db)
    with pytest.raises(ValueError): _prepare(settings, {'scout':{'max_targets':8}}, db)
    layout = Layout()
    layout.player_slots = [(.9,.1+i*.08) for i in range(8)]
    path = tmp_path/'layout.json'
    layout.save(path)
    assert Layout.load(path).player_slots == layout.player_slots


def test_main_loop_pauses_read_and_push_and_spectate_never_enters_stabilizer(setup, db, rules):
    from conftest import make_state
    from jccbrain.context import AppContext
    from jccbrain.modes.assistant import AssistantMode
    from test_assistant import ListVision
    from jccbrain.vision.base import VisionError
    ctx, runner, cap, state, _, _ = setup
    main = AppContext(ctx.settings)
    main.db, main.rules, main.scout = db, rules, ctx.scout
    main.guard, main.layout, main.executor = ctx.guard, ctx.layout, ctx.executor
    own = make_state(stage='4-1',timer=20,players=[{'name':'自己','self':True,'slot':0,'hp':80},{'name':'甲','slot':1,'hp':70},{'name':'乙','slot':2,'hp':60}],board=[{'name':'奥恩'}])
    foreign = {**own,'phase':'spectate','viewing':{'slot':1},'board':[{'name':'苍蓝哨戒'}]}
    main.vision = ListVision([own, foreign, {**own,'phase':'unknown'}, own])
    main.llm = SimpleNamespace(ready=False)
    returned = []
    class MainSink(Sink):
        def on_state(self, st):
            assert not runner.scanning
            returned.append(st)
        def on_advice(self, adv): assert not runner.paused
    sink = MainSink()
    mode = AssistantMode(main,sink)
    mode.scout = runner
    ctx.settings.scout.enabled = True
    runner.request()  # 原扫描期间隔离断言通过手动即时入口验证。
    cap.hook = lambda n: (len(returned)==1) or pytest.fail('扫描期间主视觉还在读/推')
    with pytest.raises(VisionError): mode.loop()
    finish(runner)
    assert len(returned)==3  # 两帧自己的画面，外加退出时清空界面；观战/等待自己仍丢弃。
    assert all([u.name for u in st.board]==['奥恩'] for st in returned[:-1])
    assert returned[-1].stage is None and not returned[-1].board
    assert mode.stabilizer.acc['board'][0]['name']=='奥恩'


def test_llm_reply_does_not_push_until_fresh_own_frame(setup, db, rules):
    from test_assistant import Ctx, AdviceSink
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.engine.llm import LLMReply
    ctx, runner, _, state, _, _ = setup
    main = Ctx(db,rules,[])
    main.advisor.scout = ctx.scout
    sink = AdviceSink()
    mode = AssistantMode(main,sink)
    mode.scout = runner
    adv = main.advisor.advise(state)
    from ai_fakes import authorize_ai
    authorize_ai(mode,adv,state)
    mode._latest = adv
    main.llm.advise = lambda *_: LLMReply('存钱',[], '')
    runner._awaiting_own = True
    mode._ask_llm(state,adv)
    assert not sink.advices
    runner.mark_own()
    mode._ask_llm(state,adv)
    assert len(sink.advices)==1


def test_state_replay_cannot_send_real_clicks(setup):
    ctx, runner, cap, state, sink, _ = setup
    ctx.settings.vision.backend = 'replay'
    runner.run(state, manual=True)
    assert cap.calls==0 and not ctx.executor.history
    assert '只支持演练' in sink.logs[-1]


def test_real_click_path_requires_foreground_and_manual_never_activates(setup, monkeypatch):
    ctx, runner, _, state, sink, _ = setup
    # 只伪造Mac执行器类型与前台查询，不触碰AppKit/Quartz。
    import jccbrain.modes.scout as module
    monkeypatch.setattr(module,'MacExecutor',Ex)
    front = [999]
    runner._foreground_pid = lambda: front[0]
    ctx.locator = SimpleNamespace(find=lambda _: WINDOW, activate=lambda: front.__setitem__(0,WINDOW.pid))
    ctx.settings.scout.enabled = True
    sink.scout_toast_capable = lambda: True
    sink.scout_hud_visible = lambda window=None: window == 'toast'
    runner.maybe_run(state)
    assert not ctx.executor.history and '游戏不是前台' in sink.logs[-1]
    runner.run(state,manual=True)
    finish(runner)
    assert not clicks(ctx) and front[0] == 999
    front[0] = WINDOW.pid
    runner.run(state, manual=True)
    finish(runner)
    assert len(clicks(ctx)) == 3


def test_late_same_round_result_uses_capture_time_not_recognition_time(setup):
    import time
    ctx, runner, _, state, _, _ = setup
    entered, release = threading.Event(), threading.Event()
    def read(_):
        entered.set()
        assert release.wait(2)
        return {'board':[{'name':'绯红树怪','star':2}]}
    runner._reader_factory = lambda: SimpleNamespace(read_scout=read)
    runner.run(state, manual=True)
    assert entered.wait(1)
    ctx.scout.add(ScoutShot(slot=1,round='4-1',t=time.time()+1,board=[Unit('奥恩')]))
    release.set()
    finish(runner)
    assert next(row for row in ctx.scout.latest() if row.slot==1).units[0].name=='奥恩'
