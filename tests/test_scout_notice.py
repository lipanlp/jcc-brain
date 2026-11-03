"""自动预告授权边界；全部使用假截图/假执行器，不发送系统输入。"""
import threading
from dataclasses import replace

import pytest

from jccbrain.core.models import Phase
from jccbrain.modes.scout import NOTICE_ACK_TIMEOUT
from test_scout_runner import setup, finish, clicks  # noqa: F401 - 复用受控夹具


def announce(setup):
    ctx, runner, _, state, sink, _ = setup
    ctx.settings.scout.enabled = True
    sink.scout_toast_capable = lambda: True
    sink.scout_hud_visible = lambda window=None: window in (None, 'toast')
    assert not runner.maybe_run(state)
    n = runner.status['notice']
    assert n and n['seconds'] == 3
    assert runner.busy and not runner.paused
    assert not ctx.executor.history
    return n['id']


def test_ack_starts_three_seconds_not_emit_and_second_window_does_not_restart(setup):
    ctx, runner, _, state, _, _ = setup
    id = announce(setup)
    runner.clock.sleep(1)
    assert not runner.maybe_run(state)
    assert runner.notice_shown(id, 'toast')
    start = runner.clock()
    actions = []
    ctx.executor.listeners.append(lambda a, real: actions.append(runner.clock()))
    assert not runner.maybe_run(state)
    runner.clock.sleep(1)
    assert not runner.notice_shown(id, 'hud_panel')
    assert not runner.notice_shown(id, 'hud')
    assert runner.notice_shown(id, 'toast')
    assert not runner.maybe_run(state) and runner.status['notice']['seconds'] == 2
    runner.clock.sleep(1)
    assert not runner.maybe_run(state) and runner.status['notice']['seconds'] == 1
    runner.clock.sleep(.99)
    assert not runner.maybe_run(state) and not ctx.executor.history
    runner.clock.sleep(.01)
    assert runner.maybe_run(state)
    finish(runner)
    assert actions[0] - start >= 3 and len(clicks(ctx)) == 3
    assert not runner.maybe_run(state) and runner.status['notice'] is None


@pytest.mark.parametrize('why', ['cancel', 'stop', 'guard', 'mode_stop', 'game', 'round',
                                  'spectate', 'ended', 'unknown', 'combat', 'timer', 'disabled',
                                  'rounds', 'own', 'targets', 'preflight', 'self', 'toast'])
def test_notice_invalidation_never_starts_any_action(setup, why):
    ctx, runner, cap, state, sink, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    if why == 'cancel': assert runner.cancel_notice(id)
    if why == 'stop': runner.stop()
    if why == 'guard': ctx.guard.stop('急停')
    if why == 'mode_stop': runner.stopped = lambda: True
    if why == 'game': ctx.scout.reset()
    if why == 'round': state.stage = '4-2'
    if why in ('spectate', 'ended', 'unknown', 'combat'): state.phase = Phase(why)
    if why == 'timer': state.timer = 14
    if why == 'disabled': ctx.settings.scout.enabled = False
    if why == 'rounds': ctx.settings.scout.rounds = []
    if why == 'own': state.viewing = {'slot': 1}
    if why == 'targets': state.players = state.players[:1]
    if why == 'preflight': ctx.executor.problems = ['权限已失效']
    if why == 'self': state.players[0].slot = None
    if why == 'toast': sink.scout_hud_visible = lambda window=None: False
    runner.clock.sleep(3)
    assert not runner.maybe_run(state)
    assert (why == 'game' or not runner.status['notice']) and not ctx.executor.history and cap.calls == 0
    assert not runner.notice_shown(id, 'toast')
    if why not in ('round', 'game'):
        assert not runner.maybe_run(state)  # 同轮无重复预告


def test_absent_hud_timeout_and_late_ack(setup):
    ctx, runner, _, state, sink, _ = setup
    ctx.settings.scout.enabled = True
    assert not runner.maybe_run(state)  # CLI 无 WebSink 不能假装已呈现
    assert '提示窗口' in runner.status['reason'] and not ctx.executor.history
    state.stage = '4-3'
    ctx.settings.scout.rounds.append('4-3')
    sink.scout_toast_capable = lambda: True
    sink.scout_hud_visible = lambda window=None: window == 'toast'
    assert not runner.maybe_run(state)
    id = runner.status['notice']['id']
    runner.clock.sleep(NOTICE_ACK_TIMEOUT)
    assert not runner.notice_shown(id, 'toast')
    assert not runner.maybe_run(state)
    assert '未确认呈现' in runner.status['reason'] and not ctx.executor.history


def test_short_timer_never_even_announces_and_next_round_can_announce(setup):
    ctx, runner, _, state, sink, _ = setup
    ctx.settings.scout.enabled = True
    sink.scout_toast_capable = lambda: True
    sink.scout_hud_visible = lambda window=None: window == 'toast'
    state.timer = ctx.settings.scout.min_timer + 2.99
    assert not runner.maybe_run(state) and runner.status['notice'] is None
    state.timer = 20
    assert not runner.maybe_run(state) and runner.status['notice'] is None
    state.stage = '4-3'; ctx.settings.scout.rounds.append('4-3')
    assert not runner.maybe_run(state) and runner.status['notice']


def test_cancel_then_new_game_same_round_creates_new_id(setup):
    ctx, runner, _, state, _, _ = setup
    id = announce(setup)
    assert runner.cancel_notice(id)
    assert not runner.maybe_run(state) and runner.status['notice'] is None
    ctx.scout.reset()
    assert not runner.maybe_run(state)
    assert runner.status['notice']['id'] != id
    assert not runner.notice_shown(id, 'toast')


def test_cancel_in_real_thread_at_last_preflight_is_not_cleared(setup):
    ctx, runner, cap, state, _, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    runner.clock.sleep(3)
    entered, release = threading.Event(), threading.Event()
    original = ctx.executor.preflight
    def preflight():
        entered.set()
        assert release.wait(2)
        return original()
    ctx.executor.preflight = preflight
    worker = threading.Thread(target=runner.maybe_run, args=(state,))
    worker.start()
    assert entered.wait(1)
    assert runner.cancel_notice(id)
    release.set(); worker.join(3)
    assert not worker.is_alive() and not ctx.executor.history and cap.calls == 0


def test_cancel_after_start_authorization_before_run_leaves_no_cleanup_click(setup):
    ctx, runner, cap, state, sink, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    runner.clock.sleep(3)
    def changed():
        if runner._auto_permit:
            assert runner.cancel_notice(id)
    sink.on_scout_changed = changed
    runner.maybe_run(state)
    assert not ctx.executor.history and cap.calls == 0


def test_manual_does_not_require_hud_ack_or_delay(setup):
    ctx, runner, _, state, _, _ = setup
    runner.request()
    assert runner.maybe_run(state)
    finish(runner)
    assert len(clicks(ctx)) == 3 and runner.clock() < 3


def test_unacknowledged_second_window_cannot_substitute_for_hidden_ack_window(setup):
    ctx, runner, _, state, sink, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    sink.scout_hud_visible = lambda window=None: window != 'toast'
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not ctx.executor.history


def test_lost_foreground_or_window_after_ack_never_activates(setup, monkeypatch):
    from types import SimpleNamespace
    from test_scout_runner import Ex, WINDOW
    import jccbrain.modes.scout as module
    ctx, runner, _, state, _, _ = setup
    monkeypatch.setattr(module, 'MacExecutor', Ex)
    front, activated = [WINDOW.pid], []
    ctx.locator = SimpleNamespace(find=lambda _: WINDOW, activate=lambda: activated.append(True))
    runner._foreground_pid = lambda: front[0]
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    front[0] = 999
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not ctx.executor.history and not activated


@pytest.mark.parametrize('invalid', ['spectate', 'unknown', 'ended', 'combat', 'no_hud'])
def test_assistant_raw_early_return_revokes_notice_and_waiting_keeps_processing(setup, db, rules, invalid):
    from conftest import make_state
    from test_assistant import Ctx, AdviceSink
    from jccbrain.core.models import GameState
    from jccbrain.modes.assistant import AssistantMode
    ctx, runner, _, state, sink, _ = setup
    main = Ctx(db, rules, [])
    main.advisor.scout = ctx.scout
    mode = AssistantMode(main, AdviceSink()); mode.scout = runner
    id = announce(setup)
    # 正常等待期间主状态/建议仍更新，不 paused、不等待计时器。
    row = make_state(stage='4-1', timer=20, gold=30)
    assert mode._process_frame(GameState.from_dict(row)) is not None
    row['gold'] = 31
    assert mode._process_frame(GameState.from_dict(row)) is not None
    assert runner.status['notice']['id'] == id and not ctx.executor.history
    raw = GameState(phase=Phase.COMBAT) if invalid == 'no_hud' else GameState.from_dict({**row, 'phase': invalid})
    mode._process_frame(raw)
    assert runner.status['notice'] is None and not ctx.executor.history


def test_window_changed_after_ack_and_before_first_action_is_rejected(setup, monkeypatch):
    from types import SimpleNamespace
    from test_scout_runner import Ex, WINDOW
    import jccbrain.modes.scout as module
    ctx, runner, _, state, _, _ = setup
    monkeypatch.setattr(module, 'MacExecutor', Ex)
    win = [WINDOW]
    ctx.locator = SimpleNamespace(find=lambda _: win[0], activate=lambda: pytest.fail('不得激活'))
    runner._foreground_pid = lambda: WINDOW.pid
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    win[0] = replace(WINDOW, window_id=777)
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not ctx.executor.history


def test_stop_in_thread_after_permit_during_run_preflight_has_zero_cleanup(setup):
    ctx, runner, cap, state, _, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    runner.clock.sleep(3)
    entered, release = threading.Event(), threading.Event()
    def preflight():
        if runner._auto_permit:
            entered.set(); assert release.wait(2)
        return []
    ctx.executor.preflight = preflight
    worker = threading.Thread(target=runner.maybe_run, args=(state,));worker.start()
    assert entered.wait(1)
    runner.stop()
    release.set(); worker.join(3)
    assert not worker.is_alive() and runner._cancel.is_set()
    assert not ctx.executor.history and cap.calls == 0 and not runner.paused


def test_hiding_ack_window_immediately_revokes_even_if_other_unacked_window_is_open(setup):
    ctx, runner, _, state, sink, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    sink.scout_hud_visible = lambda window=None: window != 'toast'
    runner.hud_changed()
    assert runner.status['notice'] is None
    assert not runner.notice_shown(id, 'hud_panel')
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not ctx.executor.history


def test_new_eligible_round_replaces_old_notice_without_skipping_next_round(setup):
    ctx, runner, _, state, _, _ = setup
    id = announce(setup)
    assert runner.notice_shown(id, 'toast')
    ctx.settings.scout.rounds.append('4-3')
    state.stage = '4-3'
    assert not runner.maybe_run(state)
    assert runner.status['notice'] and runner.status['notice']['id'] != id
    assert runner.status['notice']['stage'] == '4-3'
    assert not runner.notice_shown(id, 'toast')
    assert not ctx.executor.history


@pytest.mark.parametrize('change',['game','disabled','phase','stage','guard'])
def test_ack_rejects_changed_authority_even_before_next_main_loop_poll(setup,change):
    ctx,runner,_,state,_,_=setup
    id=announce(setup)
    if change=='game':ctx.scout.reset()
    if change=='disabled':ctx.settings.scout.enabled=False
    if change=='phase':state.phase=Phase.COMBAT
    if change=='stage':state.stage='4-3'
    if change=='guard':ctx.guard.stop('controlled stop')
    assert not runner.notice_shown(id,'toast')
    assert not ctx.executor.history
    runner.stop()
