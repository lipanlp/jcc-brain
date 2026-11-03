"""每回合调度只改变回合集合；全程假执行器/假截图，无真实输入。"""
from dataclasses import replace
import threading

import pytest

from jccbrain.core.models import Phase
from test_scout_runner import setup, finish, clicks  # noqa: F401
from test_webui_api import setup_api, Runner, data  # noqa: F401
from jccbrain.modes.assistant import AssistantMode


def non_key(setup):
    ctx,runner,_,state,sink,_=setup
    ctx.settings.scout.enabled=True
    state.stage='4-3'
    ctx.scout.update(state)
    sink.scout_toast_capable=lambda:True
    sink.scout_hud_visible=lambda window=None: window=='toast'
    assert state.stage not in ctx.settings.scout.rounds
    return ctx,runner,state,sink


def test_every_round_non_key_announces_only_after_hud_ack_full_three_seconds(setup):
    ctx,runner,state,_=non_key(setup)
    assert not runner.maybe_run(state) and runner.status['notice'] is None
    ctx.settings.scout.schedule='every_round'
    assert not runner.maybe_run(state)
    notice=runner.status['notice'];assert notice and notice['stage']=='4-3'
    runner.clock.sleep(2)
    assert not runner.maybe_run(state) and not ctx.executor.history
    assert runner.notice_shown(notice['id'],'toast')
    runner.clock.sleep(2.99)
    assert not runner.maybe_run(state) and not ctx.executor.history
    runner.clock.sleep(.01)
    assert runner.maybe_run(state)
    finish(runner)
    assert len(clicks(ctx)) == 3
    assert not runner.maybe_run(state) and not runner.status['notice']
    state.stage='4-4';ctx.scout.update(state)
    assert not runner.maybe_run(state) and runner.status['notice']


@pytest.mark.parametrize('why',['short_timer','cancel','game','round','spectate','unknown','ended','combat',
    'foreign','players_stale','no_self','hud','guard','disabled','invalid_round','preflight'])
def test_every_round_preserves_automatic_safety_boundaries(setup,why):
    ctx,runner,state,sink=non_key(setup)
    ctx.settings.scout.schedule='every_round'
    if why=='short_timer': state.timer=17.99
    assert not runner.maybe_run(state)
    notice=runner.status['notice']
    if why=='short_timer':
        assert notice is None and not ctx.executor.history
        return
    assert notice and runner.notice_shown(notice['id'],'toast')
    if why=='cancel': assert runner.cancel_notice(notice['id'])
    if why=='game':ctx.scout.reset()
    if why=='round':state.stage='4-4'
    if why in ('spectate','unknown','ended','combat'):state.phase=Phase(why)
    if why=='foreign':state.viewing={'slot':1}
    if why=='players_stale':state.players_stale=True
    if why=='no_self':state.players[0].is_self=False
    if why=='hud':sink.scout_hud_visible=lambda window=None:False
    if why=='guard':ctx.guard.stop('急停')
    if why=='disabled':ctx.settings.scout.enabled=False
    if why=='invalid_round':state.stage='4-99'
    if why=='preflight':ctx.executor.problems=['权限不可用']
    runner.clock.sleep(3)
    assert not runner.maybe_run(state)
    assert not ctx.executor.history
    assert not runner.notice_shown(notice['id'],'toast')
    if why not in ('round','game'):
        assert not runner.maybe_run(state) and runner.status['notice'] is None


def test_manual_remains_available_with_auto_disabled_but_rejects_foreign_or_stale_input(setup):
    ctx,runner,_,state,_,_=setup
    assert not ctx.settings.scout.enabled
    for invalid in [replace(state,viewing={'slot':1}),replace(state,players_stale=True),replace(state,stage=None)]:
        runner.request()
        assert not runner.maybe_run(invalid) and not ctx.executor.history
    runner.request()
    assert runner.maybe_run(state)
    finish(runner)
    assert len(clicks(ctx))==3 and runner.clock()<3


def test_saving_schedule_invalidates_old_permit_and_next_round_recovers(setup,setup_api):
    ctx,runner,state,sink=non_key(setup)
    api,main,_=setup_api
    # API 持久设置与假扫描器共用；运行器不创建线程，不访问游戏。
    ctx.settings=main.settings
    ctx.settings.scout.enabled=True;ctx.settings.scout.schedule='every_round'
    api._runner=Runner();mode=AssistantMode(main,api._sink);mode.scout=runner;api._runner.start(mode)
    assert not runner.maybe_run(state)
    old=runner.status['notice']['id'];assert runner.notice_shown(old,'toast')
    data(api.save_feature_settings({'scout':{'schedule':'key_rounds'}}))
    assert api._runner.mode is mode and runner.status['notice'] is None and runner._cancel.is_set()
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not runner.notice_shown(old,'toast') and not ctx.executor.history
    data(api.save_feature_settings({'scout':{'schedule':'every_round'}}))
    assert not runner.maybe_run(state) and runner.status['notice'] is None  # 同轮不可重新授权。
    state.stage='4-4';ctx.scout.update(state)
    assert not runner.maybe_run(state)
    fresh=runner.status['notice']['id'];assert fresh!=old and not runner._cancel.is_set()
    assert runner.notice_shown(fresh,'toast')
    runner.clock.sleep(3)
    assert runner.maybe_run(state)
    finish(runner)
    assert len(clicks(ctx))==3


def test_disabling_auto_during_scanning_discards_late_results_and_never_starts_second_scan(setup,setup_api):
    ctx,runner,state,sink=non_key(setup)
    api,main,_=setup_api
    ctx.settings=main.settings;ctx.settings.scout.enabled=True;ctx.settings.scout.schedule='every_round'
    api._runner=Runner();mode=AssistantMode(main,api._sink);mode.scout=runner;api._runner.start(mode)
    assert not runner.maybe_run(state)
    notice=runner.status['notice']['id'];assert runner.notice_shown(notice,'toast')
    runner.clock.sleep(3)
    changed=[]
    def disable(action,real):
        if not changed:
            changed.append(action)
            data(api.save_feature_settings({'scout':{'enabled':False}}))
            assert not runner.maybe_run(state)
    ctx.executor.listeners.append(disable)
    runner.maybe_run(state);finish(runner)
    assert changed and not runner.status['scanning'] and not runner.status['processing']
    assert not ctx.scout.latest() and runner._cancel.is_set()
    count=len(ctx.executor.history)
    assert not runner.maybe_run(state) and len(ctx.executor.history)==count


def test_raw_foreign_input_publishes_unavailable_without_reusing_last_own_state(setup,setup_api):
    ctx,runner,state,sink=non_key(setup)
    api,main,bus=setup_api
    ctx.settings=main.settings;api._runner=Runner()
    mode=AssistantMode(main,api._sink);mode.scout=runner;api._runner.start(mode)
    runner.sink=api._sink;api._last_state=state
    runner.observe(state)
    assert data(api.status())['scout_unavailable_reason']==''
    raw=replace(state,viewing={'slot':1})
    runner.observe(raw)
    assert data(api.status())['scout_unavailable_reason']=='请先返回自己的棋盘'
    assert not api.scout_now()['ok'] and not runner.status['pending']
    assert any(kind=='status' and payload['scout_unavailable_reason']=='请先返回自己的棋盘' for kind,payload in bus.events)


@pytest.mark.parametrize('why',['window','foreground'])
def test_every_round_window_or_foreground_loss_rejects_old_authorization(setup,monkeypatch,why):
    from types import SimpleNamespace
    from test_scout_runner import Ex, WINDOW
    import jccbrain.modes.scout as module
    ctx,runner,state,sink=non_key(setup)
    ctx.settings.scout.schedule='every_round'
    monkeypatch.setattr(module,'MacExecutor',Ex)
    win=[WINDOW];front=[WINDOW.pid]
    ctx.locator=SimpleNamespace(find=lambda _:win[0],activate=lambda:pytest.fail('自动侦察不得抢焦点'))
    runner._foreground_pid=lambda:front[0]
    assert not runner.maybe_run(state)
    notice=runner.status['notice']['id'];assert runner.notice_shown(notice,'toast')
    if why=='window':win[0]=replace(WINDOW,window_id=999)
    else:front[0]=999
    runner.clock.sleep(3)
    assert not runner.maybe_run(state) and not ctx.executor.history
    assert runner.status['notice'] is None
