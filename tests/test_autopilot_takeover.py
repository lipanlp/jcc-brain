"""真实托管loop，假视觉序列与演练执行器，无系统输入。"""
from types import SimpleNamespace

import pytest

from conftest import make_state
from jccbrain.actuator.actions import BuyShop, SellBench, BuyXP, Reroll, BenchToBoard, PickAugment
from jccbrain.core.models import GameState, Phase
from jccbrain.modes.autopilot import AutopilotMode
from jccbrain.modes.base import ModeSink
from test_autoroll import FakeCtx, ScriptedVision

ALL = [BuyShop(0), SellBench(0), BuyXP(), Reroll(), BenchToBoard(0, 1, 1)]


def frame(phase='planning', *, number=None, stage='2-1', **kw):
    return GameState.from_dict(make_state(phase=phase, stage=stage, gold=50, level=5,
        viewing={'slot': 0}, players=[{'name': 'me', 'hp': 90, 'self': True, 'slot': 0}],
        shop={'open': True}, frame=number, **kw))


def augment(**kw):
    return frame('augment', augments=[{'name': 'A'}, {'name': 'B'}, {'name': 'C'}], **kw)


class Sink(ModeSink):
    def __init__(self):
        self.changes = []
        self.callback = lambda: None
    def on_mode_changed(self):
        self.changes.append(self.mode.status)
        self.callback()


def build(db, rules, monkeypatch, frames, *, selection='manual', before_read=None):
    vision = ScriptedVision(frames + [frame('ended')])
    ctx = FakeCtx(db, rules, vision)
    ctx.settings.autopilot.augment_selection = selection
    ctx.game_watch_ids, ctx.game_lock_id, ctx.game_augments = [], '', []
    ctx.effective_lock_id = lambda: None
    ctx.enabled_comp = lambda key: None
    # 保留 Mode.compute_advice/GameWatcher 链，只替换纯建议计算以控制动作。
    advice = SimpleNamespace(headline='fake', target=None)
    ctx.advisor = SimpleNamespace(advise=lambda *a, **k: advice, resolve_augment=lambda name: name)
    plans = []
    def plan(st, adv):
        plans.append(st)
        return [PickAugment(0)] if st.phase == Phase.AUGMENT else ALL
    ctx.planner = SimpleNamespace(plan=plan)
    sink = Sink()
    mode = AutopilotMode(ctx, sink)
    sink.mode = mode
    monkeypatch.setattr(mode, 'sleep', lambda s: None)
    original = vision.read
    def read(fresh=True):
        if before_read:
            before_read(vision.read_count, mode)
        return original(fresh)
    monkeypatch.setattr(vision, 'read', read)
    return mode, ctx, sink, plans


def test_manual_default_pauses_all_actions_and_never_auto_resumes(db, rules, monkeypatch):
    mode, ctx, sink, plans = build(db, rules, monkeypatch, [augment(), frame(), frame(), frame()])
    assert mode.augment_selection == 'manual'
    assert '对局结束' in mode.loop()
    assert ctx.executor.history == [] and plans == []
    assert any(s['can_resume'] for s in sink.changes)
    assert not mode.status['paused'] and mode.status['token'] is None


@pytest.mark.parametrize('bad', ['options', 'one', 'foreign', 'units_stale', 'players_stale', 'unknown', 'view_unknown', 'gold_unknown', 'same_frame'])
def test_cannot_resume_until_two_valid_clear_own_frames(db, rules, monkeypatch, bad):
    invalid = frame()
    if bad == 'options': invalid.augments = augment().augments
    if bad == 'foreign': invalid.viewing = {'slot': 1}
    if bad == 'units_stale': invalid.units_stale = True
    if bad == 'players_stale': invalid.players_stale = True
    if bad == 'unknown': invalid.phase = Phase.UNKNOWN
    if bad == 'view_unknown': invalid.viewing = None
    if bad == 'gold_unknown': invalid.gold = None
    if bad == 'same_frame': invalid.frame = 99
    seq = [augment(), frame(), invalid]
    if bad == 'one': seq = [augment(), frame()]
    if bad == 'same_frame': seq = [augment(), invalid, invalid]
    failures = []
    def before(index, mode):
        if index == len(seq):
            with pytest.raises(ValueError, match='两帧'):
                mode.resume(mode.status['token'])
            failures.append(True)
    mode, ctx, _, _ = build(db, rules, monkeypatch, seq, before_read=before)
    mode.loop()
    assert failures and not ctx.executor.history


def test_two_frames_still_need_explicit_confirmation_and_token_is_once(db, rules, monkeypatch):
    confirmations = []
    def before(index, mode):
        if index == 1:
            with pytest.raises(ValueError): mode.resume(mode.status['token'])
        if index == 3:
            assert mode.status['can_resume']
            token = mode.status['token']
            mode.resume(token)
            confirmations.append(True)
            with pytest.raises(ValueError): mode.resume(token)
    mode, ctx, _, _ = build(db, rules, monkeypatch, [augment(), frame(), frame(), frame()], before_read=before)
    assert '对局结束' in mode.loop()
    assert confirmations and ctx.executor.history == ALL


def test_round_change_invalidates_old_token_and_setting_only_next_start(db, rules, monkeypatch):
    tokens = []
    def before(index, mode):
        if index == 1:
            tokens.append(mode.status['token'])
            mode.ctx.settings.autopilot.augment_selection = 'auto'
        if index == 4:
            assert mode.status['can_resume']
            assert mode.status['token'] != tokens[0]
            with pytest.raises(ValueError): mode.resume(tokens[0])
            mode.resume(mode.status['token'])
    mode, ctx, _, _ = build(db, rules, monkeypatch,
        [augment(), frame(stage='2-2'), frame(stage='2-2'), frame(stage='2-2'), frame(stage='2-2')], before_read=before)
    mode.loop()
    assert mode.augment_selection == 'manual' and ctx.executor.history == ALL


@pytest.mark.parametrize('stop', ['event', 'guard'])
def test_stop_or_emergency_never_resumes(db, rules, monkeypatch, stop):
    def before(index, mode):
        if index == 3:
            if stop == 'event': mode.stop_event.set()
            else: mode.ctx.guard.stop('测试急停')
            with pytest.raises(ValueError, match='停止'): mode.resume(mode.status['token'])
    mode, ctx, _, _ = build(db, rules, monkeypatch, [augment(), frame(), frame(), frame()], before_read=before)
    assert mode.loop() == '已停止'
    assert not ctx.executor.history and not mode.status['paused']


def test_auto_picks_once_through_option_and_phase_jitter_then_next_session(db, rules, monkeypatch):
    jitter = frame('unknown', stage='2-2')
    mode, ctx, _, _ = build(db, rules, monkeypatch,
        [augment(), augment(), jitter, augment(), frame(), augment(), frame(), frame(), augment(stage='3-2'), augment(stage='3-2')], selection='auto')
    mode.loop()
    assert ctx.executor.history == [PickAugment(0), *ALL, PickAugment(0)]


def test_new_game_clears_pause_and_old_authorization(db, rules, monkeypatch):
    mode, ctx, _, _ = build(db, rules, monkeypatch, [])
    mode._observe_augments(augment())
    old = mode.status['token']
    mode._on_new_game()
    assert not mode.status['paused'] and mode.status['token'] is None
    with pytest.raises(ValueError): mode.resume(old)


def test_new_game_detected_by_actual_advice_loop_clears_old_pause(db, rules, monkeypatch):
    mode, ctx, sink, _ = build(db, rules, monkeypatch,
        [augment(stage='5-2'), frame(stage='1-1'), frame(stage='1-1'), frame(stage='1-1')])
    mode.loop()
    assert ctx.executor.history == ALL
    assert any(s['paused'] for s in sink.changes)


def test_new_read_in_progress_revokes_old_continue_readiness(db, rules, monkeypatch):
    mode, ctx, _, _ = build(db, rules, monkeypatch, [frame()])
    mode._observe_augments(augment())
    mode._observe_augments(frame())
    mode._observe_augments(frame())
    token = mode.status['token']
    assert mode.status['can_resume']
    mode.read()  # 新鲜帧尚未由状态机验证；旧的两帧授权不可使用。
    assert not mode.status['can_resume']
    with pytest.raises(ValueError): mode.resume(token)


@pytest.mark.parametrize('invalid', ['foreign', 'unknown'])
def test_automatic_does_not_pick_from_invalid_ownership(db, rules, monkeypatch, invalid):
    st = augment()
    st.viewing = {'slot': 1} if invalid == 'foreign' else None
    mode, ctx, _, _ = build(db, rules, monkeypatch, [st, st], selection='auto')
    mode.loop()
    assert not ctx.executor.history
