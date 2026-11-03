"""Semantic AI lifetime through real mode/cache/API; no network or OS input."""
from dataclasses import replace
import threading
import time

import pytest

from test_assistant_buy_projection import running_api, isolated, raw
from test_webui_api import setup_api
from jccbrain.engine.llm import LLMReply
from jccbrain.webui import serialize
from jccbrain.webui.events import EventBus

OPTIONS = [{'name': name, 'desc': '测试说明'} for name in ['选项甲', '选项乙', '选项丙']]


def frame(mode, n=1, **kw):
    changes = dict(phase='augment', shop={'open': False}, augments=OPTIONS)
    changes.update(kw)
    for _ in range(n):
        mode._process_frame(raw(**changes))


def answer(api, ctx, mode, text='选项乙'):
    ctx.llm.advise = lambda *_: LLMReply(text, [], '')
    mode._ask_llm(api._last_state, mode._latest)
    return api._ai_projection


def test_unchanged_reads_timer_and_coordinates_keep_one_cached_reply(running_api):
    api, ctx, _, mode = running_api
    board = [{'name':'绯红树怪','star':2,'x':.2,'y':.4}]
    frame(mode, 4, board=board, board_count=[1,6])
    calls = []
    ctx.llm.advise = lambda *_: calls.append(1) or LLMReply('选项乙', [], '')
    mode._ask_llm(api._last_state, mode._latest)
    projection = api._ai_projection
    assert projection.valid and projection.augment == '选项乙'
    for i in range(20):
        frame(mode, timer=30-i, board=[{**board[0],'x':.2+i*.01,'y':.4}], board_count=[1,6])
        assert api._ai_projection is projection and mode._latest.llm_text == '选项乙'
        assert mode._latest.ai_augment == '选项乙'
    assert calls == [1]


def test_delayed_request_accepts_after_unchanged_reads_without_relabel(running_api):
    api, ctx, _, mode = running_api
    frame(mode, 4)
    entered, release = threading.Event(), threading.Event()
    def reply(*_):
        entered.set(); assert release.wait(3)
        return LLMReply('选项丙', [], '')
    ctx.llm.advise = reply
    st, adv = api._last_state, mode._latest
    binding = mode._ai_frames[(adv.buy_source.input_epoch, adv.buy_source.frame)]
    worker = threading.Thread(target=mode._ask_llm, args=(st, adv)); worker.start(); assert entered.wait(1)
    frame(mode, 5)
    assert mode._latest.buy_source.frame > adv.buy_source.frame
    release.set(); worker.join(3)
    assert not worker.is_alive() and api._ai_projection.valid
    assert api._ai_projection.source.revision == binding.revision
    assert api._ai_projection.augment == '选项丙'


@pytest.mark.parametrize('change', ['input', 'options', 'aba', 'gold', 'phase', 'new_game', 'ended', 'stop', 'ownership', 'selection'])
def test_late_reply_cannot_cross_real_semantic_change(running_api, change):
    api, ctx, _, mode = running_api
    frame(mode, 4)
    entered, release = threading.Event(), threading.Event()
    def reply(*_):
        entered.set(); assert release.wait(3)
        return LLMReply('old-context', [], '')
    ctx.llm.advise = reply
    old_state, old_advice = api._last_state, mode._latest
    before = mode._ai.projection.source.revision
    worker = threading.Thread(target=mode._ask_llm, args=(old_state, old_advice));worker.start();assert entered.wait(1)
    if change == 'input': mode._input_changed('fixture', False)
    elif change in ('options', 'aba'):
        frame(mode, augments=[{'name':'短暂新选项'}])
        assert not api._ai_projection.valid  # raw-first, even if stable options stay old
        if change == 'aba': frame(mode)
    elif change == 'gold': frame(mode, gold=1)
    elif change == 'phase': frame(mode, phase='combat')
    elif change == 'new_game': mode._on_new_game()
    elif change == 'ended': frame(mode, phase='ended')
    elif change == 'stop': mode.stop_event.set()
    elif change == 'ownership': frame(mode, viewing={'self':False,'slot':2})
    else:
        ctx.game_lock_id=''; assert mode.refresh_selection()
    assert mode._ai.projection.source.revision > before
    release.set(); worker.join(3); assert not worker.is_alive()
    assert not api._ai_projection.valid
    assert mode._latest is None or 'old-context' not in (mode._latest.llm_text or '')
    if change in ('input', 'options', 'aba', 'gold', 'phase', 'ownership'):
        frame(mode, 5)
        fresh = answer(api, ctx, mode, 'new-context')
        assert fresh.valid and fresh.text == 'new-context'


def test_ai_revoke_does_not_wait_for_mode_or_api_state_locks(running_api):
    api, ctx, _, mode = running_api
    frame(mode, 4);assert answer(api,ctx,mode).valid
    with mode._lock, api._lock, api._state_lock:
        done=threading.Event()
        t=threading.Thread(target=lambda:(mode._input_changed('fixture',False),done.set()))
        t.start();assert done.wait(.1)
        assert not api._ai_projection.valid
    t.join(1);assert not t.is_alive()


def test_api_and_bus_fence_old_projection_on_reconnect(running_api):
    api, ctx, _, mode = running_api
    frame(mode, 4);old=answer(api,ctx,mode)
    mode._input_changed('fixture',False);cleared=api._ai_projection
    assert api._on_ai_projection(old) is False and api._ai_projection is cleared
    events=[];bus=EventBus(events.append)
    try:
        bus.emit('ai_projection',serialize.ai_projection(old))
        bus.emit('ai_projection',serialize.ai_projection(cleared))
        bus.emit('ai_projection',serialize.ai_projection(old))
        bus.detach('main');bus.attach(events.append);time.sleep(.15)
        assert events and all('old-context' not in row for row in events)
        assert all('"valid": true' not in row for row in events)
    finally:bus.close()


def test_stale_raw_revoke_cannot_roll_input_epoch_back(running_api):
    api, ctx, _, mode = running_api
    frame(mode, 4);assert answer(api,ctx,mode).valid
    old_epoch=mode.buy_input_epoch;original=mode._ai.revoke;once=[True]
    def interleave(**kw):
        if once[0]:
            once[0]=False
            mode._input_changed('input-between-raw-check-and-revoke',False)
        return original(**kw)
    mode._ai.revoke=interleave
    mode._ai.observe_raw('changed-before-input',usable=True,input_epoch=old_epoch)
    mode._ai.revoke=original
    assert mode._ai.projection.source.input_epoch==mode.buy_input_epoch==old_epoch+1
    frame(mode,5);assert answer(api,ctx,mode,'recovered').valid


def test_delayed_nonpurchase_advice_delivery_uses_semantics_not_sample_frame(running_api):
    import json
    api, ctx, _, mode = running_api
    frame(mode, 4); answer(api,ctx,mode)
    original = api._serialized_advice(api._last_advice)
    original_frame = original['buy_source']['frame']
    frame(mode, 6)
    latest = serialize.buy_projection(api._buy_projection)
    assert latest['source']['frame'] > original_frame
    messages=[];bus=EventBus(messages.append)
    try:
        # A full snapshot queued before six samples keeps its ORIGINAL frame.
        with bus._lock:
            bus.emit('advice',original)
            bus.emit('buy_projection',latest)
            bus.emit('ai_projection',serialize.ai_projection(api._ai_projection))
        time.sleep(.22)
        events=[json.loads(s[len('window.__jcc_emit('):-1]) for s in messages]
        assert events[0]['type']=='ai_projection'
        assert any(e['type']=='advice' and e['payload']['buy_source']['frame']==original_frame for e in events)
        # Repeated context A can reuse strategy data; its old reply cannot return.
        mode._input_changed('revoke-old-full',False)
        bus.emit('buy_projection',serialize.buy_projection(api._buy_projection))
        bus.emit('ai_projection',serialize.ai_projection(api._ai_projection))
        messages.clear();bus._send('advice',original)
        assert not messages
        frame(mode,5);answer(api,ctx,mode,'fresh-epoch')
        fresh=api._serialized_advice(api._last_advice)
        frame(mode,6)
        with bus._lock:
            bus.emit('advice',fresh)
            bus.emit('ai_projection',serialize.ai_projection(api._ai_projection))
            bus.emit('buy_projection',serialize.buy_projection(api._buy_projection))
        time.sleep(.22)
        events=[json.loads(s[len('window.__jcc_emit('):-1]) for s in messages]
        assert any(e['type']=='advice' and e['payload']['ai_source']['input_epoch']==fresh['ai_source']['input_epoch'] for e in events)
    finally:bus.close()
