"""加载时序、识别反馈和实测截图消费回归；不发送系统输入。"""
from copy import deepcopy
import json
import logging
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from jccbrain.core.models import GameState, Player, TraitStatus, Unit
from jccbrain.core.settings import Settings
from jccbrain.engine.scout import ScoutShot
from jccbrain.modes.scout import SCOUT_REGIONS, ScoutCapture, thumbnails, trait_signature, validate_shot
from test_scout_runner import WINDOW, setup, finish, clicks

FIXTURES = Path(__file__).parent / 'fixtures' / 'scout-loading'


def begin(runner, state):
    runner._initial, runner._started = state, runner.clock()
    runner._timer_value, runner._timer_at = state.timer, runner.clock()
    runner._window = None
    runner._last_mouse = None


def frame(value=100):
    return np.full((160, 240, 3), value, np.uint8)


def fill(image, key, value):
    x0,y0,x1,y1 = SCOUT_REGIONS[key]
    h,w = image.shape[:2]
    image[round(h*y0):round(h*y1), round(w*x0):round(w*x1)] = value
    return image


class Frames:
    def __init__(self, clock, images):
        self.clock, self.images = clock, iter(images)
        self.current, self.calls = None, 0
    def grab(self):
        self.clock.sleep(.02)
        self.calls += 1
        self.current = next(self.images, self.current)
        return self.current.copy(), WINDOW


def wait_sequence(runner, state, images):
    begin(runner, state)
    runner.activity.start(automatic=False)
    runner._capture = Frames(runner.clock, images)
    result = runner._wait_stable(thumbnails(frame(0)))
    return result, runner._capture.calls


def test_dwell_then_three_stable_frames(setup):
    ctx,runner,_,state,_,_ = setup
    result,calls = wait_sequence(runner,state,[frame()])
    assert result.loaded and result.timing['total_ms'] >= 400
    assert result.timing['cut_ms'] == 20 and result.timing['settle_ms'] >= 380
    assert calls >= 14 and result.timing['total_ms'] < 1200


def test_stability_resets_and_needs_three_consecutive_frames(setup):
    ctx,runner,_,state,_,_ = setup
    ctx.settings.scout.min_dwell_ms = 0
    result,calls = wait_sequence(runner,state,[frame(),frame(),frame(),frame(200),frame(200),frame(200),frame(200)])
    assert result.loaded and calls == 7


@pytest.mark.parametrize('region',['board','traits','players'])
def test_each_region_must_stabilize(setup,region):
    ctx,runner,_,state,_,_ = setup
    ctx.settings.scout.min_dwell_ms = 0
    ctx.settings.scout.switch_timeout_ms = 240
    images = [fill(frame(),region,50 if i%2 else 200) for i in range(30)]
    result,_ = wait_sequence(runner,state,images)
    assert not result.loaded and result.image is not None and result.timing['total_ms'] >= 240


def test_unchanged_traits_wait_until_timeout(setup):
    ctx,runner,_,state,_,_ = setup
    ctx.settings.scout.min_dwell_ms = 0
    ctx.settings.scout.switch_timeout_ms = 240
    result,_ = wait_sequence(runner,state,[fill(frame(),'traits',0)])
    assert not result.loaded and result.timing['cut_ms'] == 20
    assert result.timing['total_ms'] >= 240


def test_board_cut_is_required_even_if_other_areas_changed(setup):
    ctx,runner,_,state,_,_ = setup
    ctx.settings.scout.min_dwell_ms = 0
    ctx.settings.scout.switch_timeout_ms = 240
    result,_ = wait_sequence(runner,state,[frame(25)])  # ==cut_threshold不算已切换。
    assert not result.loaded and result.timing['cut_ms'] is None and result.timing['settle_ms'] is None
    assert np.array_equal(result.image,frame(25))


@pytest.mark.parametrize('viewing',[{'slot':7},{'name':'别的玩家'}])
def test_identity_mismatch_never_enters_book(setup,viewing):
    ctx,runner,_,state,sink,_ = setup
    runner._reader_factory = lambda: SimpleNamespace(read_scout=lambda _: {'viewing':viewing,'traits':[{'name':'峡谷野怪','count':5}],'board':[{'name':'绯红树怪'}]})
    runner.run(state, manual=True)
    finish(runner)
    assert not ctx.scout.latest() and any('viewing_mismatch' in line for line in sink.logs)
    assert any('身份不符，丢弃' in line for line in sink.logs)


@pytest.mark.parametrize('min_timer,expected',[(15,1),(50,0)])
@pytest.mark.parametrize('issue',['board','traits','timeout'])
def test_retry_feedback_is_appended_once_if_time_remains(setup,min_timer,expected,issue):
    ctx,runner,cap,state,_,_ = setup
    ctx.settings.scout.min_timer = min_timer
    ready = threading.Event()
    original = runner._request_retry
    def request(target):
        original(target)
        ready.set()
    runner._request_retry = request
    if issue=='traits':
        state.traits=[TraitStatus('峡谷野怪',3)]
    if issue=='timeout':
        wait=runner._wait_stable
        waits=[]
        def timed_out(*args,**kwargs):
            result=wait(*args,**kwargs)
            if not waits:result.loaded=False
            waits.append(result)
            return result
        runner._wait_stable=timed_out
    def read(image):
        if image[0,0,0] == 100:
            return {'traits':[{'name':'峡谷野怪','count':3}] if issue!='timeout' else [],
                    'board':[] if issue=='board' else [{'name':'奥恩'}]}
        return {'board':[{'name':'奥恩'}]}
    runner._reader_factory = lambda: SimpleNamespace(read_scout=read)
    def gate(_):
        if runner.status['captured']==1:
            assert ready.wait(1), '未收到首次识别反馈'
    cap.hook = gate
    runner.run(state, manual=True)
    finish(runner)
    assert [a.label for a in clicks(ctx)] == ['侦察：玩家栏第2位','侦察：玩家栏第3位'] + (['侦察：玩家栏第2位'] if expected else []) + ['侦察：玩家栏第1位']
    assert runner.status['retries'] == expected and runner.status['captured']==2
    assert runner.status['processed']==2 and runner.status['recognized']==2


def test_late_suspect_result_does_not_restart_finished_scan(setup):
    ctx,runner,_,state,_,_ = setup
    entered,release = threading.Event(),threading.Event()
    def read(_):
        entered.set()
        assert release.wait(2)
        return {'traits':[{'name':'峡谷野怪','count':5}], 'board':[]}
    runner._reader_factory = lambda: SimpleNamespace(read_scout=read)
    runner.run(state, manual=True)
    assert entered.wait(1) and not runner.scanning
    release.set()
    finish(runner)
    assert runner.status['retries']==0 and len(clicks(ctx))==3


def test_retry_uses_latest_timer_not_only_initial_timer(setup):
    ctx,runner,_,state,_,_ = setup
    begin(runner,state)
    runner._accept_retries = True
    runner._timer_value = 7
    runner._request_retry(state.players[1])
    assert not runner._retry_targets
    runner._timer_value = 8
    runner._request_retry(state.players[1])
    assert runner._next_retry() is state.players[1]
    runner._request_retry(state.players[1])
    assert runner._next_retry() is None  # 同一人最多一次。


@pytest.mark.parametrize('loaded,flags,trusted',[(False,[],False),(True,['board_not_loaded'],False),(True,[],True)])
def test_only_trusted_board_can_refine_comps_or_reduce_pool(setup,loaded,flags,trusted):
    ctx,_,_,_,_,_ = setup
    shot=ScoutShot(slot=1,round='4-1',traits=[TraitStatus('峡谷野怪',1)],level=7,
                   board=[Unit(h.name,2) for h in ctx.db.comp('104').heroes],bench=[Unit('奥恩')],
                   loaded=loaded,flags=flags,timing={'cut_ms':170,'settle_ms':520,'total_ms':690})
    row=ctx.scout.add(shot)
    assert bool(row.units)==bool(row.carries)==bool(ctx.scout.taken())==trusted
    assert row.comp_guess['by']==('units' if trusted else 'traits') and row.level==7
    saved=json.loads(ctx.scout.path.read_text().splitlines()[0])
    assert saved['loaded']==loaded and saved['flags']==flags and saved['timing']==shot.timing
    assert saved['board']  # 原记录可审计，拒绝的是消费，不是删掉原始识别。
    restored=ScoutShot.from_dict(saved)
    assert restored.loaded==loaded and restored.flags==flags and restored.timing==shot.timing
    assert row.loaded==loaded and row.flags==flags and row.timing==shot.timing


def test_six_real_images_and_recorded_results(setup):
    ctx,runner,_,state,sink,_ = setup
    records=json.loads((FIXTURES/'recorded.json').read_text())
    assert len(records)==6
    state.stage='4-3'
    state.players=[state.players[0]]+[Player(r['recorded_shot']['name'],70,False,r['recorded_shot']['slot']) for r in records]
    ctx.scout.update(state)
    begin(runner,state)
    jobs=queue.Queue()
    responses={}
    for record in records:
        image=cv2.imread(str(FIXTURES/record['image']))
        assert image is not None and image.shape[:2]==(461,720)
        raw=record['recorded_shot']
        responses[image.tobytes()]=deepcopy(raw)
        target=next(p for p in state.players if p.slot==raw['slot'])
        jobs.put((target,ScoutCapture(image,True,{'cut_ms':170,'settle_ms':520,'total_ms':690}),raw['t']))
    # 使用实际保存的识别结果隔离视觉变化；真实像素经过后台读取/保存路径。
    runner._reader_factory=lambda: SimpleNamespace(read_scout=lambda im: deepcopy(responses[im.tobytes()]))
    done=threading.Event();done.set()
    runner._recognize(jobs,done,ctx.scout.started,'4-3')
    rows={row.slot:row for row in ctx.scout.latest()}
    assert 'traits_same_as_prev' in rows[3].flags and 'traits_same_as_prev' not in rows[2].flags
    assert {slot for slot,row in rows.items() if 'board_not_loaded' in row.flags}=={1,2,3,6}
    assert rows[2].traits==rows[3].traits and rows[2].level==7 and rows[3].level==6
    assert not ctx.scout.taken()  # 4/5也只有未知?单位，不猜名字。
    assert all(row.traits for row in rows.values()) and len(ctx.scout.path.read_text().splitlines())==6
    assert any('slot3 隋毅' in line and 'traits_same_as_prev' in line for line in sink.logs)


def test_scout_info_reaches_jcc_log(setup):
    from jccbrain.cli import setup_logging
    ctx,runner,_,state,_,_=setup
    root=logging.getLogger()
    existing,old_level=set(root.handlers),root.level
    try:
        setup_logging()
        runner.run(state, manual=True)
        finish(runner)
        for handler in root.handlers:handler.flush()
        from jccbrain.core.paths import log_dir
        text=(log_dir()/'jcc.log').read_text()
        assert 'INFO jccbrain.scout:' in text
        assert 'slot1 甲：cut ' in text and '/ settle ' in text and '/ loaded / flags=none' in text
        assert '侦察 4-1：截到 2/2' in text and '后台识别' in text and '补截 0' in text
    finally:
        for handler in set(root.handlers)-existing:
            root.removeHandler(handler);handler.close()
        root.setLevel(old_level)


def test_loading_settings_are_editable_and_validated(db):
    from jccbrain.webui.settings_schema import _prepare,schema
    settings=Settings()
    assert settings.scout.min_dwell_ms==350 and settings.scout.stable_frames==3 and settings.scout.switch_timeout_ms==1200
    group=next(s for s in schema(settings,db,full=True) if s['key']=='scout')
    assert {'min_dwell_ms','stable_frames'} <= {f['key'] for f in group['fields']}
    changed=_prepare(settings,{'scout':{'min_dwell_ms':200,'stable_frames':4}},db)
    assert changed.scout.min_dwell_ms==200 and changed.scout.stable_frames==4
    for values in [{'stable_frames':0},{'stable_frames':1.5},{'min_dwell_ms':-1},{'min_dwell_ms':1200}]:
        with pytest.raises(ValueError):_prepare(settings,{'scout':values},db)
