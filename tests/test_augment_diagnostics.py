"""Bounded evidence only; synthetic clocks/pixels are not live performance."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import copy
import json
import os

import numpy as np
import pytest

from jccbrain.vision.eye.augment_diagnostics import AugmentDiagnostics, EvidenceStore, FORMAT
from test_unit_learning_auto import auto


@pytest.fixture
def frame():
    return np.zeros((1136,1774,3),np.uint8),NS(window_id=1,width=1774,height=1136)


def data(**kw):
    return dict(stage='4-3',phase='planning',players=[{}]*8,augments_taken=[],**kw)


def state(**kw):
    d=data();d.update(kw);return d


def events(root):
    return [json.loads(p.read_text()) for p in sorted(root.glob('session-*/event-*.json'))]


def test_two_before_current_two_after_cooldown_dedupe_and_small_memory(tmp_path,frame):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diagnostics')
    for at in (0.,1.):d.observe(im,state(),win,now=at)
    d.observe(im,state(augments_taken=['A']),win,now=2.)
    # Confirmed change inside cooldown must retain its reason even when SHA hits.
    d.observe(im,state(augments_taken=['A'],phase='augment'),win,now=2.1)
    for at in (3.,4.,5.,6.):d.observe(im,state(augments_taken=['A'],phase='augment'),win,now=at)
    ev=events(d.root)[0]
    assert [f['role'] for f in ev['frames']]==['before','before','current','after','after']
    assert len(ev['frames'])==5 and ev['encoded_image_bytes']<=512*1024
    assert set(ev['reasons'])=={'confirmed_changed','enter_augment'}
    assert len(list(d.root.glob('*/roi-*.jpg')))==4  # stage/right/permanent/choice
    assert len(d.ring)==2 and 'images' not in ev
    assert d.events==1 and not d.status()['errors']


def test_1000_repeats_no_disk_scan_or_encoding_outside_event(tmp_path,frame,monkeypatch):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag')
    scan=Mock(side_effect=AssertionError('non-event disk traversal'))
    monkeypatch.setattr(EvidenceStore,'_scan',scan)
    encode=Mock(side_effect=AssertionError('non-event encoding'))
    monkeypatch.setattr('cv2.imencode',encode)
    for i in range(1000):d.observe(im,state(),win,now=i/20)
    assert d.events==0 and len(d.ring)==2 and not d.root.exists()
    assert not d.status()['errors'];scan.assert_not_called();encode.assert_not_called()
    assert all(not np.shares_memory(patch,im) and patch.shape!=im.shape
               for _,rois in d.ring for _,patch in rois)


def test_suspicion_is_persistent_unknown_not_three_augments_truth(tmp_path,frame):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag')
    for at in (0.,1.):d.observe(im,state(players=[],augments_taken=[]),win,now=at)
    assert not d.root.exists()
    d.observe(im,state(players=[],augments_taken=[]),win,now=2.)
    ev=events(d.root)[0]
    assert ev['reasons']==['unknown_owner']
    assert ev['frames'][-1]['ownership']=='unknown_owner'
    assert ev['frames'][-1]['augments_taken']==[]
    for i in range(3,1000):d.observe(im,state(players=[]),win,now=float(i))
    assert len(events(d.root))==1


def test_12_events_five_frames_and_512kib_noise_cap(tmp_path,frame):
    im,win=frame;im[:]=np.random.default_rng(1).integers(0,256,im.shape,dtype=np.uint8)
    d=AugmentDiagnostics(tmp_path/'diag')
    for i in range(30):
        d.observe(im,state(augments_taken=[str(i)]),win,now=i*12.)
    evs=events(d.root)
    assert len(evs)==12 and all(len(e['frames'])<=5 and e['encoded_image_bytes']<=512*1024 for e in evs)
    assert sum(p.stat().st_size for p in d.root.glob('session-*/*'))<=8*1024*1024
    assert d.status()['last_error']=='session_event_quota'


def test_different_noisy_preframes_really_reach_event_byte_limit(tmp_path,frame):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag');rng=np.random.default_rng(2)
    for i in range(3):
        im[:]=rng.integers(0,256,im.shape,dtype=np.uint8)
        d.observe(im,state(augments_taken=['A'] if i==2 else []),win,now=float(i))
    ev=events(d.root)[0]
    assert ev['limited']=='event_image_quota' and ev['encoded_image_bytes']<=512*1024
    assert sum(f['role']=='current' for f in ev['frames'])==1
    assert all(len(f['rois'])<=3 for f in ev['frames'])


def test_window_new_game_ended_clear_ring_and_new_random_session(tmp_path,frame):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag')
    d.observe(im,state(augments_taken=['A']),win,now=0.)
    first=d.store.session
    d.observe(im,state(phase='ended',augments_taken=['A']),win,now=.1)
    assert json.loads((first/'manifest.json').read_text())['complete'] and not d.ring
    d.observe(im,state(stage='1-1',augments_taken=['A']),win,now=1.)
    second=d.store.session;assert second!=first
    d.observe(im,state(stage='2-1',augments_taken=['A']),win,now=12.)
    d.observe(im,state(stage='1-2',augments_taken=['A']),win,now=24.)
    assert d.store.session!=second
    assert len(json.loads((d.store.session/'event-00.json').read_text())['frames'])==1
    old=d.store.session;win.window_id=2
    d.observe(im,state(stage='1-2',augments_taken=['A']),win,now=25.)
    assert d.store.session!=old
    assert d.active['frames'][0]['window_id']==2


def test_ten_sessions_only_complete_owned_oldest_removed(tmp_path,frame):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag');first=None
    for i in range(13):
        win.window_id=i+1;d.observe(im,state(augments_taken=['A']),win,now=i*20.)
        if i==0:first=d.store.session
    assert len(list(d.root.glob('session-*')))==10 and not first.exists()
    assert sum(p.stat().st_size for p in d.root.rglob('*') if p.is_file())<=100*1024*1024


@pytest.mark.parametrize('foreign',['file','symlink','session_extra','root_symlink'])
def test_never_delete_or_follow_foreign_or_symlink(tmp_path,frame,foreign):
    im,win=frame;root=tmp_path/'diag';outside=tmp_path/'outside';outside.mkdir()
    secret=outside/'keep';secret.write_text('unchanged')
    if foreign=='root_symlink':root.symlink_to(outside,target_is_directory=True)
    elif foreign in ('file','symlink'):
        root.mkdir();(root/'foreign').write_text('foreign') if foreign=='file' else (root/'foreign').symlink_to(outside,target_is_directory=True)
    else:
        d=AugmentDiagnostics(root);d.observe(im,state(augments_taken=['A']),win,now=0.)
        (d.store.session/'external').symlink_to(secret)
        win.window_id=2
    d=AugmentDiagnostics(root);d.observe(im,state(augments_taken=['A']),win,now=20.)
    assert d.status()['errors']==1 and secret.read_text()=='unchanged'
    assert (root/'foreign').is_symlink() if foreign=='symlink' else True


def test_no_overwrite_hardlink_target(tmp_path):
    root=tmp_path/'diag';store=EvidenceStore(root)
    with store.locked():store.start()
    target=tmp_path/'target';target.write_bytes(b'keep')
    os.link(target,store.session/'event-00.json')
    with pytest.raises(OSError):store._put('event-00.json',b'bad')
    assert target.read_bytes()==b'keep'
    with pytest.raises(OSError,match='filename'):store._put('../target',b'bad')
    assert target.read_bytes()==b'keep'
    os.close(store.fd);store.fd=None


def test_low_space_and_error_only_once_then_read_observation_stops(tmp_path,frame,monkeypatch):
    im,win=frame;d=AugmentDiagnostics(tmp_path/'diag')
    monkeypatch.setattr('shutil.disk_usage',lambda _:NS(free=0))
    for i in range(1000):d.observe(im,state(augments_taken=[str(i)]),win,now=i*11.)
    assert d.status()['errors']==1 and not list(d.root.glob('session-*'))


def test_incomplete_sessions_not_removed_when_ten_session_quota_is_full(tmp_path):
    root=tmp_path/'diag';stores=[]
    for _ in range(10):
        s=EvidenceStore(root)
        with s.locked():s.start()
        stores.append(s)
    other=EvidenceStore(root)
    with other.locked(),pytest.raises(OSError,match='quota'):other.start()
    assert all(s.session.exists() for s in stores)
    for s in stores:os.close(s.fd);s.fd=None


def test_session_and_global_byte_quotas_refuse_new_output(tmp_path,monkeypatch):
    from jccbrain.vision.eye import augment_diagnostics as module
    s=EvidenceStore(tmp_path/'diag')
    with s.locked():s.start()
    monkeypatch.setattr(module,'SESSION_BYTES',1000)
    with s.locked(),pytest.raises(OSError,match='byte quota'):
        s.write('roi-'+64*'a'+'.jpg',b'x'*1001)
    assert not s.files
    os.close(s.fd);s.fd=None
    monkeypatch.setattr(module,'TOTAL_BYTES',1000)
    other=EvidenceStore(tmp_path/'other')
    with other.locked(),pytest.raises(OSError,match='quota'):other.start()
    assert not list(other.root.glob('session-*'))


def test_provider_hook_failure_and_cached_read_do_not_affect_result(tmp_path,frame,monkeypatch):
    from jccbrain.vision.pyeye import PyEyeProvider
    im,win=frame;monkeypatch.setenv('JCC_HOME',str(tmp_path/'runtime'))
    p=PyEyeProvider();p.start=lambda:None;p.interval_override_s=0
    original=state(augments_taken=['A']);old=copy.deepcopy(original)
    p._capture=NS(grab=Mock(return_value=(im,win)))
    p._reader=NS(read=Mock(side_effect=lambda _:copy.deepcopy(original)),last_ocr_cells=0)
    bad=NS(observe=Mock(side_effect=OSError('diagnostic disk failure')))
    p._augment_diagnostics=bad
    result=p.read();assert result.raw['augments_taken']==old['augments_taken'] and original==old
    assert p.read(fresh=False) is result and bad.observe.call_count==1
    p.read();assert bad.observe.call_count==1 and p._capture.grab.call_count==2


def test_scout_and_learning_fast_sample_never_observe_diagnostics(tmp_path,frame,monkeypatch):
    from jccbrain.vision.pyeye import PyEyeProvider
    im,win=frame;monkeypatch.setenv('JCC_HOME',str(tmp_path/'runtime'))
    p=PyEyeProvider();p._reader=NS(read_scout=Mock(return_value={'phase':'spectate'}))
    p._augment_diagnostics=NS(observe=Mock(side_effect=AssertionError('wrong path')))
    assert p.read_scout(im)=={'phase':'spectate'}
    assert p.sample_unit_learning() is None
    p._augment_diagnostics.observe.assert_not_called()


def test_configured_learning_temporary_capture_does_not_observe_diagnostics(auto):
    e=auto;observer=NS(observe=Mock())
    e.provider._augment_diagnostics=observer
    e.provider.read();observer.observe.assert_called_once()
    observer.observe.reset_mock()
    assert e.provider.sample_unit_learning() is not None
    observer.observe.assert_not_called()


def test_failed_capture_or_parse_never_observes_diagnostics(tmp_path,frame,monkeypatch):
    from jccbrain.vision.pyeye import PyEyeProvider
    from jccbrain.vision.eye.capture import CaptureError
    from jccbrain.vision.base import VisionError
    im,win=frame;monkeypatch.setenv('JCC_HOME',str(tmp_path/'runtime'))
    p=PyEyeProvider();p.start=lambda:None;p.interval_override_s=0
    p._capture=NS(grab=Mock(side_effect=CaptureError('offline failure')))
    p._reader=NS(read=Mock(),last_ocr_cells=0);p._augment_diagnostics=NS(observe=Mock())
    with pytest.raises(VisionError):p.read()
    p._capture.grab.side_effect=None;p._capture.grab.return_value=im,win
    p._reader.read.side_effect=ValueError('main parser failed')
    with pytest.raises(ValueError):p.read()
    p._augment_diagnostics.observe.assert_not_called()


def live_diagnostic_provider(tmp_path,frame,monkeypatch,*,read=None):
    from jccbrain.vision.pyeye import PyEyeProvider
    im,win=frame;monkeypatch.setenv('JCC_HOME',str(tmp_path/'runtime'))
    p=PyEyeProvider();p.start=lambda:None;p.interval_override_s=0
    p._capture=NS(grab=Mock(return_value=(im,win)))
    p._reader=NS(read=Mock(side_effect=read or (lambda _:state(augments_taken=['A']))),last_ocr_cells=0)
    return p


def test_twelve_normal_provider_stops_allow_rotation_without_incomplete_sessions(tmp_path,frame,monkeypatch):
    root=tmp_path/'runtime/augment-diagnostics';first=None
    for i in range(12):
        p=live_diagnostic_provider(tmp_path,frame,monkeypatch)
        assert p.read().augments_taken==['A']
        d=p._augment_diagnostics;s=d.store
        if i==0:first=s.session
        assert s.fd is not None and not s.complete
        p.stop();p.stop()
        assert s.fd is None and d.closed and not d.ring and d.active is None
        assert s.complete and json.loads((s.session/'manifest.json').read_text())['complete']
        assert not d.status()['errors'] and p._augment_diagnostics is None
    markers=list(root.glob('*/manifest.json'))
    assert len(markers)==10 and all(json.loads(m.read_text())['complete'] for m in markers)
    assert not first.exists()  # Only normally completed own evidence was rotated.


def test_same_provider_stop_cached_read_fresh_read_restart_and_no_event_close(tmp_path,frame,monkeypatch):
    p=live_diagnostic_provider(tmp_path,frame,monkeypatch);last=p.read();d=p._augment_diagnostics
    reader,capture=p._reader,p._capture;first=d.store.session
    p.stop();p.stop()
    assert p._last is last and p._reader is reader and p._capture is capture
    assert p._learning_cancel.is_set()  # Original stop semantics remain cancellation.
    assert p.read(fresh=False) is last and p._augment_diagnostics is None
    p.read();new=p._augment_diagnostics
    assert new is not d and new.store.session!=first and not new.status()['errors']
    p.stop()
    empty=tmp_path/'empty'
    d=AugmentDiagnostics(empty);d.observe(frame[0],state(),frame[1],now=0.)
    assert d.ring and d.store is None
    d.close();d.close();d.observe(frame[0],state(augments_taken=['A']),frame[1],now=12.)
    assert d.closed and not d.ring and d.store is None and not empty.exists()
    never=live_diagnostic_provider(tmp_path/'never',frame,monkeypatch)
    never.stop();never.stop()
    assert not (tmp_path/'never/runtime/augment-diagnostics').exists()


@pytest.mark.parametrize('failure',['lock_open','flock'])
def test_lock_acquisition_failure_closes_every_acquired_descriptor(tmp_path,monkeypatch,failure):
    from jccbrain.vision.eye import augment_diagnostics as module
    real_open,real_close=os.open,os.close;opened=[];closed=[]
    def acquire(path,*args,**kwargs):
        if str(path)=='.lock' and failure=='lock_open':raise PermissionError('temporary .lock denial')
        fd=real_open(path,*args,**kwargs);opened.append(fd);return fd
    def release(fd):closed.append(fd);return real_close(fd)
    monkeypatch.setattr(module.os,'open',acquire);monkeypatch.setattr(module.os,'close',release)
    if failure=='flock':
        def busy(fd,mode):
            assert mode==module.fcntl.LOCK_EX|module.fcntl.LOCK_NB
            raise BlockingIOError('lock already held')
        monkeypatch.setattr(module.fcntl,'flock',busy)
    with pytest.raises(OSError):
        with EvidenceStore(tmp_path/'diag').locked():pytest.fail('lock unexpectedly acquired')
    assert opened and sorted(opened)==sorted(closed)
    for fd in opened:
        with pytest.raises(OSError):os.fstat(fd)


def test_manifest_constructor_failure_releases_session_fd_without_deleting_partial_evidence(tmp_path,monkeypatch):
    s=EvidenceStore(tmp_path/'diag');opened=[];real_open=os.open
    def acquire(path,*args,**kwargs):
        fd=real_open(path,*args,**kwargs)
        if str(path).split('/')[-1].startswith('session-'):opened.append(fd)
        return fd
    monkeypatch.setattr(os,'open',acquire)
    with s.locked():
        monkeypatch.setattr(s,'_put',Mock(side_effect=OSError('manifest initialization denied')))
        with pytest.raises(OSError):s.start()
        assert s.fd is None and s.session.exists() and not (s.session/'manifest.json').exists()
        assert len(opened)==1
        with pytest.raises(OSError):os.fstat(opened[0])


def test_finish_failure_releases_fd_retains_incomplete_evidence_and_does_not_break_stop(tmp_path,frame,monkeypatch):
    p=live_diagnostic_provider(tmp_path,frame,monkeypatch);p.read()
    d=p._augment_diagnostics;s=d.store;manifest=s.session/'manifest.json'
    monkeypatch.setattr(s,'finish',Mock(side_effect=OSError('finish denied')))
    p.stop();p.stop()
    assert s.fd is None and d.closed and d.store is None and not d.ring
    assert not json.loads(manifest.read_text())['complete'] and manifest.exists()
    assert d.status()['errors']==1 and p._augment_diagnostics is None


def test_stop_sends_cancel_before_waiting_for_inflight_read_then_closes_after_read(tmp_path,frame,monkeypatch):
    import threading
    entered,cancel_seen,release,stopped=(threading.Event() for _ in range(4))
    saved=[];errors=[]
    def blocked(_):
        entered.set()
        assert p._learning_cancel.wait(2), 'stop blocked before sending cancellation'
        cancel_seen.set()
        assert release.wait(2)
        return state(augments_taken=['A'])
    p=live_diagnostic_provider(tmp_path,frame,monkeypatch,read=blocked)
    observe=p._observe_augment_diagnostics
    def observe_and_keep(*args):
        observe(*args);saved.append((p._augment_diagnostics,p._augment_diagnostics.store))
    p._observe_augment_diagnostics=observe_and_keep
    def invoke(fn):
        try:fn()
        except Exception as exc:errors.append(exc)
    read_thread=threading.Thread(target=lambda:invoke(p.read))
    stop_thread=threading.Thread(target=lambda:(invoke(p.stop),stopped.set()))
    read_thread.start()
    try:
        assert entered.wait(2);stop_thread.start()
        assert cancel_seen.wait(2) and p._learning_cancel.is_set()
        assert not stopped.is_set()  # Still waits for successful read/observer.
    finally:
        release.set();read_thread.join(3)
        if stop_thread.ident is not None:stop_thread.join(3)
    assert not read_thread.is_alive() and not stop_thread.is_alive() and not errors
    assert len(saved)==1 and saved[0][0].closed and saved[0][1].fd is None
    assert json.loads((saved[0][1].session/'manifest.json').read_text())['complete']
    assert p._last.augments_taken==['A'] and p._capture.grab.call_count==1


def test_stop_recovers_failed_observer_latch_and_isolates_unexpected_close_error(tmp_path,frame,monkeypatch):
    p=live_diagnostic_provider(tmp_path,frame,monkeypatch)
    p._augment_diagnostics=NS(observe=Mock(side_effect=OSError('optional observer failed')),
                              close=Mock(side_effect=OSError('optional close failed')))
    assert p.read().augments_taken==['A'] and p._augment_diagnostics_failed
    p.stop();p.stop()
    assert p._augment_diagnostics is None and not p._augment_diagnostics_failed
    assert p.read().augments_taken==['A']
    d=p._augment_diagnostics;store=d.store
    assert store.fd is not None and not d.status()['errors']
    p.stop()
    assert d.closed and store.fd is None and json.loads((store.session/'manifest.json').read_text())['complete']
