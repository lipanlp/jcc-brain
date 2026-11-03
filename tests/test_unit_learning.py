"""Learning contracts: no self-labels, verifiable captures, atomic evidence and actual matching."""
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from jccbrain.vision.eye import units as units_module
from jccbrain.vision.eye.ocr import Line
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.unit_learning import (ClickTransaction, FrameSource, StoreCorrupt,
    UnitLearningController, UnitLearningStore)
from jccbrain.vision.eye.unit_panel import PanelObservation, PanelReader, bind_selection
from jccbrain.vision.eye.units import Bar, Gallery, TrackedUnit, UnitTracker

POOL = {'甲', '乙'}
IMAGE = np.zeros((120,180,3),np.uint8)
VEC = np.array([1.,0.],np.float32)


def store_at(path, **kw):
    return UnitLearningStore(path,'s',POOL,**kw)


def pending(store, name='甲', fill=0, **kw):
    image = np.full((12,12,3),fill,np.uint8)
    return store.record(dict(source='panel',name=name,cost=1,traits=['法师'],confirmation=True,
                             reason='two_fresh_consistent_panels',**kw),{'unit':image,'panel':image})


def worker(path, i):
    store = store_at(path)
    pending(store, fill=i)


def test_queue_dedup_bounds_do_not_delete_verified_evidence(tmp_path):
    store=store_at(tmp_path,max_pending=2)
    sid=pending(store)
    assert pending(store)==sid
    answer=store.promote(sid,VEC)
    assert answer['status']=='verified'
    evidence=store.verified/answer['id']/'evidence.json'
    old=evidence.read_bytes()
    for i in range(1,12):pending(store,fill=i)
    assert len(store.entries('review'))==2
    assert evidence.read_bytes()==old
    assert len(store.active_vectors())==1


def test_queue_disk_cap(tmp_path):
    store=store_at(tmp_path,max_pending=20,max_bytes=1000)
    for i in range(10):pending(store,fill=i)
    total=sum(p.stat().st_size for r in store.entries('review') for p in (store.queue/r['id']).rglob('*') if p.is_file())
    assert total<=1000


def test_atomic_index_failure_never_exposes_half_verified_entry(tmp_path,monkeypatch):
    store=store_at(tmp_path);sid=pending(store)
    original=store._save
    def fail(directory,rows):
        if directory==store.verified:raise OSError('interrupted before commit')
        return original(directory,rows)
    monkeypatch.setattr(store,'_save',fail)
    with pytest.raises(OSError):store.promote(sid,VEC)
    assert store.active_vectors()==[]
    assert list(store.verified.glob('*/vector.npy'))  # orphan is evidence, not a valid entry


def test_corrupt_index_cannot_erase_verified_history(tmp_path):
    store=store_at(tmp_path);sid=pending(store);store.promote(sid,VEC)
    index=store.verified/'index.json';index.write_text('{broken')
    assert store.active_vectors()==[]
    with pytest.raises(StoreCorrupt):store.promote(pending(store,fill=2),VEC)
    assert index.read_text()=='{broken'


def test_tampered_vector_or_evidence_is_not_loaded(tmp_path):
    store=store_at(tmp_path);sid=store.promote(pending(store),VEC)['id']
    (store.verified/sid/'vector.npy').write_bytes(b'bad')
    assert store.active_vectors()==[]


def test_process_lock_preserves_concurrent_records(tmp_path):
    ctx=multiprocessing.get_context('spawn')
    processes=[ctx.Process(target=worker,args=(str(tmp_path),i)) for i in range(6)]
    for p in processes:p.start()
    for p in processes:p.join(20);assert p.exitcode==0
    store=store_at(tmp_path)
    assert len(store.entries('review'))==6
    assert len({r['id'] for r in store.entries('review')})==6


def test_capacity_and_feature_conflict_preserve_names(tmp_path):
    store=store_at(tmp_path)
    assert store.promote(pending(store),VEC)['status']=='verified'
    ans=store.promote(pending(store,name='乙',fill=2),VEC)
    assert ans['status']=='conflict'
    assert store.entries('review')[-1]['status']=='conflict'
    assert [n for n,_ in store.active_vectors()]==['甲']
    for fill in (3,4):assert store.promote(pending(store,fill=fill),VEC)['status']=='verified'
    assert store.promote(pending(store,fill=5),VEC)['reason']=='hero_capacity'
    # A candidate outside MATCH_THRESHOLD isn't a false identity conflict.
    s2=store_at(tmp_path/'other')
    assert s2.promote(pending(s2,name='乙'),np.array([1.,.65],np.float32),existing=[('甲',VEC)])['status']=='verified'


@pytest.mark.parametrize('source,confirmed,status', [('record',True,'pending'),('legacy',True,'pending'),
    ('panel',False,'pending'),('panel',True,'rejected')])
def test_only_confirmed_allowed_provenance_can_promote(tmp_path,source,confirmed,status):
    store=store_at(tmp_path)
    sid=store.record(dict(source=source,name='甲',confirmation=confirmed),{'unit':IMAGE,'panel':IMAGE},status=status)
    assert store.promote(sid,VEC)['status']=='rejected'
    assert store.active_vectors()==[]


def test_actual_tracker_consumes_new_vectors_and_revoke_hot_reloads(tmp_path):
    store=store_at(tmp_path/'learn');gallery=Gallery(tmp_path/'gallery','s',POOL,verified_store=store)
    assert gallery.match(VEC) is None
    sid=store.promote(pending(store),VEC)['id']
    tracker=UnitTracker(gallery)
    u=TrackedUnit('board',None,.4,.3,1,fp=VEC,fp_prepared=True,crop=IMAGE)
    out=tracker.observe(IMAGE,'planning',None,prepared=({},[u]))
    assert out['board'][0]['name']=='甲'
    assert gallery.names=={'甲':1}
    assert store.revoke(sid)
    assert gallery.match(VEC) is None and gallery.names=={}
    assert (store.verified/sid/'evidence.json').exists()
    assert Gallery(tmp_path/'wrong','other',POOL,verified_store=store).match(VEC) is None


def test_running_tracker_cached_unknown_and_groups_hot_reload_then_revoke(tmp_path):
    store=store_at(tmp_path/'learning');gallery=Gallery(tmp_path/'gallery','s',POOL,verified_store=store)
    tracker=UnitTracker(gallery)
    def current():
        return TrackedUnit('board',None,.4,.3,1,fp=VEC,fp_prepared=True,crop=IMAGE,hist=np.array([1.,0.]))
    tracker.observe(IMAGE,'planning',None,prepared=({},[current()]))
    sid=store.promote(pending(store),VEC)['id']
    answer=tracker.observe(IMAGE,'planning',None,prepared=({},[current()]))
    assert answer['board'][0]['name']=='甲'
    assert tracker.group_name  # previously remembered name cannot rescue revocation
    store.revoke(sid)
    answer=tracker.observe(IMAGE,'planning',None,prepared=({},[current()]))
    assert answer['board'][0]['name'].startswith('?') and not tracker.group_name


def test_legacy_is_not_automatically_promoted_and_manual_source_is_explicit(tmp_path):
    g=Gallery(tmp_path,'s',POOL);g.add('甲',VEC,None)
    assert g.provenance()[0]['source']=='manual'
    index=g.dir/'index.json';rows=json.loads(index.read_text());rows[0].pop('source');index.write_text(json.dumps(rows))
    old=index.read_bytes();again=Gallery(tmp_path,'s',POOL)
    assert again.match(VEC)==('甲',0.)
    assert again.provenance()[0]['source']=='legacy' and index.read_bytes()==old


def test_capture_receipts_distinguish_paused_new_capture_and_reused_frame(monkeypatch):
    source=FrameSource();times=iter([1.,1.3]);monkeypatch.setattr('jccbrain.vision.eye.unit_learning.time.monotonic',lambda:next(times))
    calls=[]
    def grab():
        calls.append(True)
        return IMAGE,SimpleNamespace(window_id=7)
    capturer=SimpleNamespace(grab=grab)
    _,_,ticket=source.grab(capturer);a=source.frame_from_capture(ticket,'planning')
    with pytest.raises(ValueError):source.frame_from_capture(ticket,'planning')
    _,_,ticket=source.grab(capturer);b=source.frame_from_capture(ticket,'planning')
    assert len(calls)==2
    assert a.pixel_hash==b.pixel_hash and a.capture_id!=b.capture_id
    assert source.validates(a) and source.validates(b)
    assert not source.validates(replace(a,capture_id='forged',monotonic=5.))
    with pytest.raises(ValueError):source._issue(IMAGE,'new',1.2,7,'planning',None,None)


def test_archived_receipts_require_recorded_hash_event_and_once_only(tmp_path):
    cv2.imwrite(str(tmp_path/'one.png'),IMAGE)
    from jccbrain.vision.eye.unit_learning import file_hash
    event=dict(kind='frame',file='one.png',sha256=file_hash(tmp_path/'one.png'),monotonic=10.,
               window=dict(window_id=7,width=180,height=120))
    (tmp_path/'events.jsonl').write_text(json.dumps(event)+'\n')
    source=FrameSource();frame=source.archived(tmp_path,event,'planning')
    assert source.validates(frame)
    with pytest.raises(ValueError):source.archived(tmp_path,event,'planning')
    with pytest.raises(ValueError):FrameSource().archived(tmp_path,dict(event,monotonic=11.),'planning')
    (tmp_path/'one.png').write_bytes(b'changed')
    with pytest.raises(ValueError):FrameSource().archived(tmp_path,event,'planning')


class FakeOCR:
    name='fake'
    def __init__(self,name='甲',cost='1',trait='法师'):
        self.fields=[name,name,cost,trait]
    def prepare(self,im):return im
    def recognize_batch(self,imgs,*a,**kw):return [[Line(s,1.,0.,0.,1.,1.)] for s in self.fields]


@pytest.mark.parametrize('name,cost,trait,reason', [('甲','1','法师','consistent_fields'),
    ('甲乙','1','法师','missing_or_ambiguous_exact_name'),('甲','2','法师','cost_mismatch'),
    ('甲','','法师','missing_or_ambiguous_cost'),('甲','1','猎人','trait_mismatch')])
def test_panel_exact_name_cost_and_trait_gate(tmp_path,monkeypatch,name,cost,trait,reason):
    monkeypatch.setattr('jccbrain.vision.eye.unit_panel.presence',lambda _:1.)
    reader=PanelReader(Dictionary({'甲':1},champion_traits={'甲':['法师']}),FakeOCR(name,cost,trait))
    assert reader.read(IMAGE).reason==reason


def test_panel_presence_avoids_ocr_and_no_pool_cannot_confirm(monkeypatch):
    reader=PanelReader(Dictionary({'甲':1}),FakeOCR())
    assert not reader.read(IMAGE).present and reader.calls==0
    monkeypatch.setattr('jccbrain.vision.eye.unit_panel.presence',lambda _:1.)
    assert PanelReader(Dictionary(),FakeOCR()).read(IMAGE).reason=='no_current_pool'


def selected_image(two=False):
    im=np.zeros((1136,1774,3),np.uint8)
    bars=[Bar(675,742,450,455,.3994,.3983)]
    if two:bars.append(Bar(1000,1067,450,455,.5826,.3983))
    units=[]
    for b in bars:
        x=int(b.cx*1774)
        cv2.rectangle(im,(x-28,470),(x+28,550),(255,220,0),2)
        units.append(TrackedUnit('board',None,b.cx,b.cy,1,bar=b,crop=IMAGE,fp=VEC,fp_prepared=True))
    return im,units


def test_selection_requires_outline_unique_and_transaction_target():
    im,us=selected_image()
    assert bind_selection(im,us)[0] is us[0]
    assert bind_selection(im,us,(.9,.9))[2]=='target_mismatch'
    assert bind_selection(im*0,us)[0] is None
    im,us=selected_image(True)
    assert bind_selection(im,us,(us[0].x,.45))[0] is None
    us[0].bar=None
    assert bind_selection(im,[us[0]])[0] is None


def controller(tmp_path,monkeypatch):
    gallery=Gallery(tmp_path/'gallery','s',POOL);tracker=UnitTracker(gallery)
    im,us=selected_image()
    monkeypatch.setattr(tracker,'detect',lambda _:({},us))
    reader=SimpleNamespace(dict=Dictionary({'甲':1},champion_traits={'甲':['法师']}),ocr=FakeOCR(),units=tracker)
    source=FrameSource();store=store_at(tmp_path/'learning');ctl=UnitLearningController(reader,store,source,mode='panel')
    monkeypatch.setattr(ctl.panel,'read',lambda _:PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields'))
    monkeypatch.setattr('jccbrain.vision.eye.unit_learning.feature_print',lambda _:VEC)
    return ctl,source,im


def issue(source,im,seconds,window=7,phase='planning',viewing=None,stage='1-4'):
    return source._issue(im,str(seconds),seconds,window,phase,viewing,stage)


def test_two_distinct_receipts_required_same_frame_cannot_confirm(tmp_path,monkeypatch):
    ctl,source,im=controller(tmp_path,monkeypatch)
    a=issue(source,im,1.)
    assert ctl.process(a)['reason']=='awaiting_second_capture'
    assert ctl.process(a)['reason']=='stale_or_invalid_capture'
    b=issue(source,im,1.4)
    assert ctl.process(b)['status']=='pending'
    c=issue(source,im,1.8)
    assert ctl.process(c)['status']=='verified'
    assert ctl.reader.units.gallery.match(VEC)==('甲',0.)


@pytest.mark.parametrize('change', ['window','stage','phase','viewing','expired','old_click'])
def test_transaction_cannot_cross_scene_or_use_stale_target(tmp_path,monkeypatch,change):
    ctl,source,im=controller(tmp_path,monkeypatch)
    before=issue(source,im,1.)
    frame=issue(source,im,20. if change=='expired' else 1.5,
                window=8 if change=='window' else 7,phase='combat' if change=='phase' else 'planning',
                viewing='other' if change=='viewing' else None,stage='2-1' if change=='stage' else '1-4')
    tx=ClickTransaction(before,(.40,.45),.5 if change=='old_click' else 1.2,7)
    answer=ctl.process(frame,tx)
    assert answer['status']=='rejected'
    assert not ctl.store.active_vectors()


def test_record_mode_never_promotes_even_with_visible_panel(tmp_path,monkeypatch):
    ctl,source,im=controller(tmp_path,monkeypatch);ctl.mode='record'
    a=ctl.process(issue(source,im,1.));b=ctl.process(issue(source,im,1.4))
    assert a['promoted']==b['promoted']==0
    assert len(a['records'])==1 and b['records']==[] and b['deduplicated']==1
    assert len(ctl.store.entries('review'))==1
    assert not ctl.store.entries()


def test_real_cg_selection_patches_reject_unselected_blue_unit():
    base=Path(__file__).with_name('fixtures')/'unit_learning_l1'
    metadata=json.loads(base.with_suffix('.json').read_text())
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as arrays:
        for row in metadata['entries']:
            w,h=row['window'];im=np.zeros((h,w,3),np.uint8)
            x0,y0,x1,y1=row['rect'];im[y0:y1,x0:x1]=arrays[row['key']]
            bar=Bar(**row['bar']);u=TrackedUnit(row['zone'],row['slot'],bar.cx,bar.cy,1,bar=bar)
            assert (bind_selection(im,[u])[0] is u)==row['selected']


def test_provider_explicit_interface_reuses_existing_capture_and_cached_frame(tmp_path,monkeypatch):
    from jccbrain.vision.pyeye import PyEyeProvider
    im,us=selected_image();gallery=Gallery(tmp_path/'gallery','s',POOL)
    tracker=UnitTracker(gallery);monkeypatch.setattr(tracker,'detect',lambda _:({},us))
    fake_reader=SimpleNamespace(dict=Dictionary({'甲':1},champion_traits={'甲':['法师']}),ocr=FakeOCR(),units=tracker,
                               last_ocr_cells=0,read=lambda _:dict(phase='planning',stage='1-4'))
    provider=PyEyeProvider(min_interval_s=0,gallery_dir=tmp_path/'gallery')
    provider._reader=fake_reader
    calls=[]
    provider._capture=SimpleNamespace(grab=lambda:(calls.append(True) or im,SimpleNamespace(window_id=7)))
    provider.configure_unit_learning(tmp_path/'learning',mode='record')
    monkeypatch.setattr('jccbrain.vision.eye.unit_learning.feature_print',lambda _:VEC)
    provider.read()
    first=provider.current_learning_frame();assert first is not None and len(calls)==1
    provider.min_interval_s=100
    provider.read(fresh=False)
    assert provider.current_learning_frame() is first and len(calls)==1
    assert provider.last_learning['promoted']==0


def test_overlapping_other_unit_crop_cannot_become_a_training_sample():
    from jccbrain.vision.eye.unit_learning import isolated_crop
    im,us=selected_image()
    assert isolated_crop(us[0],us,im)
    b=replace(us[0].bar,min_x=us[0].bar.min_x+10,max_x=us[0].bar.max_x+10,cx=us[0].x+.006)
    other=TrackedUnit('board',None,b.cx,b.cy,1,bar=b,crop=IMAGE)
    assert not isolated_crop(us[0],[us[0],other],im)


def test_revoked_crop_cannot_silently_become_a_new_sample(tmp_path):
    store=store_at(tmp_path);record=pending(store);sid=store.promote(record,VEC)['id']
    store.revoke(sid)
    assert store.promote(record,VEC)['reason']=='revoked_crop'
    assert not store.active_vectors()
    assert store.entries('review')[0]['status']=='rejected'
    with pytest.raises(ValueError):
        store.record(dict(source='manual',season='other',name='甲'),{'unit':IMAGE})
