"""Optional learning must have bounded I/O and must never take down main vision."""
import importlib.util
import json
import os
from pathlib import Path
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from jccbrain.vision.base import VisionError
from jccbrain.vision.eye.capture import CaptureError
from jccbrain.vision.eye import unit_learning as learning
from jccbrain.vision.eye.units import Gallery, TrackedUnit, UnitTracker

SPEC=importlib.util.spec_from_file_location('learning_runtime_probe',Path(__file__).resolve().parents[1]/'scripts/eval_unit_learning_runtime.py')
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)


def learned(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    gallery=Gallery(tmp_path/'gallery','probe',store.pool,verified_store=store)
    tracker=UnitTracker(gallery)
    vector=np.zeros(768,np.float32)
    image=np.zeros((120,180,3),np.uint8)
    def observe():
        u=TrackedUnit('board',None,.4,.3,1,fp=vector,fp_prepared=True,crop=image,hist=np.array([1.,0.]))
        return tracker.observe(image,'planning',None,prepared=({},[u]))['board'][0]['name']
    assert observe()=='压力0'
    return store,gallery,tracker,vector,observe


@pytest.mark.parametrize('count',[2,30,195])
def test_hot_match_does_not_read_hash_or_load_unchanged_content(tmp_path,count):
    result=probe.cache_probe(tmp_path,count,repeats=3)
    assert result['valid_vectors']==result['cache_entries']==count
    assert result['hot_increment']==dict(index_reads=0,evidence_reads=0,hash_calls=0,np_loads=0)
    assert result['warm_counts']['np_loads']==count
    assert result['forced_cold_increment']['np_loads']==count*3


@pytest.mark.parametrize('kind',['index_read','asset_hash','asset_read','evidence_read','vector_load'])
def test_transient_content_io_recovers_next_refresh_with_identical_stat(tmp_path,kind):
    result=probe.transient_read_probe(tmp_path,kind)
    assert result['next_refresh_count']==1 and result['running_tracker_recovered']
    assert result['stats_before']==result['stats_during']==result['stats_after']
    assert all(n==0 for n in result['hot_increment'].values())


@pytest.mark.parametrize('kind',['index_read','asset_read','vector_load'])
def test_provider_io_recovery_requires_two_new_receipts_without_touching_assets(tmp_path,kind):
    result=probe.transient_provider_probe(tmp_path,kind)
    assert result['stat_unchanged_before_promotion'] and result['captures_during_failure']==1
    assert result['recovery']==['pending','verified'] and result['verified_counts']==[1,1,2]


@pytest.mark.parametrize('filename',['vector.npy','evidence.json','unit.png','panel.png'])
@pytest.mark.parametrize('operation',['rewrite','delete'])
def test_asset_change_invalidates_match_revision_and_persistent_tracker(tmp_path,filename,operation):
    store,gallery,tracker,vector,observe=learned(tmp_path)
    row=store.entries()[0];path=store.verified/row['id']/filename
    old=path.read_bytes();stamp=path.stat();revision=store.revision()
    if operation=='delete':
        path.unlink()
    else:
        # Same size and restored mtime: ctime must still invalidate the cache.
        path.write_bytes(bytes([old[0]^1])+old[1:])
        os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert gallery.match(vector) is None
    assert store.revision()!=revision and store.last_error
    assert observe().startswith('?') and not tracker.group_name
    path.write_bytes(old)
    assert gallery.match(vector)==('压力0',0.)
    assert observe()=='压力0'


@pytest.mark.parametrize('change',['revoke','rename','season','corrupt','missing','unreadable'])
def test_external_index_changes_fail_closed_and_clear_remembered_identity(tmp_path,monkeypatch,change):
    store,gallery,tracker,vector,observe=learned(tmp_path)
    path=store.verified/'index.json';old=path.read_bytes();revision=store.revision()
    other=learning.UnitLearningStore(tmp_path/'store','probe',store.pool)
    if change=='revoke':
        other.revoke(other.entries()[0]['id'])
    elif change=='corrupt':path.write_text('{bad')
    elif change=='missing':path.unlink()
    elif change=='unreadable':
        original=store._stat
        def denied(p):
            if p==path:raise PermissionError('index denied')
            return original(p)
        monkeypatch.setattr(store,'_stat',denied)
    else:
        data=json.loads(old)
        if change=='rename':data['entries'][0]['name']='not-in-pool'
        else:data['season']='different-season'
        path.write_text(json.dumps(data))
    assert gallery.match(vector) is None
    assert store.revision()!=revision
    assert observe().startswith('?') and not tracker.group_name
    monkeypatch.undo()
    path.write_bytes(old)
    assert observe()=='压力0'


def test_index_rename_cannot_relabel_unchanged_evidence(tmp_path):
    store=probe.pressure_store(tmp_path/'store',4)
    assert len(store.active_vectors())==4
    index=store.verified/'index.json';data=json.loads(index.read_text())
    data['entries'][0]['name']='压力1';index.write_text(json.dumps(data))
    assert len(store.active_vectors())==3
    assert 'Inconsistent verified identity' in store.last_error


def test_external_add_next_match_and_cache_is_not_history_of_revoked_entries(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    gallery=Gallery(tmp_path/'gallery','probe',store.pool,verified_store=store)
    vector=np.zeros(768,np.float32)
    assert gallery.match(vector)==('压力0',0.)
    other=learning.UnitLearningStore(tmp_path/'store','probe',store.pool)
    sid=other.record(dict(source='manual',name='压力0',confirmation=True),{'unit':np.ones((12,12,3),np.uint8)})
    answer=other.promote(sid,vector)
    assert answer['status']=='verified'
    assert len(store.active_vectors())==2 and len(store._vector_cache)==2
    for row in other.entries():other.revoke(row['id'])
    assert gallery.match(vector) is None and not store._vector_cache and not store._cache_rows


def test_missing_index_preserves_evidence_and_cannot_silently_rebuild(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    path=store.verified/'index.json';path.unlink()
    assert store.active_vectors()==[] and store.last_error
    sid=store.record(dict(source='manual',name='压力0',confirmation=True),{'unit':np.ones((12,12,3),np.uint8)})
    with pytest.raises(learning.StoreCorrupt):store.promote(sid,np.zeros(768,np.float32))
    assert not path.exists() and len(list(store.verified.glob('*/vector.npy')))==1


def test_corrupt_existing_sample_cannot_be_skipped_during_promotion(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    row=store.entries()[0]
    (store.verified/row['id']/'unit.png').unlink()
    sid=store.record(dict(source='manual',name='压力0',confirmation=True),{'unit':np.ones((12,12,3),np.uint8)})
    with pytest.raises(learning.StoreCorrupt):store.promote(sid,np.zeros(768,np.float32))
    assert len(store.entries())==1 and store.active_vectors()==[]


def test_returned_vectors_cannot_poison_cached_matching_or_cross_season(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    rows=store.active_vectors();vector=rows[0][1]
    with pytest.raises(ValueError):vector[0]=99
    with pytest.raises(ValueError):vector.setflags(write=True)
    assert isinstance(vector.base,bytes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        vector.shape=(384,2)
        vector.dtype=np.float64
    rows.clear()
    assert len(store.active_vectors())==1 and store.active_vectors()[0][1].shape==(768,)
    assert store.active_vectors()[0][1].dtype==np.float32 and store.active_vectors()[0][1][0]==0
    foreign=Gallery(tmp_path/'foreign','other',store.pool,verified_store=store)
    assert foreign.match(np.zeros(768,np.float32)) is None


def test_oversized_external_index_does_not_grow_cache(tmp_path):
    store=probe.pressure_store(tmp_path/'store',1)
    path=store.verified/'index.json';data=json.loads(path.read_text())
    row=data['entries'][0]
    data['entries']=[dict(row,id=f'{i+1:032x}') for i in range(100)]
    path.write_text(json.dumps(data))
    assert store.active_vectors()==[] and not store._vector_cache
    assert 'exceeds' in store.last_error


@pytest.mark.parametrize('kind',['record_disk','panel_ocr','receipt_ticket','receipt_issue','lock','transient_lock','index'])
def test_optional_failure_preserves_single_capture_main_result_and_recovers(tmp_path,kind):
    result=probe.fault_probe(tmp_path,kind)
    assert result['captures_during_failure']==1
    assert result['error']['status']=='error'
    assert result['main_read_returned'] and result['old_receipt_invalid']


def test_index_read_failure_does_not_break_main_or_overwrite_history(tmp_path,monkeypatch):
    provider,ctl=probe.fake_provider(tmp_path)
    path=ctl.store.verified/'index.json'
    ctl.store._save(ctl.store.verified,[]);old=path.read_bytes()
    original=Path.read_text
    def denied(p,*a,**kw):
        if p==path:raise PermissionError('index not readable')
        return original(p,*a,**kw)
    monkeypatch.setattr(Path,'read_text',denied)
    assert provider.read().gold==17
    result=provider.process_unit_learning()
    assert result['status']=='error' and 'PermissionError' in result['error']
    assert provider._capture.grab.call_count==1 and path.read_bytes()==old
    assert provider.current_learning_frame() is None


def test_explicit_interface_contains_unexpected_learning_exception(tmp_path,monkeypatch):
    provider,ctl=probe.fake_provider(tmp_path)
    provider.read()
    monkeypatch.setattr(ctl,'process',lambda *_:(_ for _ in ()).throw(OSError('unexpected write error')))
    assert provider.process_unit_learning()['status']=='error'
    assert provider.current_learning_frame() is None and ctl.previous is None
    assert provider._capture.grab.call_count==1


def test_record_read_contains_unexpected_controller_exception(tmp_path):
    provider,ctl=probe.fake_provider(tmp_path,'record')
    with patch.object(ctl,'process',side_effect=OSError('unexpected controller failure')):
        assert provider.read().gold==17
    assert provider.last_learning['status']=='error'
    assert provider.last_learning['operation']=='capture_learning'
    assert provider.current_learning_frame() is None and ctl.previous is None
    assert provider._capture.grab.call_count==1


def test_invalid_main_schema_is_not_disguised_as_a_learning_error(tmp_path):
    provider,ctl=probe.fake_provider(tmp_path)
    with patch('jccbrain.vision.pyeye.GameState.from_dict',side_effect=ValueError('invalid main state')):
        with pytest.raises(ValueError,match='invalid main state'):provider.read()
    assert provider._last is None and provider.last_learning is None
    assert provider.current_learning_frame() is None and ctl.previous is None
    assert provider._capture.grab.call_count==1


def test_warning_is_throttled_but_error_state_and_recovery_are_not_disabled(tmp_path,caplog):
    provider,ctl=probe.fake_provider(tmp_path,'record')
    with patch.object(ctl,'_process',side_effect=OSError('disk full')),patch.object(learning.time,'monotonic',return_value=100.):
        for _ in range(6):
            assert ctl.process(None)['status']=='error'
    assert len([r for r in caplog.records if 'Unit learning failed' in r.message])==1
    # A later failure reports how many messages were suppressed.
    with patch.object(learning.time,'monotonic',return_value=131.):ctl.fail(OSError('still full'))
    warnings=[r for r in caplog.records if 'Unit learning failed' in r.message]
    assert len(warnings)==2 and 'suppressed=5' in warnings[-1].message
    assert ctl.mode=='record' and ctl.previous is None


@pytest.mark.parametrize('failure',[KeyboardInterrupt,SystemExit])
def test_control_interrupts_not_swallowed_by_learning_boundaries(tmp_path,failure):
    provider,ctl=probe.fake_provider(tmp_path)
    provider.read()
    with patch.object(ctl.panel,'read',side_effect=failure):
        with pytest.raises(failure):provider.process_unit_learning()
    with patch.object(ctl.source,'_capture_ticket',side_effect=failure):
        with pytest.raises(failure):provider.read()


@pytest.mark.parametrize('learning_enabled',[False,True])
def test_capture_and_main_reader_failures_keep_original_contract(tmp_path,learning_enabled):
    provider,ctl=probe.fake_provider(tmp_path)
    if not learning_enabled:
        provider._learning=provider._learning_source=None
        provider._reader.units.gallery.verified_store=None
    provider._capture.grab.side_effect=CaptureError('actual capture failed')
    with pytest.raises(VisionError,match='actual capture failed'):provider.read()
    assert provider._reader.read.call_count==0 and provider._capture.grab.call_count==1
    provider._capture.grab.side_effect=None
    provider._reader.read.side_effect=RuntimeError('actual main reader failed')
    with pytest.raises(RuntimeError,match='actual main reader failed'):provider.read()
    assert provider._reader.read.call_count==1 and provider._capture.grab.call_count==2
    assert provider._last is None


def test_disabled_provider_does_not_use_learning_store_or_receipts(tmp_path):
    provider,ctl=probe.fake_provider(tmp_path)
    provider._learning=provider._learning_source=None
    provider._reader.units.gallery.verified_store=None
    with patch.object(ctl.store,'active_vectors',side_effect=AssertionError('optional path used')), \
            patch.object(ctl.source,'_capture_ticket',side_effect=AssertionError('receipt used')):
        state=provider.read()
    assert state.gold==17 and provider.last_learning is None
    assert provider.current_learning_frame() is None and provider._capture.grab.call_count==1


def test_configuration_errors_remain_explicit(tmp_path):
    provider,ctl=probe.fake_provider(tmp_path)
    source=provider._learning_source
    with pytest.raises(ValueError,match='unsupported non-click mode'):
        provider.configure_unit_learning(tmp_path/'other',mode='invalid')
    assert provider._learning is ctl and provider._learning_source is source
