"""Opt-in learning I/O and failure-boundary probes (synthetic, no game/capture).

Storage pressure uses distinct, schema-valid synthetic IDs and vectors, never an
accuracy claim. Costs are raw observations; the normal process-load rules apply.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from jccbrain.vision.eye import unit_learning as learning
from jccbrain.vision.eye.text import Dictionary
from jccbrain.vision.eye.unit_panel import PanelObservation
from jccbrain.vision.eye.units import Bar, Gallery, TrackedUnit, UnitTracker
from jccbrain.vision.pyeye import PyEyeProvider


def pressure_store(root, count):
    """Synthetic immutable rows; <=3 per hero, distinct IDs/crop hashes."""
    pool = {f'压力{i//3}' for i in range(count)}
    store = learning.UnitLearningStore(root, 'probe', pool)
    rows = []
    for i in range(count):
        sid, name = f'{i+1:032x}', f'压力{i//3}'
        image = np.full((12, 12, 3), i, np.uint8)
        vector = np.zeros(768, np.float32)
        vector[0] = i//3 * 2
        metadata = dict(season='probe', name=name, source='manual', status='verified',
                        confirmation=True, crop_hash=learning.image_hash(image),
                        reason='synthetic_storage_pressure_only')
        hashes = store._commit_dir(store.verified, sid, metadata, {'unit':image, 'panel':image}, vector)
        rows.append(dict(id=sid, name=name, source='manual', revoked=False,
                         crop_hash=metadata['crop_hash'], hashes=hashes))
    store._save(store.verified, rows)
    return store


@contextmanager
def count_io(store):
    counts = Counter(index_reads=0, evidence_reads=0, hash_calls=0, np_loads=0)
    old_bytes, old_text, old_hash, old_load = Path.read_bytes, Path.read_text, learning.file_hash, np.load
    def relevant(path):
        return Path(path).is_relative_to(store.verified)
    def record_read(path):
        if relevant(path):
            counts['index_reads' if Path(path).name == 'index.json' else 'evidence_reads'] += 1
    def read_bytes(path, *args, **kw):
        record_read(path)
        return old_bytes(path, *args, **kw)
    def read_text(path, *args, **kw):
        record_read(path)
        return old_text(path, *args, **kw)
    def file_hash(path):
        if relevant(path): counts['hash_calls'] += 1
        return old_hash(path)
    def load(path, *args, **kw):
        if isinstance(path, (str, Path)) and relevant(path): counts['np_loads'] += 1
        return old_load(path, *args, **kw)
    with patch.object(Path, 'read_bytes', read_bytes), patch.object(Path, 'read_text', read_text), \
            patch.object(learning, 'file_hash', file_hash), patch.object(np, 'load', load):
        yield counts


def cache_probe(root, count, repeats=5):
    store = pressure_store(root/'store', count)
    gallery = Gallery(root/'gallery', 'probe', store.pool, verified_store=store)
    vector = np.zeros(768, np.float32)
    with count_io(store) as counts:
        start = time.perf_counter()
        assert gallery.match(vector) == ('压力0', 0.)
        cold_ms = (time.perf_counter()-start)*1000
        warm = dict(counts)
        start = time.perf_counter()
        for _ in range(repeats):
            assert gallery.match(vector) == ('压力0', 0.)
            store.revision()
            gallery.names
        hot_ms = (time.perf_counter()-start)*1000/repeats
        hot = {k:counts[k]-warm[k] for k in counts}
        # Same final implementation with cache eviction models the original
        # repeated verification cost; no evidence checks are bypassed in either.
        before = dict(counts)
        start = time.perf_counter()
        for _ in range(repeats):
            store._vector_cache.clear()
            store._cache_index_stat = None
            assert gallery.match(vector) == ('压力0', 0.)
        uncached_ms = (time.perf_counter()-start)*1000/repeats
        uncached = {k:counts[k]-before[k] for k in counts}
    assert all(n == 0 for n in hot.values()), hot
    return dict(count=count, valid_vectors=len(store.active_vectors()), cache_entries=len(store._vector_cache),
                warm_counts=warm, hot_increment=hot, forced_cold_increment=uncached, repeats=repeats,
                raw_ms=dict(first_match=cold_ms, hot_match_plus_revision_and_names=hot_ms,
                            forced_cold_match=uncached_ms),
                note='Synthetic storage pressure; forced cold evicts final cache, not an accuracy/performance gate.')


IMAGE = np.zeros((1136,1774,3),np.uint8)
VECTOR = np.array([1.,0.],np.float32)


def fake_provider(root, mode='panel'):
    """No native capture/OCR: inject only into the optional boundary."""
    image = IMAGE.copy()
    bar = Bar(675,742,450,455,.3994,.3983)
    cv2.rectangle(image,(680,470),(736,550),(255,220,0),2)
    def unit():
        return TrackedUnit('board',None,bar.cx,bar.cy,1,bar=bar,crop=image[470:550,680:736],
                           fp=VECTOR,fp_prepared=True,hist=np.array([1.,0.]))
    gallery = Gallery(root/'gallery','probe',{'甲'})
    tracker = UnitTracker(gallery)
    tracker.detect = lambda _: ({},[unit()])
    reader = SimpleNamespace(dict=Dictionary({'甲':1},champion_traits={'甲':['法师']}),
                             ocr=SimpleNamespace(name='fake'), units=tracker,last_ocr_cells=0)
    def read(im):
        return dict(phase='planning',stage='1-4',gold=17,
                    **tracker.observe(im,'planning',None,prepared=({},[unit()])))
    reader.read = Mock(side_effect=read)
    provider = PyEyeProvider(min_interval_s=0,gallery_dir=root/'gallery')
    provider._reader = reader
    provider._capture = SimpleNamespace(grab=Mock(return_value=(image,SimpleNamespace(window_id=7))))
    controller = provider.configure_unit_learning(root/'learning',mode=mode)
    controller.panel.read = Mock(return_value=PanelObservation(True,1.,'甲',1,['法师'],{},'consistent_fields'))
    return provider, controller


def fault_probe(root, kind):
    provider, ctl = fake_provider(root, 'record' if kind=='record_disk' else 'panel')
    with patch.object(learning, 'feature_print', return_value=VECTOR):
        # Establish a real outstanding receipt and a pending confirmation chain.
        provider.read()
        before = provider.current_learning_frame()
        if ctl.mode == 'panel':
            assert provider.process_unit_learning()['reason'] == 'awaiting_second_capture'
            assert ctl.previous is not None
        if kind=='record_disk':
            ctl._record_seen.clear()
            target = patch.object(ctl.store,'record',side_effect=OSError('injected disk full'))
        elif kind=='panel_ocr':
            target = patch.object(ctl.panel,'read',side_effect=RuntimeError('injected panel OCR'))
        elif kind=='receipt_ticket':
            target = patch.object(ctl.source,'_capture_ticket',side_effect=ValueError('injected receipt'))
        elif kind=='receipt_issue':
            target = patch.object(ctl.source,'frame_from_capture',side_effect=ValueError('injected receipt'))
        elif kind=='lock':
            target = patch.object(ctl.store,'locked',side_effect=OSError('injected lock denied'))
        elif kind=='transient_lock':
            original, attempts = ctl.store.locked, []
            def once():
                if not attempts:
                    attempts.append(True)
                    raise OSError('injected transient lock error during main matching')
                return original()
            target = patch.object(ctl.store,'locked',side_effect=once)
        elif kind=='index':
            (ctl.store.verified/'index.json').write_text('{broken')
            target = nullcontext()  # leave corruption in place during read
        else:
            raise ValueError(kind)
        calls = provider._capture.grab.call_count
        with target:
            state = provider.read()
            result = provider.last_learning
            if kind in ('panel_ocr','lock','transient_lock','index'):
                result = provider.process_unit_learning()
        assert state.gold == 17 and state.phase.value == 'planning'
        assert provider._capture.grab.call_count == calls+1
        assert result['status']=='error', result
        assert ctl.previous is None and provider.current_learning_frame() is None
        assert not ctl.source.validates(before)
        if kind=='index':
            # Explicit repair, never auto-overwrite the corrupt file.
            assert (ctl.store.verified/'index.json').read_text()=='{broken'
            ctl.store._save(ctl.store.verified,[])
        assert ctl.store.entries()==[]
        # Recovery must require two newly captured receipts, even after an error
        # in the second confirmation. The probe only delays its synthetic clock.
        provider.read()
        if ctl.mode == 'panel':
            one = provider.process_unit_learning()
            assert one['reason']=='awaiting_second_capture'
            with patch.object(learning.time,'monotonic',return_value=ctl.source._last_time+.3):
                provider.read()
                two = provider.process_unit_learning()
            assert two['status']=='verified', two
            recovered = [one['status'], two['status']]
        else:
            assert provider.last_learning['status']=='recorded'
            recovered = [provider.last_learning['status']]
    return dict(kind=kind,main_read_returned=True,main_gold=state.gold,captures_during_failure=1,
                error=result,confirmation_chain_cleared=True,old_receipt_invalid=True,recovery=recovered)


@contextmanager
def transient_read_error(store, kind):
    """Fail actual content I/O only; leave all file contents/stat untouched."""
    if kind in ('index_read', 'evidence_read', 'asset_read'):
        method = 'read_bytes' if kind=='asset_read' else 'read_text'
        original = getattr(Path, method)
        def read(path, *args, **kw):
            selected = (path == store.verified/'index.json' if kind=='index_read' else
                        path.is_relative_to(store.verified) and
                        path.name == ('evidence.json' if kind=='evidence_read' else 'unit.png'))
            if selected:
                raise PermissionError(f'injected temporary {kind}')
            return original(path, *args, **kw)
        target = patch.object(Path, method, read)
    elif kind=='asset_hash':
        original = learning.file_hash
        def hash_file(path):
            if path.is_relative_to(store.verified) and path.name=='unit.png':
                raise OSError('injected temporary asset hash read')
            return original(path)
        target = patch.object(learning, 'file_hash', hash_file)
    elif kind=='vector_load':
        original = np.load
        def load(path, *args, **kw):
            if isinstance(path,(str,Path)) and Path(path).is_relative_to(store.verified):
                raise OSError('injected temporary np.load read')
            return original(path, *args, **kw)
        target = patch.object(np, 'load', load)
    else:
        raise ValueError(kind)
    with target:
        yield


def verified_stats(store):
    return {str(p.relative_to(store.verified)):store._stat(p)
            for p in store.verified.rglob('*') if p.is_file()}


def transient_read_probe(root, kind):
    store = pressure_store(root/'store',1)
    gallery = Gallery(root/'gallery','probe',store.pool,verified_store=store)
    tracker = UnitTracker(gallery)
    vector = np.zeros(768,np.float32)
    image = np.zeros((120,180,3),np.uint8)
    def observe():
        unit = TrackedUnit('board',None,.4,.3,1,fp=vector,fp_prepared=True,crop=image,hist=np.array([1.,0.]))
        return tracker.observe(image,'planning',None,prepared=({},[unit]))['board'][0]['name']
    before = verified_stats(store)
    with transient_read_error(store,kind):
        assert store.active_vectors()==[]
        error = store.last_error
        assert gallery.match(vector) is None and observe().startswith('?')
        during = verified_stats(store)
    # The SAME Store must recover on the first refresh, with no touch/rebuild.
    next_count = len(store.active_vectors())
    assert next_count==1 and store.last_error is None, (kind,next_count,store.last_error)
    assert gallery.match(vector)==('压力0',0.) and observe()=='压力0'
    after = verified_stats(store)
    with count_io(store) as counts:
        for _ in range(5):
            gallery.match(vector)
            store.revision()
        hot = dict(counts)
    assert before==during==after and all(n==0 for n in hot.values())
    return dict(kind=kind,rejected_during_fault=True,error=error,next_refresh_count=next_count,
                recovered_name='压力0',running_tracker_recovered=True,stat_unchanged=True,
                stats_before=before,stats_during=during,stats_after=after,hot_increment=hot)


def transient_provider_probe(root, kind):
    provider, ctl = fake_provider(root)
    with patch.object(learning,'feature_print',return_value=VECTOR):
        provider.read()
        old_frame = provider.current_learning_frame()
        assert provider.process_unit_learning()['reason']=='awaiting_second_capture'
        # Add a healthy, distinct verified sample via the actual store interface.
        # It doesn't match this selected unit and cannot deduplicate its crop.
        record = ctl.store.record(dict(source='manual',name='甲',confirmation=True),
                                  {'unit':np.full((12,12,3),125,np.uint8)})
        seed_vector = VECTOR+3
        assert ctl.store.promote(record,seed_vector)['status']=='verified'
        before = verified_stats(ctl.store)
        calls = provider._capture.grab.call_count
        with transient_read_error(ctl.store,kind):
            state = provider.read()
            error = provider.process_unit_learning()
        assert state.gold==17 and provider._capture.grab.call_count==calls+1
        assert error['status']=='error' and ctl.previous is None
        assert provider.current_learning_frame() is None and not ctl.source.validates(old_frame)
        assert len(ctl.store.entries())==1  # no promotion during failure
        assert ctl.reader.units.gallery.match(seed_vector)==('甲',0.)
        after = verified_stats(ctl.store)
        assert before==after
        provider.read()
        first_frame = provider.current_learning_frame()
        first = provider.process_unit_learning()
        assert first['reason']=='awaiting_second_capture' and len(ctl.store.entries())==1
        with patch.object(learning.time,'monotonic',return_value=ctl.source._last_time+.3):
            provider.read()
            second_frame = provider.current_learning_frame()
            second = provider.process_unit_learning()
        assert second['status']=='verified' and len(ctl.store.entries())==2
        assert len({old_frame.capture_id,first_frame.capture_id,second_frame.capture_id})==3
    return dict(kind=kind,main_read_returned=True,captures_during_failure=1,error=error,
                old_receipt_invalid=True,stat_unchanged_before_promotion=True,
                recovery=[first['status'],second['status']],verified_counts=[1,1,2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory; never overwrites old evidence')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.eval_vision_b8 import process_snapshot
    before = process_snapshot()
    storage = [cache_probe(args.output/f'pressure-{n}',n) for n in (2,30,195)]
    faults = [fault_probe(args.output/f'fault-{kind}',kind) for kind in
              ('record_disk','panel_ocr','receipt_ticket','receipt_issue','lock','transient_lock','index')]
    reads = [transient_read_probe(args.output/f'read-{kind}',kind) for kind in
             ('index_read','asset_hash','asset_read','evidence_read','vector_load')]
    providers = [transient_provider_probe(args.output/f'provider-{kind}',kind) for kind in
                 ('index_read','asset_read','vector_load')]
    result = dict(storage=storage,faults=faults,transient_reads=reads,transient_providers=providers,
                  processes_before=before,processes_after=process_snapshot(),
                  timing_status='unmeasured', timing_reason='Synthetic boundary probe, with active external loads retained.',
                  sources={str(p.relative_to(ROOT)):learning.file_hash(p) for p in
                           (Path(__file__),ROOT/'src/jccbrain/vision/eye/unit_learning.py',ROOT/'src/jccbrain/vision/pyeye.py')})
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(storage=storage,faults=faults,transient_reads=reads,transient_providers=providers),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
