"""F2-1 OFFLINE original-frame equivalence, quotas and diagnostic-only timings.

No capture/input is performed. Source images are CG decoded. Window receipt and
monotonic cadence below are explicit offline adapters, never live fresh samples.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.eval_augment_ownership import CASES,POSITIVE,manifest,sha,write,validate_output

BASE=ROOT/'var/vision-module-optimization/batch-F2'


def audit_store(root):
    sessions=[]
    for marker in root.glob('session-*/manifest.json'):
        d=json.loads(marker.read_text());events=[]
        for p in marker.parent.glob('event-*.json'):
            e=json.loads(p.read_text());events.append(e)
            assert len(e['frames'])<=5 and sum(f['role']=='after' for f in e['frames'])<=2
            assert e['encoded_image_bytes']<=512*1024
            assert all(len(f['rois'])<=3 for f in e['frames'])
            # Verify physical unique image bytes, not just the writer's counter.
            images={r['file'] for f in e['frames'] for r in f['rois']}
            assert sum((marker.parent/name).stat().st_size for name in images)==e['encoded_image_bytes']
            assert sum(f['role']=='before' for f in e['frames'])<=2
            assert sum(f['role']=='current' for f in e['frames'])==1
        assert len(events)<=12
        size=sum(p.stat().st_size for p in marker.parent.iterdir())
        assert size<=8*1024*1024
        sessions.append(dict(session=marker.parent.name,events=events,bytes=size))
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    assert len(sessions)<=10 and total<=100*1024*1024
    return dict(sessions=sessions,total_bytes=total)


def process_review():
    raw=subprocess.check_output(['ps','-axo','pid,ppid,pcpu,command'],text=True)
    rows=[];parents={}
    for line in raw.splitlines()[1:]:
        cells=line.strip().split(None,3)
        if len(cells)!=4:continue
        try:pid,ppid,cpu=int(cells[0]),int(cells[1]),float(cells[2])
        except ValueError:continue
        parents[pid]=ppid;rows.append((pid,ppid,cpu,cells[3]))
    ancestors={os.getpid()};pid=os.getpid()
    while pid in parents and parents[pid] not in ancestors:
        pid=parents[pid];ancestors.add(pid)
    suspects=[dict(pid=pid,cpu=cpu,command=cmd) for pid,_,cpu,cmd in rows
              if pid not in ancestors and cpu>1 and any(t in cmd.lower() for t in
              ('jkchess','playcover','record_assistant','jccbrain','pytest','eval_vision','eval_unit','eval_augment'))]
    return dict(raw=raw,excluded_own_ancestors=sorted(ancestors),active_external_load=suspects)


def percentile(values,q):
    values=sorted(values);return values[min(len(values)-1,int(q*(len(values)-1)))]


def performance(root,image,state,window):
    from jccbrain.vision.eye.augment_diagnostics import AugmentDiagnostics
    before=process_review();ordinary=AugmentDiagnostics(root/'ordinary')
    # Actual existing full-roster state, warm event outside measured loop.
    for i in (0.,1.,2.):ordinary.observe(image,state,window,now=i)
    on=[];off=[]
    for i in range(1000):
        now=10+i*.1
        for enabled in ((True,False) if i%2==0 else (False,True)):
            t=time.perf_counter()
            if enabled:ordinary.observe(image,state,window,now=now)
            else:pass
            (on if enabled else off).append((time.perf_counter()-t)*1000)
    # Each trial is a fresh isolated observer. Real BGR+unchanged existing result;
    # initial reference-set precondition is controlled, no new model output.
    triggered=[];disabled=[];trial_roots=[]
    for i in range(20):
        d=AugmentDiagnostics(root/f'trigger-{i:02d}')
        # Fill two preframes without inventing an additional OCR/choice result.
        d.names=frozenset(state.get('augments_taken') or [])
        d.previous_phase=state.get('phase')
        for at in (0.,1.):d.observe(image,state,window,now=at)
        # A genuine existing selection text change would be a trigger; resetting
        # observer's previous set is the explicit controlled trigger precondition.
        d.names=frozenset()
        for enabled in ((True,False) if i%2==0 else (False,True)):
            t=time.perf_counter()
            if enabled:d.observe(image,state,window,now=2.)
            else:pass
            (triggered if enabled else disabled).append((time.perf_counter()-t)*1000)
        trial_roots.append(audit_store(d.root))
        store=d.store;d.close()
        trial_roots[-1]['observer_closed']=d.closed and store.fd is None
    ordinary.close()
    after=process_review()
    dirty=bool(before['active_external_load'] or after['active_external_load'])
    return dict(timing_status='未测（环境不净；保留离线观察）' if dirty else 'measured offline diagnostic increment',
        meaning='Only observe() overhead, NOT Reader, capture, learning transaction or live latency.',
        trigger_precondition='Controlled diagnostic previous-name-set reset, real BGR and existing result unchanged. Same-pixel ROI SHA dedupe retained; not a moving-scene worst-case measurement.',
        before=before,after=after,ordinary_enabled_ms=on,ordinary_disabled_ms=off,
        triggered_enabled_ms=triggered,triggered_disabled_ms=disabled,
        ordinary_p95_ms=percentile(on,.95),triggered_p95_ms=percentile(triggered,.95),
        ordinary_target=percentile(on,.95)<=3,triggered_target=percentile(triggered,.95)<=25,
        trigger_trials=trial_roots)


def provider_lifecycle(root,image,state,window):
    """Controlled Reader adapter, real Provider/hook/storage/stop, no native I/O."""
    from jccbrain.vision.pyeye import PyEyeProvider
    from jccbrain.vision.eye.augment_diagnostics import AugmentDiagnostics
    rows=[]
    for i in range(12):
        p=PyEyeProvider();p.start=lambda:None;p.interval_override_s=0
        p._reader=NS(read=lambda _:copy.deepcopy(state),last_ocr_cells=0)
        p._capture=NS(grab=lambda:(image,copy.copy(window)))
        p._augment_diagnostics=AugmentDiagnostics(root)
        p.read();d=p._augment_diagnostics;store=d.store
        p.stop();p.stop()
        rows.append(dict(index=i,closed=d.closed,fd_released=store.fd is None,
                         complete=json.loads((store.session/'manifest.json').read_text())['complete'],
                         errors=d.status()['errors']))
    previous=store.session;p.read();d=p._augment_diagnostics;store=d.store
    resumed=store.session!=previous and not d.closed and not d.status()['errors']
    p.stop()
    audit=audit_store(root)
    complete=all(json.loads(m.read_text())['complete'] for m in root.glob('*/manifest.json'))
    resumed_closed=d.closed and store.fd is None
    return dict(note='Lifecycle adapter only: existing real pixels/result, no additional OCR or native capture.',
                stops=rows,same_provider_resumed=resumed,sessions_after_resume=len(audit['sessions']),
                all_complete=complete,resumed_closed=resumed_closed,
                passed=all(r['closed'] and r['fd_released'] and r['complete'] and not r['errors'] for r in rows)
                       and resumed and resumed_closed and complete and len(audit['sessions'])<=10)


def evaluate(args):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import PyEyeProvider,build_reader
    from jccbrain.vision.eye.augment_diagnostics import AugmentDiagnostics
    from jccbrain.data.db import GameDB
    from jccbrain.core import paths
    import shutil
    out=validate_output(args.output,(args.input,args.samples,ROOT/'src'),base=BASE)
    before=manifest(args.input);assert before==json.loads((args.input.parent/'input-manifest.json').read_text())
    out.mkdir(parents=True);write(out/'inputs-before.json',before)
    runtime=out/'runtime';runtime.mkdir();os.environ['JCC_HOME']=str(runtime)
    shutil.copy2(args.input/'jcc.sqlite3',runtime/'db.sqlite3')
    for name in ('augment_icons','unit-learning'):shutil.copytree(args.input/name,runtime/name)
    shutil.copytree(ROOT/'var/item_icons',runtime/'item_icons')
    paths.legacy_gallery_dir=lambda:ROOT/'game-eye/gallery'
    db=GameDB(runtime/'db.sqlite3');images={};source_sha={}
    for name in CASES:
        path=ROOT/'game-eye/samples/result_2nd.jpg' if name=='RESULT' else args.samples/name
        images[name]=load_image(path);source_sha[name]=sha(path)
    clock=[0.];window=NS(window_id=1,width=1774,height=1136)
    def provider(label,enabled):
        p=PyEyeProvider(db=db,gallery_dir=runtime/label);p.start=lambda:None;p.interval_override_s=0
        p._reader=build_reader(db,runtime/label,'rapid')
        p._augment_diagnostics=AugmentDiagnostics(runtime/'augment-diagnostics',clock=lambda:clock[0])
        if not enabled:p._observe_augment_diagnostics=lambda *_:None
        return p
    old,new=provider('baseline',False),provider('candidate',True)
    results=[];duplicates=[];stops=[]
    for i,name in enumerate(CASES):
        if i>=len(POSITIVE):
            window.window_id=i+2;old,new=provider(f'base-{i}',False),provider(f'new-{i}',True)
        clock[0]=i*20.;im=images[name]
        def both():
            rows=[]
            for p in (old,new):
                p._capture=NS(grab=lambda:(im,copy.copy(window)))
                raw=copy.deepcopy(p.read().raw);raw.pop('_ms',None);rows.append(raw)
            return dict(baseline=rows[0],candidate=rows[1],equal=rows[0]==rows[1])
        r=both();results.append(dict(id=name,source_sha=source_sha[name],**r))
        if name=='augment-taken/0006.png':
            clock[0]+=3.;duplicates.append(dict(id=name,clock=clock[0],note='same archived pixels repeated, explicit offline persistence probe, not fresh live capture',**both()))
        if i==3:
            # Exercise the actual Provider lifecycle, not a direct store.finish.
            d=new._augment_diagnostics;store=d.store
            old.stop();new.stop()
            stops.append(dict(closed=d.closed,fd_released=store.fd is None,
                              complete=json.loads((store.session/'manifest.json').read_text())['complete']))
        elif i>=len(POSITIVE):
            old.stop();new.stop()
    evidence=audit_store(runtime/'augment-diagnostics')
    perf=performance(out/'performance',images[CASES[0]],results[0]['candidate'],window)
    write(out/'performance.json',perf)
    lifecycle=provider_lifecycle(out/'lifecycle',images[CASES[0]],results[0]['candidate'],window)
    write(out/'stop-lifecycle.json',lifecycle)
    after=manifest(args.input);write(out/'inputs-after.json',after)
    source_after={name:sha(ROOT/'game-eye/samples/result_2nd.jpg' if name=='RESULT' else args.samples/name)
                  for name in CASES}
    write(out/'source-images.json',dict(before=source_sha,after=source_after))
    events=[e for s in evidence['sessions'] for e in s['events']]
    checks=dict(all_14_original_results_equal=len(results)==14 and all(r['equal'] for r in results),
        repeated_0006_equal=all(r['equal'] for r in duplicates),
        original_counts=[len(r['candidate']['augments_taken']) for r in results[:4]]==[1,2,2,3],
        opponents_not_promoted=all(not r['candidate']['augments_taken'] for r in results[4:]),
        unknown_owner_recorded=any('unknown_owner' in e['reasons'] and any(f['players_count']==0 and f['stage']=='4-3' and len(f['augments_taken'])==2 for f in e['frames']) for e in events),
        recovery_three_recorded=any('confirmed_changed' in e['reasons'] and any(len(f['augments_taken'])==3 for f in e['frames']) for e in events),
        inputs_unchanged=before==after,
        source_images_unchanged=source_sha==source_after,
        provider_stop_complete=all(r['closed'] and r['fd_released'] and r['complete'] for r in stops),
        twelve_stops_and_resume=lifecycle['passed'],
        performance_targets=perf['ordinary_target'] and perf['triggered_target'])
    summary=dict(checks=checks,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/jccbrain/vision/eye/augment_diagnostics.py',ROOT/'src/jccbrain/vision/pyeye.py',Path(__file__).resolve())},
        reader_source=sys.modules['jccbrain.vision.eye.reader'].__file__,
        diagnostics_source=sys.modules['jccbrain.vision.eye.augment_diagnostics'].__file__,
        note='Offline controlled window IDs/cadence, no native capture/input. Diagnostics capability only; no augment ownership/recognition fix.',
        results=results,duplicates=duplicates,evidence=evidence,
        timing_status=perf['timing_status'],ordinary_p95_ms=perf['ordinary_p95_ms'],triggered_p95_ms=perf['triggered_p95_ms'])
    write(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('results','duplicates','evidence')},ensure_ascii=False,indent=2))
    if not all(v for k,v in checks.items() if k!='performance_targets') or (
            perf['timing_status'].startswith('measured') and not checks['performance_targets']):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=BASE/'input')
    p.add_argument('--samples',type=Path,default=ROOT/'var/vision-module-optimization/replay-samples')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for k in vars(a):setattr(a,k,getattr(a,k).absolute())
    evaluate(a)
