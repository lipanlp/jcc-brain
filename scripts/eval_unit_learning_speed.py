"""P2 paired OFFLINE scene/transaction replay. Never capture or send native input.

Each worker imports --repo/src, warms the same OCR backend, then processes the
same eight L2 images in order. Archived receipt times stay original. Transaction
replay has controlled input/guard/capture adapters; its duration is NOT live
latency. Scene CPU/wall timings are real processing measurements (no mock OCR).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'var/vision-module-optimization'
SCENE_KEYS = ('stage','timer','phase','viewing','banner','next_round','round_type',
              'players','players_stale','right_panel','hp','placement')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path):
    return {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def freeze(reader):
    attrs = ('cache','cell_cache','_frame_id','augment_desc_cache','phase','last_stage','last_timer',
             'roster','last_players','round_bar','items','augments_taken')
    state = {k:getattr(reader,k,None) for k in attrs}
    state['units'] = {k:v for k,v in reader.units.__dict__.items() if k not in ('gallery','items')}
    state['ocr'] = {k:getattr(reader.ocr,k,None) for k in ('cache','words')}
    return {k:hashlib.sha256(pickle.dumps(v)).hexdigest() for k,v in state.items()}


def inputs(args):
    # Hash only immutable input; never hash/write live user paths.
    return dict(snapshot=manifest(args.input), live_frames=manifest(args.live/'frames'),
                live_captures=sha(args.live/'captures.jsonl'),live_trace=sha(args.live/'trace.jsonl'),
                l1=manifest(args.l1), gallery=manifest(args.gallery),
                db=sha(args.runtime/'jcc.sqlite3'), icons=manifest(args.runtime/'augment_icons'),
                item_icons=manifest(ROOT/'var/item_icons'),replay=manifest(args.replay),
                seeds=manifest(ROOT/'game-eye/gallery'),
                ended=sha(ROOT/'game-eye/samples/result_2nd.jpg'),
                no_hud=sha(BASE/'batch-F1/executor/current-1.png'))


def worker(args):
    sys.path.insert(0, str(args.repo));sys.path.insert(0, str(args.repo/'src'))
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader
    from jccbrain.data.db import GameDB
    from jccbrain.core import paths
    out=args.output;out.mkdir()
    home=out/'home';home.mkdir();os.environ['JCC_HOME']=str(home)
    shutil.copy2(args.runtime/'jcc.sqlite3',home/'db.sqlite3')
    shutil.copytree(args.runtime/'augment_icons',home/'augment_icons')
    shutil.copytree(ROOT/'var/item_icons',home/'item_icons')
    shutil.copytree(args.gallery,home/'gallery')
    db=GameDB(home/'db.sqlite3')
    # A git archive has no ignored legacy seeds. Supply exactly the same
    # immutable input directory to BOTH repositories, independently of cwd.
    paths.legacy_gallery_dir=lambda:ROOT/'game-eye/gallery'
    reader=build_reader(db,home/'gallery',args.ocr)
    # build_reader already loaded cached icons. prepare() downloads missing
    # entries, so it must NEVER be called in this offline evaluator.
    if warmup:=getattr(reader.ocr,'warmup',None):warmup()
    images=[load_image(args.live/'frames'/f'{i:04d}.png') for i in range(2,11)]
    # Same ordinary warm read on both sides, then one unmeasured private pass.
    initial=reader.read(images[0]);ctx=reader._foreign_context()
    fast=hasattr(ctx,'_read_learning_scene')
    read=ctx._read_learning_scene if fast else ctx.read
    for im in images[1:]:read(im)
    ctx=reader._foreign_context();read=ctx._read_learning_scene if fast else ctx.read
    operations=[]
    original=ctx._begin_cells
    def cells(im,only=None,*a,**kw):
        operations.append(dict(kind='regions',only=sorted(only) if only is not None else 'full'))
        return original(im,only,*a,**kw)
    ctx._begin_cells=cells
    for name in ('prepare','observe','read_current'):
        fn=getattr(ctx.units,name)
        def call(*a,_fn=fn,_name=name,**kw):
            operations.append(dict(kind='units',method=_name));return _fn(*a,**kw)
        setattr(ctx.units,name,call)
    frozen=freeze(reader);rows=[]
    for i,im in enumerate(images[1:],3):
        wall=time.perf_counter();cpu=time.process_time()
        result=read(im)
        wall_ms=(time.perf_counter()-wall)*1000;cpu_ms=(time.process_time()-cpu)*1000
        rows.append(dict(file=f'{i:04d}.png',sha256=sha(args.live/'frames'/f'{i:04d}.png'),
            wall_ms=wall_ms,cpu_ms=cpu_ms,phases_ms=dict(ctx.last_ms),
            scene={k:result[k] for k in SCENE_KEYS if k in result},output=result))
    measured_frozen=freeze(reader)==frozen
    # Ordinary read remains the default, compare complete outputs independently.
    normal=build_reader(db,home/'normal-gallery',args.ocr)
    normal.read(images[0])
    normal_rows=[normal.read(im) for im in images[1:]]
    negative=[]
    for label,path,entrance in (
        ('spectate',args.replay/'spectate/0001.png',True),
        ('carousel',args.replay/'carousel/0001.png',True),
        ('dialog',args.replay/'dialog/0001.png',True),
        ('ended',ROOT/'game-eye/samples/result_2nd.jpg',True),
        ('no_hud',BASE/'batch-F1/executor/current-1.png',True),
        ('old_stage',args.l1/'board-b-before.png',False),
        ('short_timer',args.l1/'board-a-before.png',True)):
        context=reader._foreign_context();read_negative=context._read_learning_scene if fast else context.read
        image=load_image(path)
        result=read_negative(image)
        negative.append(dict(label=label,path=str(path),sha256=sha(path),state=result,
                             reason=scene_gate(result,initial,entrance,size=image.shape[:2][::-1])))
    negative.append(dict(label='window_changed',path=str(args.live/'frames/0002.png'),state=initial,
        metadata_controlled=True,note='real pixels; simulated window-id change at the scene gate',
        reason=scene_gate(initial,initial,False,window_changed=True)))
    flow=replay_transaction(args, reader, fast, out)
    panels=replay_l1_panels(args, reader, fast, out)
    result=dict(repo=str(args.repo),reader_source=sys.modules['jccbrain.vision.eye.reader'].__file__,
        evaluator_sha=sha(Path(__file__)),
        source_sha=manifest(args.repo/'src/jccbrain/vision'),runner_sha=sha(args.repo/'src/jccbrain/modes/unit_learning.py'),
        backend=reader.ocr.name,fast=fast,seed_source=str(ROOT/'game-eye/gallery'),
        warmup='model + identical ordinary initial + eight unmeasured private reads',
        rows=rows,operations=operations,main_frozen=measured_frozen,normal_outputs=normal_rows,
        negatives=negative,flow=flow,l1_panels=panels)
    write(out/'result.json',result)


def scene_gate(state,initial,entrance=True,*,size=(1774,1136),window_changed=False):
    """Call the product scene gate with genuine pixel outputs, no input adapter."""
    from jccbrain.modes.unit_learning import UnitLearningRunner
    from jccbrain.vision.eye.unit_learning import LearningCancelled
    from jccbrain.core.settings import Settings
    from jccbrain.actuator.window import WindowInfo
    from dataclasses import replace
    window=WindowInfo('game','title',86880,3380,151,1774,1136,39973)
    r=UnitLearningRunner(NS(settings=Settings()),NS(),clock=lambda:100.)
    frame=NS(phase=state.get('phase'),viewing=state.get('viewing'),stage=state.get('stage'),
             monotonic=100.,window=replace(window,width=size[0],height=size[1],
                 window_id=window.window_id+int(window_changed)),size=tuple(size))
    r._initial=None if entrance else NS(state=initial,frame=NS(window=window))
    try:r._scene(NS(state=state,frame=frame),entrance=entrance)
    except LearningCancelled as exc:return str(exc)
    return 'accepted'


def replay_l1_panels(args, reader, fast, out):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.eye.unit_learning import FrameSource, UnitLearningStore, UnitLearningController, ClickTransaction
    events=[json.loads(l) for l in (args.l1/'events.jsonl').read_text().splitlines()]
    frames={e['file']:e for e in events if e['kind']=='frame'}
    clicks={e['label']:e for e in events if e['kind']=='click'}
    rows=[];frozen=freeze(reader)
    for group in ('board-a','board-b','board-c'):
        source=FrameSource();context=reader._foreign_context()
        read=context._read_learning_scene if fast else context.read
        store=UnitLearningStore(out/('l1-offline-'+group),reader.units.gallery.season,reader.units.gallery.pool)
        ctl=UnitLearningController(reader,store,source,mode='panel')
        def frame(name):
            state=read(load_image(args.l1/name));ev=frames[name]
            return source.archived(args.l1,ev,state.get('phase'),state.get('viewing'),state.get('stage')),state
        before,state=frame(group+'-before.png');click=clicks[group]
        tx=ClickTransaction(before,tuple(click['point']),click['monotonic'],click['window']['window_id'])
        results=[]
        for i in range(2):
            current,state=frame(f'{group}-{i}.png')
            results.append(dict(file=f'{group}-{i}.png',state={k:state[k] for k in SCENE_KEYS if k in state},
                                result=ctl.process(current,tx)))
        rows.append(dict(group=group,results=results,verified=[e['name'] for e in store.entries()],
                         fresh_ocr_calls=ctl.panel.calls,main_frozen=freeze(reader)==frozen))
    return rows


def replay_transaction(args, reader, fast, out):
    """Real archived pixels/times; controlled input observations, never native I/O.

    The old two redundant captures remain in the original stream. The candidate
    does not consume 0007 (close entry) or 0010 (restore) unless freshness requires
    them. The close before_do receives the SAME 0008 on both sides. All consumed
    frames retain original receipt times; no synthetic fresh capture is issued.
    """
    from jccbrain.actuator.actions import ClickPoint
    from jccbrain.actuator.executor import Executor
    from jccbrain.actuator.window import WindowInfo
    from jccbrain.core.models import GameState
    from jccbrain.core.settings import Settings
    from jccbrain.modes.unit_learning import UnitLearningRunner
    from jccbrain.vision.pyeye import PyEyeProvider
    from jccbrain.vision.eye.input_monitor import InputSnapshot, EVENT_NAMES
    from jccbrain.vision.eye.unit_learning import LearningSample
    from jccbrain.vision.eye.capture import load_image
    captures=[json.loads(l) for l in (args.live/'captures.jsonl').read_text().splitlines()]
    original_trace=[json.loads(l) for l in (args.live/'trace.jsonl').read_text().splitlines()]
    receipts={r['frame']['pixel_hash']:r['frame'] for r in original_trace if r['kind']=='sample'}
    archive=out/'archive';archive.mkdir();events=[]
    for c in captures:
        if c['pixel_hash'] not in receipts:continue
        f=receipts[c['pixel_hash']]
        shutil.copy2(args.live/'frames'/c['file'],archive/c['file'])
        events.append(dict(kind='frame',file=c['file'],sha256=c['sha256'],monotonic=f['monotonic'],
                           window=c['window'],source='L2 ORIGINAL receipt, converted metadata only'))
    (archive/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    settings=Settings();settings.vision.unit_learning_mode='auto'
    p=PyEyeProvider();p._reader=reader
    ctl=p.configure_unit_learning(out/'offline-learning',mode='panel',automatic_record=False)
    context=reader._foreign_context();read=context._read_learning_scene if fast else context.read
    operations=[];begin=context._begin_cells
    def cells(im,only=None,*a,**kw):
        operations.append(dict(kind='regions',only=sorted(only) if only is not None else 'full'))
        return begin(im,only,*a,**kw)
    context._begin_cells=cells
    for name in ('prepare','observe','read_current'):
        fn=getattr(context.units,name)
        def call(*a,_fn=fn,_name=name,**kw):
            operations.append(dict(kind='units',method=_name));return _fn(*a,**kw)
        setattr(context.units,name,call)
    virtual=[events[0]['monotonic']+.18];counts={n:0 for n in EVENT_NAMES};mouse=[4480.03125,740.16015625]
    window=WindowInfo(**events[0]['window']);actions=[];samples=[];trace=[];waits=[]
    class Monitor:
        def start(self):return True
        def stop(self):return True
        def clear_expected(self):pass
        def snapshot(self):
            return InputSnapshot(virtual[0],True,10.,tuple(counts.items()),False,tuple(mouse),window.pid,window,
                                 generation=1,own_counters=tuple(counts.items()),tap_healthy=True)
        def expect_action(self,*_):return len(actions)+1
        def finish_action(self,*_,**__):return self.snapshot()
    def wait(seconds):waits.append(seconds);virtual[0]+=seconds
    class ReplayExecutor(Executor):
        dry_run=False
        def _do(self,action):
            actions.append(dict(action=action.describe(),at=virtual[0],native_input_sent=False))
            names=('MouseMoved','LeftMouseDown','LeftMouseUp') if isinstance(action,ClickPoint) else ('MouseMoved',)
            for n in names:counts[n]+=1
            mouse[:]=window.to_screen(*action.point) if isinstance(action,ClickPoint) else action.screen_point
    guard=NS(stopped=False,check=lambda:None)
    ctx=NS(settings=settings,vision=p,guard=guard,executor=ReplayExecutor(None,guard))
    r=UnitLearningRunner(ctx,NS(on_log=lambda *_:None,on_action=lambda *_:None),monitor=Monitor(),
                         clock=lambda:virtual[0],wait=wait,trace=lambda k,d:trace.append((k,d)))
    r.provider=p;r._configured='auto';r._configure=lambda:True
    def archived(index):
        ev=next(e for e in events if e['file']==f'{index:04d}.png')
        start=time.perf_counter();state=read(load_image(archive/ev['file']));ms=(time.perf_counter()-start)*1000
        virtual[0]=max(virtual[0],ev['monotonic'])+ms/1000
        frame=ctl.source.archived(archive,ev,state.get('phase'),state.get('viewing'),state.get('stage'))
        samples.append(dict(file=ev['file'],original_monotonic=frame.monotonic,state=state,scene_wall_ms=ms))
        s=LearningSample(frame,state)
        if hasattr(s,'timings'):s.timings.update(scene_ms=ms)
        return s
    initial=archived(2)
    p._last=GameState.from_dict(initial.state);p._learning_frame=initial.frame
    queue=iter((3,4,5,6,8,9,10) if fast else (3,4,5,6,7,8,9,10))
    p.sample_unit_learning=lambda check=None:checked_sample(check)
    def checked_sample(check):
        if check:check()
        sample=archived(next(queue))
        if check:check()
        return sample
    # Account actual panel processing in the virtual replay clock separately;
    # panel OCR/store/feature work is REAL, clock advance is an offline adapter.
    process=p.process_unit_learning
    def timed_process(*a,**kw):
        start=time.perf_counter()
        try:return process(*a,**kw)
        finally:virtual[0]+=time.perf_counter()-start
    p.process_unit_learning=timed_process
    frozen=freeze(reader);start=time.perf_counter();replay_start=virtual[0];ran=r.maybe_run(p._last)
    trace_rows=[]
    for k,d in trace:
        if k in ('sample','input'):continue
        row=dict(kind=k,**d)
        if 'real' in row:row['product_execute_flag']=row.pop('real');row['native_input_sent']=False
        trace_rows.append(row)
    return dict(note='OFFLINE controlled replay: no native capture/input, duration is not live latency',
        adapter_timing='capture_ms/guard_ms=0 are placeholders, not live measurements; waits advance original archive clock',
        ran=ran,status=r.status,actions=actions,samples=samples,waits_s=waits,
        real_processing_wall_ms=(time.perf_counter()-start)*1000,replay_duration_ms=(virtual[0]-replay_start)*1000,
        main_frozen=freeze(reader)==frozen,operations=operations,panel_fresh_ocr_calls=ctl.panel.calls,
        verified=[dict(name=e['name'],id=e['id'],evidence=json.loads((ctl.store.verified/e['id']/'evidence.json').read_text()))
                  for e in ctl.store.entries()],
        original_capture_durations_ms=[(c['input_after']['at']-c['input_before']['at'])*1000 for c in captures],
        trace=trace_rows)


def main(args):
    if args.worker:return worker(args)
    sys.path.insert(0,str(ROOT))
    from scripts.eval_vision_b8 import process_snapshot
    if args.output.exists():raise ValueError('output must be a new directory')
    for protected in (args.input,args.live,args.l1,args.runtime,args.gallery,args.replay,args.baseline_repo):
        if args.output==protected or protected in args.output.parents:
            raise ValueError('output must be outside immutable inputs/source: '+str(protected))
    baseline_files=('src/jccbrain/vision/eye/reader.py','src/jccbrain/vision/pyeye.py',
                    'src/jccbrain/vision/eye/unit_learning.py','src/jccbrain/modes/unit_learning.py')
    for name in baseline_files:
        expected=subprocess.check_output(['git','show','03b1c54:'+name],cwd=ROOT)
        if (args.baseline_repo/name).read_bytes()!=expected:
            raise ValueError('baseline must contain unchanged git archive 03b1c54: '+name)
    args.output.mkdir(parents=True)
    before=inputs(args);write(args.output/'inputs-before.json',before)
    results=[]
    for repeat in range(args.repeat):
        for label,repo in (('baseline',args.baseline_repo),('candidate',ROOT)):
            name=f'{repeat+1}-{label}';out=args.output/name
            write(args.output/f'{name}-processes.json',process_snapshot())
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--repo',str(repo),
                 '--output',str(out),'--ocr',args.ocr]
            for key in ('input','live','l1','runtime','gallery','replay'):cmd.extend(['--'+key,str(getattr(args,key))])
            with (args.output/f'{name}.log').open('w') as log:
                p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(repo/'src')},stdout=log,stderr=log)
            if p.returncode:raise RuntimeError(f'{name} failed: see {name}.log')
            results.append((label,json.loads((out/'result.json').read_text())))
    pairs=[]
    for n in range(args.repeat):
        old,new=results[n*2][1],results[n*2+1][1]
        changes=[dict(file=a['file'],old=a['scene'],new=b['scene']) for a,b in zip(old['rows'],new['rows']) if a['scene']!=b['scene']]
        negatives=[dict(label=a['label'],old=a['reason'],new=b['reason'],safe=b['reason']!='accepted')
                   for a,b in zip(old['negatives'],new['negatives'])]
        pairs.append(dict(scene_changes=changes,normal_equal=old['normal_outputs']==new['normal_outputs'],
                          negatives=negatives,main_frozen=all(r['main_frozen'] and r['flow']['main_frozen'] for r in (old,new)),
                          success_both=all(r['flow']['status']['verified']==1 and len(r['flow']['actions'])==3 for r in (old,new)),
                          l1_panels=all(a['verified']==b['verified']
                              and a['fresh_ocr_calls']==b['fresh_ocr_calls']==2 and a['main_frozen'] and b['main_frozen']
                              and [(s['result']['status'],s['result']['reason'],s['result']['panel']) for s in a['results']]
                                 ==[(s['result']['status'],s['result']['reason'],s['result']['panel']) for s in b['results']]
                              for a,b in zip(old['l1_panels'],new['l1_panels']))))
    metrics={}
    for key in ('wall_ms','cpu_ms'):
        values={label:statistics.median(sum(row[key] for row in r['rows']) for l,r in results if l==label)
                for label in ('baseline','candidate')}
        values['reduction']=1-values['candidate']/values['baseline']
        values['unit']='median eight-frame batch ms'
        frame_values={label:statistics.median(statistics.median(row[key] for row in r['rows'])
                      for l,r in results if l==label) for label in ('baseline','candidate')}
        frame_values['reduction']=1-frame_values['candidate']/frame_values['baseline']
        values['per_frame_median']=frame_values;metrics[key]=values
    after=inputs(args);write(args.output/'inputs-after.json',after)
    checks=dict(inputs_unchanged=before==after,scene_equal=all(not p['scene_changes'] for p in pairs),
        normal_equal=all(p['normal_equal'] for p in pairs),negative_safe=all(n['safe'] for p in pairs for n in p['negatives']),
        main_frozen=all(p['main_frozen'] for p in pairs),transactions=all(p['success_both'] for p in pairs),
        l1_two_panels=all(p['l1_panels'] for p in pairs),
        scene_wall_target=metrics['wall_ms']['per_frame_median']['reduction']>=.30,
        scene_cpu_target=metrics['cpu_ms']['per_frame_median']['reduction']>=.30,
        scene_batch_wall_target=metrics['wall_ms']['reduction']>=.30,scene_batch_cpu_target=metrics['cpu_ms']['reduction']>=.30,
        no_full_work=all(not any(op['kind']=='units' or op.get('only')=='full' for op in r['operations']+r['flow']['operations'])
                         for label,r in results if label=='candidate'))
    summary=dict(checks=checks,pairs=pairs,metrics=metrics,repeat=args.repeat,argv=sys.argv,evaluator_sha=sha(Path(__file__)),
                 absolute_live_performance='未测（离线回放；用户游戏运行，未干预）',
                 note='Same ordered images, alternating independent processes, same OCR warmup. Paired offline observation only.')
    write(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo',type=Path,help='git archive 03b1c54 directory (required outside workers)')
    parser.add_argument('--output',type=Path,required=True,help='new directory, refuses overwrite')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--ocr',choices=('rapid','vision'),default='rapid')
    for key,default in dict(input=BASE/'batch-P2/input',live=BASE/'batch-L2/conductor/live-rework-2',
        l1=BASE/'batch-L1/conductor/live-preparation',runtime=BASE/'batch-F1/input',
        gallery=BASE/'batch-B8/data/jcc-vf-gallery',replay=BASE/'replay-samples').items():
        parser.add_argument('--'+key,type=Path,default=default)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--repo',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    for key in ('baseline_repo','output','input','live','l1','runtime','gallery','replay','repo'):
        if getattr(args,key,None) is not None:setattr(args,key,getattr(args,key).resolve())
    if args.repeat<1 or (not args.worker and args.baseline_repo is None):parser.error('baseline repo and positive repeat required')
    main(args)
