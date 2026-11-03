"""L2 offline archive/interface audit (default), or bounded product live probe.

Default NEVER queries/clicks the game. L1's archive has only one clean pre-click
sample per target and timers 10/10/7: it cannot prove a natural active transaction
with the L2 >=15s entrance gate. We retain that rejection, audit product candidate
geometry, then associate the ORIGINAL logged clicks with real two-panel receipts
through the product controller. This is decomposed offline interface evidence,
not a simulated action presented as an actual automatic game click.

--live-dry-run captures/observes with the actual product scheduler, zero actions
or promotions. --live explicitly permits the same scheduler/executor to inspect
at most two targets within 45 seconds, without activating a window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import cv2

from scripts.eval_unit_learning import require_new_output, write, predict_holdout
from scripts.eval_vision_b8 import manifest, process_snapshot
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.modes.base import ModeSink
from jccbrain.modes.unit_learning import UnitLearningRunner, body_point, candidates, crop_bounds, window_key
from jccbrain.vision.eye.capture import load_image
from jccbrain.vision.eye.unit_learning import FrameSource, ClickTransaction, LearningSample, image_hash, file_hash, LearningCancelled
from jccbrain.vision.eye.unit_panel import bind_selection, presence
from jccbrain.vision.eye.units import Gallery, UnitTracker, feature_print
from jccbrain.vision.pyeye import PyEyeProvider, build_reader

BASE=ROOT/'var/vision-module-optimization'


def main_state(reader):
    """Recognition memory only; shared immutable models/store I/O caches excluded."""
    names=('cache','cell_cache','_frame_id','augment_desc_cache','phase','last_stage','last_timer','roster','last_players')
    state={k:getattr(reader,k,None) for k in names}
    state['round_bar']=reader.round_bar.__dict__
    state['units']={k:v for k,v in reader.units.__dict__.items() if k not in ('gallery','items')}
    state['gallery']={k:v for k,v in reader.units.gallery.__dict__.items() if k!='verified_store'}
    state['items']={k:v for k,v in reader.items.__dict__.items() if k!='last_ms'} if reader.items else None
    state['augments']=reader.augments_taken.__dict__ if reader.augments_taken else None
    state['ocr']={k:getattr(reader.ocr,k,None) for k in ('cache','words')}
    return {k:hashlib.sha256(pickle.dumps(v)).hexdigest() for k,v in state.items()}


def prepare(args):
    out=require_new_output(args.output,[args.archive,args.gallery])
    before=dict(db=file_hash(args.db),gallery=manifest(args.gallery),archive=manifest(args.archive),
                icons={n:manifest(ROOT/'var'/n) for n in ('item_icons','augment_icons')},
                seeds=manifest(ROOT/'game-eye/gallery'))
    write(out/'inputs-before.json',before)
    home=out/'home';home.mkdir();shutil.copy2(args.db,home/'db.sqlite3');shutil.copytree(args.gallery,home/'gallery')
    for n in ('item_icons','augment_icons'):shutil.copytree(ROOT/'var'/n,home/n)
    os.environ['JCC_HOME']=str(home)
    from jccbrain.data.db import GameDB
    db=GameDB(home/'db.sqlite3')
    write(out/'source.json',dict(vision=manifest(ROOT/'src/jccbrain/vision'),modes=manifest(ROOT/'src/jccbrain/modes'),
        settings=file_hash(ROOT/'src/jccbrain/core/settings.py'),schema=file_hash(ROOT/'src/jccbrain/webui/settings_schema.py'),
        script=file_hash(Path(__file__)),argv=sys.argv,reader_source=__import__('jccbrain.vision.pyeye',fromlist=['']).__file__))
    return out,home,db,before


def finish_inputs(args,out,before):
    after=dict(db=file_hash(args.db),gallery=manifest(args.gallery),archive=manifest(args.archive),
               icons={n:manifest(ROOT/'var'/n) for n in ('item_icons','augment_icons')},seeds=manifest(ROOT/'game-eye/gallery'))
    write(out/'inputs-after.json',after)
    return before==after


def offline(args):
    out,home,db,inputs=prepare(args)
    reader=build_reader(db,home/'gallery',args.ocr)
    provider=PyEyeProvider(db=db,gallery_dir=home/'gallery');provider._reader=reader
    ctl=provider.configure_unit_learning(home/'unit-learning',mode='panel');source=ctl.source
    gallery=reader.units.gallery
    baseline=Gallery(home/'baseline',gallery.season,gallery.pool)
    baseline.items=[(n,v.copy()) for n,v in gallery.items]
    events=[json.loads(l) for l in (args.archive/'events.jsonl').read_text().splitlines()]
    frames={r['file']:r for r in events if r['kind']=='frame'}
    clicks={r['label']:r for r in events if r['kind']=='click'}
    settings=Settings();settings.vision.unit_learning_mode='auto'
    executor=NS(dry_run=False,execute=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('offline must not send input')))
    guard=NS(stopped=False)
    ctx=NS(vision=provider,settings=settings,executor=executor,guard=guard)
    clock=[0.];runner=UnitLearningRunner(ctx,ModeSink(),clock=lambda:clock[0])
    # Adapter uses genuine archived receipts as the already-read sample. It does
    # not forge a capturer invocation or change original times/window/timer.
    runner.provider=provider;runner._configured='auto'
    crops=out/'crops';crops.mkdir()
    records=dict(groups=[],train=[],holdout=[],actions=[],note='OFFLINE: no native input; original click logs are association evidence only.')
    for key in ('board-a','board-b','board-c'):
        ev=frames[key+'-before.png'];im=load_image(args.archive/ev['file'])
        context=reader.read(im)
        before=source.archived(args.archive,ev,context.get('phase'),context.get('viewing'),context.get('stage'))
        provider._last=GameState.from_dict(context);provider._learning_frame=before;provider._last_img=im
        clock[0]=before.monotonic+.2
        # This runs the actual product entrance. The real timer rejection is
        # retained; no input-monitor mock can turn it into a natural active pass.
        runner.maybe_run(provider._last)
        gate=runner.status
        frozen=main_state(reader)
        temp=reader._foreign_context()
        _,detected=reader.units.detect(im)
        geometry=[];overlay=im.copy()
        for u in detected:
            point=body_point(u,im)
            row=dict(x=u.x,y=u.y,zone=u.zone,slot=u.slot,body_point=point,crop_bounds=crop_bounds(u,im))
            geometry.append(row)
            if point:
                cv2.circle(overlay,(round(point[0]*im.shape[1]),round(point[1]*im.shape[0])),7,(0,0,255),2)
                cv2.imwrite(str(crops/f'{key}-{len(geometry)}.png'),u.crop)
        cv2.imwrite(str(out/f'{key}-body-points.jpg'),overlay)
        click=clicks[key]
        tx=ClickTransaction(before,tuple(click['point']),click['monotonic'],click['window']['window_id'])
        group=dict(key=key,before=before.metadata(),actual_entrance=gate,geometry=geometry,
                   archived_click=click,active_automatic_transaction_measured=False,
                   before_has_panel=presence(im)>=.9,
                   original_click_was_not_sent_by_this_script=True)
        # Three genuine receipts, private scene OCR, original click association.
        # Only the controller can bind/name/promote; manual-review is not open.
        for i in range(3):
            ev=frames[f'{key}-{i}.png'];current_im=load_image(args.archive/ev['file'])
            scene=temp.read(current_im)
            frame=source.archived(args.archive,ev,scene.get('phase'),scene.get('viewing'),scene.get('stage'))
            proposals=candidates(reader,before,frame)
            result=ctl.process(frame,tx)
            records['train'].append(dict(file=ev['file'],transaction=key,result=result,
                automatic_geometry_candidates=[vars(t) for t in proposals],capture=frame.metadata()))
            if i==0:
                bound,_,_=bind_selection(current_im,reader.units.detect(current_im)[1])
                group['selected_geometry']=next((r for r in geometry if bound and r['zone']==bound.zone
                    and r['slot']==bound.slot and abs(r['x']-bound.x)<.012 and abs(r['y']-bound.y)<.02),None)
        group['main_before']=frozen;group['main_after']=main_state(reader)
        group['main_frozen']=group['main_before']==group['main_after']
        records['groups'].append(group)
        if key=='board-b':records['holdout'].append(predict_holdout('holdout-hide-1.png',NS(live=args.archive),reader,baseline,gallery,ctl.store))
    records['holdout'].append(predict_holdout('c-hide-0.png',NS(live=args.archive),reader,baseline,gallery,ctl.store))
    # An actual persistent tracker consumes confirmed vectors on later pixels.
    tracker=UnitTracker(gallery);records['ongoing_reader']=[]
    for name in ('holdout-hide-1.png','c-hide-0.png'):
        im=load_image(args.archive/name)
        records['ongoing_reader'].append(dict(file=name,prediction=tracker.observe(im,'planning',None,
            prepared=tracker.prepare(im))))
    records['verified']=ctl.store.entries()
    # Separate recovery experiment, same genuine timestamps/captures, no actions.
    # An injected user-takeover callback cancels frame0 during OCR; only the two
    # later archived receipts (1/2) may form the new PASSIVE confirmation chain.
    from jccbrain.vision.eye.unit_learning import UnitLearningController, UnitLearningStore
    recovery_reader=build_reader(db,home/'recovery-gallery',args.ocr)
    recovery_source=FrameSource()
    recovery_store=UnitLearningStore(home/'cancel-recovery',gallery.season,gallery.pool)
    recovery=UnitLearningController(recovery_reader,recovery_store,recovery_source,mode='panel')
    recovery_context=recovery_reader._foreign_context()
    recovery_reader.read(load_image(args.archive/'board-b-before.png'))
    # Initialize private scene from a fresh own frame, as in the real Provider.
    recovery_context=recovery_reader._foreign_context()
    recovery_results=[]
    for i in range(3):
        event=frames[f'board-b-{i}.png'];scene=recovery_context.read(load_image(args.archive/event['file']))
        frame=recovery_source.archived(args.archive,event,scene.get('phase'),scene.get('viewing'),scene.get('stage'))
        checks=[0]
        def cancel_after_ocr():
            checks[0]+=1
            if checks[0]==2:raise LearningCancelled('simulated_user_input_after_ocr')
        result=recovery.process(frame,check=cancel_after_ocr if i==0 else None)
        recovery_results.append(dict(file=event['file'],result=result,verified=len(recovery_store.entries())))
    records['cancel_recovery']=dict(input_change_simulated=True,actions=0,results=recovery_results)
    write(out/'predictions.json',records)
    write(out/'prediction-seal.json',dict(predictions_sha=file_hash(out/'predictions.json'),identity_truth_read=False,
        heldout_used_for_training=False,source_sha=manifest(ROOT/'src/jccbrain'),
        archive_events_sha=file_hash(args.archive/'events.jsonl')))
    # Human labels only enter after predictions, targets and vectors are sealed.
    human=json.loads((args.archive/'manual-review.json').read_text())
    accepted=[]
    for row in records['train']:
        if row['result']['status']!='verified':continue
        label=next(t for t in human['training'] if row['file'] in t['confirm'])
        accepted.append(dict(file=row['file'],actual=row['result']['panel']['name'],expected=label['name'],
                             correct=row['result']['panel']['name']==label['name']))
    held=[]
    for label in human['held_out']:
        row=next(r for r in records['holdout'] if r['file']==label['frame'])
        x,y=label['click'];us=[u for u in row['units'] if u['rect'][0]<=x*1774<u['rect'][2] and u['rect'][1]<=y*1136<u['rect'][3]]
        u=us[0] if len(us)==1 else None
        held.append(dict(file=label['frame'],expected=label['name'],prediction=u,
            unknown_to_correct=bool(u and u['before_name'] is None and u['after_name']==label['name']),
            wrong_new=bool(u and u['before_name']!=u['after_name'] and u['after_name'] not in (None,label['name']))))
    summary=dict(offline=True,actions_simulated=True,actual_actions=0,accepted=accepted,heldout=held,
        verified=len(records['verified']),main_frozen=all(r['main_frozen'] for r in records['groups']),
        archive_active_gates=[g['actual_entrance']['reason'] for g in records['groups']],
        natural_active_transaction_measured=False,live_click_learning='未测（由指挥实机验收）',
        limitation='Only one clean before sample per target; genuine timers 10/10/7 fail >=15. Original archived clicks associate controller proof; no timestamps/window/timers fabricated.',
        cancel_recovery=[r['result']['status'] for r in recovery_results],
        inputs_unchanged=finish_inputs(args,out,inputs))
    summary['interface_passed']=bool(len(accepted)==2 and all(r['correct'] for r in accepted)
        and len({r['expected'] for r in held if r['unknown_to_correct']})>=2 and not any(r['wrong_new'] for r in held)
        and summary['main_frozen'] and summary['inputs_unchanged']
        and summary['cancel_recovery']==['cancelled','pending','verified'])
    write(out/'summary.json',summary);db.close()
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary['interface_passed'] else 1


def live_probe(args):
    from jccbrain.actuator.window import WindowLocator
    from jccbrain.actuator.executor import MacExecutor, DryRunExecutor
    from jccbrain.actuator.layout import Layout
    from jccbrain.actuator.safety import SafetyGuard
    from jccbrain.vision.eye.capture import WindowCapture, CaptureError
    from jccbrain.vision.eye.input_monitor import HIDMonitor

    out,home,db,inputs=prepare(args)
    settings=Settings();settings.vision.unit_learning_mode='auto';settings.actuator.dry_run=not args.live
    settings.actuator.window_owner=args.window_owner
    original=WindowLocator(args.window_owner);first=original.find(0)
    write(out/'first-window.json',vars(first) if first else None)
    class BoundLocator:
        def find(self,*_):
            current=original.find(0)
            return current if window_key(current)==window_key(first) else None
    locator=BoundLocator();monitor=HIDMonitor(locator)
    guard=SafetyGuard(settings.actuator.max_actions_per_sec,settings.actuator.failsafe_corner)
    layout=Layout()
    executor=MacExecutor(layout,guard,settings.actuator,locator) if args.live else DryRunExecutor(layout,guard,0)
    provider=PyEyeProvider(args.window_owner,db=db,gallery_dir=home/'gallery',ocr=args.ocr)
    trace=(out/'trace.jsonl').open('w');captures=(out/'captures.jsonl').open('w')
    (out/'frames').mkdir()
    def emit(kind,data):
        if kind=='sample':
            sample=data['sample'];data=dict(frame=sample.frame.metadata(),state=sample.state)
        trace.write(json.dumps(dict(at=time.monotonic(),kind=kind,**data),ensure_ascii=False,default=str)+'\n');trace.flush()
    raw_capture=WindowCapture(locator,backend='cg')
    class LoggedCapture:
        backend='cg'
        def __init__(self):self.locator=locator;self.i=0
        def grab(self):
            pre=monitor.snapshot();im,win=raw_capture.grab();post=monitor.snapshot()
            self.i+=1;file=f'{self.i:04}.png';cv2.imwrite(str(out/'frames'/file),im)
            captures.write(json.dumps(dict(file=file,sha256=file_hash(out/'frames'/file),pixel_hash=image_hash(im),
                at=time.monotonic(),window=vars(win),input_before=pre.to_json(),input_after=post.to_json()))+'\n');captures.flush()
            return im,win
    provider._reader=build_reader(db,home/'gallery',args.ocr);provider._capture=LoggedCapture()
    warm=getattr(provider._reader.ocr,'warmup',None)
    if warm:warm()
    ctx=NS(settings=settings,vision=provider,guard=guard,executor=executor,locator=locator)
    deadline=time.monotonic()+args.max_seconds
    runner=UnitLearningRunner(ctx,ModeSink(),stopped=lambda:time.monotonic()>=deadline,monitor=monitor,trace=emit)
    observations=[];error=None;processes=process_snapshot();start=time.monotonic()
    preflight=executor.preflight() if args.live else []
    try:
        if first is None:raise CaptureError('no_initial_window')
        if preflight:raise RuntimeError('; '.join(preflight))
        while time.monotonic()<deadline and runner.status.get('attempted',0)<args.max_targets:
            state=provider.read();frozen=main_state(provider._reader)
            last=provider._last;last_t=provider._last_t;last_img=provider._last_img
            ran=runner.maybe_run(state)
            after=main_state(provider._reader)
            observations.append(dict(state=state.raw,ran=ran,status=runner.status,main_before=frozen,main_after=after,
                main_frozen=frozen==after and provider._last is last and provider._last_t==last_t and provider._last_img is last_img))
            if runner.status.get('verified',0):
                # Later independent capture, never passed into learning, for
                # actual matching/heldout review. No manual hero selection.
                runner.close();later=provider.read()
                emit('heldout',dict(state=later.raw,frame_index=provider._capture.i,used_for_training=False))
                break
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';emit('error',dict(error=error))
    finally:
        runner.close();trace.close();captures.close()
    write(out/'observations.json',observations)
    summary=dict(live=bool(args.live),dry_run=not args.live,elapsed_s=time.monotonic()-start,max_seconds=args.max_seconds,
        max_targets=args.max_targets,status=runner.status,error=error,preflight=preflight,
        main_frozen=all(r['main_frozen'] for r in observations),processes=processes,
        timing_status='未测（环境不净）' if processes['suspects'] else 'not_benchmarked',
        inputs_unchanged=finish_inputs(args,out,inputs),
        note='Actual product scheduler and existing executor only. No activation. Input counters never reset after own actions. Inspect trace/input snapshots for synthetic-event vs HID relationship; no automatic live-accuracy claim.')
    write(out/'summary.json',summary);db.close();print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if not error and summary['inputs_unchanged'] and summary['main_frozen'] else 1


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--db',type=Path,default=BASE/'batch-B8/data/jcc-vf-db.sqlite3')
    parser.add_argument('--gallery',type=Path,default=BASE/'batch-B8/data/jcc-vf-gallery')
    parser.add_argument('--archive',type=Path,default=BASE/'batch-L1/conductor/live-preparation')
    parser.add_argument('--ocr',choices=['rapid','vision'],default='rapid')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--live',action='store_true',help='EXPLICIT actual inspection clicks; at most two targets / 45s, never activate')
    group.add_argument('--live-dry-run',action='store_true',help='Native read-only capture/input probe; zero actions/promotions')
    parser.add_argument('--window-owner',default='金铲铲')
    parser.add_argument('--max-seconds',type=float,default=45)
    parser.add_argument('--max-targets',type=int,default=2)
    args=parser.parse_args();args.output=args.output.resolve()
    if not 0<args.max_seconds<=45 or not 1<=args.max_targets<=2:parser.error('bounded probe requires seconds<=45 and targets<=2')
    return live_probe(args) if args.live or args.live_dry_run else offline(args)


if __name__=='__main__':raise SystemExit(main())
