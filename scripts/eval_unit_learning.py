"""Offline panel/selection/learning evaluation, sealed predictions before human scoring.

Defaults: 25 heroes, 27+20 legacy negatives, fixed 37 full-session samples, and
L1 genuine capture transactions. No clicks. Existing refs/manual review are only
read by score_saved(), after all predictions and holdout outputs are saved.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import cv2
from scripts.eval_vision_b8 import manifest,process_snapshot
from jccbrain.vision.eye.capture import load_image
from jccbrain.vision.eye.unit_panel import PanelReader,PANEL_REGION,presence,bind_selection
from jccbrain.vision.eye.unit_learning import (UnitLearningStore,UnitLearningController,FrameSource,
    ClickTransaction,file_hash)
from jccbrain.vision.eye.image import crop
from jccbrain.vision.eye.units import UnitTracker,Gallery,feature_print,fp_distance

BASE=ROOT/'var/vision-module-optimization'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def require_new_output(output,inputs):
    output=output.resolve()
    if any(output==p.resolve() or output.is_relative_to(p.resolve()) for p in inputs if p.is_dir()):
        raise ValueError('output must be outside frozen inputs')
    output.mkdir(parents=True,exist_ok=False)
    return output


def compare_identity_outputs(old_rows,new_rows):
    """L1 exact non-identity comparison; no phase/opponent/items exemptions."""
    from scripts.eval_vision_b8 import differences
    if [r['_file'] for r in old_rows]!=[r['_file'] for r in new_rows]:
        raise ValueError('frame alignment mismatch')
    allowed=[];rejected=[];distribution={'before':Counter(),'after':Counter()}
    for a,b in zip(old_rows,new_rows):
        for side,row in [('before',a),('after',b)]:
            for zone in ('board','bench'):
                distribution[side].update(u['name'] for u in row['prediction'].get(zone,[])
                                          if u.get('name') and not u['name'].startswith('?'))
        for path,old,new in differences(a['prediction'],b['prediction']):
            field='.'.join(map(str,path))
            removed_inference=(path and path[0]=='auto_labeled' and 'auto_labeled' in a['prediction']
                               and 'auto_labeled' not in b['prediction'])
            permit=(removed_inference or (len(path)==3 and path[0] in ('board','bench')
                     and isinstance(path[1],int) and path[2] in ('name','unsure')))
            target=allowed if permit else rejected
            target.append(dict(_file=a['_file'],field=field,old=old,new=new))
    return dict(allowed=allowed,rejected=rejected,known_names={k:dict(v) for k,v in distribution.items()})


def rect(unit,image):
    h,w=image.shape[:2];bw=unit.bar.width
    return [max(0,int(unit.x*w-.8*bw)),unit.bar.max_y+max(1,round(.005*h)),
            min(w,int(unit.x*w+.8*bw)),min(h,unit.bar.max_y+max(1,round(.005*h))+round(1.9*bw))]


def scene_analysis(path,reader,pr,output,key):
    image=load_image(path);panel=pr.read(image);_,units=reader.units.detect(image)
    bound,diagnostics,reason=bind_selection(image,units)
    row=dict(file=str(path),sha256=file_hash(path),dimensions=list(image.shape[:2][::-1]),
        panel=panel.to_json(),binding=diagnostics,binding_reason=reason,
        bound=None if bound is None else dict(x=bound.x,y=bound.y,zone=bound.zone,slot=bound.slot),
        units=[dict(x=u.x,y=u.y,zone=u.zone,slot=u.slot,rect=rect(u,image)) for u in units])
    if panel.present:
        cv2.imwrite(str(output/(key+'-panel.png')),crop(image,PANEL_REGION))
    if bound:
        cv2.imwrite(str(output/(key+'-bound.png')),bound.crop)
    return row


def run_evaluation(args):
    inputs=[args.heroes,args.live,args.samples,args.session,args.gallery]
    out=require_new_output(args.output,inputs);shots=out/'crops';shots.mkdir()
    before=dict(db=file_hash(args.db),gallery=manifest(args.gallery),heroes=manifest(args.heroes),
                live=manifest(args.live),session=manifest(args.session),truth=manifest(ROOT/'tests/fixtures'))
    write(out/'inputs-before.json',before)
    home=out/'runtime';home.mkdir();shutil.copy2(args.db,home/'db.sqlite3')
    shutil.copytree(args.gallery,home/'gallery')
    for name in ('item_icons','augment_icons'):shutil.copytree(ROOT/'var'/name,home/name)
    os.environ['JCC_HOME']=str(home)
    from jccbrain.data.db import GameDB
    from jccbrain.vision.pyeye import build_reader
    db=GameDB(home/'db.sqlite3');reader=build_reader(db,home/'gallery',args.ocr)
    gallery=reader.units.gallery
    pr=PanelReader(reader.dict,reader.ocr)
    records={'heroes':[],'negative':[],'live':[],'train':[],'holdout':[]}
    process=process_snapshot()
    for path in sorted(args.heroes.glob('*.png')):
        records['heroes'].append(scene_analysis(path,reader,pr,shots,'heroes-'+path.stem))
    selection=json.loads((BASE/'batch-B9/preparation/audit-selection.json').read_text())
    # Selection contains filenames only; never read B9 identity/phase truth here.
    if isinstance(selection,dict):
        selection=selection.get('frames',selection.get('selection',[]))
    session_names=[r if isinstance(r,str) else r.get('file',r.get('_file')) for r in selection]
    negatives=[*sorted(args.samples.glob('*.jpg')),*sorted((args.samples/'track').glob('*.jpg'))]
    negatives.extend(args.session/'frames'/name for name in session_names)
    for i,path in enumerate(negatives):
        records['negative'].append(scene_analysis(path,reader,pr,shots,f'negative-{i}'))
    events=[json.loads(l) for l in (args.live/'events.jsonl').read_text().splitlines()]
    frames={r['file']:r for r in events if r['kind']=='frame'}
    clicks={r['label']:r for r in events if r['kind']=='click'}
    # Analyze current pixels before loading any human labels. No live/holdout
    # analysis is ever fed into the controller's training state or Gallery.add.
    for i,path in enumerate(sorted(args.live.glob('*.png'))):
        records['live'].append(scene_analysis(path,reader,pr,shots,f'live-{i}'))
    source=FrameSource();store=UnitLearningStore(home/'learning',gallery.season,gallery.pool)
    ctl=UnitLearningController(reader,store,source,mode='panel')
    baseline=Gallery(home/'baseline-gallery',gallery.season,gallery.pool,seed_from=None)
    # Baseline must include exactly the frozen user vectors as well as bundled refs.
    baseline.items=[(n,v.copy()) for n,v in gallery.items]
    source_paths=[]
    for key in ('board-a','board-b','board-c'):
        before_event=frames[key+'-before.png'];image=load_image(args.live/before_event['file'])
        context=reader.read(image)
        before_frame=source.archived(args.live,before_event,context.get('phase','unknown'),context.get('viewing'),context.get('stage'))
        click=clicks[key]
        tx=ClickTransaction(before_frame,tuple(click['point']),click['monotonic'],click['window']['window_id'])
        for i in range(3):
            event=frames[f'{key}-{i}.png'];context=reader.read(load_image(args.live/event['file']))
            frame=source.archived(args.live,event,context.get('phase','unknown'),context.get('viewing'),context.get('stage'))
            result=ctl.process(frame,tx)
            records['train'].append(dict(file=event['file'],transaction=key,result=result))
            source_paths.append(event['file'])
            print(event['file'],result['status'],result['reason'],flush=True)
        if key=='board-b':
            records['holdout'].append(predict_holdout('holdout-hide-1.png',args,reader,baseline,gallery,store))
    records['holdout'].append(predict_holdout('c-hide-0.png',args,reader,baseline,gallery,store))
    # Record mode on genuinely distinct archived captures: dedup, disk bound,
    # no promotion and no writes to either frozen/user gallery.
    record_source=FrameSource();record_store=UnitLearningStore(home/'record-proof',gallery.season,gallery.pool,max_pending=3,max_bytes=150000)
    recorder=UnitLearningController(reader,record_store,record_source,mode='record')
    record_results=[]
    repeated_capture_proof=None
    for key in ('board-a','board-b','board-c'):
        for i in range(3):
            ev=frames[f'{key}-{i}.png'];fr=record_source.archived(args.live,ev,'planning',stage='1-4')
            record_results.append(dict(file=ev['file'],result=recorder.process(fr)))
            if repeated_capture_proof is None:
                queue_before=manifest(record_store.root)
                replay=recorder.process(fr)
                repeated_capture_proof=dict(result=replay,unchanged=queue_before==manifest(record_store.root),
                                             note='Same genuine Frame receipt reused, not a fabricated fresh sample.')
    write(out/'record-proof.json',dict(results=record_results,queue=record_store.entries('review'),
        repeat=repeated_capture_proof,verified=record_store.entries(),record_bytes=sum(p.stat().st_size for r in record_store.entries('review')
            for p in (record_store.queue/r['id']).rglob('*') if p.is_file())))
    gallery.verified_store=store
    # Revoke the new samples and recheck the already measured holdout pixels.
    revoked=[]
    for row in store.entries():
        store.revoke(row['id']);revoked.append(row['id'])
    for row in records['holdout']:
        im=load_image(args.live/row['file']);_,us=reader.units.detect(im)
        row['after_revoke']=[gallery.match(fp) if (fp:=feature_print(u.crop)) is not None else None for u in us]
    records['revoked']=revoked
    write(out/'predictions.json',records)
    write(out/'prediction-seal.json',dict(sha256=file_hash(out/'predictions.json'),
        identity_truth_read=False,training_files=source_paths,heldout_files=['holdout-hide-1.png','c-hide-0.png'],
        heldout_used_for_training=False,source_sha=manifest(ROOT/'src/jccbrain/vision'),script_sha=file_hash(Path(__file__))))
    # Labels first enter the evaluation here, after immutable raw predictions.
    metrics=score_saved(records,args.live)
    metrics['record_proof_passed']=bool(repeated_capture_proof['unchanged'] and
        repeated_capture_proof['result']['reason']=='stale_or_invalid_capture' and not record_store.entries() and
        len(record_store.entries('review'))<=3)
    metrics['revoke_restores_baseline']=all(
        u['before_name']==(match[0] if match else None) for row in records['holdout']
        for u,match in zip(row['units'],row['after_revoke']))
    metrics['quality_passed'] &= metrics['record_proof_passed'] and metrics['revoke_restores_baseline']
    after=dict(db=file_hash(args.db),gallery=manifest(args.gallery),heroes=manifest(args.heroes),
               live=manifest(args.live),session=manifest(args.session),truth=manifest(ROOT/'tests/fixtures'))
    write(out/'inputs-after.json',after)
    metrics['inputs_unchanged']=before==after
    metrics['timing_status']='未测（环境不净）' if process['suspects'] else 'not_benchmarked'
    metrics['processes']=process
    write(out/'summary.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return 0 if metrics['quality_passed'] and metrics['inputs_unchanged'] else 1



def predict_holdout(name,args,reader,baseline,gallery,store):
    image=load_image(args.live/name);_,us=reader.units.detect(image)
    rows=[]
    for u in us:
        fp=feature_print(u.crop) if u.crop is not None else None
        if fp is None:continue
        old=baseline.match(fp);new=gallery.match(fp)
        u.fp,u.fp_prepared=fp,True
        rows.append(dict(x=u.x,y=u.y,zone=u.zone,slot=u.slot,rect=rect(u,image),
            before_name=old[0] if old else None,after_name=new[0] if new else None,
            before_distance=old[1] if old else None,after_distance=new[1] if new else None,
            distances=[dict(name=n,distance=fp_distance(fp,v)) for n,v in store.active_vectors()]))
    product=UnitTracker(gallery).observe(image,'planning',None,prepared=({},us))
    return dict(file=name,sha256=file_hash(args.live/name),units=rows,product=product)

def score_saved(records,live):
    refs=json.loads((ROOT/'src/jccbrain/vision/eye/unit_refs/unit_refs.json').read_text())
    reference={Path(r['frame']).name:r for r in refs['entries']}
    detail=[];names=Counter();bindings=[]
    for row in records['heroes']:
        label=reference.get(Path(row['file']).name)
        if label:
            p=row['panel'];correct=p['name']==label['name']
            detail.append(dict(file=Path(row['file']).name,expected=label['name'],actual=p['name'],correct=correct,reason=p['reason']))
            if correct:names[p['name']]+=1
            if row['bound']:
                b=row['bound'];x,y=label['bar'][4:]
                bindings.append(dict(file=Path(row['file']).name,correct=label['mode']=='detected' and abs(b['x']-x)<.012 and abs(b['y']-y)<.02))
    human=json.loads((live/'manual-review.json').read_text())
    accepted=[]
    for row in records['train']:
        if row['result']['status']=='verified':
            label=next(t for t in human['training'] if row['file'] in t['confirm'])
            target=row['result']['target'];click=label['click']
            accepted.append(dict(file=row['file'],expected=label['name'],actual=row['result']['panel']['name'],
                name_correct=row['result']['panel']['name']==label['name'],
                binding_correct=any(d['selected'] and d['x']==target['x'] and d['y']==target['y'] and
                    d['rect'][0]<=click[0]*1774<d['rect'][2] and d['rect'][1]<=click[1]*1136<d['rect'][3]
                    for d in row['result']['binding'])))
    heldout=[]
    for label in human['held_out']:
        row=next(r for r in records['holdout'] if r['file']==label['frame'])
        x,y=label['click'];candidates=[u for u in row['units'] if u['rect'][0]<=x*1774<u['rect'][2] and u['rect'][1]<=y*1136<u['rect'][3]]
        u=candidates[0] if len(candidates)==1 else None
        heldout.append(dict(expected=label['name'],file=label['frame'],prediction=u,
            unknown_to_correct=bool(u and u['before_name'] is None and u['after_name']==label['name']),
            wrong_new=bool(u and u['before_name']!=u['after_name'] and u['after_name'] not in (None,label['name']))))
    negative_panels=[r['file'] for r in records['negative'] if r['panel']['present']]
    accepted_wrong=sum(not r['name_correct'] or not r['binding_correct'] for r in accepted)
    improvement_names={r['expected'] for r in heldout if r['unknown_to_correct']}
    return dict(panel_names=dict(correct=sum(r['correct'] for r in detail),labeled=len(detail),distinct=len(names),details=detail),
        static_binding=dict(correct=sum(r['correct'] for r in bindings),total=len(bindings),details=bindings),
        accepted=accepted,accepted_wrong=accepted_wrong,heldout=heldout,independent_improvement_heroes=sorted(improvement_names),
        negative_panel_candidates=negative_panels,
        conservative_reasons=dict(Counter(r['result']['reason'] for r in records['train'])),
        calibration=dict(positive_presence=[r['panel']['presence_score'] for r in records['heroes'] if Path(r['file']).name!='0006.png'],
            negative_presence=[r['panel']['presence_score'] for r in records['negative']],
            all_binding_measurements=[dict(file=r['file'],**b) for r in records['heroes']+records['live']+records['negative'] for b in r['binding']]),
        quality_passed=bool(accepted and accepted_wrong==0 and len(names)>=4 and len(improvement_names)>=2
                            and all(r['correct'] for r in bindings) and not any(r['wrong_new'] for r in heldout) and not negative_panels))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--db',type=Path,default=BASE/'batch-B8/data/jcc-vf-db.sqlite3')
    p.add_argument('--gallery',type=Path,default=BASE/'batch-B8/data/jcc-vf-gallery')
    p.add_argument('--heroes',type=Path,default=BASE/'replay-samples/heroes')
    p.add_argument('--samples',type=Path,default=ROOT/'game-eye/samples')
    p.add_argument('--session',type=Path,default=BASE/'replay-sessions/live-20261023-205340')
    p.add_argument('--live',type=Path,default=BASE/'batch-L1/conductor/live-preparation')
    p.add_argument('--ocr',default='rapid',choices=['rapid','vision'])
    args=p.parse_args();args.output=args.output.resolve()
    return run_evaluation(args)

if __name__=='__main__':raise SystemExit(main())
