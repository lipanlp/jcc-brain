"""S7-0: real CG pixels -> unchanged read_scout -> actual ScoutRunner background.

Original JSONL target names are MACHINE observations, not human truth. No click,
input monitor, capture, target-name correction or candidate-word injection. The
initial context is a cold same-frame main read, NOT a reconstructed pre-scan own
state. Truth is opened only after both independent workers seal predictions.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'var/vision-module-optimization/batch-S7'
BASE_COMMIT='afb6f183eac347e33bbf9edf20ddedaed14a489a'


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def manifest(path):return {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))


def protected(args):
    return dict(input=manifest(args.input),runtime=manifest(args.runtime),
                item_icons=manifest(ROOT/'var/item_icons'),seeds=manifest(ROOT/'game-eye/gallery'))


def validate_output(args):
    if args.output.exists():raise ValueError('output must be a new directory')
    if BASE.resolve() not in args.output.parents:raise ValueError('output must stay inside batch-S7')
    for path in (args.input,args.runtime,args.baseline_repo,ROOT/'src'):
        if path in args.output.parents or args.output in path.parents:raise ValueError('output overlaps input/source')


def consume(reader,db,initial,target,image,record,directory):
    from jccbrain.core.settings import Settings
    from jccbrain.engine.scout import ScoutBook
    from jccbrain.modes.base import ModeSink
    from jccbrain.modes.scout import ScoutRunner,ScoutCapture
    book=ScoutBook(db,comps_getter=lambda:[],directory=directory/'book');book.update(initial)
    logs=[];sink=ModeSink();sink.on_log=logs.append
    monitor_calls=[]
    def forbidden_start():
        monitor_calls.append('start');raise RuntimeError('offline replay must not start an input monitor')
    runner=ScoutRunner(NS(db=db,settings=Settings(),advisor=NS(scout=book)),sink,reader_factory=lambda:reader,
                       input_monitor=NS(start=forbidden_start))
    runner._initial=initial;runner._status.update(captured=1,total=1)
    timing={k:(record.get('timing') or {}).get(k) for k in ('cut_ms','settle_ms','total_ms')}
    jobs=queue.Queue();jobs.put((target,ScoutCapture(image,bool(record.get('loaded',False)),timing),record['t']))
    done=threading.Event();done.set();errors=[];observed=[];inner=[]
    original=reader.read_scout
    def read(im):
        result=original(im);observed.append(deepcopy(result));return result
    reader.read_scout=read
    class Handler(logging.Handler):
        def emit(self,r):
            if r.exc_info:inner.append(dict(type=r.exc_info[0].__name__,traceback=''.join(traceback.format_exception(*r.exc_info))))
    handler=Handler();logger=logging.getLogger('jccbrain.scout');logger.addHandler(handler)
    def run():
        try:runner._recognize(jobs,done,book.started,record['round'])
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,traceback=traceback.format_exc()))
    thread=threading.Thread(target=run,name='s7-background',daemon=True)
    try:
        thread.start();thread.join(30)
        if thread.is_alive():raise RuntimeError('offline background exceeded 30 seconds')
    finally:
        reader.read_scout=original;logger.removeHandler(handler)
    shots=[json.loads(line) for line in book.path.read_text().splitlines()] if book.path else []
    rejected=('viewing_mismatch' if any('viewing_mismatch' in line for line in logs) else
              'background_exception' if errors or inner else 'book_rejected' if not shots else None)
    return dict(raw=observed,outer_errors=errors,inner_errors=inner,logs=logs,
                type_errors=sum(r['type']=='TypeError' for r in errors+inner),
                status=runner.status,unfinished_tasks=jobs.unfinished_tasks,shots=shots,
                profiles=[asdict(r) for r in book.latest()],rejection=rejected,
                input_actions=0,input_monitor_started=bool(monitor_calls))


def worker(args):
    validate_output(args)
    sys.path.insert(0,str(args.repo/'src'))
    from jccbrain.core import paths
    from jccbrain.core.models import GameState,Player
    from jccbrain.data.db import GameDB
    from jccbrain.vision.pyeye import build_reader
    from jccbrain.vision.eye.capture import load_image
    before=protected(args);out=args.output;out.mkdir()
    home=out/'runtime';home.mkdir();os.environ['JCC_HOME']=str(home)
    shutil.copy2(args.runtime/'jcc.sqlite3',home/'db.sqlite3')
    for name in ('augment_icons','unit-learning'):shutil.copytree(args.runtime/name,home/name)
    shutil.copytree(ROOT/'var/item_icons',home/'item_icons')
    paths.legacy_gallery_dir=lambda:ROOT/'game-eye/gallery'
    db=GameDB(home/'db.sqlite3');rows=[]
    for i,path in enumerate(sorted((args.input/'scout').glob('*/*.png'))):
        records=[json.loads(l) for l in (args.input/'scout'/(path.parent.name+'.jsonl')).read_text().splitlines()]
        at=float(path.stem.split('-slot')[0]);record=next(r for r in records if abs(r['t']-at)<.00001)
        reader=build_reader(db,home/f'gallery-{i:02d}','rapid')
        image=load_image(path)
        # Current machine context only; never replace its traits/name/self fields
        # with recorded/human values. The queued target is the original JSONL.
        main=reader._foreign_context().read(image);initial=GameState.from_dict(main)
        hp=next((p.hp for p in initial.players if p.slot==record['slot']),None)
        target=Player(record['name'],hp,False,record['slot'])
        raw=reader.read_scout(image)
        if scout:=getattr(reader,'_scout_reader',None):scout.reset()
        result=consume(reader,db,initial,target,image,record,out/f'frame-{i:02d}')
        rows.append(dict(file=str(path.relative_to(args.input)),sha256=sha(path),main=main,raw=raw,
                         machine_target=asdict(target),record=record,backend=result,
                         initial_unknown_traits=sum(t.name is None for t in initial.traits),
                         raw_unknown_traits=sum(t.get('name') is None for t in raw.get('traits',[])),
                         context='historical machine target + cold same-frame main state; no original own pre-scan/input proof'))
    # Separate controlled missing-catalog probes, never part of the 15 real-frame
    # accuracy denominator. Actual OCR creates None names; no injected trait text
    # or human target is supplied. Exercise both startup and result validation.
    from jccbrain.vision.eye.text import Dictionary
    probe_source=rows[3];probe_image=load_image(args.input/probe_source['file']);probes=[]
    for location in ('initial','result'):
        limited=build_reader(db,home/('missing-catalog-'+location),'rapid')
        limited.dict=Dictionary(limited.dict.champions,['主宰'])
        probe_raw=limited.read_scout(probe_image)
        initial=GameState.from_dict(probe_source['main'])
        if location=='initial':initial.traits=GameState.from_dict(probe_raw).traits
        target=Player(**{k:v for k,v in probe_source['machine_target'].items() if k in ('name','hp','is_self','slot','y')})
        if scout:=getattr(limited,'_scout_reader',None):scout.reset()
        backend=consume(limited,db,initial,target,probe_image,probe_source['record'],out/('controlled-'+location))
        probes.append(dict(location=location,file=probe_source['file'],sha256=probe_source['sha256'],
                           fault='controlled catalog reduced to one existing trait; not independent live failure evidence',
                           raw=probe_raw,unknown_traits=sum(t.get('name') is None for t in probe_raw.get('traits',[])),
                           parsed_traits=[asdict(t) for t in GameState.from_dict(probe_raw).traits],
                           backend=backend))
    sources=['src/jccbrain/modes/scout.py','src/jccbrain/vision/eye/reader.py','src/jccbrain/vision/eye/scout.py',
             'src/jccbrain/vision/eye/scout_units.py','src/jccbrain/core/models.py','src/jccbrain/engine/scout.py']
    result=dict(repo=str(args.repo),label=args.label,source_sha={f:sha(args.repo/f) for f in sources},
                modes_source=sys.modules['jccbrain.modes.scout'].__file__,reader_source=sys.modules['jccbrain.vision.eye.reader'].__file__,
                evaluator_sha=sha(Path(__file__)),rows=rows,controlled_unknown_probes=probes,inputs_unchanged=before==protected(args),
                native_capture=0,native_input=0)
    write(out/'predictions.json',result)
    write(out/'prediction-seal.json',dict(sha256=sha(out/'predictions.json'),human_truth_opened=False))


def main(args):
    if args.worker:return worker(args)
    validate_output(args)
    for f in ('src/jccbrain/modes/scout.py','src/jccbrain/vision/eye/reader.py','src/jccbrain/vision/eye/scout.py','src/jccbrain/core/models.py','src/jccbrain/engine/scout.py'):
        if (args.baseline_repo/f).read_bytes()!=subprocess.check_output(['git','show',BASE_COMMIT+':'+f],cwd=ROOT):
            raise ValueError('baseline is not the exact afb6f18 source: '+f)
    before=protected(args)
    if before['input']!=json.loads((args.input.parent/'input-manifest.json').read_text()):raise ValueError('input manifest differs')
    args.output.mkdir(parents=True);write(args.output/'inputs-before.json',before)
    results={}
    for label,repo in (('baseline',args.baseline_repo),('candidate',ROOT)):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--label',label,'--repo',str(repo),
             '--input',str(args.input),'--runtime',str(args.runtime),'--output',str(args.output/label)]
        with (args.output/(label+'.log')).open('w') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(repo/'src')},stdout=log,stderr=log)
        if proc.returncode:raise RuntimeError(label+' worker failed; see log')
        p=args.output/label/'predictions.json';seal=json.loads((p.parent/'prediction-seal.json').read_text())
        assert sha(p)==seal['sha256'];results[label]=json.loads(p.read_text())
    # Sealed predictions first; old manual truth is used only for reporting.
    truth=json.loads((args.input/'truth-v1.json').read_text());old,new=results['baseline'],results['candidate']
    details=[]
    for a,b,human in zip(old['rows'],new['rows'],truth['rows']):
        assert a['file']==b['file']==human['_file'] and a['sha256']==b['sha256']==human['sha256']
        profile=b['backend']['profiles'];board=b['raw'].get('board',[]);bench=b['raw'].get('bench',[])
        known=[u for u in board+bench if u.get('name') and not u['name'].startswith('?')]
        details.append(dict(file=b['file'],raw_equal=encoded(a['raw'])==encoded(b['raw']),
            main_equal=encoded(a['main'])==encoded(b['main']),old_type_errors=a['backend']['type_errors'],
            new_type_errors=b['backend']['type_errors'],new_done=b['backend']['unfinished_tasks']==0 and not b['backend']['status']['processing'],
            rejection=b['backend']['rejection'],profiles=len(profile),
            profile_wrong=sum(r['name']!=human['viewing']['name'] or r['slot']!=human['viewing']['slot'] for r in profile),
            raw_name=(b['raw'].get('viewing') or {}).get('name'),expected_name=human['viewing']['name'],
            board_detected=len(board),board_truth=human['board_count'],bench_detected=len(bench),bench_truth=human['bench_count'],
            known=len(known),unknown=len(board)+len(bench)-len(known)))
    after=protected(args);write(args.output/'inputs-after.json',after)
    checks=dict(fifteen=len(details)==len(old['rows'])==len(new['rows'])==len(truth['rows'])==15,
        raw_recognition_equal=all(r['raw_equal'] for r in details),main_context_equal=all(r['main_equal'] for r in details),
        background_recognition_equal=all(encoded(r['backend']['raw'])==encoded([r['raw']]) for r in old['rows']+new['rows']),
        candidate_no_type_error=all(r['new_type_errors']==0 for r in details),candidate_jobs_done=all(r['new_done'] for r in details),
        candidate_no_other_background_exception=all(not r['backend']['outer_errors'] and not r['backend']['inner_errors'] for r in new['rows']),
        controlled_probe_has_unknown=all(p['unknown_traits']>0 for p in new['controlled_unknown_probes']),
        controlled_probe_old_type_error=all(p['backend']['type_errors']>0 for p in old['controlled_unknown_probes']),
        controlled_probe_candidate_done=all(p['backend']['type_errors']==0 and p['backend']['unfinished_tasks']==0
                                           and not p['backend']['status']['processing'] and len(p['backend']['profiles'])==1
                                           and encoded(p['backend']['profiles'][0]['traits'])==encoded(p['parsed_traits'])
                                           for p in new['controlled_unknown_probes']),
        controlled_probe_raw_equal=all(encoded(a['raw'])==encoded(b['raw']) for a,b in zip(old['controlled_unknown_probes'],new['controlled_unknown_probes'])),
        no_input_monitor_started=all(not r['backend']['input_monitor_started'] for r in old['rows']+new['rows']
                                     +old['controlled_unknown_probes']+new['controlled_unknown_probes']),
        inputs_unchanged=before==after and old['inputs_unchanged'] and new['inputs_unchanged'])
    summary=dict(checks=checks,details=details,baseline_commit=BASE_COMMIT,
                 candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 evaluator_sha=sha(Path(__file__)),truth_sha=sha(args.input/'truth-v1.json'),
                 original_exception_frames=sum(bool(r['old_type_errors']) for r in details),
                 controlled_probe_summary={side:[dict(location=p['location'],unknown_traits=p['unknown_traits'],
                     type_errors=p['backend']['type_errors'],unfinished_tasks=p['backend']['unfinished_tasks'],
                     profiles=len(p['backend']['profiles'])) for p in r['controlled_unknown_probes']] for side,r in results.items()},
                 candidate_profiles=sum(r['profiles'] for r in details),candidate_wrong_profiles=sum(r['profile_wrong'] for r in details),
                 board_detected=sum(r['board_detected'] for r in details),board_truth=sum(r['board_truth'] for r in details),
                 known=sum(r['known'] for r in details),unknown=sum(r['unknown'] for r in details),
                 note='Sorting repair only. Existing identity rejections/fallbacks remain; no 15/15 name/profile claim. Timing and initial-state adapter are not a live scout reconstruction.')
    write(args.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=BASE/'input')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=ROOT/'var/vision-module-optimization/batch-F2/input')
    p.add_argument('--baseline-repo',type=Path,default=BASE/'baseline-src')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--repo',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--label',help=argparse.SUPPRESS)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    main(a)
