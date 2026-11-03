"""F2-0: finite OFFLINE current-owner evidence probe, never a product write.

Two fixed raw-pixel name crops per existing coarse legend bar. No candidate
words, fuzzy word cache, text completion, new capture, native input or downloads.
Human review is opened only AFTER predictions are sealed. Capture provenance is
scoring truth, never an OCR input or a substitute for current identity pixels.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from collections import OrderedDict, deque
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import threading

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'var/vision-module-optimization/batch-F2'
TRUTH = ROOT / 'tests/fixtures/augment_ownership_truth.json'
POSITIVE = tuple(f'augment-taken/{i:04d}.png' for i in (3, 5, 6, 7))
CASES = POSITIVE + ('augment-taken/0001.png',) + tuple(
    f'spectate/{i:04d}.png' for i in (1, 2, 3, 4, 5, 6)) + (
    'carousel/0001.png', 'dialog/0001.png', 'RESULT')
# The first box is the EXISTING opponent-name geometry. The second only narrows
# horizontal/background coverage, still raw BGR. Exactly two fixed probes; no
# search over thresholds/transforms. Both are window ratios at every resolution.
CROPS = ((-.11, -.04, .22, .035), (-.09, -.033, .18, .030))
PANEL = (.80, .105, .19, .40)
# Human-position upper bound for the two already reviewed icon-occluded frames.
# Frozen in batch-F2/executor/preflight/supplement-probe-plan.json BEFORE OCR.
# These are diagnostic oracle hints, NEVER automatic localization/owner evidence.
MANUAL_HINTS = {f'augment-taken/{i:04d}.png':(.26,.23,.18,.09) for i in (6,7)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def manifest(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def validate_output(output, protected, *, base=None):
    """Resolve symlinks before checking containment; refuse ANY input overlap."""
    output = output.resolve()
    if output.exists():
        raise ValueError('output must be a new directory')
    if base is not None and base.resolve() not in output.parents:
        raise ValueError('all output must stay inside batch-F2')
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError('output overlaps protected input/source: ' + str(path))
    return output


def private_ocr(ocr):
    if ocr.name != 'rapid':
        raise RuntimeError('probe requires actual Rapid backend, not a fallback')
    engine = copy.copy(ocr)
    engine.cache, engine.cache_size = OrderedDict(), 128
    engine.words = deque(maxlen=0)  # Never retain fuzzy word/candidate matches.
    engine.lock, engine._stats_lock = threading.Lock(), threading.Lock()
    engine._warm_lock = threading.Lock()
    engine.reset_stats()
    engine.fallback = False
    return engine


def memory(reader):
    """Semantic state + hashes of exact mutable OCR/unit caches (not locks)."""
    aug = reader.augments_taken
    digest = lambda obj: hashlib.sha256(pickle.dumps(obj)).hexdigest()
    return dict(phase=reader.phase, last_stage=reader.last_stage, last_timer=reader.last_timer,
                roster=copy.deepcopy(reader.roster), last_players=copy.deepcopy(reader.last_players),
                augments=dict(confirmed=copy.deepcopy(aug.confirmed),
                              pending=copy.deepcopy(aug.pending), remembered=aug.remembered()),
                cache_sha=digest(reader.cache), cell_cache_sha=digest(reader.cell_cache),
                ocr_cache_sha=digest(reader.ocr.cache), ocr_words_sha=digest(reader.ocr.words),
                unit_state_sha=digest({k:v for k,v in reader.units.__dict__.items()
                                       if k not in ('gallery', 'items')}))


def single_text(lines):
    """Do not pick the longest/familiar candidate if multiple name lines exist."""
    words = [l['text'].strip() for l in lines if any(c.isalpha() for c in l['text'])]
    return words[0] if len(words) == 1 else None


def agreeing_name(crops):
    names = [r['single_text'] for r in crops]
    return names[0] if len(names) == 2 and names[0] and names[0] == names[1] else None


def inspect_pixels(image, ocr, output, *, manual_hint=None):
    from jccbrain.vision.eye.image import Region, crop
    from jccbrain.vision.eye.units import Bar, detect_bars, green_mask
    from jccbrain.vision.eye.opponent import enemy_bars
    from jccbrain.vision.eye.augments import spectate_score
    from jccbrain.vision.eye.scout import viewing_slot
    output.mkdir()
    h, w = image.shape[:2]
    # EXACT existing UnitTracker coarse-bar threshold, not a new owner classifier.
    green = [b for b in detect_bars(image) if b.thick >= 9*h/1136]
    _, orange = enemy_bars(image)
    engine = private_ocr(ocr)  # Empty private cache PER frame, shared loaded model only.
    before = (copy.deepcopy(ocr.cache), copy.deepcopy(ocr.words))
    plates, patches, refs = [], [], []
    manual_bars = []
    if manual_hint is not None:
        import numpy as np
        hint = Region(*manual_hint)
        x0,y0,_,_ = hint.px(image)
        _,_,stats,_ = cv2.connectedComponentsWithStats(green_mask(crop(image,hint)).astype(np.uint8))
        if len(stats)>1:
            # Raw largest component INSIDE a human-provided area, no completion.
            x,y,bw,bh,_ = max(stats[1:],key=lambda s:s[4])
            manual_bars=[Bar(int(x+x0),int(x+x0+bw-1),int(y+y0),int(y+y0+bh-1),
                             float(x+x0+(bw-1)/2)/w,float(y+y0+(bh-1)/2)/h)]
    for colour, bars in (('green', green), ('orange', orange), ('manual-green',manual_bars)):
        for bar in bars:
            plate = dict(colour=colour, bar=asdict(bar), crops=[])
            for dx, dy, width, height in CROPS:
                region = Region(bar.cx+dx, bar.min_y/h+dy, width, height)
                patch = crop(image, region)
                if patch is None:
                    plate['crops'].append(dict(region=asdict(region), lines=[], single_text=None))
                    continue
                path = output / f'name-{len(plates):02d}-{len(plate["crops"])}.png'
                cv2.imwrite(str(path), patch)
                row = dict(region=asdict(region), rect=region.px(image), file=path.name,
                           pixel_sha=hashlib.sha256(patch.tobytes()).hexdigest(), lines=[])
                plate['crops'].append(row); patches.append(engine.prepare(patch)); refs.append(row)
            plates.append(plate)
    region = Region(*PANEL); panel = crop(image, region)
    panel_row = dict(region=asdict(region), rect=region.px(image), file='panel.png', lines=[])
    cv2.imwrite(str(output/'panel.png'), panel)
    patches.append(engine.prepare(panel)); refs.append(panel_row)
    results = engine.recognize_batch(patches, ['zh-Hans', 'en-US'], ['ownership']*len(patches))
    for row, lines in zip(refs, results):
        row['lines'] = [asdict(l) for l in lines]
        row['single_text'] = single_text(row['lines'])
    for plate in plates:
        plate['exact_two_crop_name'] = agreeing_name(plate['crops'])
    canvas = image.copy()
    for i, plate in enumerate(plates):
        b = plate['bar']; colour = (0, 255, 0) if plate['colour']=='green' else (0, 140, 255)
        cv2.rectangle(canvas, (b['min_x'], b['min_y']), (b['max_x'], b['max_y']), colour, 2)
        cv2.putText(canvas, str(i), (b['min_x'], b['min_y']-4), cv2.FONT_HERSHEY_SIMPLEX,.6,colour,1)
    cv2.imwrite(str(output/'bars.jpg'), canvas)
    # The main word cache contains arrays, so equality is exact via pickle hash.
    hash_cache = lambda obj: hashlib.sha256(pickle.dumps(obj)).hexdigest()
    assert hash_cache(before) == hash_cache((ocr.cache, ocr.words)), 'private OCR mutated main cache'
    return dict(plates=[p for p in plates if p['colour']!='manual-green'],
                manual_upper_bound=[p for p in plates if p['colour']=='manual-green'],
                manual_hint=manual_hint,manual_hint_is_owner_evidence=False,
                panel=panel_row, raw_eye_score=spectate_score(image),
                current_eye_slot=viewing_slot(image), private_main_cache_unchanged=True,
                private_fuzzy_words=len(engine.words), private_backend=engine.name)


def anchor_candidate(row):
    """Exploratory anchor from existing full-roster trusted context only.

    Not a new product permission: one current green plate, two EXACT readings,
    unique self row, no eye/spectate. Human names are not passed to this function.
    """
    state, probe = row['reader'], row['probe']
    players = state.get('players', [])
    names = [p['exact_two_crop_name'] for p in probe['plates'] if p['colour']=='green']
    if (len(players) >= 5 and sum(bool(p.get('self')) for p in players)==1
            and state.get('phase') in ('planning','combat') and state.get('viewing') is None
            and probe['raw_eye_score'] < .60 and probe['current_eye_slot'] is None
            and len(names)==1 and names[0]):
        return names[0]
    return None


def strict_probe(row, anchor):
    """Report a stringent trial AND, NOT authorization to bypass the product gate.

    Even a pass still needs independent visibility/occlusion calibration before
    implementation. A partial matching suffix can NEVER pass this exact test.
    """
    state, probe = row['reader'], row['probe']
    green = [p['exact_two_crop_name'] for p in probe['plates'] if p['colour']=='green']
    orange = [p['exact_two_crop_name'] for p in probe['plates'] if p['colour']=='orange']
    panel_text = [l['text'] for l in probe['panel']['lines']]
    checks = dict(anchor_present=bool(anchor), planning=state.get('phase')=='planning',
                  no_current_viewing=state.get('viewing') is None and probe['current_eye_slot'] is None,
                  no_eye=probe['raw_eye_score'] < .60,
                  unique_green=len(green)==1,
                  exact_current_name=bool(anchor) and len(green)==1 and green[0]==anchor,
                  no_orange_conflict=not orange,
                  owned_panel_title=any('已拥有' in text for text in panel_text))
    return dict(checks=checks, trial_and=all(checks.values()),
                note='Offline sufficiency trial only. Title/green/no-eye/history alone are never proof; no product promotion.')


def evaluate(args):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader
    from jccbrain.data.db import GameDB
    from jccbrain.core import paths
    sources = {name: (args.legacy/'result_2nd.jpg' if name=='RESULT' else args.samples/name) for name in CASES}
    out = validate_output(args.output, (args.input,args.samples,args.legacy,TRUTH,ROOT/'src',ROOT/'var/item_icons',ROOT/'game-eye/gallery'), base=BASE)
    before = manifest(args.input)
    expected_manifest = args.input.parent/'input-manifest.json'
    if expected_manifest.is_file() and before != json.loads(expected_manifest.read_text()):
        raise ValueError('frozen input manifest differs')
    source_before = {name:sha(path) for name,path in sources.items()}
    out.mkdir(parents=True)
    write(out/'inputs-before.json',before)
    write(out/'sources-before.json',source_before)
    runtime=out/'runtime';runtime.mkdir();os.environ['JCC_HOME']=str(runtime)
    shutil.copy2(args.input/'jcc.sqlite3',runtime/'db.sqlite3')
    for name in ('augment_icons','unit-learning'):
        shutil.copytree(args.input/name,runtime/name)
    shutil.copytree(ROOT/'var/item_icons',runtime/'item_icons')
    paths.legacy_gallery_dir=lambda:ROOT/'game-eye/gallery'
    db=GameDB(runtime/'db.sqlite3')
    reader=build_reader(db,runtime/'gallery','rapid')
    rows=[]
    for i,name in enumerate(CASES):
        # Only the four original own frames share a real source-order context.
        # Every negative has a NEW reader/memory, never a copied own snapshot.
        if i>=len(POSITIVE):
            reader=build_reader(db,runtime/f'negative-gallery-{i}','rapid')
        image=load_image(sources[name]);prior=memory(reader)
        state=reader.read(image,debug=True);after=memory(reader)
        probe=inspect_pixels(image,reader.ocr,out/f'frame-{i:02d}',manual_hint=MANUAL_HINTS.get(name))
        candidates=[dict(center=x,name=n,score=s,second_score=t,margin=s-t,
                         ambiguous=n in reader.augments_taken.ambiguous_names)
                    for x,n,s,t in reader.augments_taken.candidates(image)]
        rows.append(dict(id=name,sha256=source_before[name],shape=list(image.shape),
                         context='original-own-chronology' if i<len(POSITIVE) else 'independent-cold-negative',
                         reader=state,memory_before=prior,memory_after=after,probe=probe,augment_candidates=candidates))
    # Seal raw model outputs BEFORE opening the human name/owner review.
    source_sha=manifest(ROOT/'src/jccbrain/vision')
    result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                evaluator_sha=sha(Path(__file__)),reader_source=sys.modules['jccbrain.vision.eye.reader'].__file__,
                source_sha={k:v for k,v in source_sha.items() if k.endswith('.py')},rows=rows,
                crop_plan=CROPS,synthetic_cases=0,native_capture=0,native_input=0)
    write(out/'predictions.json',result)
    write(out/'prediction-seal.json',dict(sha256=sha(out/'predictions.json'),human_truth_opened=False))
    truth=json.loads(TRUTH.read_text());write(out/'truth.json',truth)
    truth_rows={r['id']:r for r in truth['rows']}
    assert set(truth_rows)==set(CASES), 'truth/source set differs'
    assert all(r['sha256']==truth_rows[r['id']]['sha256'] for r in rows), 'truth/source SHA mismatch'
    anchors=[dict(id=r['id'],name=anchor_candidate(r)) for r in rows[:len(POSITIVE)]]
    names={r['name'] for r in anchors if r['name']};anchor=next(iter(names)) if len(names)==1 else None
    scored=[]
    for row in rows:
        review=truth_rows[row['id']];trial=strict_probe(row,anchor)
        scored.append(dict(id=row['id'],truth=review,anchor_name=anchor_candidate(row),trial=trial,
                           augment_count=len(row['reader'].get('augments_taken',[])),
                           players=len(row['reader'].get('players',[])),phase=row['reader'].get('phase'),
                           warning='Manual occlusion audit forbids treating OCR-completed hidden glyph as evidence'
                           if review.get('green_legend_name_fully_visible') is False else None))
    after=manifest(args.input);write(out/'inputs-after.json',after)
    source_after={name:sha(path) for name,path in sources.items()};write(out/'sources-after.json',source_after)
    by={r['id']:r for r in scored}
    negatives=[r for r in scored if not r['truth']['eligible_context']]
    checks=dict(inputs_unchanged=before==after,source_images_unchanged=source_before==source_after,
                four_original_own_frames=len(rows[:len(POSITIVE)])==4,
                at_least_eight_real_negatives=sum(r['truth']['real'] and r['truth']['board_owner'] in ('other','shared')
                                                   for r in negatives)>=8,
                reproduced_0006_two=by['augment-taken/0006.png']['augment_count']==2,
                reproduced_0007_three=by['augment-taken/0007.png']['augment_count']==3,
                probe_did_not_mutate_main_ocr=all(r['probe']['private_main_cache_unchanged'] for r in rows),
                no_fuzzy_words=all(r['probe']['private_fuzzy_words']==0 for r in rows),
                negative_trial_accepts_zero=not any(r['trial']['trial_and'] for r in negatives),
                masked_0006_not_sufficient=not by['augment-taken/0006.png']['trial']['trial_and'])
    summary=dict(checks=checks,anchors=anchors,independent_anchor=anchor,scored=scored,
                 real_negative_count=len(negatives),trial_accepts=sum(r['trial']['trial_and'] for r in scored),
                 trial_precision=None if not any(r['trial']['trial_and'] for r in scored) else 'manual review required',
                 truth_sha=sha(TRUTH),predictions_sha=sha(out/'predictions.json'),evaluator_sha=sha(Path(__file__)),
                 conclusion='Current 0006 identity is NOT established. Do not bypass own_view. Defer until visible trusted own evidence or a separately bound own selection event.',
                 limits='Old replay mechanism only, not the user forgotten third augment; no new product rule approved; no live performance measurement.')
    write(out/'summary.json',summary)
    print(json.dumps(dict(checks=checks,anchor=anchor,real_negatives=len(negatives),trial_accepts=summary['trial_accepts'],
                         augment_counts=[r['augment_count'] for r in scored[:4]]),ensure_ascii=False,indent=2))
    if not all(checks.values()):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BASE/'input')
    parser.add_argument('--samples',type=Path,default=ROOT/'var/vision-module-optimization/replay-samples')
    parser.add_argument('--legacy',type=Path,default=ROOT/'game-eye/samples')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for k in vars(args):setattr(args,k,getattr(args,k).absolute())
    evaluate(args)


if __name__=='__main__':main()
