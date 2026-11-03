"""CG-decode B8 opponent evidence; saved source crops, geometry and annotations.

No truth or input mutation. Coordinates are supplementary audit labels only;
experiments/detectors receive image pixels, never a frame name or truth positions.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
import types
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jccbrain.vision.eye.capture import load_image
from jccbrain.vision.eye import opponent
from jccbrain.vision.eye.opponent import enemy_mask, enemy_bars
from jccbrain.vision.eye.units import detect_bars, SCAN


def seeds(image):
    h,w=image.shape[:2]
    x0,y0=round(SCAN.x*w),round(SCAN.y*h)
    patch=image[y0:round((SCAN.y+SCAN.h)*h),x0:round((SCAN.x+SCAN.w)*w)]
    count,_,stats,_=cv2.connectedComponentsWithStats(enemy_mask(patch).astype(np.uint8))
    return [dict(rect=[int(x+x0),int(y+y0),int(cw),int(ch)], area=int(area), fill=float(area/(cw*ch)))
            for x,y,cw,ch,area in stats[1:] if cw>=6/1774*w and ch<=30/1136*h]


def bar_json(b):
    return dict(rect=[b.min_x,b.min_y,b.width,b.thick],x=round(b.cx,4),y=round(b.cy,4))


def current_audit(output, samples):
    output.mkdir(parents=True,exist_ok=True)
    truth=[json.loads(s) for s in (ROOT/'tests/fixtures/opponent_truth.jsonl').read_text().splitlines() if s]
    rows=[];crop_rows=[];patches={}
    for expected in truth:
        name=expected['_file'];image=load_image(samples/name);h,w=image.shape[:2]
        raw=detect_bars(image,mask_fn=enemy_mask,min_width=8/1774)
        found,legends=enemy_bars(image);all_seeds=seeds(image)
        gray=image.max(axis=2)
        audits=[]
        for i,point in enumerate(expected['enemy_positions']):
            x,y=round(point['x']*w),round(point['y']*h)
            x0,x1=max(0,x-85),min(w,x+85);y0,y1=max(0,y-22),min(h,y+42)
            key=f'{Path(name).stem}-{i}'
            patches[key]=image[y0:y1,x0:x1].copy()
            nearby=[]
            for seed in all_seeds:
                sx,sy,cw,ch=seed['rect']
                if sx<x+85 and sx+cw>x-85 and abs(sy+ch/2-y)<15:
                    nearby.append(seed)
            matched=any(abs(b.cx-point['x'])<.02 and abs(b.cy-point['y'])<.03 for b in found)
            audits.append(dict(expected=point,baseline_matched=matched,orange_components=nearby))
            crop_rows.append(dict(key=key,source=name,origin=[x0,y0],expected=point,baseline_matched=matched))
        annotated=image.copy()
        for seed in all_seeds:
            x,y,cw,ch=seed['rect'];cv2.rectangle(annotated,(x,y),(x+cw,y+ch),(255,255,0),1)
        for b in found:cv2.rectangle(annotated,(b.min_x,b.min_y),(b.max_x,b.max_y),(255,0,255),2)
        for i,p in enumerate(expected['enemy_positions']):
            x,y=round(p['x']*w),round(p['y']*h)
            cv2.circle(annotated,(x,y),10,(255,255,255),1)
            cv2.putText(annotated,str(i),(x-5,y-14),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(output/f'{Path(name).stem}-baseline.jpg'),annotated)
        rows.append(dict(_file=name,truth=expected,baseline=[bar_json(b) for b in found],
                         raw=[bar_json(b) for b in raw],legends=[bar_json(b) for b in legends],
                         seeds=all_seeds,truth_audit=audits))
    np.savez_compressed(output/'truth-crops.npz',**patches)
    (output/'truth-crops.json').write_text(json.dumps(crop_rows,ensure_ascii=False,indent=2)+'\n')
    (output/'baseline-audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    for page in range(0,len(crop_rows),12):
        sheet=np.zeros((4*110,3*340,3),np.uint8)
        for k,entry in enumerate(crop_rows[page:page+12]):
            col,row=k%3,k//3;patch=patches[entry['key']]
            patch=cv2.resize(patch,None,fx=2,fy=1.3,interpolation=cv2.INTER_NEAREST)
            ph,pw=patch.shape[:2];sheet[row*110+25:row*110+25+ph,col*340:col*340+pw]=patch
            cv2.putText(sheet,f"{entry['key']} old={int(entry['baseline_matched'])}",(col*340,row*110+18),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.imwrite(str(output/f'truth-sheet-{page//12}.png'),sheet)
    print('truth',sum(r['truth']['board_count'] for r in rows),'baseline',sum(len(r['baseline']) for r in rows))
    for row in rows:print(row['_file'],len(row['baseline']),row['truth']['board_count'])


def calibration(rows, samples):
    accepted = [entry for row in rows for entry in row['diagnostics'] if 'accepted' in entry]
    rejected = [entry for row in rows for entry in row['diagnostics'] if 'rejected' in entry]
    groups = {}
    for label, entries in (('accepted_seed_options', accepted), ('rejected_seed_options', rejected)):
        fills = [entry['seed']['fill'] for entry in entries]
        caps = [value for entry in entries for value in entry['measurements']['caps']]
        widths = [value for entry in entries for value in entry['measurements']['widths']]
        groups[label] = dict(count=len(entries), fill_min=min(fills), fill_max=max(fills),
                             paired_trials=sum(entry['measurements']['paired'] for entry in entries),
                             width_ok=sum(entry['measurements']['width_ok'] for entry in entries),
                             width_min=min(widths) if widths else None, width_max=max(widths) if widths else None,
                             cap_min=min(caps) if caps else None, cap_max=max(caps) if caps else None,
                             selected_cap_min=min(min(entry['accepted']['caps']) for entry in entries)
                                              if label == 'accepted_seed_options' else None)
    # These explicit hard-negative source rectangles are manually selected from
    # the CG originals, independent of any detector output. Other rejected seeds
    # are not automatically labelled negatives: they can be unresolved enemies.
    from make_opponent_b8_fixtures import ENTRIES
    negatives = []
    for key, file, rect, _point, enemy in ENTRIES:
        if enemy:
            continue
        image = load_image(samples/file)
        x,y,width,height = rect
        patch = image[y:y+height,x:x+width]
        _, _, stats, _ = cv2.connectedComponentsWithStats(enemy_mask(patch).astype(np.uint8))
        components = [dict(width=int(cw), height=int(ch), area=int(area), fill=float(area/(cw*ch)))
                      for _x,_y,cw,ch,area in stats[1:] if cw >= 6]
        row = next(row for row in rows if row['source'].endswith('/'+file) and '/track/' not in row['source'])
        nearby = [entry for entry in row['diagnostics']
                  if x <= entry['seed']['rect'][0]+entry['seed']['rect'][2]/2 <= x+width
                  and y <= entry['seed']['rect'][1]+entry['seed']['rect'][3]/2 <= y+height]
        negatives.append(dict(key=key, source=file, rect=rect, components=components,
                              selected_seed_count=len(nearby), accepted=sum('accepted' in entry for entry in nearby)))
    return dict(seed_options=groups, hard_negatives=negatives,
                note='Seed fill overlaps. Only the combined color+pair+width+caps criterion separates observed slot evidence; rejected seeds can include unresolved true enemies.')


def audit53(output, samples, replay, baseline_ref, review_path=None):
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.run(['git', 'show', baseline_ref+':src/jccbrain/vision/eye/opponent.py'],
                            cwd=ROOT, check=True, text=True, capture_output=True).stdout
    baseline = types.ModuleType('b8_baseline_opponent')
    exec(compile(source, '<git:'+baseline_ref+'>', 'exec'), baseline.__dict__)
    (output/'baseline-opponent.py').write_text(source)
    files = (sorted(samples.glob('*.jpg')) + sorted((samples/'track').glob('*.jpg'))
             + sorted((replay/'spectate').glob('*.png')))
    truth = {row['_file']: row for row in map(json.loads,
             (ROOT/'tests/fixtures/opponent_truth.jsonl').read_text().splitlines())}
    reviewed = json.loads(review_path.read_text()) if review_path else []
    reviews = {(row['source'], tuple(row['rect'])): row for row in reviewed}
    rows, crops, patches = [], [], {}
    for index, file in enumerate(files):
        image = load_image(file)
        h, w = image.shape[:2]
        file_key = str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        old, old_legends = baseline.enemy_bars(image)
        current, legends = opponent.enemy_bars(image)
        old_json, new_json = [bar_json(b) for b in old], [bar_json(b) for b in current]
        added = [bar for bar in current if bar_json(bar) not in old_json]
        removed = [bar_json(bar) for bar in old if bar_json(bar) not in new_json]
        diagnostics = []
        opponent.recover_enemy_bars(image, diagnostics)
        expected = truth.get(file.name) if file.parent.resolve() == samples.resolve() else None
        entries = []
        canvas = image.copy()
        for bar in old:
            cv2.rectangle(canvas, (bar.min_x, bar.min_y), (bar.max_x, bar.max_y), (255, 0, 255), 1)
        for number, bar in enumerate(added, 1):
            geometry = bar_json(bar)
            matched = [i for i, point in enumerate((expected or {}).get('enemy_positions', []))
                       if abs(bar.cx-point['x']) < .02 and abs(bar.cy-point['y']) < .03]
            seed_records = [row for row in diagnostics if row.get('accepted') and
                            abs((row['accepted']['rect'][0]+row['accepted']['rect'][1])/2/w-bar.cx) < .02 and
                            abs((row['accepted']['rect'][2]+row['accepted']['rect'][3])/2/h-bar.cy) < .01]
            review = reviews.get((file_key, tuple(geometry['rect'])))
            if review and review['source_sha256'] != sha:
                raise ValueError('review source hash changed: '+file_key)
            entry = dict(**geometry, matches_enemy_positions=matched if expected else None,
                         evidence=seed_records, review=review)
            entries.append(entry)
            key = f'{index:02d}-{number}'
            x0, x1 = max(0, bar.min_x-30), min(w, bar.max_x+35)
            y0, y1 = max(0, bar.min_y-15), min(h, bar.max_y+45)
            patches[key] = image[y0:y1, x0:x1].copy()
            crops.append(dict(key=key, source=file_key, source_sha256=sha, origin=[x0, y0], candidate=entry))
            cv2.rectangle(canvas, (bar.min_x, bar.min_y), (bar.max_x, bar.max_y), (0, 255, 255), 2)
            cv2.putText(canvas, str(number), (bar.min_x, bar.min_y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 255, 255), 1)
        if entries or expected or file.name == '0060.jpg':
            cv2.imwrite(str(output/f'{index:02d}-{file.stem}-audit.jpg'), canvas)
        rows.append(dict(source=file_key, source_sha256=sha, shape=list(image.shape),
                         old=old_json, current=new_json, added=entries, removed=removed,
                         legends_unchanged=[bar_json(b) for b in old_legends] == [bar_json(b) for b in legends],
                         truth=expected, seeds=seeds(image), diagnostics=diagnostics))
    for page in range(0, len(crops), 12):
        sheet = np.zeros((4*120, 3*340, 3), np.uint8)
        for k, entry in enumerate(crops[page:page+12]):
            col, row = k%3, k//3
            patch = patches[entry['key']]
            ph, pw = patch.shape[:2]
            sheet[row*120+27:row*120+27+ph, col*340:col*340+pw] = patch
            text = f"{entry['key']} {Path(entry['source']).parent.name}/{Path(entry['source']).stem}"
            cv2.putText(sheet, text, (col*340, row*120+17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.imwrite(str(output/f'added-sheet-{page//12}.png'), sheet)
    np.savez_compressed(output/'added-crops.npz', **patches)
    (output/'added-crops.json').write_text(json.dumps(crops, ensure_ascii=False, indent=2)+'\n')
    added_count = sum(len(row['added']) for row in rows)
    reviewed_count = sum(item['review'] is not None for row in rows for item in row['added'])
    report = dict(baseline_ref=baseline_ref, product_import=opponent.__file__,
                  product_sha256=hashlib.sha256(Path(opponent.__file__).read_bytes()).hexdigest(),
                  uses_product_enemy_bars=True, frame_count=len(rows), added_count=added_count,
                  removed_count=sum(len(row['removed']) for row in rows), reviewed_count=reviewed_count,
                  unreviewed_count=added_count-reviewed_count,
                  false_new_count=sum(item['review'] is not None and item['review']['label'] != 'visible_enemy_slot'
                                      for row in rows for item in row['added']),
                  calibration=calibration(rows,samples), frames=rows)
    (output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, ensure_ascii=False, indent=2))
    for row in rows:
        if row['truth']:
            print(row['source'], len(row['old']), '->', len(row['current']), '/', row['truth']['board_count'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for command in ('baseline', 'audit53'):
        parser = sub.add_parser(command)
        parser.add_argument('--output', type=Path, required=True, help='新建证据目录，不覆盖已有审计')
        parser.add_argument('--samples', type=Path, default=ROOT/'game-eye/samples')
        if command == 'audit53':
            parser.add_argument('--replay', type=Path, default=ROOT/'var/vision-module-optimization/replay-samples')
            parser.add_argument('--baseline-ref', default='3a923b1', help='仅从git show读取旧opponent源码')
            parser.add_argument('--review', type=Path, help='人工逐新增复核记录；严格核对源文件哈希/实际框')
    args = ap.parse_args()
    # Resolve repo-relative paths without following the samples symlink so audit
    # labels remain portable inside this worktree; decoding still reads real files.
    args.output = args.output.absolute()
    args.samples = args.samples.absolute()
    if args.command == 'baseline':
        if args.output.exists():
            raise FileExistsError(args.output)
        current_audit(args.output, args.samples)
    else:
        audit53(args.output, args.samples, args.replay.absolute(), args.baseline_ref, args.review)


if __name__=='__main__':main()
