"""Rebuild small CG-derived L1 selection patches; no full screenshots in Git."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from jccbrain.vision.eye.capture import load_image
from jccbrain.vision.eye.units import Gallery,UnitTracker


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.with_suffix('.npz').exists() or a.output.with_suffix('.json').exists():raise ValueError('output exists')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    source=ROOT/'var/vision-module-optimization/batch-L1/conductor/live-preparation'
    cases=[('board-a-0.png',.4002254791,.5488556338,True),('board-b-0.png',.5329763247,.5132042254,True),
           ('board-c-0.png',.4391206313,.3424295775,True),('board-a-0.png',.2421082300,.6452464789,False),
           ('board-b-0.png',.3041149944,.6558098592,False)]
    arrays={};rows=[];tracker=UnitTracker(Gallery(a.output.parent/'.disabled-gallery',pool=None))
    for i,(name,x,y,selected) in enumerate(cases):
        image=load_image(source/name);h,w=image.shape[:2]
        _,units=tracker.detect(image)
        unit=next(u for u in units if abs(u.x-x)<.00001 and abs(u.y-y)<.00001)
        b=unit.bar;bw=b.width
        rect=[int(unit.x*w-.8*bw),b.max_y+max(1,round(.005*h)),int(unit.x*w+.8*bw),b.max_y+max(1,round(.005*h))+round(1.9*bw)]
        key=f'patch{i}';arrays[key]=image[rect[1]:rect[3],rect[0]:rect[2]].copy()
        rows.append(dict(key=key,file=name,sha256=hashlib.sha256((source/name).read_bytes()).hexdigest(),
            window=[w,h],rect=rect,bar=vars(b),zone=unit.zone,slot=unit.slot,selected=selected))
    np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
    a.output.with_suffix('.json').write_text(json.dumps(dict(version=1,source=str(source.relative_to(ROOT)),
        notes='Real cyan outline versus unselected blue/silver unit patches; manual reviewed live targets, no predictor labels.',entries=rows),ensure_ascii=False,indent=2)+'\n')
    print(sum(p.stat().st_size for p in (a.output.with_suffix('.npz'),a.output.with_suffix('.json'))))

if __name__=='__main__':main()
