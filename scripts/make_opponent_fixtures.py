"""真实橙条/绿条小图回归，所有像素经CoreGraphics读取。"""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from jccbrain.vision.eye.capture import load_image


def main():
    main=Path('game-eye/samples')
    # 先按原图目视确认颜色和暗框，再记录小图；不存整帧。
    entries=[('enemy-shield','0015.jpg',836,222,True),('enemy','0033.jpg',1004,307,True),
             ('green','0033.jpg',1100,304,False)]
    arrays={};rows=[]
    for key,file,x,y,enemy in entries:
        image=load_image(main/file);x0,y0=x-20,y-10
        arrays[key]=image[y0:y0+40,x0:x0+144].copy()
        rows.append(dict(key=key,origin=[x0,y0],source=file,enemy=enemy))
    target=ROOT/'tests/fixtures/opponent_cells.npz';np.savez_compressed(target,**arrays)
    target.with_suffix('.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(target.stat().st_size+target.with_suffix('.json').stat().st_size)


if __name__=='__main__':main()
