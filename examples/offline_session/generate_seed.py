"""Generate the procedural offline fixture; no photographs or model output."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def seed_image():
    w,h=1344,768
    x=np.arange(w)[None,:];y=np.arange(h)[:,None]
    pixels=np.stack(np.broadcast_arrays(48+(x//96%2)*22+y//24,65+(y//64%2)*20+x//32,92+(x+y)//48),axis=-1).clip(20,220).astype(np.uint8)
    image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
    for row in range(5):
        for col in range(9):
            left=30+col*148;top=25+row*140
            draw.rectangle((left,top,left+96,top+92),outline=(215,180,125),width=4)
            draw.ellipse((left+20,top+20,left+65,top+65),fill=(80+row*20,130+col*8,125))
            draw.line((left,top+92,left+96,top),fill=(190,110,170),width=2)
    draw.text((28,735),'OFFLINE PROCEDURAL FIXTURE - NOT A MODEL',fill='white')
    return image

def generate(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    seed_image().save(path)
    if path.stat().st_size>=200000:raise RuntimeError('Fixture seed exceeds size limit')
    return path

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    generate(parser.parse_args().out)
