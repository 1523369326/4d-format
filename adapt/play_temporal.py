"""Direct optimized AV1/sidecar -> Gaussian -> CUDA display (no PLY staging)."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmark'))
from av1_codec import iter_tables
from render_psnr import cameras,gpu_attributes,render


def main():
    import cv2
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoded',type=Path,default=ROOT/'benchmark/results_single_thread_v2/morton_av1_opt/encoded')
    parser.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--width',type=int,default=960)
    args=parser.parse_args()
    torch.set_num_threads(1)
    cams=cameras(args.width)
    view=0
    count=0
    iterator=iter_tables(args.encoded,args.ffmpeg)
    try:
        quit_requested=False
        for i,timestamp,table in iterator:
            started=time.perf_counter()
            attributes=gpu_attributes(table)
            if args.headless:
                image=render(attributes,cams[view][1])
                torch.cuda.synchronize()
            else:
                paused=False
                while True:
                    image=render(attributes,cams[view][1])
                    rgb=image.detach().permute(1,2,0).cpu().numpy()
                    bgr=np.rint(rgb[:,:,::-1]*255).astype(np.uint8)
                    cv2.putText(bgr,f'Frame {i+1}  {timestamp:.3f}s  view {view+1}  1-4:view Space:pause Esc:quit',
                                (12,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1,cv2.LINE_AA)
                    cv2.imshow('Temporal AV1 -> 3DGS',bgr)
                    delay=max(1,round(1000*(1/30-(time.perf_counter()-started)))) if not paused else 30
                    key=cv2.waitKey(delay)&255
                    if key==27:
                        quit_requested=True
                        break
                    if key==32:
                        paused=not paused
                    if key in map(ord,'1234'):
                        view=key-ord('1')
                        continue
                    if not paused:
                        break
            count+=1
            if quit_requested or (args.limit and count>=args.limit):
                break
    finally:
        iterator.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'Direct optimized AV1 playback: {count} frames; atlas + full quantized-table checksums verified')


if __name__=='__main__':
    main()
