"""Stream AV1 byte-plane MKV -> Gaussian attributes -> CUDA renderer -> local player.

Run with the existing cu121-3dgs Python. Keys 1..4 switch benchmark cameras;
space pauses and Esc quits. --headless --limit 2 tests the direct path without UI.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmark'))
from adapt.common import read_json
from adapt.av1_planes import read_exact,table_from_atlas
from render_psnr import cameras,gpu_attributes,render


def main():
    import torch
    import cv2
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded',type=Path,default=ROOT/'benchmark/results/morton_av1/encoded')
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--width',type=int,default=960)
    args=p.parse_args()
    torch.set_num_threads(2)
    config=read_json(args.encoded/'manifest.json')
    cams=cameras(args.width)
    command=[str(args.ffmpeg),'-v','error','-nostdin','-threads','4','-i',str(args.encoded/config['video']),
             '-f','rawvideo','-pix_fmt','gray','pipe:1']
    count=0
    view=0
    quit_requested=False
    with tempfile.TemporaryFile() as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for i,points in enumerate(config['counts']):
                if args.limit and i>=args.limit:
                    break
                started=time.perf_counter()
                data=read_exact(process.stdout,config['width']*config['height'])
                if hashlib.sha256(data).hexdigest()!=config['atlas_sha256'][i]:
                    raise ValueError(f'Attribute plane corruption at frame {i}')
                atlas=np.frombuffer(data,dtype=np.uint8).reshape(config['height'],config['width'])
                table=table_from_atlas(atlas,points,config)
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
                        cv2.putText(bgr,f'{i+1}/{config["frame_count"]}  view {view+1}  1-4:view Space:pause Esc:quit',
                                    (12,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1,cv2.LINE_AA)
                        cv2.imshow('AV1 byte planes -> 3DGS',bgr)
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
                if quit_requested:
                    break
            if not args.limit and not quit_requested:
                if process.stdout.read(1):
                    raise ValueError('Unexpected extra video frames')
                if process.wait()!=0:
                    log.seek(0)
                    raise RuntimeError(log.read().decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            if not args.headless:
                cv2.destroyAllWindows()
    print(f'Direct AV1 -> Gaussian -> CUDA playback: {count} frames, checksums verified, device={torch.cuda.get_device_name(0)}')


if __name__=='__main__':
    main()
