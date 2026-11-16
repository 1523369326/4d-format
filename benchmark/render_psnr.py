"""Render all source/decoded frames from four fixed cameras with one CUDA renderer.

Run in the existing cu121-3dgs environment. Pixel MSE uses clamped float RGB,
before PNG quantization, over the entire image and all RGB channels.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_ply, read_json, write_json, sequence_paths


def rotation_matrix(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]],dtype=np.float64)


def cameras(width=None):
    original=read_json(ROOT/'dataset/high_discussion/camera.json')
    w=width or original['width']
    h=round(original['height']*w/original['width'])
    R=rotation_matrix(original['camquat'])
    C=np.array(original['campos'],dtype=np.float64)
    target=C+R[:,2]*2.5
    output=[]
    for name,local_offset in [('source',[0,0,0]),('right',[.2,0,0]),('left',[-.2,0,0]),('up',[0,-.15,0])]:
        center=C+R@np.array(local_offset)
        if name=='source':
            rotation=R
        else:
            forward=target-center
            forward/=np.linalg.norm(forward)
            right=np.cross(R[:,1],forward)
            right/=np.linalg.norm(right)
            down=np.cross(forward,right)
            rotation=np.column_stack((right,down,forward))
        world_view=np.eye(4)
        world_view[:3,:3]=rotation.T
        world_view[:3,3]=-rotation.T@center
        znear,zfar=.01,1000.
        projection=np.zeros((4,4))
        projection[0,0]=1/math.tan(original['fovx']/2)
        projection[1,1]=1/math.tan(original['fovy']/2)
        projection[2,2]=zfar/(zfar-znear)
        projection[2,3]=-zfar*znear/(zfar-znear)
        projection[3,2]=1
        view=torch.tensor(world_view.T,dtype=torch.float32,device='cuda')
        proj=torch.tensor(projection.T,dtype=torch.float32,device='cuda')
        settings=GaussianRasterizationSettings(image_height=h,image_width=w,
            tanfovx=math.tan(original['fovx']/2),tanfovy=math.tan(original['fovy']/2),
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=view,
            projmatrix=view@proj,sh_degree=3,campos=torch.tensor(center,dtype=torch.float32,device='cuda'),
            prefiltered=False,debug=False,antialiasing=False)
        output.append((name,GaussianRasterizer(raster_settings=settings),
            {'name':name,'camera_to_world_rotation':rotation.tolist(),'position':center.tolist(),
             'width':w,'height':h,'fovx':original['fovx'],'fovy':original['fovy']}))
    return output


def gpu_attributes(table):
    sh=np.concatenate((table[:,6:9,None].transpose(0,2,1),
                       table[:,9:54].reshape(-1,3,15).transpose(0,2,1)),axis=1)
    xyz=torch.tensor(table[:,:3].copy(),device='cuda')
    rotations=torch.tensor(table[:,58:62].copy(),device='cuda')
    rotations=rotations/torch.linalg.vector_norm(rotations,dim=1,keepdim=True).clamp_min(1e-20)
    return {'means3D':xyz,'means2D':torch.zeros_like(xyz),
        'shs':torch.tensor(sh.copy(),device='cuda'),
        'opacities':torch.sigmoid(torch.tensor(table[:,54:55].copy(),device='cuda')),
        'scales':torch.exp(torch.tensor(table[:,55:58].copy(),device='cuda')),
        'rotations':rotations}


@torch.no_grad()
def render(attributes,rasterizer):
    result=rasterizer(**attributes)
    image=result[0].clamp(0,1)
    if not torch.isfinite(image).all():
        raise ValueError('Nonfinite rendered image')
    return image


def save_png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    array=image.detach().permute(1,2,0).cpu().numpy()
    Image.fromarray(np.rint(array*255).astype(np.uint8)).save(path)


def psnr(mse):
    return -10*math.log10(mse) if mse>0 else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--methods',nargs='+',default=['texture_h264','gsd_v1','gsd_v2','tsog','morton_av1','morton_av1_opt','smv'])
    p.add_argument('--results',type=Path,default=ROOT/'benchmark/results_single_thread_v2')
    p.add_argument('--sequence',type=Path,default=ROOT/'benchmark/input_first60.json')
    p.add_argument('--width',type=int)
    p.add_argument('--limit',type=int)
    p.add_argument('--reference-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    paths,timestamps=sequence_paths(args.sequence)
    if args.limit:
        paths=paths[:args.limit]
    cams=cameras(args.width)
    write_json(args.results/'cameras.json',{'views':[c[2] for c in cams],
        'renderer':'diff_gaussian_rasterization CUDA','sh_degree':3,'background':[0,0,0],
        'antialiasing':False,'pixel_domain':'float RGB clamped to [0,1], full-frame RGB MSE before PNG quantization'})
    rows={method:[] for method in args.methods}
    if not args.reference_only:
        for method in args.methods:
            manifest=read_json(args.results/method/'decoded/sequence.json')
            if len(manifest['frames'])!=len(timestamps):
                raise ValueError(f'Decoded sequence frame count mismatch: {method}')
            if [f['timestamp'] for f in manifest['frames']]!=timestamps:
                raise ValueError(f'Decoded sequence timestamps mismatch: {method}')
    sample_frames={0,len(paths)//2,len(paths)-1}
    for i,path in enumerate(paths):
        source=read_ply(path)
        attributes=gpu_attributes(source)
        reference={name:render(attributes,rasterizer) for name,rasterizer,_ in cams}
        if i in sample_frames:
            for name,image in reference.items():
                save_png(args.results/'renders/reference'/f'frame_{i:06d}_{name}.png',image)
        if not args.reference_only:
            save_png(args.results/'renders/playback_reference'/f'frame_{i:06d}.png',reference['source'])
        del attributes
        if args.reference_only:
            continue
        for method in args.methods:
            decoded=read_ply(args.results/method/'decoded'/f'frame_{i:06d}.ply')
            if len(decoded)!=len(source):
                raise ValueError(f'Decoded point count mismatch: {method}/{i}')
            attributes=gpu_attributes(decoded)
            for name,rasterizer,_ in cams:
                image=render(attributes,rasterizer)
                mse=float(torch.mean((image.double()-reference[name].double())**2).item())
                rows[method].append({'frame':i,'view':name,'mse':mse,'psnr_db':psnr(mse)})
                if i in sample_frames:
                    save_png(args.results/'renders'/method/f'frame_{i:06d}_{name}.png',image)
                if method in ['morton_av1','morton_av1_opt','smv'] and name=='source':
                    save_png(args.results/'renders'/f'playback_{method}'/f'frame_{i:06d}.png',image)
            del attributes
        if (i+1)%10==0:
            print(f'Rendered and compared {i+1}/{len(paths)} frames x 4 views',flush=True)
    if args.reference_only:
        print('Reference camera renders written',flush=True)
        return
    summaries={}
    for method,entries in rows.items():
        per_view={name:psnr(float(np.mean([e['mse'] for e in entries if e['view']==name]))) for name,_,_ in cams}
        mse=float(np.mean([e['mse'] for e in entries]))
        summaries[method]={'frames':len(paths),'view_count':4,'images_compared':len(entries),
            'global_rgb_mse':mse,'pooled_psnr_db':psnr(mse),'per_view_pooled_psnr_db':per_view,
            'mean_image_psnr_db':float(np.mean([e['psnr_db'] for e in entries if e['psnr_db'] is not None]))}
        write_json(args.results/method/'quality.json',{**summaries[method],'per_frame_view':entries})
    write_json(args.results/'quality_summary.json',summaries)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    main()
