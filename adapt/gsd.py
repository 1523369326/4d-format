"""Native GSD v1/v2 encoders, and a reader for their actual emitted layout."""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import lz4.block
from .common import ROOT, activated_table, save_frame, save_sequence, write_json, read_json


def encode(paths,timestamps,folder,ffmpeg=None,version=1,threads=None):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'third_party/4dgs-converter'))
    if version==1:
        from app.pipeline.ply_to_gsd import convert_ply_to_gsd
        stats=convert_ply_to_gsd(str(paths[0].parent),str(folder/'scene.gsd'),'discussion_first60',
            target_fps=30,sh_degree=3,position_precision=0,rotation_precision=1,
            scale_opacity_precision=1,sh_precision=1,max_workers=2 if threads is None else threads,
            start_frame=0,end_frame=len(paths)-1,progress_callback=print)
    else:
        from app.pipeline.ply_to_gsd_v2 import convert_ply_to_gsd_v2
        np.random.seed(42)
        stats=convert_ply_to_gsd_v2(str(paths[0].parent),str(folder/'scene.gsd'),'discussion_first60',
            target_fps=30,max_workers=2 if threads is None else threads,start_frame=0,end_frame=len(paths)-1,progress_callback=print)
    write_json(folder/'timeline.json',{'timestamps':timestamps})
    return {'native_stats':stats,'sh_degree':3 if version==1 else 0,
        'implementation':f'unmodified native conversion API, {2 if threads is None else threads} worker process(es)',
        'settings':'v1: FP32 position; FP16 rotation/activated scale/alpha/SH' if version==1 else 'v2: VQ256 rotation/scale/DC, FP16 position, no higher SH'}


def take_block(f):
    size_bytes=f.read(4)
    if len(size_bytes)!=4:
        raise ValueError('Truncated GSD size')
    size=struct.unpack('<I',size_bytes)[0]
    data=f.read(size)
    if len(data)!=size:
        raise ValueError('Truncated GSD blob')
    return data


def v1_table(blob,header,info):
    pixels=header['textureWidth']*header['textureHeight']
    precisions=[header['positionPrecision'],header['rotationPrecision'],header['scaleOpacityPrecision']]+[header['shPrecision']]*12
    raw=lz4.block.decompress(blob,uncompressed_size=sum(pixels*(16 if p==0 else 8) for p in precisions))
    offset=0
    textures=[]
    for p in precisions:
        bpp=16 if p==0 else 8
        section=np.frombuffer(raw,dtype=np.uint8,count=pixels*bpp,offset=offset)
        unshuffled=section.reshape(bpp,pixels).T.copy()
        textures.append(unshuffled.reshape(-1).view('<f4' if p==0 else '<f2').reshape(pixels,4).astype(np.float32))
        offset+=pixels*bpp
    n=info['gaussianCount']
    pos,rot,so=[t[:n] for t in textures[:3]]
    positions=np.column_stack((pos[:,1],-pos[:,2],pos[:,0]))
    quats=np.column_stack((rot[:,3],rot[:,1],-rot[:,2],rot[:,0]))
    scales=np.column_stack((so[:,1],so[:,2],so[:,0]))
    sh=np.concatenate(textures[3:],axis=1)[:n]
    # Native textures: DC RGB, then coefficient-major RGB higher bands.
    rest=sh[:,3:].reshape(n,15,3).transpose(0,2,1).reshape(n,45)
    return activated_table(positions,scales,quats,sh[:,:3],so[:,3],rest)


def v2_table(blob,header,info):
    sizes=struct.unpack('<8I',blob[:32])
    parts=[]
    offset=32
    for size in sizes:
        parts.append(blob[offset:offset+size])
        offset+=size
    if offset!=len(blob):
        raise ValueError('GSD v2 section sizes mismatch')
    pixels=header['textureWidth']*header['textureHeight']
    k=header['vqK']
    rot_cb=np.frombuffer(parts[0],dtype='<f4').reshape(k,4)
    scale_cb=np.frombuffer(parts[1],dtype='<f4').reshape(k,3)
    dc_cb=np.frombuffer(parts[2],dtype='<f4').reshape(k,3)
    raw=lz4.block.decompress(parts[3],uncompressed_size=pixels*8)
    pos=np.frombuffer(raw,dtype=np.uint8).reshape(8,pixels).T.copy().reshape(-1).view('<f2').reshape(pixels,4).astype(np.float32)
    indices=[np.frombuffer(lz4.block.decompress(p,uncompressed_size=pixels),dtype=np.uint8) for p in parts[4:]]
    n=info['gaussianCount']
    positions=np.column_stack((pos[:n,1],-pos[:n,2],pos[:n,0]))
    return activated_table(positions,np.exp(scale_cb[indices[1][:n]]),rot_cb[indices[0][:n]],
                           dc_cb[indices[3][:n]],indices[2][:n].astype(np.float32)/255)


def decode(folder,output,ffmpeg=None,threads=None):
    folder=Path(folder)
    with (folder/'scene.gsd').open('rb') as f:
        if f.read(4)!=b'GSD1':
            raise ValueError('GSD magic mismatch')
        header=json.loads(take_block(f))
        for i,info in enumerate(header['frames']):
            blob=take_block(f)
            table=v1_table(blob,header,info) if header['version']==1 else v2_table(blob,header,info)
            save_frame(output,i,table)
        if f.read(1):
            raise ValueError('Unexpected trailing GSD data')
    save_sequence(output,read_json(folder/'timeline.json')['timestamps'])
    return {'frame_count':header['frameCount'],'sh_degree':header['shDegree']}
