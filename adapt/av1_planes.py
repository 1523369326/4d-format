"""Morton -> per-attribute uint16 -> two gray8 planes -> lossless AV1/MKV.

One PLY maps to ONE video frame containing a tiled atlas of all byte planes.
No RGB/YUV matrix, chroma subsampling or byte reinterpretation is used.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .common import read_ply, read_json, write_json, save_frame, save_sequence

COLUMNS=[0,1,2]+list(range(6,62))


def morton_order(xyz,lo,hi):
    extent=np.where(hi>lo,hi-lo,1)
    q=np.clip((xyz.astype(np.float64)-lo)/extent*2097151,0,2097151).astype(np.uint64)
    def spread(x):
        x=(x|(x<<32))&np.uint64(0x1f00000000ffff)
        x=(x|(x<<16))&np.uint64(0x1f0000ff0000ff)
        x=(x|(x<<8))&np.uint64(0x100f00f00f00f00f)
        x=(x|(x<<4))&np.uint64(0x10c30c30c30c30c3)
        return (x|(x<<2))&np.uint64(0x1249249249249249)
    code=spread(q[:,0])|(spread(q[:,1])<<1)|(spread(q[:,2])<<2)
    return np.argsort(code,kind='stable')


def atlas_from_table(table,config):
    lo=np.array(config['mins'],dtype=np.float64)
    hi=np.array(config['maxs'],dtype=np.float64)
    order=morton_order(table[:,:3],lo[:3],hi[:3])
    values=table[order][:,COLUMNS].astype(np.float64)
    codes=np.rint(np.clip((values-lo)/np.where(hi>lo,hi-lo,1),0,1)*65535).astype(np.uint16)
    side=config['tile_side']
    plane_count=config['tile_columns']*config['tile_rows']
    planes=np.zeros((plane_count,side*side),dtype=np.uint8)
    planes[0:2*len(COLUMNS):2,:len(table)]=(codes&255).T
    planes[1:2*len(COLUMNS):2,:len(table)]=(codes>>8).T
    return planes.reshape(config['tile_rows'],config['tile_columns'],side,side).transpose(0,2,1,3).reshape(config['height'],config['width'])


def table_from_atlas(atlas,count,config):
    side=config['tile_side']
    planes=atlas.reshape(config['tile_rows'],side,config['tile_columns'],side).transpose(0,2,1,3).reshape(-1,side*side)
    low=planes[0:2*len(COLUMNS):2,:count].T.astype(np.uint16)
    high=planes[1:2*len(COLUMNS):2,:count].T.astype(np.uint16)
    codes=low|(high<<8)
    lo=np.array(config['mins'],dtype=np.float64)
    hi=np.array(config['maxs'],dtype=np.float64)
    table=np.zeros((count,62),dtype=np.float32)
    table[:,COLUMNS]=(codes.astype(np.float64)/65535*(hi-lo)+lo).astype(np.float32)
    return table


def encode(paths,timestamps,folder,ffmpeg,threads=4):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    lo=np.full(len(COLUMNS),np.inf)
    hi=-lo.copy()
    counts=[]
    for path in paths:
        table=read_ply(path)
        lo=np.minimum(lo,table[:,COLUMNS].min(axis=0))
        hi=np.maximum(hi,table[:,COLUMNS].max(axis=0))
        counts.append(len(table))
    side=math.ceil(math.sqrt(max(counts)))
    cols=12
    rows=math.ceil(2*len(COLUMNS)/cols)
    config={'format':'morton-av1-byteplanes','version':1,'quantization_bits':16,
        'columns':COLUMNS,'mins':lo.tolist(),'maxs':hi.tolist(),'counts':counts,
        'timestamps':timestamps,'frame_count':len(paths),'tile_side':side,
        'tile_columns':cols,'tile_rows':rows,'width':side*cols,'height':side*rows,
        'plane_order':'attribute 0 low, attribute 0 high, attribute 1 low, attribute 1 high, ...',
        'morton_bits_per_axis':21,'morton_bounds':'same position bounds as attribute quantization',
        'video':'attributes.mkv','codec':'libaom-av1','pixel_format':'gray',
        'video_lossless':True,'crf':0,'cpu_used':8,'threads':threads,'atlas_sha256':[]}
    args=[str(ffmpeg),'-v','error','-y','-filter_threads',str(threads),'-threads',str(threads),'-f','rawvideo','-pix_fmt','gray',
        '-s:v',f'{config["width"]}x{config["height"]}','-r','30','-i','pipe:0','-an',
        '-c:v','libaom-av1','-cpu-used','8','-crf','0','-b:v','0','-pix_fmt','gray',
        '-row-mt','0' if threads==1 else '1','-threads',str(threads),'-g','30',str(folder/'attributes.mkv')]
    config['ffmpeg_command']=args
    with (folder/'encode.log').open('wb') as log:
        p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i,path in enumerate(paths):
                atlas=atlas_from_table(read_ply(path),config)
                data=atlas.tobytes()
                config['atlas_sha256'].append(hashlib.sha256(data).hexdigest())
                p.stdin.write(data)
                if (i+1)%10==0:
                    print(f'AV1 encoded input {i+1}/{len(paths)}',flush=True)
            p.stdin.close()
            result=p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise
        finally:
            p.stdin.close()
    if result:
        raise RuntimeError((folder/'encode.log').read_text(errors='replace'))
    write_json(folder/'manifest.json',config)
    return {'sh_degree':3,'algorithm':'Morton, global uint16 attributes, gray8 low/high atlas, lossless AV1',
            'one_video_frame_per_ply':True}


def read_exact(stream,size):
    chunks=[]
    remaining=size
    while remaining:
        chunk=stream.read(remaining)
        if not chunk:
            raise ValueError('Truncated video frame')
        chunks.append(chunk)
        remaining-=len(chunk)
    return b''.join(chunks)


def decode(folder,output,ffmpeg,threads=4):
    folder,output=Path(folder),Path(output)
    config=read_json(folder/'manifest.json')
    args=[str(ffmpeg),'-v','error','-nostdin','-filter_threads',str(threads),'-threads',str(threads),'-i',str(folder/config['video']),
        '-f','rawvideo','-pix_fmt','gray','-threads',str(threads),'pipe:1']
    with tempfile.TemporaryFile() as log:
        p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=log)
        try:
            size=config['width']*config['height']
            for i,count in enumerate(config['counts']):
                data=read_exact(p.stdout,size)
                if hashlib.sha256(data).hexdigest()!=config['atlas_sha256'][i]:
                    raise ValueError(f'AV1 byte-plane checksum mismatch at frame {i}')
                atlas=np.frombuffer(data,dtype=np.uint8).reshape(config['height'],config['width'])
                save_frame(output,i,table_from_atlas(atlas,count,config))
            if p.stdout.read(1):
                raise ValueError('Unexpected extra AV1 frames')
            result=p.wait()
            if result:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors='replace'))
        except BaseException:
            p.kill()
            p.wait()
            raise
        finally:
            p.stdout.close()
    save_sequence(output,config['timestamps'])
    return {'all_byte_plane_checksums_match':True}


def main():
    import argparse
    from .common import ROOT,sequence_paths
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['encode','decode'])
    p.add_argument('--sequence',type=Path)
    p.add_argument('--encoded',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    args=p.parse_args()
    if args.output.exists():
        p.error('Output path already exists; choose a new directory')
    if args.operation=='encode':
        if not args.sequence:
            p.error('--sequence is required for encode')
        paths,timestamps=sequence_paths(args.sequence)
        encode(paths,timestamps,args.output,args.ffmpeg)
    else:
        if not args.encoded:
            p.error('--encoded is required for decode')
        decode(args.encoded,args.output,args.ffmpeg)


if __name__=='__main__':
    main()
