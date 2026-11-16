"""AV1 v2: preserve v1's quantized attributes, remove representation redundancy.

Stable Morton order per GOP; exact SH dictionary; immutable attributes and static
geometry once per GOP; only dynamic xyz/quaternion modulo-uint16 deltas in AV1.
All numerical attributes use EXACTLY the v1 global 16-bit quantizer.
"""
from __future__ import annotations
import os
if __name__=='__main__':
    # Set limits before importing NumPy or any numerical dependencies.
    for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT']:
        os.environ[_name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['OMP_DYNAMIC']='FALSE'
    os.environ['MKL_DYNAMIC']='FALSE'
import hashlib
import io
import lzma
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from adapt.common import read_ply,read_json,write_json,save_frame,save_sequence
from adapt.av1_planes import COLUMNS,morton_order,read_exact
from smv_codec import analyze

MOTION=[0,1,2,55,56,57,58]
MISC=[3,4,5,51,52,53,54]


def quantize(table,config):
    lo=np.asarray(config['mins'],dtype=np.float64)
    hi=np.asarray(config['maxs'],dtype=np.float64)
    return np.rint(np.clip((table[:,COLUMNS].astype(np.float64)-lo)/np.where(hi>lo,hi-lo,1),0,1)*65535).astype(np.uint16)


def dequantize(codes,config):
    lo=np.asarray(config['mins'],dtype=np.float64)
    hi=np.asarray(config['maxs'],dtype=np.float64)
    table=np.zeros((len(codes),62),dtype=np.float32)
    table[:,COLUMNS]=(codes.astype(np.float64)/65535*(hi-lo)+lo).astype(np.float32)
    return table


def pack(residual,side):
    planes=np.zeros((16,side*side),dtype=np.uint8)
    n=len(residual)
    planes[0:14:2,:n]=(residual&255).T
    planes[1:14:2,:n]=(residual>>8).T
    return planes.reshape(4,4,side,side).transpose(0,2,1,3).reshape(4*side,4*side)


def unpack(atlas,count,side):
    planes=atlas.reshape(4,side,4,side).transpose(0,2,1,3).reshape(16,side*side)
    return planes[0:14:2,:count].T.astype(np.uint16)|(planes[1:14:2,:count].T.astype(np.uint16)<<8)


def zigzag(current,predictor):
    signed=(current.astype(np.int32)-predictor.astype(np.int32)).astype(np.int16).astype(np.int32)
    return ((signed<<1)^(signed>>15)).astype(np.uint16)


def unzigzag(code,predictor):
    signed=(code.astype(np.int32)>>1)^-(code.astype(np.int32)&1)
    return ((predictor.astype(np.int32)+signed)&65535).astype(np.uint16)


def encode(paths,timestamps,folder,ffmpeg,threads=1,predictor='previous'):
    if threads!=1:
        raise ValueError('This optimization experiment requires a single CPU thread')
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    frames=[{'path':p,'timestamp':t} for p,t in zip(paths,timestamps)]
    segments=analyze(frames,30)
    lo=np.full(59,np.inf)
    hi=-lo.copy()
    # Positions/quaternions vary, so inspect every frame for v1-identical ranges.
    for p in paths:
        table=read_ply(p)
        lo=np.minimum(lo,table[:,COLUMNS].min(axis=0))
        hi=np.maximum(hi,table[:,COLUMNS].max(axis=0))
    side=max(16,math.ceil(math.sqrt(max(int(s['dynamic'].sum()) for s in segments))))
    config={'format':'morton-av1-temporal','version':2,'quantization_bits':16,'mins':lo.tolist(),'maxs':hi.tolist(),
        'columns':COLUMNS,'motion_columns_in_codes':MOTION,'misc_columns_in_codes':MISC,
        'timestamps':timestamps,'frame_count':len(paths),'tile_side':side,'width':4*side,'height':4*side,
        'predictor':predictor,'residual':'signed modulo-uint16 delta, zigzag uint16',
        'video':'motion.mkv','sidecar':'keyframes.npz.xz','codec':'libaom-av1','pixel_format':'gray',
        'sidecar_codec':'stored NPZ + single-thread LZMA2, preset6, 32MiB dictionary',
        'video_lossless':True,'threads':1,'cpu_used':8,'crf':0,'gops':[],
        'atlas_sha256':[],'quantized_frame_sha256':[],'preserves_v1_quantizer':True}
    arrays={}
    for g,s in enumerate(segments):
        base=s['base']
        static=np.flatnonzero(~s['dynamic'])
        dynamic=np.flatnonzero(s['dynamic'])
        static=static[morton_order(base[static,:3],lo[:3],hi[:3])]
        dynamic=dynamic[morton_order(base[dynamic,:3],lo[:3],hi[:3])]
        order=np.concatenate((static,dynamic))
        codes=quantize(base[order],config)
        palette,labels=np.unique(codes[:,6:51],axis=0,return_inverse=True)
        p=f'gop_{g}_'
        arrays[p+'order']=order.astype(np.int32)
        arrays[p+'misc']=codes[:,MISC]
        arrays[p+'sh_palette']=palette
        arrays[p+'sh_labels']=labels.astype(np.uint16 if len(palette)<=65536 else np.uint32)
        arrays[p+'static_motion']=codes[:len(static),MOTION]
        arrays[p+'motion_reference']=codes[len(static):,MOTION]
        config['gops'].append({'gop':g,'start_frame':s['start'],'frame_count':s['end']-s['start'],
            'point_count':len(base),'static_count':len(static),'dynamic_count':len(dynamic),
            'sh_unique_vectors':len(palette),'keyframe_sort':'static Morton then dynamic Morton, reused for whole GOP'})
        del s['base']
        s['order']=order
    payload=io.BytesIO()
    np.savez(payload,**arrays)
    compressed=lzma.compress(payload.getvalue(),filters=[{'id':lzma.FILTER_LZMA2,'preset':6,'dict_size':32*1024*1024}])
    (folder/config['sidecar']).write_bytes(compressed)
    config['sidecar_sha256']=hashlib.sha256(compressed).hexdigest()
    force=','.join(str(s['start_frame']/30) for s in config['gops'])
    command=[str(ffmpeg),'-v','error','-y','-filter_threads','1','-threads','1','-f','rawvideo',
        '-pix_fmt','gray','-s:v',f'{4*side}x{4*side}','-r','30','-i','pipe:0','-an',
        '-c:v','libaom-av1','-cpu-used','8','-crf','0','-b:v','0','-pix_fmt','gray',
        '-row-mt','0','-threads','1','-g','30','-force_key_frames',force,str(folder/'motion.mkv')]
    config['ffmpeg_command']=command
    with (folder/'encode.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for g,(segment,info) in enumerate(zip(segments,config['gops'])):
                p=f'gop_{g}_'
                previous=arrays[p+'motion_reference']
                reference=previous
                order=segment['order']
                static=info['static_count']
                for i in range(info['start_frame'],info['start_frame']+info['frame_count']):
                    raw=quantize(read_ply(paths[i]),config)
                    config['quantized_frame_sha256'].append(hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest())
                    codes=raw[order]
                    if (not np.array_equal(codes[:,MISC],arrays[p+'misc'])
                            or not np.array_equal(codes[:,6:51],arrays[p+'sh_palette'][arrays[p+'sh_labels']])
                            or not np.array_equal(codes[:static,MOTION],arrays[p+'static_motion'])):
                        raise ValueError('A supposedly immutable/static field changed within the GOP')
                    motion=codes[static:,MOTION]
                    pred=previous if predictor=='previous' else reference
                    atlas=pack(zigzag(motion,pred),side)
                    previous=motion
                    data=atlas.tobytes()
                    config['atlas_sha256'].append(hashlib.sha256(data).hexdigest())
                    process.stdin.write(data)
                print(f'Optimized AV1 GOP {g+1}/{len(segments)}: {info["static_count"]:,} static, '
                      f'{info["dynamic_count"]:,} dynamic, {info["sh_unique_vectors"]} exact SH vectors',flush=True)
            process.stdin.close()
            result=process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdin.close()
    if result:
        raise RuntimeError((folder/'encode.log').read_text(errors='replace'))
    config['raw_motion_atlas_bytes']=4*side*4*side*len(paths)
    config['raw_full_attribute_bytes']=sum(s['point_count']*s['frame_count']*118 for s in config['gops'])
    write_json(folder/'manifest.json',config)
    return {'sh_degree':3,'algorithm':'stable GOP Morton; exact SH dictionary; static/immutable once; AV1 dynamic delta byte planes',
        'threads':1,'preserves_v1_quantizer':True,'gop_count':len(segments),
        'motion_atlas_dimensions':[4*side,4*side],'predictor':predictor,
        'keyframe_bytes':(folder/config['sidecar']).stat().st_size,'video_bytes':(folder/'motion.mkv').stat().st_size,
        'keyframe_codec':config['sidecar_codec']}


def iter_tables(folder,ffmpeg):
    folder=Path(folder)
    config=read_json(folder/'manifest.json')
    if hashlib.sha256((folder/config['sidecar']).read_bytes()).hexdigest()!=config['sidecar_sha256']:
        raise ValueError('Keyframe sidecar checksum mismatch')
    command=[str(ffmpeg),'-v','error','-nostdin','-filter_threads','1','-threads','1','-i',str(folder/config['video']),
        '-f','rawvideo','-pix_fmt','gray','-threads','1','pipe:1']
    payload=lzma.decompress((folder/config['sidecar']).read_bytes())
    with tempfile.TemporaryFile() as log, np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for info in config['gops']:
                g=info['gop']
                p=f'gop_{g}_'
                order=arrays[p+'order']
                static=info['static_count']
                template=np.zeros((info['point_count'],59),dtype=np.uint16)
                template[:,MISC]=arrays[p+'misc']
                template[:,6:51]=arrays[p+'sh_palette'][arrays[p+'sh_labels']]
                template[:static,MOTION]=arrays[p+'static_motion']
                reference=arrays[p+'motion_reference']
                previous=reference
                for i in range(info['start_frame'],info['start_frame']+info['frame_count']):
                    data=read_exact(process.stdout,config['width']*config['height'])
                    if hashlib.sha256(data).hexdigest()!=config['atlas_sha256'][i]:
                        raise ValueError(f'AV1 atlas checksum mismatch: {i}')
                    atlas=np.frombuffer(data,dtype=np.uint8).reshape(config['height'],config['width'])
                    residual=unpack(atlas,info['dynamic_count'],config['tile_side'])
                    pred=previous if config['predictor']=='previous' else reference
                    motion=unzigzag(residual,pred)
                    template[static:,MOTION]=motion
                    previous=motion
                    codes=np.empty_like(template)
                    codes[order]=template
                    if hashlib.sha256(codes.astype('<u2').tobytes()).hexdigest()!=config['quantized_frame_sha256'][i]:
                        raise ValueError(f'Quantized attribute checksum mismatch: {i}')
                    yield i,config['timestamps'][i],dequantize(codes,config)
            if process.stdout.read(1):
                raise ValueError('Extra video frames')
            if process.wait()!=0:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def decode(folder,output,ffmpeg,threads=1):
    if threads!=1:
        raise ValueError('This optimization experiment requires a single CPU thread')
    timestamps=[]
    for i,timestamp,table in iter_tables(folder,ffmpeg):
        save_frame(output,i,table)
        timestamps.append(timestamp)
    save_sequence(output,timestamps)
    return {'frame_count':len(timestamps),'all_atlas_checksums_match':True,
            'all_quantized_attribute_checksums_match':True,'same_16bit_quantizer_as_v1':True}


def main():
    import argparse
    from adapt.common import ROOT,sequence_paths
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['encode','decode'])
    parser.add_argument('--sequence',type=Path)
    parser.add_argument('--encoded',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output directory exists; choose a fresh directory')
    # Reuse the workspace's audited single-core execution policy for this CLI.
    # Imported encode/decode functions remain controlled by their caller/runner.
    import importlib.util
    runtime_path=ROOT/'benchmark/optimization/runtime.py'
    spec=importlib.util.spec_from_file_location('_main_av1_runtime',runtime_path)
    runtime=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    cpu,mask=runtime.pin_cpu()
    runtime.audit_ffmpeg(mask,[])
    print(f'Single-thread CPU codec; logical CPU={cpu}; affinity mask={mask}',flush=True)
    if args.operation=='encode':
        if not args.sequence:
            parser.error('--sequence is required')
        paths,timestamps=sequence_paths(args.sequence)
        encode(paths,timestamps,args.output,args.ffmpeg,threads=1)
    else:
        if not args.encoded:
            parser.error('--encoded is required')
        decode(args.encoded,args.output,args.ffmpeg,threads=1)


if __name__=='__main__':
    main()
