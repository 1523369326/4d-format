"""GOP codec: exact v1 quantizer, SH dictionary, cached decode and source row order.

No input point identity is assumed: a GOP continues only while row-wise immutable
attributes remain exactly equal. All changing xyz/quaternion rows are transmitted.
New version-5 streams retain the GOP permutation and restore each input frame's
row order. Version-4 streams remain readable in their historical GOP Morton order.
"""
from __future__ import annotations
if __name__ == '__main__':
    from benchmark.optimization.runtime import configure_environment
    configure_environment()
import hashlib
import io
import lzma
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from .common import read_ply,read_json,write_json,save_frame,save_sequence
from .av1_planes import COLUMNS,morton_order,read_exact

MOTION = [0,1,2,55,56,57,58]
MISC = [3,4,5,51,52,53,54]
RAW_MOTION = [COLUMNS[i] for i in MOTION]


def immutable_hash(table):
    return hashlib.sha256(np.ascontiguousarray(table[:,6:58]).tobytes()).hexdigest()


def quantize(values,lo,hi):
    return np.rint(np.clip((values.astype(np.float64)-lo)/np.where(hi>lo,hi-lo,1),0,1)*65535).astype(np.uint16)


def dequantize(codes,lo,hi):
    return (codes.astype(np.float64)/65535*(hi-lo)+lo).astype(np.float32)


def analyze(paths,max_gop):
    groups=[]
    group=None
    lo=np.full(59,np.inf)
    hi=-lo.copy()
    for i,path in enumerate(paths):
        table=read_ply(path)
        values=table[:,COLUMNS]
        lo=np.minimum(lo,values.min(axis=0))
        hi=np.maximum(hi,values.max(axis=0))
        digest=immutable_hash(table)
        if group is None or len(table)!=len(group['base']) or digest!=group['immutable_hash'] or i-group['start']>=max_gop:
            if group is not None:
                group['end']=i
                groups.append(group)
            group={'start':i,'base':table,'dynamic':np.zeros(len(table),bool),'immutable_hash':digest}
        base=group['base']
        group['dynamic'] |= np.any(table[:,:3]!=base[:,:3],axis=1)
        group['dynamic'] |= np.any(table[:,58:62]!=base[:,58:62],axis=1)
    group['end']=len(paths)
    groups.append(group)
    return groups,lo,hi


def zigzag(current,predictor):
    difference=(current.astype(np.int32)-predictor.astype(np.int32)).astype(np.int16).astype(np.int32)
    return ((difference<<1)^(difference>>15)).astype(np.uint16)


def unzigzag(codes,predictor):
    delta=(codes.astype(np.int32)>>1)^-(codes.astype(np.int32)&1)
    return ((predictor.astype(np.int32)+delta)&65535).astype(np.uint16)


def predict(mode,previous,previous2,reference):
    if mode=='previous':
        return previous
    if mode=='reference':
        return reference
    if mode=='linear':
        return ((2*previous.astype(np.int32)-previous2.astype(np.int32))&65535).astype(np.uint16)
    raise ValueError(f'Unsupported prediction mode: {mode}')


def pack(values,side):
    planes=np.zeros((16,side*side),np.uint8)
    planes[:14:2,:len(values)]=(values&255).T
    planes[1:14:2,:len(values)]=(values>>8).T
    return planes.reshape(4,4,side,side).transpose(0,2,1,3).reshape(4*side,4*side)


def unpack(atlas,count,side):
    planes=atlas.reshape(4,side,4,side).transpose(0,2,1,3).reshape(16,side*side)
    return planes[:14:2,:count].T.astype(np.uint16)|(planes[1:14:2,:count].T.astype(np.uint16)<<8)


def encode(paths,timestamps,folder,ffmpeg,threads=1,predictor='previous',max_gop=30,planar_sidecar=True,sidecar_codec='lzma'):
    if threads!=1 or predictor not in ['previous','reference','linear'] or sidecar_codec not in ['lzma','deflate'] or max_gop<1 or len(paths)!=len(timestamps) or not paths:
        raise ValueError('Require one CPU thread, aligned sequence, valid predictor and positive GOP')
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    phases={}
    start=time.perf_counter()
    groups,lo,hi=analyze(paths,max_gop)
    phases['read_analyze_ranges_and_gop_correspondence']=time.perf_counter()-start
    side=max(16,math.ceil(math.sqrt(max(int(g['dynamic'].sum()) for g in groups))))
    config={'format':'morton-av1-gop','version':5,'columns':COLUMNS,'quantization_bits':16,
            'mins':lo.tolist(),'maxs':hi.tolist(),'timestamps':timestamps,'frame_count':len(paths),
            'tile_side':side,'width':4*side,'height':4*side,'video':'motion.mkv',
            'sidecar':'keyframes.npz.xz' if sidecar_codec=='lzma' else 'keyframes.npz','sidecar_codec':sidecar_codec,
            'predictor':predictor,'max_gop':max_gop,'planar_sidecar':bool(planar_sidecar),
            'motion_code_indices':MOTION,'misc_code_indices':MISC,'gops':[],
            'output_order':'original input PLY row order','preserves_input_row_order':True,
            'preserves_v1_quantizer':True,'codec':'libaom-av1','crf':0,'video_lossless':True,'threads':1,
            'atlas_sha256':[],'dynamic_quantized_sha256':[]}
    arrays={}
    runtime=[]
    start=time.perf_counter()
    for index,group in enumerate(groups):
        base=group['base']
        static=np.flatnonzero(~group['dynamic'])
        dynamic=np.flatnonzero(group['dynamic'])
        static=static[morton_order(base[static,:3],lo[:3],hi[:3])]
        dynamic=dynamic[morton_order(base[dynamic,:3],lo[:3],hi[:3])]
        order=np.concatenate((static,dynamic))
        codes=quantize(base[np.ix_(order,COLUMNS)],lo,hi)
        book,labels=np.unique(codes[:,6:51],axis=0,return_inverse=True)
        prefix=f'gop_{index}_'
        def store(name,value):
            arrays[prefix+name]=np.ascontiguousarray(value.T if planar_sidecar and value.ndim==2 else value)
        store('order',order.astype(np.uint32 if len(base)<=np.iinfo(np.uint32).max else np.uint64))
        store('misc',codes[:,MISC])
        store('sh_palette',book)
        store('sh_labels',labels.astype(np.uint16 if len(book)<=65536 else np.uint32))
        store('static_motion',codes[:len(static),MOTION])
        store('motion_reference',codes[len(static):,MOTION])
        config['gops'].append({'gop':index,'start_frame':group['start'],'frame_count':group['end']-group['start'],
                              'point_count':len(base),'static_count':len(static),'dynamic_count':len(dynamic),
                              'sh_unique_vectors':len(book),'immutable_sha256':group['immutable_hash']})
        runtime.append({'order':order,'static_source_indices':static,'dynamic_source_indices':dynamic,
                        'static_raw_motion':base[np.ix_(static,RAW_MOTION)].copy(),
                        'reference':codes[len(static):,MOTION]})
        del group['base']
    if sidecar_codec=='lzma':
        payload=io.BytesIO()
        np.savez(payload,**arrays)
        compressed=lzma.compress(payload.getvalue(),filters=[{'id':lzma.FILTER_LZMA2,'preset':6,'dict_size':32*1024*1024}])
        (folder/config['sidecar']).write_bytes(compressed)
    else:
        np.savez_compressed(folder/config['sidecar'],**arrays)
    del arrays
    config['sidecar_sha256']=hashlib.sha256((folder/config['sidecar']).read_bytes()).hexdigest()
    phases['quantize_keyframes_dictionary_and_sidecar']=time.perf_counter()-start
    command=[str(ffmpeg),'-v','error','-y','-filter_threads','1','-threads','1','-f','rawvideo','-pix_fmt','gray',
             '-s:v',f'{4*side}x{4*side}','-r','30','-i','pipe:0','-an','-c:v','libaom-av1','-cpu-used','8',
             '-crf','0','-b:v','0','-pix_fmt','gray','-row-mt','0','-threads','1','-g',str(max_gop),
             '-force_key_frames',','.join(str(g['start_frame']/30) for g in config['gops']),str(folder/'motion.mkv')]
    config['ffmpeg_command']=command
    start=time.perf_counter()
    with (folder/'encode.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for info,cache in zip(config['gops'],runtime):
                reference=cache['reference']
                previous=reference
                previous2=reference
                for frame in range(info['start_frame'],info['start_frame']+info['frame_count']):
                    table=read_ply(paths[frame])
                    if len(table)!=info['point_count'] or immutable_hash(table)!=info['immutable_sha256']:
                        raise ValueError(f'GOP immutable fields changed after analysis: {frame}')
                    if not np.array_equal(table[np.ix_(cache['static_source_indices'],RAW_MOTION)],cache['static_raw_motion']):
                        raise ValueError(f'Static motion changed after analysis: {frame}')
                    motion=quantize(table[np.ix_(cache['dynamic_source_indices'],RAW_MOTION)],lo[MOTION],hi[MOTION])
                    config['dynamic_quantized_sha256'].append(hashlib.sha256(motion.astype('<u2').tobytes()).hexdigest())
                    prediction=predict(predictor,previous,previous2,reference)
                    data=pack(zigzag(motion,prediction),side).tobytes()
                    config['atlas_sha256'].append(hashlib.sha256(data).hexdigest())
                    process.stdin.write(data)
                    previous2,previous=previous,motion
                print(f'AV1 v4 GOP {info["gop"]+1}/{len(config["gops"])}: {info["dynamic_count"]} dynamic, {info["sh_unique_vectors"]} exact SH vectors',flush=True)
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
    phases['validate_immutable_quantize_dynamic_and_av1']=time.perf_counter()-start
    config['encode_phase_seconds']=phases
    write_json(folder/'manifest.json',config)
    return {'sh_degree':3,'algorithm':'verified I/P GOP; exact SH dictionary; dynamic AV1 deltas; static decode cache; source row restoration',
            'format_version':5,'preserves_input_row_order':True,
            'preserves_v1_quantizer':True,'predictor':predictor,'phase_seconds':phases,'gop_count':len(groups),
            'sidecar_bytes':(folder/config['sidecar']).stat().st_size,'sidecar_codec':sidecar_codec,
            'video_bytes':(folder/'motion.mkv').stat().st_size}


def iter_tables(folder,ffmpeg,reuse_buffer=False):
    folder=Path(folder)
    config=read_json(folder/'manifest.json')
    if config.get('format')!='morton-av1-gop' or config.get('version') not in [4,5]:
        raise ValueError('Unsupported GOP format/version')
    sidecar=(folder/config['sidecar']).read_bytes()
    if hashlib.sha256(sidecar).hexdigest()!=config['sidecar_sha256']:
        raise ValueError('Keyframe sidecar checksum mismatch')
    if config.get('sidecar_codec','deflate')=='lzma':
        sidecar=lzma.decompress(sidecar)
    lo,hi=np.asarray(config['mins'],np.float64),np.asarray(config['maxs'],np.float64)
    command=[str(ffmpeg),'-v','error','-nostdin','-filter_threads','1','-threads','1','-i',str(folder/config['video']),
             '-f','rawvideo','-pix_fmt','gray','-threads','1','pipe:1']
    with tempfile.TemporaryFile() as log,np.load(io.BytesIO(sidecar),allow_pickle=False) as arrays:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for info in config['gops']:
                prefix=f'gop_{info["gop"]}_'
                def load(name):
                    value=arrays[prefix+name]
                    return value.T if config['planar_sidecar'] and value.ndim==2 else value
                codes=np.zeros((info['point_count'],59),np.uint16)
                book=load('sh_palette')
                labels=load('sh_labels')
                if len(book)!=info['sh_unique_vectors'] or (labels>=len(book)).any():
                    raise ValueError('Invalid SH codebook/index')
                static=info['static_count']
                codes[:,MISC]=load('misc')
                codes[:,6:51]=book[labels]
                codes[:static,MOTION]=load('static_motion')
                reference=load('motion_reference')
                codes[static:,MOTION]=reference
                template=np.zeros((len(codes),62),np.float32)
                template[:,COLUMNS]=dequantize(codes,lo,hi)
                del codes
                if config['version']==5:
                    order=load('order')
                    if (order.shape!=(info['point_count'],) or not np.issubdtype(order.dtype,np.integer)
                            or not np.array_equal(np.sort(order),np.arange(info['point_count']))):
                        raise ValueError('Invalid source-order permutation')
                    restored=np.empty_like(template)
                    restored[order]=template
                    template=restored
                    dynamic_ids=order[static:]
                else:
                    dynamic_ids=np.arange(static,len(template))
                float_index=np.ix_(dynamic_ids,RAW_MOTION)
                previous=reference
                previous2=reference
                for frame in range(info['start_frame'],info['start_frame']+info['frame_count']):
                    data=read_exact(process.stdout,config['width']*config['height'])
                    if hashlib.sha256(data).hexdigest()!=config['atlas_sha256'][frame]:
                        raise ValueError(f'AV1 atlas checksum mismatch: {frame}')
                    atlas=np.frombuffer(data,np.uint8).reshape(config['height'],config['width'])
                    residual=unpack(atlas,info['dynamic_count'],config['tile_side'])
                    prediction=predict(config['predictor'],previous,previous2,reference)
                    motion=unzigzag(residual,prediction)
                    if hashlib.sha256(motion.astype('<u2').tobytes()).hexdigest()!=config['dynamic_quantized_sha256'][frame]:
                        raise ValueError(f'Dynamic quantized attribute mismatch: {frame}')
                    # The PLY writer opts into reuse; public iteration owns each frame by default.
                    template[float_index]=dequantize(motion,lo[MOTION],hi[MOTION])
                    previous2,previous=previous,motion
                    yield frame,config['timestamps'][frame],template if reuse_buffer else template.copy()
            if process.stdout.read(1):
                raise ValueError('Unexpected extra AV1 frames')
            if process.wait():
                log.seek(0)
                raise RuntimeError(log.read().decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def decode(folder,output,ffmpeg,threads=1):
    if threads!=1:
        raise ValueError('GOP experiments require one CPU thread')
    timestamps=[]
    for i,timestamp,table in iter_tables(folder,ffmpeg,reuse_buffer=True):
        save_frame(output,i,table)
        timestamps.append(timestamp)
    save_sequence(output,timestamps)
    config=read_json(Path(folder)/'manifest.json')
    restores_rows=config['version']==5
    return {'frame_count':len(timestamps),'all_atlas_checksums_match':True,'all_dynamic_quantized_checksums_match':True,
            'static_attributes_cached_per_gop':True,'preserves_input_row_order':restores_rows,
            'output_order':'original input PLY row order' if restores_rows else 'legacy GOP static/dynamic Morton'}


def main():
    import argparse
    import json
    from .common import ROOT,sequence_paths
    from benchmark.optimization.runtime import pin_cpu,audit_ffmpeg
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['encode','decode'])
    p.add_argument('--sequence',type=Path)
    p.add_argument('--encoded',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--predictor',choices=['previous','reference','linear'],default='previous')
    p.add_argument('--max-gop',type=int,default=30)
    p.add_argument('--point-major-sidecar',action='store_true')
    p.add_argument('--sidecar-codec',choices=['lzma','deflate'],default='lzma')
    args=p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    _,mask=pin_cpu()
    audit_ffmpeg(mask,[])
    with threadpool_limits(limits=1):
        if args.operation=='encode':
            if not args.sequence:
                p.error('--sequence is required')
            paths,timestamps=sequence_paths(args.sequence)
            details=encode(paths,timestamps,args.output,args.ffmpeg,predictor=args.predictor,max_gop=args.max_gop,planar_sidecar=not args.point_major_sidecar,sidecar_codec=args.sidecar_codec)
        else:
            if not args.encoded:
                p.error('--encoded is required')
            details=decode(args.encoded,args.output,args.ffmpeg)
    print(json.dumps(details,indent=2),flush=True)


if __name__=='__main__':
    main()
