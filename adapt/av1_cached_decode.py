"""Decode existing v2 NPZ/XZ streams with cached static float attributes.

The encoder is an explicitly frozen reference supplied to encode(), unchanged.
Point order, v1 quantizer and every atlas/full-quantized checksum are preserved.
The decoder needs only encoded files and supports both earlier NPZ and newer XZ.
"""
from __future__ import annotations
if __name__=='__main__':
    from benchmark.optimization.runtime import configure_environment
    configure_environment()
import hashlib
import importlib.util
import io
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .common import read_json,save_frame,save_sequence
from .av1_planes import COLUMNS,read_exact

MOTION=[0,1,2,55,56,57,58]
MISC=[3,4,5,51,52,53,54]
RAW_MOTION=[COLUMNS[i] for i in MOTION]


def load_encoder(path):
    spec=importlib.util.spec_from_file_location('adapt._frozen_temporal_encoder',Path(path))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def encode(paths,timestamps,folder,ffmpeg,threads=1,encoder_source=None,**options):
    if threads!=1 or encoder_source is None:
        raise ValueError('One CPU thread and an explicit frozen encoder_source are required')
    encoder=load_encoder(encoder_source)
    details=encoder.encode(paths,timestamps,folder,ffmpeg,threads=1,**options)
    return {**details,'encoder_source_sha256':hashlib.sha256(Path(encoder_source).read_bytes()).hexdigest(),
            'decoder_change':'cache static float attributes only; encoded format and encoder unchanged'}


def dequantize(codes,lo,hi):
    return (codes.astype(np.float64)/65535*(hi-lo)+lo).astype(np.float32)


def iter_tables(folder,ffmpeg,reuse_buffer=False):
    folder=Path(folder)
    config=read_json(folder/'manifest.json')
    if config.get('format')!='morton-av1-temporal' or config.get('version')!=2 or config['columns']!=COLUMNS:
        raise ValueError('Unsupported legacy temporal format/version')
    payload=(folder/config['sidecar']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=config['sidecar_sha256']:
        raise ValueError('Keyframe sidecar checksum mismatch')
    if payload.startswith(b'\xfd7zXZ\x00'):
        payload=lzma.decompress(payload)
    lo,hi=np.asarray(config['mins'],np.float64),np.asarray(config['maxs'],np.float64)
    command=[str(ffmpeg),'-v','error','-nostdin','-filter_threads','1','-threads','1','-i',str(folder/config['video']),
             '-f','rawvideo','-pix_fmt','gray','-threads','1','pipe:1']
    with tempfile.TemporaryFile() as log,np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for info in config['gops']:
                prefix=f'gop_{info["gop"]}_'
                n=info['point_count']
                static=info['static_count']
                order=arrays[prefix+'order']
                if len(order)!=n or not np.array_equal(np.sort(order),np.arange(n)):
                    raise ValueError('Invalid source-order permutation')
                ordered_codes=np.zeros((n,59),np.uint16)
                ordered_codes[:,MISC]=arrays[prefix+'misc']
                ordered_codes[:,6:51]=arrays[prefix+'sh_palette'][arrays[prefix+'sh_labels']]
                ordered_codes[:static,MOTION]=arrays[prefix+'static_motion']
                reference=arrays[prefix+'motion_reference']
                ordered_codes[static:,MOTION]=reference
                codes=np.empty_like(ordered_codes)
                codes[order]=ordered_codes
                del ordered_codes
                template=np.zeros((n,62),np.float32)
                template[:,COLUMNS]=dequantize(codes,lo,hi)
                dynamic_ids=order[static:]
                code_index=np.ix_(dynamic_ids,MOTION)
                float_index=np.ix_(dynamic_ids,RAW_MOTION)
                previous=reference
                for frame in range(info['start_frame'],info['start_frame']+info['frame_count']):
                    data=read_exact(process.stdout,config['width']*config['height'])
                    if hashlib.sha256(data).hexdigest()!=config['atlas_sha256'][frame]:
                        raise ValueError(f'AV1 atlas checksum mismatch: {frame}')
                    side=config['tile_side']
                    planes=np.frombuffer(data,np.uint8).reshape(4,side,4,side).transpose(0,2,1,3).reshape(16,side*side)
                    count=info['dynamic_count']
                    residual=planes[:14:2,:count].T.astype(np.uint16)|(planes[1:14:2,:count].T.astype(np.uint16)<<8)
                    prediction=previous if config['predictor']=='previous' else reference
                    delta=(residual.astype(np.int32)>>1)^-(residual.astype(np.int32)&1)
                    motion=((prediction.astype(np.int32)+delta)&65535).astype(np.uint16)
                    codes[code_index]=motion
                    if hashlib.sha256(codes.astype('<u2',copy=False).tobytes()).hexdigest()!=config['quantized_frame_sha256'][frame]:
                        raise ValueError(f'Quantized attribute checksum mismatch: {frame}')
                    template[float_index]=dequantize(motion,lo[MOTION],hi[MOTION])
                    previous=motion
                    yield frame,config['timestamps'][frame],template if reuse_buffer else template.copy()
            if process.stdout.read(1):
                raise ValueError('Unexpected extra video frames')
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
        raise ValueError('Cached decoder requires one CPU thread')
    timestamps=[]
    for frame,timestamp,table in iter_tables(folder,ffmpeg,reuse_buffer=True):
        save_frame(output,frame,table)
        timestamps.append(timestamp)
    save_sequence(output,timestamps)
    return {'frame_count':len(timestamps),'all_atlas_checksums_match':True,'all_quantized_attribute_checksums_match':True,
            'static_float_attributes_cached_per_gop':True,'preserves_existing_v2_source_order':True,
            'supports_npz_and_xz_sidecars':True}


def main():
    import argparse
    from .common import ROOT
    from benchmark.optimization.runtime import pin_cpu,audit_ffmpeg
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    args=p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    _,mask=pin_cpu()
    audit_ffmpeg(mask,[])
    with threadpool_limits(limits=1):
        print(decode(args.encoded,args.output,args.ffmpeg))


if __name__=='__main__':
    main()
