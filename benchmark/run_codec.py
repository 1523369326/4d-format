"""Run one isolated codec encode/decode measurement; keep GPU rendering separate."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method',choices=['texture_h264','gsd_v1','gsd_v2','tsog','morton_av1','morton_av1_opt','smv'])
    p.add_argument('--sequence',type=Path,default=ROOT/'benchmark/input_first60.json')
    p.add_argument('--results',type=Path,default=ROOT/'benchmark/results')
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--threads',type=int,choices=[1,2,4],default=1)
    args=p.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]=str(args.threads)
    from threadpoolctl import threadpool_limits
    from adapt.common import sequence_paths,write_json,read_json
    mapping={'texture_h264':'texture_video','gsd_v1':'gsd','gsd_v2':'gsd','tsog':'tsog_adapter',
             'morton_av1':'av1_planes','morton_av1_opt':'av1_temporal','smv':'smv_adapter'}
    codec=importlib.import_module('av1_codec' if args.method=='morton_av1_opt' else 'adapt.'+mapping[args.method])
    paths,timestamps=sequence_paths(args.sequence)
    folder=args.results/args.method
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'encoded').exists() or (folder/'decoded').exists():
        raise FileExistsError(f'Use a fresh results directory: {folder}')
    kwargs={'version':1 if args.method=='gsd_v1' else 2} if args.method.startswith('gsd') else {}
    report={'method':args.method,'frames':len(paths),'input_bytes':sum(p.stat().st_size for p in paths),
        'python':sys.executable,'cpu_math_threads':args.threads,'compression_hardware':'CPU only',
        'timing_scope':'end-to-end encode, end-to-end decode to standard binary PLY; excludes renderer, outer process startup and adapter module import; lazy imports inside codec functions are included',
        'benchmark_repetitions':1,'cache_policy':'OS page cache not flushed; methods run serially'}
    if args.sequence.resolve()==(ROOT/'benchmark/input_first60.json').resolve():
        report['dataset_sha256']=read_json(ROOT/'benchmark/environment.json')['dataset_sha256']
    write_json(folder/'timing.json',{**report,'status':'encoding'})
    with threadpool_limits(limits=args.threads):
        start=time.perf_counter()
        report['encode_details']=codec.encode(paths,timestamps,folder/'encoded',args.ffmpeg,threads=args.threads,**kwargs)
        report['encode_seconds']=time.perf_counter()-start
        report['encoded_bytes']=sum(p.stat().st_size for p in (folder/'encoded').rglob('*') if p.is_file() and p.name!='encode.log')
        report['compression_ratio']=report['input_bytes']/report['encoded_bytes']
        write_json(folder/'timing.json',{**report,'status':'decoding'})
        start=time.perf_counter()
        report['decode_details']=codec.decode(folder/'encoded',folder/'decoded',args.ffmpeg,threads=args.threads)
        report['decode_seconds']=time.perf_counter()-start
    report['status']='completed'
    report['encode_ms_per_frame']=1000*report['encode_seconds']/len(paths)
    report['decode_ms_per_frame']=1000*report['decode_seconds']/len(paths)
    write_json(folder/'timing.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
