"""Measure exactly one immutable-source codec on one CPU; CUDA renderer is separate."""
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True

from runtime import configure_environment, pin_cpu, affinity, audit_ffmpeg
configure_environment()  # Must precede numpy / BLAS imports.

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=['morton_av1', 'tsog', 'shdict', 'temporal', 'predictive'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', type=Path, default=ROOT/'benchmark/input_first60.json')
    p.add_argument('--ffmpeg', type=Path, default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--candidate', type=Path)
    p.add_argument('--options', default='{}')
    p.add_argument('--cpu', type=int)
    args = p.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cpu, mask = pin_cpu(args.cpu)
    sys.path.insert(0, str(source))
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from adapt.common import sequence_paths, read_ply, write_json, read_json
    mapping = {'morton_av1': 'av1_planes', 'tsog': 'tsog_adapter', 'shdict': 'av1_shdict', 'temporal': 'av1_temporal', 'predictive': 'av1_predictive'}
    if args.candidate:
        import shutil
        saved = out/'candidate.py'
        shutil.copy2(args.candidate, saved)
        spec = importlib.util.spec_from_file_location('adapt.' + mapping[args.method], saved)
        codec = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = codec
        spec.loader.exec_module(codec)
    else:
        codec = importlib.import_module('adapt.' + mapping[args.method])
    paths, timestamps = sequence_paths(args.sequence)
    inputs = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path), 'timestamp': t}
              for path, t in zip(paths, timestamps)]
    events = []
    audit_ffmpeg(mask, events)
    decoding_phase = [False]
    input_names = {str(path.resolve()).casefold() for path in paths}
    def deny_source_reads_during_decode(event, event_args):
        if decoding_phase[0] and event == 'open' and isinstance(event_args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(event_args[0])
            if str(Path(name).resolve()).casefold() in input_names:
                raise RuntimeError('Decoder attempted to read original PLY input')
    sys.addaudithook(deny_source_reads_during_decode)
    stage_times = {}
    # Inclusive times: e.g. atlas includes Morton. These are diagnostic, not summed.
    for name in ['read_ply', 'atlas_from_table', 'table_from_atlas', 'save_frame', 'morton_order']:
        if hasattr(codec, name):
            fn = getattr(codec, name)
            def wrap(*a, _fn=fn, _name=name, **kw):
                start = time.perf_counter()
                try:
                    return _fn(*a, **kw)
                finally:
                    stage_times[_name] = stage_times.get(_name, 0.0) + time.perf_counter() - start
            setattr(codec, name, wrap)
    source_manifest = read_json(source.parent/'manifest.json')
    for item in source_manifest['sources']:
        if sha(source/item['path']) != item['sha256']:
            raise ValueError(f'Frozen source changed: {item["path"]}')
    preserved_inputs = {item['path']: item for item in source_manifest['inputs']}
    for item in inputs:
        if item != preserved_inputs.get(item['path']):
            raise ValueError(f'Input is not identical to preserved baseline: {item["path"]}')
    report = {'method': args.method, 'status': 'initializing', 'frames': len(paths),
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_root': str(source), 'source_tree_sha256': source_manifest['source_tree_sha256'],
              'codec_sha256': sha(codec.__file__), 'options': json.loads(args.options),
              'runner_sha256': sha(__file__), 'runtime_sha256': sha(Path(__file__).with_name('runtime.py')),
              'sequence': str(args.sequence.resolve()), 'input_fingerprint_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
              'input_bytes': sum(item['bytes'] for item in inputs), 'inputs': inputs,
              'python': sys.executable, 'platform': platform.platform(), 'cpu': platform.processor(),
              'logical_cpu': cpu, 'affinity_mask': mask, 'cpu_math_threads': 1,
              'compression_hardware': 'CPU only; CUDA_VISIBLE_DEVICES=-1',
              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'Pillow', 'plyfile', 'threadpoolctl']},
              'ffmpeg_sha256': sha(args.ffmpeg),
              'timing_scope': 'read PLY + all required preprocessing/codebooks + encode/write; decode/read/reconstruct/write binary PLY; excludes process/module startup and post-timing validation and CUDA renderer',
              'cache_policy': 'OS cache not flushed; inputs hashed immediately before each case; serial independent processes',
              'child_process_audit': events}
    def state(status):
        report['status'] = status
        write_json(out/'timing.json', report)
    try:
        with threadpool_limits(limits=1):
            report['threadpools_before'] = threadpool_info()
            if any(pool['num_threads'] != 1 for pool in threadpool_info()):
                raise RuntimeError('Numerical library thread limit not enforced')
            state('encoding')
            start = time.perf_counter()
            report['encode_details'] = codec.encode(paths, timestamps, out/'encoded', args.ffmpeg, threads=1, **report['options'])
            report['encode_seconds'] = time.perf_counter() - start
            report['encode_inclusive_stage_seconds'] = dict(stage_times)
            stage_times.clear()
            report['encoded_bytes'] = sum(f.stat().st_size for f in (out/'encoded').rglob('*') if f.is_file() and f.name != 'encode.log')
            report['compression_ratio'] = report['input_bytes'] / report['encoded_bytes']
            state('decoding')
            start = time.perf_counter()
            decoding_phase[0] = True
            try:
                report['decode_details'] = codec.decode(out/'encoded', out/'decoded', args.ffmpeg, threads=1)
            finally:
                decoding_phase[0] = False
            report['decode_seconds'] = time.perf_counter() - start
            report['decoder_source_read_guard'] = 'original PLY reads denied during measured decode; child commands restricted to audited FFmpeg'
            report['decode_inclusive_stage_seconds'] = dict(stage_times)
            report['threadpools_after'] = threadpool_info()
            if affinity() != mask or any(pool['num_threads'] != 1 for pool in threadpool_info()):
                raise RuntimeError('Execution policy changed during codec')
            state('validating')
            manifest = read_json(out/'decoded/sequence.json')
            if [f['timestamp'] for f in manifest['frames']] != timestamps:
                raise ValueError('Decoded timeline mismatch')
            decoded = []
            for i, path in enumerate(paths):
                target = out/'decoded'/f'frame_{i:06d}.ply'
                table = read_ply(target)
                count = len(read_ply(path))
                if table.shape != (count, 62) or not np.isfinite(table).all():
                    raise ValueError(f'Invalid decoded frame {i}')
                decoded.append({'frame': i, 'count': count, 'sha256': sha(target)})
            report['decoded_frames'] = decoded
            report['encode_ms_per_frame'] = 1000 * report['encode_seconds'] / len(paths)
            report['decode_ms_per_frame'] = 1000 * report['decode_seconds'] / len(paths)
            state('completed')
            print(json.dumps({k: report[k] for k in ['method', 'status', 'encoded_bytes', 'encode_seconds', 'decode_seconds', 'affinity_mask']}), flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        state('failed')
        raise


if __name__ == '__main__':
    main()

