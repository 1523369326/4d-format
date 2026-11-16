"""v3 exact SH dictionary ablation: v1 quantizer and Morton order, software AV1.

All 45 AC coefficients are reconstructed; there is no SH truncation or clustering.
The v1 module is kept intact so old streams retain their decoder.
"""
from __future__ import annotations
if __name__ == '__main__':
    from benchmark.optimization.runtime import configure_environment
    configure_environment()
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from .common import read_ply, read_json, write_json, save_frame, save_sequence
from .av1_planes import COLUMNS, morton_order, read_exact

SCALAR_INDICES = list(range(6)) + list(range(51, 59))
SCALAR_COLUMNS = [COLUMNS[i] for i in SCALAR_INDICES]


def quantize(values, lo, hi):
    return np.rint(np.clip((values.astype(np.float64)-lo)/np.where(hi>lo,hi-lo,1),0,1)*65535).astype(np.uint16)


def pack(scalars, labels, config):
    side = config['tile_side']
    n = len(scalars)
    planes = np.zeros((config['tile_rows']*config['tile_columns'], side*side), dtype=np.uint8)
    planes[:28:2,:n] = (scalars & 255).T
    planes[1:28:2,:n] = (scalars >> 8).T
    for byte in range(config['index_bytes']):
        planes[28+byte,:n] = ((labels.astype(np.uint32) >> (8*byte)) & 255)
    return planes.reshape(config['tile_rows'],config['tile_columns'],side,side).transpose(0,2,1,3).reshape(config['height'],config['width'])


def unpack(atlas, count, config):
    side = config['tile_side']
    planes = atlas.reshape(config['tile_rows'],side,config['tile_columns'],side).transpose(0,2,1,3).reshape(-1,side*side)
    scalars = planes[:28:2,:count].T.astype(np.uint16) | (planes[1:28:2,:count].T.astype(np.uint16) << 8)
    labels = np.zeros(count, dtype=np.uint32)
    for byte in range(config['index_bytes']):
        labels |= planes[28+byte,:count].astype(np.uint32) << (8*byte)
    return scalars, labels


def encode(paths, timestamps, folder, ffmpeg, threads=1, spatial_delta=False, tile_columns=6):
    if threads != 1:
        raise ValueError('SH dictionary experiments require one CPU thread')
    if len(paths) != len(timestamps) or not paths or tile_columns < 1:
        raise ValueError('Nonempty aligned sequence and positive tile columns required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    lo = np.full(59,np.inf)
    hi = -lo.copy()
    counts, raw_books, labels = [], [], []
    phases = {}
    start = time.perf_counter()
    for path in paths:
        table = read_ply(path)
        values = table[:,COLUMNS]
        lo = np.minimum(lo,values.min(axis=0))
        hi = np.maximum(hi,values.max(axis=0))
        book, inverse = np.unique(table[:,9:54],axis=0,return_inverse=True)
        counts.append(len(table))
        raw_books.append(book)
        labels.append(inverse.astype(np.uint32))
    phases['read_ranges_and_exact_raw_sh_unique'] = time.perf_counter()-start
    books = {}
    unique_counts = []
    start = time.perf_counter()
    for i, book in enumerate(raw_books):
        q = quantize(book,lo[6:51],hi[6:51])
        unique, mapping = np.unique(q,axis=0,return_inverse=True)
        labels[i] = mapping[labels[i]].astype(np.uint32)
        books[f'frame_{i:06d}'] = unique
        unique_counts.append(len(unique))
    del raw_books
    np.savez_compressed(folder/'sh_codebooks.npz',**books)
    phases['quantize_unique_books_and_write_sidecar'] = time.perf_counter()-start
    index_bytes = 2 if max(unique_counts) <= 65536 else 4
    side = math.ceil(math.sqrt(max(counts)))
    rows = math.ceil((28+index_bytes)/tile_columns)
    config = {'format':'morton-av1-shdict','version':3,'quantization_bits':16,
              'columns':COLUMNS,'mins':lo.tolist(),'maxs':hi.tolist(),'counts':counts,
              'timestamps':timestamps,'frame_count':len(paths),'tile_side':side,
              'tile_columns':tile_columns,'tile_rows':rows,'width':side*tile_columns,'height':side*rows,
              'scalar_code_indices':SCALAR_INDICES,'index_bytes':index_bytes,'sh_unique_counts':unique_counts,
              'spatial_delta':bool(spatial_delta),'video':'attributes.mkv','sidecar':'sh_codebooks.npz',
              'sidecar_sha256':hashlib.sha256((folder/'sh_codebooks.npz').read_bytes()).hexdigest(),
              'video_lossless':True,'codec':'libaom-av1','threads':1,'crf':0,'cpu_used':8,
              'preserves_v1_quantizer':True,'preserves_v1_morton_order':True,'atlas_sha256':[],
              'morton_bits_per_axis':21,'sh_dictionary':'per frame exact quantized AC vectors; no clustering'}
    command = [str(ffmpeg),'-v','error','-y','-filter_threads','1','-threads','1','-f','rawvideo',
               '-pix_fmt','gray','-s:v',f'{config["width"]}x{config["height"]}','-r','30','-i','pipe:0','-an',
               '-c:v','libaom-av1','-cpu-used','8','-crf','0','-b:v','0','-pix_fmt','gray',
               '-row-mt','0','-threads','1','-g','30',str(folder/'attributes.mkv')]
    config['ffmpeg_command'] = command
    start = time.perf_counter()
    with (folder/'encode.log').open('wb') as log:
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i,path in enumerate(paths):
                table = read_ply(path)
                order = morton_order(table[:,:3],lo[:3],hi[:3])
                scalars = quantize(table[np.ix_(order,SCALAR_COLUMNS)],lo[SCALAR_INDICES],hi[SCALAR_INDICES])
                if spatial_delta:
                    scalars[1:] = scalars[1:] - scalars[:-1]
                atlas = pack(scalars,labels[i][order],config)
                data = atlas.tobytes()
                config['atlas_sha256'].append(hashlib.sha256(data).hexdigest())
                process.stdin.write(data)
                if (i+1)%10 == 0:
                    print(f'SH dictionary AV1 input {i+1}/{len(paths)}',flush=True)
            process.stdin.close()
            result = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdin.close()
    if result:
        raise RuntimeError((folder/'encode.log').read_text(errors='replace'))
    phases['morton_scalar_quantization_pack_and_av1'] = time.perf_counter()-start
    config['encode_phase_seconds'] = phases
    write_json(folder/'manifest.json',config)
    return {'sh_degree':3,'algorithm':'v1 quantizer/Morton; exact per-frame SH dictionary; AV1 scalar/index byte planes',
            'preserves_v1_quantizer':True,'preserves_v1_morton_order':True,'index_bytes':index_bytes,
            'sh_unique_counts':unique_counts,'spatial_delta':bool(spatial_delta),'phase_seconds':phases,
            'sidecar_bytes':(folder/'sh_codebooks.npz').stat().st_size,'video_bytes':(folder/'attributes.mkv').stat().st_size}


def iter_tables(folder, ffmpeg):
    folder = Path(folder)
    config = read_json(folder/'manifest.json')
    if config.get('format') != 'morton-av1-shdict' or config.get('version') != 3:
        raise ValueError('Unsupported SH dictionary format/version')
    if hashlib.sha256((folder/config['sidecar']).read_bytes()).hexdigest() != config['sidecar_sha256']:
        raise ValueError('SH dictionary sidecar checksum mismatch')
    lo, hi = np.asarray(config['mins'],np.float64), np.asarray(config['maxs'],np.float64)
    command = [str(ffmpeg),'-v','error','-nostdin','-filter_threads','1','-threads','1','-i',str(folder/config['video']),
               '-f','rawvideo','-pix_fmt','gray','-threads','1','pipe:1']
    with tempfile.TemporaryFile() as log, np.load(folder/config['sidecar'],allow_pickle=False) as books:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for i,count in enumerate(config['counts']):
                data = read_exact(process.stdout,config['width']*config['height'])
                if hashlib.sha256(data).hexdigest() != config['atlas_sha256'][i]:
                    raise ValueError(f'AV1 atlas checksum mismatch: {i}')
                atlas = np.frombuffer(data,np.uint8).reshape(config['height'],config['width'])
                scalars, labels = unpack(atlas,count,config)
                if config['spatial_delta']:
                    scalars = np.cumsum(scalars,axis=0,dtype=np.uint64).astype(np.uint16)
                book = books[f'frame_{i:06d}']
                if len(book) != config['sh_unique_counts'][i] or book.shape[1:] != (45,) or (labels >= len(book)).any():
                    raise ValueError(f'Invalid SH dictionary/index: {i}')
                codes = np.empty((count,59),np.uint16)
                codes[:,SCALAR_INDICES] = scalars
                codes[:,6:51] = book[labels]
                table = np.zeros((count,62),np.float32)
                table[:,COLUMNS] = (codes.astype(np.float64)/65535*(hi-lo)+lo).astype(np.float32)
                yield i,config['timestamps'][i],table
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


def decode(folder, output, ffmpeg, threads=1):
    if threads != 1:
        raise ValueError('SH dictionary experiments require one CPU thread')
    timestamps = []
    for i,timestamp,table in iter_tables(folder,ffmpeg):
        save_frame(output,i,table)
        timestamps.append(timestamp)
    save_sequence(output,timestamps)
    return {'frame_count':len(timestamps),'all_atlas_checksums_match':True,'all_sidecar_checksums_match':True}


def main():
    import argparse
    from .common import ROOT, sequence_paths
    from benchmark.optimization.runtime import pin_cpu,audit_ffmpeg
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['encode','decode'])
    p.add_argument('--sequence',type=Path)
    p.add_argument('--encoded',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--spatial-delta',action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    _,mask = pin_cpu()
    audit_ffmpeg(mask,[])
    with threadpool_limits(limits=1):
        if args.operation == 'encode':
            if not args.sequence:
                p.error('--sequence is required')
            paths,timestamps = sequence_paths(args.sequence)
            encode(paths,timestamps,args.output,args.ffmpeg,spatial_delta=args.spatial_delta)
        else:
            if not args.encoded:
                p.error('--encoded is required')
            decode(args.encoded,args.output,args.ffmpeg)


if __name__ == '__main__':
    main()
