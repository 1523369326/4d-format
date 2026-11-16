"""Experimental PLY -> SMV attribute/motion encoder with an explicit timeline.

The NPZ and HEVC payloads follow the supplied website decoder. manifest.json adds
variable-duration, non-overlapping GOPs; the unmodified website ignores this timeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROPERTIES = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
              + [f'f_rest_{i}' for i in range(45)] + ['opacity']
              + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def read_ply(path):
    """Read scalar-property binary PLYs; normalize property order to standard 3DGS."""
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
             'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'int32': 'i4',
             'uint': 'u4', 'uint32': 'u4', 'short': 'i2', 'ushort': 'u2'}
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not PLY: {path}')
        fmt, n, fields, in_vertex = None, None, [], False
        for _ in range(2048):
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            words = line.decode('ascii').strip().split()
            if not words:
                continue
            if words[0] == 'format':
                fmt = words[1]
            elif words[0] == 'element':
                in_vertex = words[1] == 'vertex'
                if in_vertex:
                    n = int(words[2])
            elif words[0] == 'property' and in_vertex:
                if words[1] not in types or len(words) != 3:
                    raise ValueError('Unsupported vertex property (list properties are not supported)')
                fields.append((words[2], types[words[1]]))
            elif words[0] == 'end_header':
                break
        else:
            raise ValueError('PLY header too long')
        if fmt not in ('binary_little_endian', 'binary_big_endian') or not n:
            raise ValueError('Require a nonempty binary 3DGS PLY')
        endian = '<' if fmt == 'binary_little_endian' else '>'
        dtype = np.dtype([(name, endian + kind) for name, kind in fields])
        raw = np.fromfile(f, dtype=dtype, count=n)
        if len(raw) != n:
            raise ValueError('Truncated vertex data')
    out = np.zeros((n, 62), dtype=np.float32)
    for i, name in enumerate(PROPERTIES):
        if name in raw.dtype.names:
            out[:, i] = raw[name]
        elif name not in ('nx', 'ny', 'nz'):
            raise ValueError(f'Missing 3DGS property {name}: {path}')
    if not np.isfinite(out).all():
        raise ValueError(f'Nonfinite PLY data: {path}')
    if np.any(np.linalg.norm(out[:, 58:62], axis=1) == 0):
        raise ValueError('Zero-length quaternion')
    return out


def normalize(q):
    q = np.asarray(q, dtype=np.float64)
    lengths = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(lengths == 0, 1, lengths)


def canonical(q):
    q = normalize(q)
    # Use the largest component to choose a deterministic quaternion sign.
    biggest = np.take_along_axis(q, np.abs(q).argmax(axis=1)[:, None], axis=1)
    return q * np.where(biggest < 0, -1, 1)


def multiply(a, b):
    w,x,y,z = np.asarray(a).T
    u,v,s,t = np.asarray(b).T
    return np.stack((w*u-x*v-y*s-z*t, w*v+x*u+y*t-z*s,
                     w*s-x*t+y*u+z*v, w*t+x*s-y*v+z*u), axis=1)


def immutable_hash(table):
    return hashlib.sha256(np.ascontiguousarray(table[:, 6:58]).tobytes()).digest()


def unique_rows(values):
    return np.unique(np.ascontiguousarray(values, dtype=np.float32), axis=0, return_inverse=True)


def palette(values, threads=1):
    centers, _ = unique_rows(values)
    original_count = len(centers)
    if len(centers) > 1024:
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            centers = MiniBatchKMeans(n_clusters=1024, random_state=0, n_init=1,
                batch_size=4096, max_iter=40, reassignment_ratio=0).fit(centers).cluster_centers_
    if len(centers) == 0:
        centers = np.zeros((1,45), dtype=np.float32)
    centers = np.pad(centers, ((0, 1024-len(centers)), (0,0)), mode='edge')
    return centers.astype(np.float32), original_count


def palette_indices(values, centers, threads=1):
    from scipy.spatial import cKDTree
    unique, inverse = unique_rows(values)
    if not len(unique):
        return np.empty(0, dtype=np.uint16)
    _, labels = cKDTree(centers).query(unique, workers=threads)
    return labels[inverse].astype(np.uint16)


def quantize(values, bits, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mn = values.min(axis=axis, keepdims=True).astype(np.float32)
    mx = values.max(axis=axis, keepdims=True).astype(np.float32)
    span = (mx-mn).astype(np.float32)
    divisor = np.where(span > 0, span, 1)
    q = np.rint(np.clip((values-mn)/divisor, 0, 1) * (2**bits-1))
    return q.astype(np.uint8 if bits == 8 else np.uint16), mn, span


def load_sequence(sequence):
    sequence = Path(sequence).resolve()
    data = json.loads(sequence.read_text(encoding='utf-8-sig'))
    frames = data['frames'] if isinstance(data, dict) else data
    if not frames:
        raise ValueError('Empty sequence')
    result = []
    for i, item in enumerate(frames):
        path = (sequence.parent / item['file']).resolve()
        timestamp = float(item['timestamp'])
        if not math.isfinite(timestamp) or (i and timestamp <= result[-1]['timestamp']):
            raise ValueError('Timestamps must be finite and strictly increasing')
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({'path': path, 'timestamp': timestamp})
    fps = 1 / float(np.median(np.diff([f['timestamp'] for f in result]))) if len(result)>1 else float(data.get('fps',30))
    return result, fps


def analyze(frames, max_gop):
    segments = []
    first, current, previous_hash = None, None, None
    for i, frame in enumerate(frames):
        t = read_ply(frame['path'])
        digest = immutable_hash(t)
        if current is None or digest != previous_hash or len(t) != len(first) or i-current['start'] >= max_gop:
            if current is not None:
                current['end'] = i
                segments.append(current)
            first = t
            current = {'start': i, 'base': first, 'dynamic': np.zeros(len(t), dtype=bool)}
            previous_hash = digest
        current['dynamic'] |= np.any(t[:,:3] != first[:,:3], axis=1)
        current['dynamic'] |= np.any(np.abs(canonical(t[:,58:62])-canonical(first[:,58:62])) > 1e-12, axis=1)
        if (i+1)%50 == 0:
            print(f'Analyzed {i+1}/{len(frames)} frames', flush=True)
    current['end'] = len(frames)
    segments.append(current)
    return segments


def table_attrs(table):
    # Bounds protect inverse activations from overflow on hostile/invalid inputs.
    if np.any(table[:,55:58] > 11):
        raise ValueError('Scale too large for this SMV encoder')
    return {'position': table[:,:3], 'rotation': canonical(table[:,58:62]).astype(np.float32),
            'scaling': np.exp(table[:,55:58].astype(np.float64)).astype(np.float32),
            'opacity': (1/(1+np.exp(-np.clip(table[:,54:55].astype(np.float64),-80,80)))).astype(np.float32),
            'base_color': table[:,6:9].reshape(-1,1,3),
            'rest': table[:,9:54].reshape(-1,3,15).transpose(0,2,1).reshape(-1,45)}


def encode_attrs(target, g, table):
    p = f'gop_{g}_'
    attrs = table_attrs(table)
    with np.errstate(over='ignore'):
        positions = attrs['position'].astype(np.float16)
    if not np.isfinite(positions).all():
        raise ValueError('Position exceeds float16 range')
    target[p+'position'] = positions
    for name, bitkey, bits, axis in [('rotation','rotation_n_bit',8,0),
            ('scaling','scaling_n_bit',16,None), ('opacity','opacity_n_bit',8,None),
            ('base_color','color_n_bit',8,0)]:
        q,mn,span = quantize(attrs[name], bits, axis)
        target[p+name],target[p+name+'_min'],target[p+name+'_range'] = q,mn,span
        target[p+bitkey] = np.array(bits, dtype=np.int64)
    q = target[p+'rotation'].astype(np.float64)/255 * target[p+'rotation_range'] + target[p+'rotation_min']
    return positions.astype(np.float32), normalize(q).astype(np.float32)


def encode_book(target, g, kind, centers):
    q,mn,span = quantize(centers,8,0)
    p = f'gop_{g}_sh_codebook_{kind}'
    target[p], target[p+'_min'], target[p+'_range'] = q,mn,span


def video_encode(rgb, side, fps, path, ffmpeg, preset, threads=1):
    pools='none' if threads==1 else str(threads)
    command = [str(ffmpeg),'-v','error','-y','-filter_threads','1','-threads',str(threads),'-f','rawvideo','-pix_fmt','rgb24',
        '-s:v',f'{side}x{side}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx265',
        '-preset',preset,'-pix_fmt','gbrp','-color_range','pc',
        '-threads',str(threads),'-x265-params',f'lossless=1:log-level=error:pools={pools}:frame-threads=1:wpp=0:lookahead-threads=1',str(path)]
    r = subprocess.run(command,input=np.ascontiguousarray(rgb).tobytes(),
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace')[-4000:])


def encode(sequence, output, ffmpeg, max_gop=30, position_bits=16, preset='fast', threads=1):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    frames, fps = load_sequence(sequence)
    segments = analyze(frames,max_gop)
    # Share only stationary rows that are identical at the same row index in every GOP.
    common = np.zeros(len(segments[0]['base']),dtype=bool)
    common[:min(len(s['base']) for s in segments)] = True
    for s in segments:
        n=min(len(common),len(s['base']))
        common[:n] &= ~s['dynamic'][:n]
        common[:n] &= np.all(s['base'][:n] == segments[0]['base'][:n],axis=1)
    shared = np.flatnonzero(common)
    static_unique=[]
    for s in segments:
        rest=table_attrs(s['base'][~s['dynamic']])['rest']
        static_unique.append(unique_rows(rest)[0])
    static_centers,static_unique_count=palette(np.concatenate(static_unique),threads)
    arrays={}
    manifest={'format':'smv-explicit-timeline','version':1,
        'payload_format':'SMV website multi-GOP NPZ + HEVC gbrp motion streams',
        'frame_count':len(frames),'fps':fps,'timestamps':[f['timestamp'] for f in frames],
        'crossfade':False,'timeline_requires_manifest':True,
        'original_website_direct_playback':False,
        'point_identity_assumption':'row index stays consistent while immutable attributes remain unchanged',
        'position_motion_bits':position_bits,'quaternion_motion_bits':8,
        'sh_codebook_entries':1024,'shared_static_count':len(shared),'gops':[]}
    with tempfile.TemporaryDirectory(prefix='smv-work-',dir=output.parent) as temp:
        temp=Path(temp)
        archive=temp/'result.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            encode_book(arrays,0,'static',static_centers)
            for g,s in enumerate(segments):
                n=len(s['base'])
                dynamic=np.flatnonzero(s['dynamic'])
                statics=np.flatnonzero(~s['dynamic'])
                shared_mask=np.zeros(n,dtype=bool)
                shared_mask[shared]=True
                private_static=statics[~shared_mask[statics]] if g else statics
                order=np.concatenate((shared,private_static,dynamic)) if g else np.concatenate((statics,dynamic))
                side=max(16,math.ceil(math.sqrt(max(1,len(dynamic)))))
                pad=side*side-len(dynamic)
                dummy=np.zeros((pad,62),dtype=np.float32)
                dummy[:,54]=-16.118095
                dummy[:,55:58]=-10
                dummy[:,58]=1
                ordered=np.concatenate((s['base'][order],dummy))
                private=ordered[len(shared):] if g else ordered
                base_xyz,base_q=encode_attrs(arrays,g,private)
                p=f'gop_{g}_'
                if g:
                    arrays[p+'dynamic_idx']=np.array(side*side,dtype=np.int64)
                else:
                    arrays[p+'static_idx']=np.array(len(statics),dtype=np.int64)
                    arrays[p+'mask_for_later']=common[statics]
                arrays[p+'row_order']=order.astype(np.int64)
                attrs=table_attrs(private)
                private_static_count=len(private_static)
                arrays[p+'spec_color_indices_static']=palette_indices(attrs['rest'][:private_static_count],static_centers,threads)
                dyn_centers,dyn_unique_count=palette(attrs['rest'][private_static_count:],threads)
                arrays[p+'spec_color_indices_dynamic']=palette_indices(attrs['rest'][private_static_count:],dyn_centers,threads)
                encode_book(arrays,g,'dynamic',dyn_centers)
                base_xyz=base_xyz[private_static_count:]
                base_q=base_q[private_static_count:]
                inverse=base_q.astype(np.float64).copy()
                inverse[:,1:]*=-1
                encoded_frames=max(1,s['end']-s['start']-1)
                buffers={key:[] for key in ['xyz_offsets_u','quat_w','quat_x','quat_y','quat_z']}
                if position_bits==16:
                    buffers['xyz_offsets_l']=[]
                for j in range(encoded_frames):
                    source_frame=min(s['start']+j+1,s['end']-1)
                    table=read_ply(frames[source_frame]['path'])
                    xyz=np.concatenate((table[dynamic,:3],dummy[:,:3]))
                    target_q=np.concatenate((canonical(table[dynamic,58:62]),dummy[:,58:62]))
                    offsets=xyz.astype(np.float64)-base_xyz
                    dq=normalize(multiply(inverse,target_q))
                    dq*=np.where(dq[:,:1]<0,-1,1)
                    # Quantize position to 16 bits; 8-bit mode drops the lower byte,
                    # exactly matching the original decoder's optional lower stream.
                    q,mn,span=quantize(offsets,16,0)
                    buffers['xyz_offsets_u'].append((q>>8).astype(np.uint8))
                    if position_bits==16:
                        buffers['xyz_offsets_l'].append((q&255).astype(np.uint8))
                    metadata={'shape':[side*side,3],'dtype':'float32',
                              'mins':mn.reshape(-1).tolist(), 'maxs':(mn.astype(np.float64)+span).reshape(-1).tolist()}
                    z.writestr(f'gop_{g}_motion_vectors_meta/xyz_offsets/{j:06d}.json',json_text(metadata))
                    for c,name in enumerate('wxyz'):
                        q,mn,span=quantize(dq[:,c:c+1],8,0)
                        buffers['quat_'+name].append(np.repeat(q,3,axis=1))
                        metadata={'shape':[side*side,1],'dtype':'float32',
                                  'mins':mn.reshape(-1).tolist(),'maxs':(mn.astype(np.float64)+span).reshape(-1).tolist()}
                        z.writestr(f'gop_{g}_motion_vectors_meta/q_{name}/{j:06d}.json',json_text(metadata))
                for key,images in buffers.items():
                    video=temp/f'gop_{g}_{key}.mkv'
                    video_encode(np.stack(images),side,fps,video,ffmpeg,preset,threads)
                    z.write(video,video.name,compress_type=zipfile.ZIP_STORED)
                manifest['gops'].append({'gop':g,'start_frame':s['start'],'frame_count':s['end']-s['start'],
                    'original_point_count':n,'encoded_point_count':len(ordered),
                    'static_count':len(statics),'dynamic_count':len(dynamic),'padding_count':pad,
                    'encoded_motion_frames':encoded_frames,'dynamic_sh_unique_rows':dyn_unique_count})
                print(f'Encoded GOP {g+1}/{len(segments)}: frames {s["start"]}..{s["end"]-1}, '
                      f'{len(statics):,} static, {len(dynamic):,} dynamic, {pad} padding',flush=True)
            npz=io.BytesIO()
            np.savez_compressed(npz,**arrays)
            z.writestr('quantized_gaussian_pruned.npz',npz.getvalue(),compress_type=zipfile.ZIP_STORED)
            z.writestr('manifest.json',json_text(manifest))
        archive.replace(output)
    input_bytes=sum(f['path'].stat().st_size for f in frames)
    report={'archive':str(output),'input_frames':len(frames),'input_ply_bytes':input_bytes,
        'archive_bytes':output.stat().st_size,'compression_ratio':input_bytes/output.stat().st_size,
        'size_reduction_percent':100*(1-output.stat().st_size/input_bytes),
        'seconds':time.monotonic()-started,'gop_count':len(segments),
        'shared_static_count':len(shared),'static_sh_unique_rows':static_unique_count,
        'original_website_direct_playback':False,
        'limitation':'Payloads use the SMV decoder; nonuniform GOP timeline requires manifest.json support.'}
    output.with_suffix(output.suffix+'.report.json').write_text(json_text(report),encoding='utf-8')
    print(json_text(report),flush=True)
    return report


def reference_decoder(path=None):
    path=Path(path) if path else WORKSPACE/'dataset/convert_smv_to_ply.py'
    spec=importlib.util.spec_from_file_location('smv_reference_decoder',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decoded_frames(archive,ffmpeg,reference=None,threads=1):
    """Use the pre-existing decoder, not inverse encoder equations, for round trips."""
    decoder=reference_decoder(reference)
    decoder.FFMPEG_THREADS=threads
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        with np.load(io.BytesIO(z.read('quantized_gaussian_pruned.npz')),allow_pickle=False) as a:
            base=decoder.attrs(a,0)
            prefix=np.flatnonzero(a['gop_0_mask_for_later'])
            static_cb=decoder.codebook(a,0,'static')
            with tempfile.TemporaryDirectory(prefix='smv-decode-') as cache:
                for s in manifest['gops']:
                    g=s['gop']
                    kf=decoder.keyframe(a,g,base,prefix,static_cb)
                    _,videos,meta=decoder.decode_motion(z,g,kf,ffmpeg,Path(cache))
                    order=a[f'gop_{g}_row_order']
                    for local in range(s['frame_count']):
                        table=decoder.ply_table(kf)
                        if local:
                            xyz,q=decoder.motion_frame(kf,videos,meta,local-1)
                            table[kf['static_idx']:, :3]=xyz
                            table[kf['static_idx']:,58:62]=q
                        restored=np.empty((s['original_point_count'],62),dtype=np.float32)
                        restored[order]=table[:s['original_point_count']]
                        index=s['start_frame']+local
                        yield index,manifest['timestamps'][index],restored


def verify(archive,sequence,ffmpeg,reference=None,threads=1):
    frames,_=load_sequence(sequence)
    sums={k:0.0 for k in ['position','rotation_degrees','scale_relative','alpha','dc','sh']}
    maxima=sums.copy()
    counts={k:0 for k in sums}
    seen=0
    for i,timestamp,reconstructed in decoded_frames(archive,ffmpeg,reference,threads):
        source=read_ply(frames[i]['path'])
        if timestamp != frames[i]['timestamp'] or source.shape != reconstructed.shape:
            raise ValueError('Round-trip timestamp/count mismatch')
        q0,q1=normalize(source[:,58:62]),normalize(reconstructed[:,58:62])
        # sign-invariant angular difference; atan2 is stable around zero.
        sign=np.where(np.sum(q0*q1,axis=1,keepdims=True)<0,-1,1)
        q1*=sign
        angle=4*np.arctan2(np.linalg.norm(q0-q1,axis=1),np.linalg.norm(q0+q1,axis=1))*180/np.pi
        errors={'position':np.linalg.norm(source[:,:3].astype(np.float64)-reconstructed[:,:3],axis=1),
            'rotation_degrees':angle,
            'scale_relative':np.abs(np.exp(reconstructed[:,55:58].astype(np.float64)-source[:,55:58])-1),
            'alpha':np.abs(1/(1+np.exp(-source[:,54].astype(np.float64)))-1/(1+np.exp(-reconstructed[:,54].astype(np.float64)))),
            'dc':np.abs(source[:,6:9].astype(np.float64)-reconstructed[:,6:9]),
            'sh':np.abs(source[:,9:54].astype(np.float64)-reconstructed[:,9:54])}
        for key,error in errors.items():
            sums[key]+=float(np.sum(error**2))
            counts[key]+=error.size
            maxima[key]=max(maxima[key],float(error.max()))
        seen+=1
        if seen%50==0:
            print(f'Round-trip verified {seen}/{len(frames)} frames',flush=True)
    if seen != len(frames):
        raise ValueError('Round-trip frame count mismatch')
    report={'status':'completed','frames_compared':seen,'timestamps_and_point_counts_match':True,
        'decoder':'Unmodified pre-existing convert_smv_to_ply.py attribute/motion functions',
        'errors':{k:{'rmse':math.sqrt(sums[k]/counts[k]),'max':maxima[k]} for k in sums},
        'note':'Lossy attribute quantization; no image-space rendering quality measurement.',
        'original_website_direct_playback':False}
    destination=Path(str(archive)+'.validation.json')
    destination.write_text(json_text(report),encoding='utf-8')
    print(json_text(report),flush=True)
    return report


def decode(archive,output,ffmpeg,reference=None,threads=1):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite directory: {output}')
    (output/'frames').mkdir(parents=True)
    decoder=reference_decoder(reference)
    frames=[]
    for i,timestamp,table in decoded_frames(archive,ffmpeg,reference,threads):
        relative=f'frames/frame_{i:06d}.ply'
        decoder.write_ply(output/relative,table)
        frames.append({'file':relative,'timestamp':timestamp})
    (output/'sequence.json').write_text(json_text({'frames':frames}),encoding='utf-8')
    print(f'Decoded {len(frames)} frames to {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['encode','verify','decode']:
        p=sub.add_parser(name)
        p.add_argument('--ffmpeg',type=Path,required=True)
        p.add_argument('--threads',type=int,choices=[1,2,4],default=1)
        if name!='encode':
            p.add_argument('--archive',type=Path,required=True)
            p.add_argument('--reference-decoder',type=Path)
        if name!='verify':
            p.add_argument('--output',type=Path,required=True)
        if name!='decode':
            p.add_argument('--sequence',type=Path,required=True)
        if name=='encode':
            p.add_argument('--max-gop',type=int,default=30)
            p.add_argument('--position-bits',type=int,choices=[8,16],default=16)
            p.add_argument('--preset',choices=['ultrafast','fast','medium'],default='fast')
    args=parser.parse_args()
    if args.command=='encode':
        if args.max_gop<2:
            parser.error('--max-gop must be >= 2')
        encode(args.sequence,args.output,args.ffmpeg,args.max_gop,args.position_bits,args.preset,args.threads)
    elif args.command=='verify':
        verify(args.archive,args.sequence,args.ffmpeg,args.reference_decoder,args.threads)
    else:
        decode(args.archive,args.output,args.ffmpeg,args.reference_decoder,args.threads)


if __name__=='__main__':
    main()
