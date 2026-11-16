"""Decode the downloaded SMV high/discussion into standard 3DGS PLY frames.

Uses only NumPy and a portable FFmpeg executable. See high_discussion/README.txt.
The equations follow the accompanying website decode*.js and dequant_worker.js.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
PROPERTIES = (
    ['x', 'y', 'z', 'nx', 'ny', 'nz']
    + [f'f_dc_{i}' for i in range(3)]
    + [f'f_rest_{i}' for i in range(45)]
    + ['opacity'] + [f'scale_{i}' for i in range(3)]
    + [f'rot_{i}' for i in range(4)]
)
FPS, STRIDE, OVERLAP = 30, 25, 5
FFMPEG_THREADS = 2


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def norm_quat(q):
    # JS normalizes in double precision then stores into Float32Array.
    q = q.astype(np.float64)
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.where(length == 0, 1, length)).astype(np.float32)


def dequant(npz, prefix, name, bit_name):
    q = npz[prefix + name].astype(np.float64)
    return (q / (2 ** int(npz[prefix + bit_name]) - 1)
            * npz[prefix + name + '_range'] + npz[prefix + name + '_min']).astype(np.float32)


def attrs(npz, g):
    p = f'gop_{g}_'
    n = len(npz[p + 'position'])
    return {
        'xyz': npz[p + 'position'].astype(np.float32),
        'quat': norm_quat(dequant(npz, p, 'rotation', 'rotation_n_bit')),
        'scale': dequant(npz, p, 'scaling', 'scaling_n_bit'),
        'opacity': dequant(npz, p, 'opacity', 'opacity_n_bit').reshape(n),
        'dc': dequant(npz, p, 'base_color', 'color_n_bit').reshape(n, 3),
    }


def codebook(npz, g, kind):
    return dequant(npz, f'gop_{g}_', f'sh_codebook_{kind}', 'color_n_bit')


def sanitize(kf):
    """Match the website's invisible-floater handling, keeping vertex order."""
    xyz = kf['xyz']
    sample = xyz[::max(1, len(xyz) // 4096)].astype(np.float64)
    valid = np.isfinite(sample).all(axis=1)
    if not valid.any():
        raise ValueError('No finite keyframe positions')
    center = sample[valid].mean(axis=0)
    distances = np.linalg.norm(sample - center, axis=1)
    distances = np.sort(distances[np.isfinite(distances)])
    threshold = (distances[int(len(distances) * .99)] or 1) * 20
    bad = ~np.isfinite(xyz).all(axis=1) | (np.linalg.norm(xyz.astype(np.float64) - center, axis=1) > threshold)
    xyz[bad] = center
    kf['opacity'][bad] = 0
    return int(bad.sum())


def keyframe(npz, g, base, prefix_idx, static_cb):
    a = attrs(npz, g)
    if g == 0:
        si = int(npz['gop_0_static_idx'])
    else:
        a = {k: np.concatenate((base[k][prefix_idx], v)) for k, v in a.items()}
        si = len(a['xyz']) - int(npz[f'gop_{g}_dynamic_idx'])
    p = f'gop_{g}_'
    static_idx = npz[p + 'spec_color_indices_static']
    if g:
        static_idx = np.concatenate((npz['gop_0_spec_color_indices_static'][prefix_idx], static_idx))
    dyn_idx = npz[p + 'spec_color_indices_dynamic']
    # Codebook layout is coefficient-major RGB. Standard 3DGS PLY is channel-major.
    cb = np.concatenate((static_cb, codebook(npz, g, 'dynamic')))
    indices = np.concatenate((static_idx, len(static_cb) + dyn_idx.astype(np.int64)))
    a['rest'] = cb[indices].reshape(-1, 15, 3).transpose(0, 2, 1).reshape(-1, 45)
    a['static_idx'] = si
    a['culled'] = sanitize(a)
    return a


def metas(z, g, kind):
    names = sorted(n for n in z.namelist() if f'gop_{g}_motion_vectors_meta/{kind}/' in n and n.endswith('.json'))
    return [json.loads(z.read(n)) for n in names]


def decode_video(z, name, ffmpeg, cache, frames, dynamic_count):
    path = cache / name
    path.write_bytes(z.read(name))
    command = [str(ffmpeg), '-v', 'error', '-threads', str(FFMPEG_THREADS), '-i', str(path),
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1']
    decoded = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = np.frombuffer(decoded.stdout, dtype=np.uint8)
    expected = frames * dynamic_count * 3
    if data.size != expected:
        raise ValueError(f'{name}: {data.size} bytes, expected {expected}')
    return data.reshape(frames, dynamic_count, 3)


def decode_motion(z, g, kf, ffmpeg, cache):
    meta = {k: metas(z, g, k) for k in ['xyz_offsets', 'q_w', 'q_x', 'q_y', 'q_z']}
    f = len(meta['xyz_offsets'])
    d = len(kf['xyz']) - kf['static_idx']
    if any(len(v) != f for v in meta.values()):
        raise ValueError('Mismatched motion metadata lengths')
    videos = {}
    for component in ['xyz_offsets_u', 'quat_w', 'quat_x', 'quat_y', 'quat_z']:
        videos[component] = decode_video(z, f'gop_{g}_{component}.mkv', ffmpeg, cache, f, d)
    lower = f'gop_{g}_xyz_offsets_l.mkv'
    if lower in z.namelist():
        videos['xyz_offsets_l'] = decode_video(z, lower, ffmpeg, cache, f, d)
    return f, videos, meta


def motion_frame(kf, videos, meta, frame):
    si = kf['static_idx']
    q = videos['xyz_offsets_u'][frame].astype(np.uint16) << 8
    if 'xyz_offsets_l' in videos:
        q += videos['xyz_offsets_l'][frame]
    m = meta['xyz_offsets'][frame]
    mn = np.asarray(m['mins'], dtype=np.float64)
    rg = np.asarray(m['maxs'], dtype=np.float64) - mn
    xyz = (kf['xyz'][si:].astype(np.float64) + q.astype(np.float64) / 65535 * rg + mn).astype(np.float32)
    deltas = []
    for component in 'wxyz':
        m = meta['q_' + component][frame]
        mn = m['mins'][0]
        rg = m['maxs'][0] - mn
        deltas.append(videos['quat_' + component][frame, :, 0].astype(np.float64) / 255 * rg + mn)
    dq = np.stack(deltas, axis=1)
    norms = np.linalg.norm(dq, axis=1, keepdims=True)
    dq /= np.where(norms == 0, 1, norms)
    w, x, y, zz = kf['quat'][si:].astype(np.float64).T
    a, b, c, dd = dq.T
    rotation = np.stack((w*a-x*b-y*c-zz*dd, w*b+x*a+y*dd-zz*c,
                         w*c-x*dd+y*a+zz*b, w*dd+x*c-y*b+zz*a), axis=1)
    return xyz, norm_quat(rotation)


def ply_table(kf):
    table = np.zeros((len(kf['xyz']), len(PROPERTIES)), dtype='<f4')
    table[:, :3] = kf['xyz']
    table[:, 6:9] = kf['dc']
    table[:, 9:54] = kf['rest']
    # Archive stores activated alpha and linear sigma; standard 3DGS PLY stores
    # inverse-sigmoid alpha and log(sigma). Clamp only to keep finite values.
    alpha = np.clip(kf['opacity'].astype(np.float64), 1e-7, 1-1e-7)
    table[:, 54] = np.log(alpha / (1-alpha))
    table[:, 55:58] = np.log(np.maximum(kf['scale'], 1e-20))
    table[:, 58:62] = kf['quat']
    return table


def write_ply(path, table):
    if not np.isfinite(table).all():
        raise ValueError(f'Nonfinite PLY data: {path}')
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Standard 3DGS parameters decoded from SMV high discussion\n'
              f'element vertex {len(table)}\n'
              + ''.join(f'property float {name}\n' for name in PROPERTIES) + 'end_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        table.tofile(f)
    return path.stat().st_size


def schedule(total, gop_count):
    jobs = defaultdict(list)
    frames = []
    for frame in range(total):
        owner = min(frame // STRIDE, gop_count - 1)
        local = frame - owner * STRIDE
        sources = [(owner, local, 1.0)]
        if owner > 0 and local < OVERLAP:
            weight = local / (OVERLAP - 1)
            sources = [(owner - 1, local + STRIDE, 1-weight), (owner, local, weight)]
            sources = [s for s in sources if s[2] > 0]
        primary = max(sources, key=lambda s: (s[2], s[0]))
        file = f'frames/frame_{frame:06d}.ply'
        layers = []
        for g, lf, weight in sources:
            target = file if (g, lf) == primary[:2] else f'transitions/frame_{frame:06d}_gop_{g:02d}.ply'
            layer = {'file': target, 'weight': weight, 'gop': g, 'local_frame': lf}
            layers.append(layer)
            jobs[g].append((lf, target, frame, layer))
        frames.append({'index': frame, 'timestamp': frame / FPS, 'duration': 1/FPS,
                       'file': file, 'gop': primary[0], 'local_frame': primary[1], 'layers': layers})
    return jobs, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'high_discussion')
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    archive = out / 'source/discussion.zip'
    for name in ['frames', 'transitions', 'source/motion_cache']:
        (out / name).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f'Corrupt ZIP entry: {bad}')
        npz_name = next(n for n in z.namelist() if n.endswith('.npz'))
        with np.load(io.BytesIO(z.read(npz_name)), allow_pickle=False) as npz:
            ids = sorted(int(re.match(r'gop_(\d+)_position', k)[1]) for k in npz.files if re.fullmatch(r'gop_\d+_position', k))
            if ids != list(range(len(ids))):
                raise ValueError('Non-contiguous GOP IDs')
            total = (len(ids) - 1) * STRIDE + len(metas(z, ids[-1], 'xyz_offsets')) + 1
            jobs, frames = schedule(total, len(ids))
            base = attrs(npz, 0)
            prefix_idx = np.flatnonzero(npz['gop_0_mask_for_later'])
            static_cb = codebook(npz, 0, 'static')
            statistics, byte_count = [], 0
            for g in ids:
                kf = keyframe(npz, g, base, prefix_idx, static_cb)
                f, videos, meta = decode_motion(z, g, kf, args.ffmpeg, out / 'source/motion_cache')
                table = ply_table(kf)
                si = kf['static_idx']
                for lf, relative, frame, layer in sorted(jobs[g]):
                    if lf:
                        xyz, quat = motion_frame(kf, videos, meta, lf-1)
                        table[si:, :3] = xyz
                        table[si:, 58:62] = quat
                    else:
                        table[:, :3] = kf['xyz']
                        table[:, 58:62] = kf['quat']
                    byte_count += write_ply(out / relative, table)
                    layer['point_count'] = len(table)
                    if relative == frames[frame]['file']:
                        frames[frame]['point_count'] = len(table)
                stat = {'gop': g, 'point_count': len(table), 'static_count': si,
                        'dynamic_count': len(table)-si, 'motion_frames': f, 'neutralized_floaters': kf['culled']}
                statistics.append(stat)
                print(f'GOP {g+1}/{len(ids)}: {len(table):,} Gaussians, {len(jobs[g])} PLY files; '
                      f'{byte_count/1e9:.2f} GB; {time.monotonic()-started:.1f}s', flush=True)
                del videos, table, kf
    scenes = json.loads((ROOT / 'scenes.json').read_text(encoding='utf-8-sig'))
    scene = next(s for s in scenes if s['url'] == 'scenes/meetroom/high/run01/discussion.zip')
    dump_json(out / 'camera.json', {**scene['cam'], 'quaternion_order': 'wxyz',
                                   'convention': 'OpenCV: x right, y down, z forward; world up = -y'})
    source_url = 'https://miya9756.github.io/smv-viewer-prototype/' + scene['url']
    manifest = {
        'schema_version': 1, 'name': 'high - discussion', 'format': '3dgs_ply_sequence',
        'fps': FPS, 'frame_count': total, 'duration': total/FPS, 'timestamp_unit': 'seconds',
        'timestamp_origin': 'relative_scene_start', 'start_timestamp': 0,
        'last_timestamp': (total-1)/FPS, 'frame_pattern': 'frames/frame_%06d.ply',
        'sh_degree': 3, 'coordinate_system': 'source world coordinates, OpenCV camera, world up -y',
        'opacity_encoding': 'logit', 'scale_encoding': 'natural_log', 'quaternion_order': 'wxyz',
        'sh_rest_order': 'channel_major: R[15], G[15], B[15]',
        'gop_stride': STRIDE, 'gop_overlap': OVERLAP,
        'transition_mode': 'render layers independently, then sum premultiplied RGBA images times weight',
        'single_file_transition_policy': 'dominant GOP (incoming GOP wins equal-weight tie)',
        'source': {'page': 'https://miya9756.github.io/smv-viewer-prototype/', 'url': source_url,
                   'archive': 'source/discussion.zip', 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                   'credit': 'Mingyang Song; SmoothMotionVectors; MeetRoom dataset', 'license_file': 'source/LICENSE'},
        'camera_file': 'camera.json', 'frames': frames,
    }
    dump_json(out / 'sequence.json', manifest)
    dump_json(out / 'conversion_report.json', {
        'frame_count': total, 'primary_ply_count': total,
        'transition_ply_count': sum(len(v) for v in jobs.values())-total,
        'ply_bytes': byte_count, 'elapsed_seconds': time.monotonic()-started, 'gops': statistics,
        'source_zip_crc_checked': True,
    })
    print(f'Complete: {out / "sequence.json"}', flush=True)


if __name__ == '__main__':
    main()
