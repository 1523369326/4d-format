"""Validate every exported PLY, plus independently recompute sampled source motion."""
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'high_discussion'


def read_ply(path):
    with path.open('rb') as f:
        lines = []
        while True:
            line = f.readline()
            assert line, f'Truncated header: {path}'
            lines.append(line)
            if line == b'end_header\n':
                break
        header = b''.join(lines)
        n = int(next(l.split()[2] for l in lines if l.startswith(b'element vertex')))
        names = [l.decode().split()[2] for l in lines if l.startswith(b'property float')]
        assert len(names) == 62
        payload = f.read()
    assert len(payload) == n * 62 * 4, path
    data = np.frombuffer(payload, dtype='<f4').reshape(n, 62)
    return data, header, payload


def unit(q):
    length = math.sqrt(sum(float(v)**2 for v in q)) or 1
    return [float(v)/length for v in q]


def raw_attribute(a, g, field, point, bit):
    p = f'gop_{g}_'
    q = a[p+field][point].reshape(-1)
    mn = a[p+field+'_min'].reshape(-1)
    rg = a[p+field+'_range'].reshape(-1)
    return np.array([float(v)/(2**int(a[p+bit])-1)*float(rg[min(i,len(rg)-1)])
                     + float(mn[min(i,len(mn)-1)]) for i,v in enumerate(q)], dtype=np.float32)


def main():
    m = json.loads((OUT/'sequence.json').read_text(encoding='utf-8'))
    report = json.loads((OUT/'conversion_report.json').read_text(encoding='utf-8'))
    assert m['frame_count'] == 300 and len(m['frames']) == 300 and m['fps'] == 30
    files, per_gop = {}, defaultdict(list)
    for i, frame in enumerate(m['frames']):
        assert frame['index'] == i
        assert abs(frame['timestamp'] - i/30) < 1e-12
        assert frame['file'] == f'frames/frame_{i:06d}.ply'
        assert frame['timestamp'] > m['frames'][i-1]['timestamp'] if i else frame['timestamp'] == 0
        assert abs(sum(l['weight'] for l in frame['layers'])-1) < 1e-12
        primary = next(l for l in frame['layers'] if l['file'] == frame['file'])
        assert frame['point_count'] == primary['point_count']
        for layer in frame['layers']:
            assert layer['file'] not in files
            files[layer['file']] = layer
            per_gop[layer['gop']].append(layer)
    assert len(files) == 333
    assert len(list((OUT/'frames').glob('*.ply'))) == 300
    assert len(list((OUT/'transitions').glob('*.ply'))) == 33
    hashes, total_bytes, max_norm_error = [], 0, 0.0
    for j,(relative,layer) in enumerate(files.items()):
        p = OUT/relative
        data, header, payload = read_ply(p)
        assert len(data) == layer['point_count']
        assert np.isfinite(data).all(), relative
        assert np.all(data[:,3:6] == 0)
        error = float(np.max(np.abs(np.linalg.norm(data[:,58:62].astype(np.float64),axis=1)-1)))
        assert error < 2e-6, (relative,error)
        max_norm_error = max(max_norm_error,error)
        assert np.exp(data[:,55:58]).min() > 0
        h = hashlib.sha256(header)
        h.update(payload)
        hashes.append(f'{h.hexdigest()}  {relative}')
        total_bytes += len(header)+len(payload)
        if (j+1)%75 == 0:
            print(f'Validated {j+1}/{len(files)} PLY files', flush=True)
    assert total_bytes == report['ply_bytes']
    (OUT/'checksums.sha256').write_text('\n'.join(hashes)+'\n',encoding='ascii')

    # Independent scalar reference: no conversion module is imported here.
    ffmpeg = next((ROOT/'tools').glob('ffmpeg*.exe'))
    max_xyz, max_quat, max_attr, sampled = 0.0, 0.0, 0.0, 0
    with zipfile.ZipFile(OUT/'source/discussion.zip') as z:
        with np.load(io.BytesIO(z.read('quantized_gaussian_pruned.npz'))) as a:
            prefix = np.flatnonzero(a['gop_0_mask_for_later'])
            for g,layers in sorted(per_gop.items()):
                layers = sorted(layers,key=lambda l:l['local_frame'])
                candidates = [layers[0],layers[len(layers)//2],layers[-1]]
                stat = report['gops'][g]
                d,si = stat['dynamic_count'],stat['static_count']
                f = stat['motion_frames']
                videos = {}
                for name in ['xyz_offsets_u','quat_w','quat_x','quat_y','quat_z']:
                    p = OUT/f'source/motion_cache/gop_{g}_{name}.mkv'
                    r = subprocess.run([str(ffmpeg),'-v','error','-threads','2','-i',str(p),
                                        '-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],
                                       check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    videos[name] = np.frombuffer(r.stdout,dtype=np.uint8).reshape(f,d,3)
                for layer in candidates:
                    data,_,_ = read_ply(OUT/layer['file'])
                    lf = layer['local_frame']
                    for point in [0,si-1,si,si+d//2,si+d-1]:
                        source_g = 0 if g and point < len(prefix) else g
                        point_src = int(prefix[point]) if g and point < len(prefix) else point-(len(prefix) if g else 0)
                        base_xyz = a[f'gop_{source_g}_position'][point_src].astype(np.float32)
                        base_q = np.asarray(unit(raw_attribute(a,source_g,'rotation',point_src,'rotation_n_bit')),dtype=np.float32)
                        expected_xyz,expected_q = base_xyz,base_q
                        if point >= si and lf:
                            dyn = point-si
                            md = json.loads(z.read(f'gop_{g}_motion_vectors_meta/xyz_offsets/'+
                                sorted(n.rsplit('/',1)[1] for n in z.namelist() if f'gop_{g}_motion_vectors_meta/xyz_offsets/' in n and n.endswith('.json'))[lf-1]))
                            vals = videos['xyz_offsets_u'][lf-1,dyn]
                            expected_xyz = np.array([float(base_xyz[c])+(int(vals[c])<<8)/65535*
                                (md['maxs'][c]-md['mins'][c])+md['mins'][c] for c in range(3)],dtype=np.float32)
                            dq = []
                            for c in 'wxyz':
                                names = sorted(n for n in z.namelist() if f'gop_{g}_motion_vectors_meta/q_{c}/' in n and n.endswith('.json'))
                                md = json.loads(z.read(names[lf-1]))
                                dq.append(int(videos['quat_'+c][lf-1,dyn,0])/255*(md['maxs'][0]-md['mins'][0])+md['mins'][0])
                            w,x,y,zz = map(float,base_q)
                            b,c,dd,e = unit(dq)
                            expected_q = np.asarray(unit([w*b-x*c-y*dd-zz*e,w*c+x*b+y*e-zz*dd,
                                                          w*dd-x*e+y*b+zz*c,w*e+x*dd-y*c+zz*b]),dtype=np.float32)
                        max_xyz = max(max_xyz,float(np.max(np.abs(data[point,:3]-expected_xyz))))
                        max_quat = max(max_quat,float(np.max(np.abs(data[point,58:62]-expected_q))))
                        expected_scale = raw_attribute(a,source_g,'scaling',point_src,'scaling_n_bit')
                        expected_dc = raw_attribute(a,source_g,'base_color',point_src,'color_n_bit')
                        max_attr = max(max_attr,float(np.max(np.abs(data[point,55:58]-np.log(expected_scale)))),
                                       float(np.max(np.abs(data[point,6:9]-expected_dc))))
                        # Independently expand the SH codebook and test channel ordering.
                        if point < si:
                            kind='static'
                            cb_g=0
                            idx_point = point_src
                        else:
                            kind='dynamic'
                            cb_g=g
                            idx_point=point-si
                        idx = int(a[f'gop_{source_g}_spec_color_indices_{kind}'][idx_point])
                        cb_name=f'gop_{cb_g}_sh_codebook_{kind}'
                        cb=(a[cb_name][idx].astype(np.float64)/255*a[cb_name+'_range'].reshape(45)
                            +a[cb_name+'_min'].reshape(45)).astype(np.float32)
                        expected_rest=np.array([cb[coef*3+channel] for channel in range(3) for coef in range(15)])
                        max_attr=max(max_attr,float(np.max(np.abs(data[point,9:54]-expected_rest))))
                        sampled+=1
                print(f'Compared GOP {g}: source keyframes, SH and decoded motion',flush=True)
                del videos
    assert max_xyz < 2e-5, max_xyz
    assert max_quat < 2e-6, max_quat
    assert max_attr < 2e-6, max_attr
    validation={'status':'passed','primary_frames':300,'auxiliary_transition_plys':33,
        'all_ply_files_checked':len(files),'all_ply_bytes_checked':total_bytes,
        'timestamps_monotonic':True,'all_layer_weights_sum_to_one':True,
        'all_values_finite':True,'all_file_sizes_match_headers':True,
        'sha256_file':'checksums.sha256','max_quaternion_norm_error':max_norm_error,
        'independent_scalar_reference_samples':sampled,
        'max_source_position_error':max_xyz,'max_source_rotation_error':max_quat,
        'max_source_scale_dc_sh_error':max_attr}
    (OUT/'validation_report.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    main()
