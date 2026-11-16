"""Measure structural differences between the v1 full-attribute atlas and TSOG."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_ply,read_json,write_json
from adapt.av1_planes import COLUMNS,morton_order
from av1_codec import quantize


def main():
    old=ROOT/'benchmark/results'
    config=read_json(old/'morton_av1/encoded/manifest.json')
    base=ROOT/'dataset/high_discussion/frames'
    t0=read_ply(base/'frame_000000.ply')
    t1=read_ply(base/'frame_000001.ply')
    lo,hi=np.array(config['mins']),np.array(config['maxs'])
    o0=morton_order(t0[:,:3],lo[:3],hi[:3])
    o1=morton_order(t1[:,:3],lo[:3],hi[:3])
    q0,q1=quantize(t0,config),quantize(t1,config)
    changed_immutable=int(np.count_nonzero(t0[:,6:58]!=t1[:,6:58]))
    categories={}
    for chunk in (old/'tsog/encoded').glob('*.tsog'):
        with zipfile.ZipFile(chunk) as z:
            for info in z.infolist():
                n=info.filename
                key=('position' if n.startswith('means_') else 'rotation' if n=='quats.webp'
                     else 'scale' if n=='scales.webp' else 'dc_alpha' if n=='sh0.webp'
                     else 'sh_palette' if n.startswith('shN_centroids') else 'sh_labels' if n.startswith('shN_labels')
                     else 'timeline' if n.startswith('timeline_') else 'metadata')
                categories[key]=categories.get(key,0)+info.compress_size
    unique={}
    for i in [0,27,52]:
        table=read_ply(base/f'frame_{i:06d}.ply')
        unique[str(i)]={'point_count':len(table),'exact_quantized_sh_vectors':len(np.unique(quantize(table,config)[:,6:51],axis=0))}
    report={'old_av1_bytes':read_json(old/'morton_av1/timing.json')['encoded_bytes'],
        'tsog_bytes':read_json(old/'tsog/timing.json')['encoded_bytes'],
        'old_av1_psnr_db':read_json(old/'morton_av1/quality.json')['pooled_psnr_db'],
        'tsog_psnr_db':read_json(old/'tsog/quality.json')['pooled_psnr_db'],
        'source_attributes':59,'expanded_sh_attributes':45,'expanded_sh_raw_fraction':45/59,
        'old_atlas_dimensions':[config['width'],config['height']],
        'old_raw_atlas_bytes':config['width']*config['height']*60,
        'old_expanded_sh_bytes_per_point':90,'exact_sh_label_bytes_per_point':2,
        'frame0_to_frame1':{'fraction_sorted_slots_with_different_point_identity':float(np.mean(o0!=o1)),
            'fraction_changed_codes_independent_morton':float(np.mean(q0[o0]!=q1[o1])),
            'fraction_changed_codes_fixed_morton':float(np.mean(q0[o0]!=q1[o0])),
            'immutable_float_values_changed':changed_immutable},
        'gop_keyframe_sh_uniqueness':unique,'tsog_encoded_payload_breakdown_bytes':categories,
        'not_same_rate_distortion_point':True,
        'optimization_goal':'preserve exact v1 global uint16 quantized attributes; remove structural redundancy before AV1'}
    write_json(ROOT/'benchmark/av1_structure_analysis.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
