"""Post-timing verification against all v1 decoded attributes, including row permutations."""
from runtime import configure_environment, pin_cpu
configure_environment()
pin_cpu()
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from adapt.common import read_ply,read_json,sequence_paths,write_json
from adapt.av1_planes import morton_order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True,help='Baseline method folder')
    p.add_argument('--candidate',type=Path,required=True,help='Candidate method folder')
    p.add_argument('--candidate-order',choices=['v1','source','gop'],default='v1')
    args = p.parse_args()
    config = read_json(args.baseline/'encoded/manifest.json')
    candidate_config = read_json(args.candidate/'encoded/manifest.json')
    paths,timestamps = sequence_paths(ROOT/'benchmark/input_first60.json')
    lo,hi = np.asarray(config['mins'],np.float64),np.asarray(config['maxs'],np.float64)
    if candidate_config['mins'] != config['mins'] or candidate_config['maxs'] != config['maxs']:
        raise ValueError('Quantizer range mismatch')
    frames=[]
    gop_orders={}
    if args.candidate_order == 'gop':
        for group in candidate_config['gops']:
            first=group['start_frame']
            end=first+group['frame_count']
            base=read_ply(paths[first])
            dynamic=np.zeros(len(base),bool)
            for frame in range(first,end):
                table=read_ply(paths[frame])
                np.testing.assert_array_equal(table[:,6:58],base[:,6:58],err_msg='GOP immutable fields are not equal')
                dynamic |= np.any(table[:,:3]!=base[:,:3],axis=1)
                dynamic |= np.any(table[:,58:62]!=base[:,58:62],axis=1)
            static_ids=np.flatnonzero(~dynamic)
            dynamic_ids=np.flatnonzero(dynamic)
            static_ids=static_ids[morton_order(base[static_ids,:3],lo[:3],hi[:3])]
            dynamic_ids=dynamic_ids[morton_order(base[dynamic_ids,:3],lo[:3],hi[:3])]
            if len(static_ids)!=group['static_count'] or len(dynamic_ids)!=group['dynamic_count']:
                raise ValueError('GOP static/dynamic counts mismatch')
            order=np.concatenate((static_ids,dynamic_ids))
            for frame in range(first,end):
                gop_orders[frame]=order
    for i,path in enumerate(paths):
        expected=read_ply(args.baseline/'decoded'/f'frame_{i:06d}.ply')
        actual=read_ply(args.candidate/'decoded'/f'frame_{i:06d}.ply')
        if args.candidate_order == 'source':
            order=morton_order(read_ply(path)[:,:3],lo[:3],hi[:3])
            actual=actual[order]
        elif args.candidate_order == 'gop':
            order=morton_order(read_ply(path)[:,:3],lo[:3],hi[:3])
            source_order=np.empty_like(expected)
            source_order[order]=expected
            expected=source_order[gop_orders[i]]
        np.testing.assert_array_equal(actual,expected,err_msg=f'v1 decoded attributes mismatch at frame {i}')
        frames.append({'frame':i,'point_count':len(actual),'attribute_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
    decoded_seq=read_json(args.candidate/'decoded/sequence.json')
    if [f['timestamp'] for f in decoded_seq['frames']] != timestamps:
        raise ValueError('Timestamp mismatch')
    report={'status':'passed','frames':len(frames),'all_62_columns_exact':True,'candidate_order':args.candidate_order,
            'baseline':str(args.baseline.resolve()),'candidate':str(args.candidate.resolve()),'per_frame':frames}
    write_json(args.candidate/'equivalence.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'}),flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
