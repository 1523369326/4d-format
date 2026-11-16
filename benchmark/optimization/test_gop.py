"""GOP format tests: topology, row shuffle, quaternion sign, singleton, empty motion."""
from runtime import configure_environment,pin_cpu,audit_ffmpeg
configure_environment()
_,mask=pin_cpu()
audit_ffmpeg(mask,[])
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from adapt import av1_gop as codec
from adapt.common import read_ply,save_frame,read_json
from adapt.av1_planes import COLUMNS,morton_order


class GOPTests(unittest.TestCase):
    def test_exact_quantizer_and_safe_gop_fallbacks(self):
        first=read_ply(ROOT/'dataset/high_discussion/frames/frame_000000.ply')[::256].copy()
        second=first.copy()
        second[:17,:3]+=.0002
        second[0,58:62]*=-1 # Equivalent rotation but different raw v1 quantized values.
        third=second.copy()
        third[:17,:3]+=.00015
        shuffled=third[::-1].copy() # Must not assume constant point count means identity.
        last=shuffled[:-3].copy()
        last[0,9]+=.001
        tables=[first,second,third,shuffled,last,last.copy()]
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark/optimization') as temp:
            temp=Path(temp)
            paths=[]
            for i,table in enumerate(tables):
                save_frame(temp/'input',i,table)
                paths.append(temp/'input'/f'frame_{i:06d}.ply')
            for predictor in ['previous','reference','linear']:
                for planar in [False,True]:
                    folder=temp/f'{predictor}_{planar}'
                    codec.encode(paths,[i/30 for i in range(len(paths))],folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe',predictor=predictor,planar_sidecar=planar,sidecar_codec='lzma' if planar else 'deflate')
                    config=read_json(folder/'manifest.json')
                    lo,hi=np.asarray(config['mins']),np.asarray(config['maxs'])
                    expected=[]
                    for group in config['gops']:
                        base=tables[group['start_frame']]
                        dynamic=np.zeros(len(base),bool)
                        for frame in range(group['start_frame'],group['start_frame']+group['frame_count']):
                            dynamic|=np.any(tables[frame][:,:3]!=base[:,:3],axis=1)
                            dynamic|=np.any(tables[frame][:,58:62]!=base[:,58:62],axis=1)
                        a,b=np.flatnonzero(~dynamic),np.flatnonzero(dynamic)
                        a=a[morton_order(base[a,:3],lo[:3],hi[:3])]
                        b=b[morton_order(base[b,:3],lo[:3],hi[:3])]
                        order=np.concatenate((a,b))
                        for frame in range(group['start_frame'],group['start_frame']+group['frame_count']):
                            table=np.zeros((len(base),62),np.float32)
                            table[:,COLUMNS]=codec.dequantize(codec.quantize(tables[frame][np.ix_(order,COLUMNS)],lo,hi),lo,hi)
                            expected.append(table)
                    actual=list(codec.iter_tables(folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'))
                    for (frame,timestamp,table),reference in zip(actual,expected):
                        np.testing.assert_array_equal(table,reference)
                        self.assertEqual(timestamp,frame/30)
                    self.assertEqual(config['gops'][-1]['dynamic_count'],0)
                    self.assertGreaterEqual(len(config['gops']),3)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
