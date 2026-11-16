"""Exact-quantizer round trips, variable-count boundaries and modulo prediction."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_ply,save_frame,read_json
import av1_codec as av1_temporal
from adapt import av1_planes


class TemporalCodecTests(unittest.TestCase):
    def test_modulo_zigzag_is_bijective(self):
        rng=np.random.default_rng(81)
        a=rng.integers(0,65536,(1000,7),dtype=np.uint16)
        b=rng.integers(0,65536,(1000,7),dtype=np.uint16)
        a[0]=[0,65535,32768,32767,1,65534,0]
        b[0]=[65535,0,0,65535,65534,1,32768]
        np.testing.assert_array_equal(av1_temporal.unzigzag(av1_temporal.zigzag(a,b),b),a)

    def test_exact_v1_quantization_with_topology_and_single_frame_gop(self):
        ffmpeg=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as temp:
            temp=Path(temp)
            source=ROOT/'dataset/high_discussion/frames'
            selection=np.linspace(0,len(read_ply(source/'frame_000000.ply'))-1,1024,dtype=int)
            tables=[read_ply(source/f'frame_{i:06d}.ply')[selection] for i in range(2)]
            # Changed point count and appearance force a singleton, fully static GOP.
            last=np.concatenate((tables[-1],tables[-1][:1]))
            last[0,6]+=.001
            tables.append(last)
            paths=[]
            for i,t in enumerate(tables):
                save_frame(temp/'input',i,t)
                paths.append(temp/'input'/f'frame_{i:06d}.ply')
            av1_temporal.encode(paths,[0,1/30,2/30],temp/'encoded',ffmpeg,threads=1)
            config=read_json(temp/'encoded/manifest.json')
            decoded=list(av1_temporal.iter_tables(temp/'encoded',ffmpeg))
            self.assertEqual([len(t) for _,_,t in decoded],[1024,1024,1025])
            legacy_side=int(np.ceil(np.sqrt(max(len(t) for t in tables))))
            legacy_config={**config,'tile_side':legacy_side,'tile_columns':12,'tile_rows':10,
                           'width':legacy_side*12,'height':legacy_side*10}
            for i,timestamp,t in decoded:
                # Compare against the independently implemented v1 encoder/decoder,
                # restoring its per-frame Morton permutation back to input row order.
                atlas=av1_planes.atlas_from_table(tables[i],legacy_config)
                legacy=av1_planes.table_from_atlas(atlas,len(tables[i]),legacy_config)
                order=av1_planes.morton_order(tables[i][:,:3],np.array(config['mins'])[:3],np.array(config['maxs'])[:3])
                expected=np.empty_like(legacy)
                expected[order]=legacy
                np.testing.assert_array_equal(t,expected)
                self.assertEqual(timestamp,i/30)
            self.assertEqual(config['gops'][-1]['dynamic_count'],0)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
