"""Meaningful format invariants: byte-exact v1 reconstruction, overflow, corruption."""
from runtime import configure_environment, pin_cpu, audit_ffmpeg
configure_environment()
cpu, mask = pin_cpu()
events = []
audit_ffmpeg(mask,events)
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from adapt import av1_planes as v1, av1_shdict as v3
from adapt.common import read_ply,save_frame,read_json


class SHDictionaryTests(unittest.TestCase):
    def test_wide_label_planes_and_scalar_extremes(self):
        config = {'tile_side':2,'tile_rows':6,'tile_columns':6,'height':12,'width':12,'index_bytes':4}
        codes = np.zeros((4,14),np.uint16)
        codes[1] = 65535
        codes[2] = 32768
        labels = np.array([0,65535,65536,2**32-1],np.uint32)
        a,b = v3.unpack(v3.pack(codes,labels,config),4,config)
        np.testing.assert_array_equal(a,codes)
        np.testing.assert_array_equal(b,labels)

    def test_exact_v1_round_trip_variable_count_constants_and_delta(self):
        source = ROOT/'dataset/high_discussion/frames'
        tables = [read_ply(source/f'frame_{i:06d}.ply')[::128].copy() for i in [0,1]]
        tables[1] = tables[1][:-3]
        for table in tables:
            table[:,9] = 0.25  # constant SH channel must reconstruct identically
            table[:,54] = 0.0  # constant scalar channel
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark/optimization') as temp:
            temp = Path(temp)
            paths = []
            for i,table in enumerate(tables):
                save_frame(temp/'input',i,table)
                paths.append(temp/'input'/f'frame_{i:06d}.ply')
            for delta in [False,True]:
                folder = temp/str(delta)
                v3.encode(paths,[0,1/30],folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe',spatial_delta=delta)
                config = read_json(folder/'manifest.json')
                decoded = list(v3.iter_tables(folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'))
                for i,timestamp,actual in decoded:
                    expected = v1.table_from_atlas(v1.atlas_from_table(tables[i],{**config,'tile_columns':12,'tile_rows':10,'width':config['tile_side']*12,'height':config['tile_side']*10}),len(tables[i]),{**config,'tile_columns':12,'tile_rows':10,'width':config['tile_side']*12,'height':config['tile_side']*10})
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(timestamp,i/30)
                sidecar = folder/config['sidecar']
                sidecar.write_bytes(sidecar.read_bytes()+b'corruption')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    list(v3.iter_tables(folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'))


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
