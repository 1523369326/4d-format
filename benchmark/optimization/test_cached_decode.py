"""Existing v2 bitstream compatibility: old NPZ and latest XZ, exact float outputs."""
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
from adapt import av1_cached_decode as codec
from adapt.common import read_ply,save_frame


class CachedDecodeTests(unittest.TestCase):
    def test_both_v2_containers_match_original_decoder(self):
        sources=[ROOT/'benchmark/optimization/baselines/historical_20261116/source/adapt/av1_temporal.py',
                 ROOT/'benchmark/optimization/references/latest_1607/av1_temporal.py']
        ffmpeg=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'
        first=read_ply(ROOT/'dataset/high_discussion/frames/frame_000000.ply')[::256].copy()
        second=first.copy()
        second[:17,:3]+=.0002
        last=second[:-3].copy()
        last[0,9]+=.001
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark/optimization') as temp:
            temp=Path(temp)
            paths=[]
            for i,table in enumerate([first,second,last,last.copy()]):
                save_frame(temp/'input',i,table)
                paths.append(temp/'input'/f'frame_{i:06d}.ply')
            for i,source in enumerate(sources):
                encoder=codec.load_encoder(source)
                encoded=temp/f'encoded_{i}'
                encoder.encode(paths,[k/30 for k in range(4)],encoded,ffmpeg,threads=1)
                expected=list(encoder.iter_tables(encoded,ffmpeg))
                actual=list(codec.iter_tables(encoded,ffmpeg))
                self.assertEqual(len(actual),4)
                for (frame,timestamp,table),(other,other_time,reference) in zip(actual,expected):
                    self.assertEqual((frame,timestamp),(other,other_time))
                    np.testing.assert_array_equal(table,reference)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
