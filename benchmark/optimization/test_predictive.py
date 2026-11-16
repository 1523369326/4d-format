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
from adapt import av1_predictive as codec,av1_planes as v1
from adapt.common import read_ply,save_frame,read_json


class PredictiveTests(unittest.TestCase):
    def test_bijective_with_overflow(self):
        values=np.random.default_rng(23).integers(0,65536,(1024,59),dtype=np.uint16)
        values[0]=0
        values[1]=65535
        values[2]=32768
        for mode in codec.MODES:
            np.testing.assert_array_equal(codec.inverse(codec.forward(values,mode),mode),values)

    def test_real_av1_transport_and_v1_restoration(self):
        ffmpeg=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'
        original_encode=v1.atlas_from_table
        original_decode=v1.table_from_atlas
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark/optimization') as temp:
            temp=Path(temp)
            source=read_ply(ROOT/'dataset/high_discussion/frames/frame_000000.ply')[::128]
            save_frame(temp/'input',0,source)
            for mode in codec.MODES:
                folder=temp/mode
                codec.encode([temp/'input/frame_000000.ply'],[0.0],folder/'encoded',ffmpeg,prediction=mode)
                codec.decode(folder/'encoded',folder/'decoded',ffmpeg)
                config=read_json(folder/'encoded/manifest.json')
                expected=original_decode(original_encode(source,config),len(source),config)
                np.testing.assert_array_equal(read_ply(folder/'decoded/frame_000000.ply'),expected)
                self.assertIs(v1.atlas_from_table,original_encode)
                self.assertIs(v1.table_from_atlas,original_decode)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
