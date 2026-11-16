"""Exercise real third-party encoders/decoders and byte-exact AV1 on small inputs."""
import importlib
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
from adapt.common import read_ply,save_frame


class AdapterSmoke(unittest.TestCase):
    def test_vectorized_texture_fill_matches_original(self):
        from adapt.texture_video import make_encoder,modules
        from plyfile import PlyData,PlyElement
        enc,_=modules()
        vertex=PlyData.read(ROOT/'dataset/high_discussion/frames/frame_000000.ply')['vertex'].data[:64].copy()
        ply=PlyData([PlyElement.describe(vertex,'vertex')],text=False,byte_order='<')
        class OriginalUniform(enc.PlyTextureEncoder):
            def _select_texture_size(self,entry_count):
                return 8,8
        original=OriginalUniform().encode(ply)
        fast=make_encoder(8).encode(ply)
        for a,b in zip(original.groups,fast.groups):
            self.assertEqual(a.name,b.name)
            self.assertEqual(a.min_values,b.min_values)
            self.assertEqual(a.max_values,b.max_values)
            np.testing.assert_array_equal(a.texture,b.texture)

    def test_all_codecs(self):
        ffmpeg=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as temp:
            temp=Path(temp)
            inputs=temp/'input'
            source=ROOT/'dataset/high_discussion/frames'
            indices=np.linspace(0,len(read_ply(source/'frame_000000.ply'))-1,1024,dtype=int)
            for i in range(2):
                save_frame(inputs,i,read_ply(source/f'frame_{i:06d}.ply')[indices])
            paths=sorted(inputs.glob('*.ply'))
            for name,module,options in [('av1','av1_planes',{}),('gsd_v1','gsd',{'version':1}),
                    ('gsd_v2','gsd',{'version':2}),('tsog','tsog_adapter',{}),('texture','texture_video',{})]:
                with self.subTest(codec=name):
                    codec=importlib.import_module('adapt.'+module)
                    encoded=temp/name/'encoded'
                    decoded=temp/name/'decoded'
                    codec.encode(paths,[0,1/30],encoded,ffmpeg,threads=1,**options)
                    codec.decode(encoded,decoded,ffmpeg,threads=1)
                    files=sorted(decoded.glob('*.ply'))
                    self.assertEqual(len(files),2)
                    for p in files:
                        table=read_ply(p)
                        self.assertEqual(table.shape,(1024,62))
                        self.assertTrue(np.isfinite(table).all())


if __name__=='__main__':
    unittest.main()
