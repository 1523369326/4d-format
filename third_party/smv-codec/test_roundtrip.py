"""Small end-to-end checks covering actual HEVC, row maps and timeline edges."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import smv_codec as codec


class RoundTrip(unittest.TestCase):
    def test_nonuniform_singleton_static_padding_and_variable_count(self):
        ffmpeg=next((codec.WORKSPACE/'tools').glob('ffmpeg*.exe'))
        ref=codec.reference_decoder()
        with tempfile.TemporaryDirectory(dir=codec.ROOT) as temp:
            root=Path(temp)
            rng=np.random.default_rng(5)
            base=np.zeros((20,62),dtype=np.float32)
            base[:,:3]=rng.normal(0,.1,(20,3))
            base[:,6:9]=rng.normal(0,.1,(20,3))
            base[:,9:54]=rng.normal(0,.01,(20,45))
            base[:,54]=-1
            base[:,55:58]=-3
            base[:,58]=1
            frames=[]
            for i in range(7):
                t=base.copy()
                if i>=1:
                    t[0,6]=.9
                if i in (2,3):
                    t[5:10,0]+=.02*(i-1)
                    theta=.1*(i-1)
                    t[5:10,58]=np.cos(theta/2)
                    t[5:10,61]=np.sin(theta/2)
                if i>=4:
                    t=np.concatenate((base,base[:2]),axis=0)
                    # The new stationary rows have indices beyond GOP-0's count.
                    t[20:,1]=.5
                p=root/f'{i}.ply'
                ref.write_ply(p,t)
                frames.append({'file':p.name,'timestamp':i/30})
            sequence=root/'sequence.json'
            sequence.write_text(json.dumps({'frames':frames}),encoding='utf-8')
            archive=root/'test.smv.zip'
            codec.encode(sequence,archive,ffmpeg,preset='ultrafast')
            report=codec.verify(archive,sequence,ffmpeg)
            self.assertEqual(report['frames_compared'],7)
            self.assertLess(report['errors']['position']['max'],1e-3)
            self.assertLess(report['errors']['rotation_degrees']['max'],1)
            self.assertLess(report['errors']['sh']['max'],.001)
            decoded=list(codec.decoded_frames(archive,ffmpeg))
            self.assertEqual([len(t) for _,_,t in decoded],[20]*4+[22]*3)
            out=root/'decoded'
            codec.decode(archive,out,ffmpeg)
            self.assertEqual(len(json.loads((out/'sequence.json').read_text())['frames']),7)
            with self.assertRaises(FileExistsError):
                codec.encode(sequence,archive,ffmpeg)


if __name__=='__main__':
    unittest.main()
