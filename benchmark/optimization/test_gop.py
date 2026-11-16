"""GOP format tests: topology, row shuffle, quaternion sign, singleton, empty motion."""
from runtime import configure_environment,pin_cpu,audit_ffmpeg
configure_environment()
_,mask=pin_cpu()
audit_ffmpeg(mask,[])
from pathlib import Path
import sys
import hashlib
import io
import lzma
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from adapt import av1_gop as codec
from adapt.common import read_ply,save_frame,read_json,write_json
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
                    self.assertEqual(config['version'],5)
                    self.assertTrue(config['preserves_input_row_order'])
                    lo,hi=np.asarray(config['mins']),np.asarray(config['maxs'])
                    expected=[]
                    for group in config['gops']:
                        base=tables[group['start_frame']]
                        for frame in range(group['start_frame'],group['start_frame']+group['frame_count']):
                            table=np.zeros((len(base),62),np.float32)
                            table[:,COLUMNS]=codec.dequantize(codec.quantize(tables[frame][:,COLUMNS],lo,hi),lo,hi)
                            expected.append(table)
                    actual=list(codec.iter_tables(folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'))
                    self.assertEqual(len(actual),len(tables))
                    for (frame,timestamp,table),reference in zip(actual,expected):
                        np.testing.assert_array_equal(table,reference)
                        self.assertEqual(timestamp,frame/30)
                    self.assertEqual(config['gops'][-1]['dynamic_count'],0)
                    self.assertGreaterEqual(len(config['gops']),3)
                    # Re-express the temporary fixture as a historical v4 stream.
                    # Its video is unchanged, but v4 had no permutation sidecar.
                    sidecar=folder/config['sidecar']
                    payload=sidecar.read_bytes()
                    if config['sidecar_codec']=='lzma':
                        payload=lzma.decompress(payload)
                    with np.load(io.BytesIO(payload),allow_pickle=False) as stored:
                        arrays={name:stored[name] for name in stored.files if not name.endswith('_order')}
                    output=io.BytesIO()
                    if config['sidecar_codec']=='lzma':
                        np.savez(output,**arrays)
                        payload=lzma.compress(output.getvalue())
                    else:
                        np.savez_compressed(output,**arrays)
                        payload=output.getvalue()
                    sidecar.write_bytes(payload)
                    config['version']=4
                    config.pop('preserves_input_row_order')
                    config['sidecar_sha256']=hashlib.sha256(payload).hexdigest()
                    write_json(folder/'manifest.json',config)
                    legacy=list(codec.iter_tables(folder,ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'))
                    for frame,_,table in legacy:
                        group=next(g for g in config['gops'] if g['start_frame']<=frame<g['start_frame']+g['frame_count'])
                        base=tables[group['start_frame']]
                        changed=np.zeros(len(base),bool)
                        for j in range(group['start_frame'],group['start_frame']+group['frame_count']):
                            changed|=np.any(tables[j][:,:3]!=base[:,:3],axis=1)
                            changed|=np.any(tables[j][:,58:62]!=base[:,58:62],axis=1)
                        a,b=np.flatnonzero(~changed),np.flatnonzero(changed)
                        a=a[morton_order(base[a,:3],lo[:3],hi[:3])]
                        b=b[morton_order(base[b,:3],lo[:3],hi[:3])]
                        np.testing.assert_array_equal(table,expected[frame][np.concatenate((a,b))])


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        unittest.main()
