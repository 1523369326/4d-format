from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/smv-codec'))
from smv_codec import read_ply, PROPERTIES


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sequence_paths(sequence):
    sequence=Path(sequence).resolve()
    frames=read_json(sequence)['frames']
    return [(sequence.parent/f['file']).resolve() for f in frames], [f['timestamp'] for f in frames]


def save_frame(folder,index,table):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if table.ndim!=2 or table.shape[1]!=62 or not np.isfinite(table).all():
        raise ValueError(f'Invalid decoded table at frame {index}')
    path=folder/f'frame_{index:06d}.ply'
    header=('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(table)}\n'
            +''.join(f'property float {name}\n' for name in PROPERTIES)+'end_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        np.asarray(table,dtype='<f4').tofile(f)


def save_sequence(folder,timestamps):
    write_json(Path(folder)/'sequence.json',{'frames':[
        {'file':f'frame_{i:06d}.ply','timestamp':t} for i,t in enumerate(timestamps)]})


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def run_ffmpeg(ffmpeg,args):
    result=subprocess.run([str(ffmpeg),'-v','error','-nostdin',*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-6000:])
    return result.stdout


def activated_table(positions,scales,quats,dc,alpha,rest=None):
    table=np.zeros((len(positions),62),dtype=np.float32)
    table[:,:3]=positions
    table[:,6:9]=dc
    if rest is not None:
        table[:,9:54]=rest
    alpha=np.clip(np.asarray(alpha).reshape(-1),1e-7,1-1e-7)
    table[:,54]=np.log(alpha/(1-alpha))
    table[:,55:58]=np.log(np.maximum(scales,1e-20))
    q=np.asarray(quats)
    lengths=np.linalg.norm(q,axis=1,keepdims=True)
    table[:,58:62]=q/np.where(lengths>0,lengths,1)
    return table
