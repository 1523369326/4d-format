"""Native TSOG API, in 10-frame chunks with discrete frame IDs and full SH."""
from pathlib import Path
import sys
import numpy as np
import importlib
from concurrent.futures import Future
from .common import ROOT, read_ply, activated_table, write_json, read_json, save_frame, save_sequence


class InlineExecutor:
    """Native TSOG pool interface executed synchronously, without worker threads."""
    def __init__(self,**kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False
    def submit(self,fn,*args,**kwargs):
        future=Future()
        try:
            future.set_result(fn(*args,**kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


def encode(paths,timestamps,folder,ffmpeg=None,chunk_size=10,threads=2):
    sys.path.insert(0,str(ROOT/'third_party/tsog/src'))
    from tsog import Scene,encode as native_encode
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for start in range(0,len(paths),chunk_size):
        batch=[read_ply(p) for p in paths[start:start+chunk_size]]
        counts=[len(t) for t in batch]
        table=np.concatenate(batch)
        del batch
        # The native API explicitly leaves palette construction to its caller.
        palette,labels=np.unique(table[:,9:54],axis=0,return_inverse=True)
        if len(palette)>65536:
            raise ValueError('TSOG SH palette exceeds 65536; clustering is needed for this input')
        ids=np.repeat(np.arange(len(counts),dtype=np.uint16),counts)
        scene=Scene(positions=table[:,:3],scales=np.exp(table[:,55:58]),quaternions=table[:,58:62],
            dc=table[:,6:9],opacity=1/(1+np.exp(-table[:,54])),frame_ids=ids,
            sh_palette=palette.reshape(-1,3,15).transpose(0,2,1),sh_labels=labels.astype(np.uint16),
            metadata={'timeline':{'N':len(counts)}})
        name=f'chunk_{len(chunks):03d}.tsog'
        native_module=importlib.import_module('tsog.codec')
        original_executor=native_module.ThreadPoolExecutor
        if threads==1:
            native_module.ThreadPoolExecutor=InlineExecutor
        try:
            native_encode(scene,folder/name,playback={'duration':len(counts)/30,'fps':30,'loop':False},
                          width=2048,workers=threads,webp_method=4)
        finally:
            native_module.ThreadPoolExecutor=original_executor
        chunks.append({'file':name,'start':start,'counts':counts})
        print(f'TSOG encoded {start+len(counts)}/{len(paths)} frames',flush=True)
    write_json(folder/'manifest.json',{'timestamps':timestamps,'chunks':chunks,'chunk_size':chunk_size})
    return {'sh_degree':3,'chunk_size':chunk_size,'implementation':'native TSOG encode; exact unique SH palette, discrete frame IDs',
            'synchronous_execution':threads==1}


def decode(folder,output,ffmpeg=None,threads=2):
    sys.path.insert(0,str(ROOT/'third_party/tsog/src'))
    from tsog import decode as native_decode
    folder=Path(folder)
    manifest=read_json(folder/'manifest.json')
    for chunk in manifest['chunks']:
        scene=native_decode(folder/chunk['file'])
        ids=np.asarray(scene.frame_ids)
        for local,count in enumerate(chunk['counts']):
            selected=ids==local
            if int(selected.sum())!=count:
                raise ValueError('TSOG frame-ID/count mismatch')
            rest=np.asarray(scene.sh_palette)[np.asarray(scene.sh_labels)[selected]].transpose(0,2,1).reshape(count,45)
            table=activated_table(np.asarray(scene.positions)[selected],np.asarray(scene.scales)[selected],
                np.asarray(scene.quaternions)[selected],np.asarray(scene.dc)[selected],np.asarray(scene.opacity)[selected],rest)
            save_frame(output,chunk['start']+local,table)
    save_sequence(output,manifest['timestamps'])
    return {'frame_count':len(manifest['timestamps']),'sh_degree':3}
