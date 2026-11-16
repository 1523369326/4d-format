"""Create a reproducible first-N frame input manifest and capture provenance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_json,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=int,default=60)
    args=p.parse_args()
    source=ROOT/'dataset/high_discussion/frames'
    paths=sorted(source.glob('frame_*.ply'))[:args.frames]
    if len(paths)!=args.frames:
        raise ValueError('Not enough input frames')
    frames=[{'file':str(p),'timestamp':i/30} for i,p in enumerate(paths)]
    write_json(ROOT/'benchmark/input_first60.json',{'frames':frames})
    inputs=[{'file':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    projects={}
    for name in ['4DGS-compression','4dgs-converter','tsog','smv-codec']:
        repo=ROOT/'third_party'/name
        commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
        changes=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True)
        source_hash=hashlib.sha256()
        source_files=[]
        for file in sorted(repo.rglob('*')):
            relative=file.relative_to(repo)
            if (not file.is_file() or any(part in ('.git','__pycache__','build','output') or part.endswith('.egg-info') for part in relative.parts)
                    or file.suffix.lower() not in ('.py','.js','.json','.md','.toml','.txt','.ps1','.bat')):
                continue
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            source_hash.update((relative.as_posix()+'\0'+digest+'\n').encode())
            source_files.append({'file':relative.as_posix(),'sha256':digest})
        projects[name]={'path':str(repo),'revision':commit.stdout.strip() if commit.returncode==0 else None,
                        'worktree_status':changes.stdout.strip() if changes.returncode==0 else None,
                        'source_tree_sha256':source_hash.hexdigest(),'source_files':source_files}
        if name=='smv-codec':
            projects[name]['provenance']='locally authored SMV payload/timeline prototype; relocated by user request, not an upstream checkout'
    packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','Pillow','opencv-python-headless','plyfile','lz4','threadpoolctl']}
    dataset_hash=hashlib.sha256(''.join(i['sha256']+'\n' for i in inputs).encode()).hexdigest()
    report={'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),
        'environment_snapshot_used':'2026-10-19T23:56:52+08:00; selected environment live-verified',
        'compression_environment':'workspace .venv-compression (Python 3.12), NumPy 2',
        'renderer_python':r'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe',
        'renderer':'preinstalled diff_gaussian_rasterization CUDA extension, PyTorch 2.5.1 CUDA 12.1',
        'packages':packages,'projects':projects,'inputs':inputs,'input_bytes':sum(i['bytes'] for i in inputs),
        'dataset_sha256':dataset_hash}
    write_json(ROOT/'benchmark/environment.json',report)
    print(json.dumps({'frames':len(paths),'input_bytes':report['input_bytes'],'packages':packages},indent=2))


if __name__=='__main__':
    main()
