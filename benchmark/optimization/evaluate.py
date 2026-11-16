"""Run the frozen CUDA quality evaluator, entirely outside codec timing."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--methods',nargs='+',required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    renderer_path = args.source.resolve()/'benchmark/render_psnr.py'
    sys.path.insert(0,str(args.source.resolve()))
    spec = importlib.util.spec_from_file_location('frozen_quality_renderer',renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.ROOT = ROOT  # Original immutable dataset/camera, not a copied 1.5GB input.
    camera = ROOT/'dataset/high_discussion/camera.json'
    provenance = {'renderer_source':str(renderer_path),'renderer_sha256':hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
                  'camera_sha256':hashlib.sha256(camera.read_bytes()).hexdigest(),'python':sys.executable,
                  'scope':'CUDA rendering and evaluation ONLY; never included in encode/decode time',
                  'methods':args.methods,'sequence':str(ROOT/'benchmark/input_first60.json')}
    sys.argv = [str(renderer_path),'--results',str(args.results.resolve()),'--methods',*args.methods]
    renderer.main()
    for method in args.methods:
        quality = json.loads((args.results/method/'quality.json').read_text())
        if quality['images_compared'] != 240 or quality['frames'] != 60 or quality['view_count'] != 4:
            raise RuntimeError(f'Incomplete quality evaluation: {method}')
        provenance.setdefault('quality_sha256',{})[method] = hashlib.sha256((args.results/method/'quality.json').read_bytes()).hexdigest()
    (args.results/'quality_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
