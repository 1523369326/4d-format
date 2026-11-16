"""Create an immutable source copy plus fingerprints; never touch historical outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    files = []
    for directory in ['adapt', 'third_party']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.md', '.toml', '.txt', '.json'} and '__pycache__' not in path.parts:
                files.append(path)
    files += [p for p in sorted((ROOT / 'benchmark').glob('*')) if p.is_file()]
    # Include root codec entry points in future snapshots; existing snapshots stay unchanged.
    files += [p for p in sorted(ROOT.glob('*.py')) if p.is_file()]
    for relative in ['README.md', 'benchmark/optimization/runtime.py']:
        path = ROOT / relative
        if path.exists():
            files.append(path)
    if (ROOT / 'AGENTS.md').exists():
        files.append(ROOT / 'AGENTS.md')
    sources = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = dest / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise RuntimeError(f'Source changed while copying: {path}')
        sources.append({'path': relative.as_posix(), 'sha256': sha(target), 'bytes': target.stat().st_size})
    history = []
    for path in sorted((ROOT / 'benchmark/results').rglob('*')):
        if path.is_file():
            history.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    seq = json.loads((ROOT / 'benchmark/input_first60.json').read_text(encoding='utf-8-sig'))
    inputs = []
    for frame in seq['frames']:
        path = (ROOT / 'benchmark' / frame['file']).resolve()
        inputs.append({'path': str(path), 'timestamp': frame['timestamp'], 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'workspace': str(ROOT),
                'sources': sources, 'historical_results': history, 'inputs': inputs,
                'ffmpeg': {'path': str(ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'), 'sha256': sha(ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')}}
    manifest['source_tree_sha256'] = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'snapshot': str(dest), 'source_files': len(sources), 'historical_files': len(history), 'input_frames': len(inputs), 'source_tree_sha256': manifest['source_tree_sha256']}), flush=True)


if __name__ == '__main__':
    main()
