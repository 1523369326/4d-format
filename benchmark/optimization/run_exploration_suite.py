"""Serial resumable repetitions. No renderer or concurrent experiments during timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import shutil

HERE = Path(__file__).resolve().parent


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['morton_av1', 'tsog'])
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--options', default='{}')
    p.add_argument('--sequence', type=Path)
    args = p.parse_args()
    if (HERE/'HOLD_NEW_SUITES').exists():
        print('New suites held at a stage boundary for experiment-plan revision; existing completed suites are preserved.',flush=True)
        raise SystemExit(75)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'source': str(args.source.resolve()), 'methods': args.methods, 'repetitions': args.repetitions,
              'candidate_sha256': hashlib.sha256(args.candidate.read_bytes()).hexdigest() if args.candidate else None,
              'options': json.loads(args.options), 'sequence': str(args.sequence.resolve()) if args.sequence else None,
              'runner_sha256': hashlib.sha256((HERE/'run_case_explore.py').read_bytes()).hexdigest(),
              'runtime_sha256': hashlib.sha256((HERE/'runtime.py').read_bytes()).hexdigest()}
    config_path = args.output/'suite_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Suite configuration changed; choose a fresh output directory')
    write(config_path, config)
    if args.candidate:
        frozen_candidate = args.output/'candidate_source.py'
        if frozen_candidate.exists():
            if hashlib.sha256(frozen_candidate.read_bytes()).hexdigest() != config['candidate_sha256']:
                raise ValueError('Frozen suite candidate changed')
        else:
            shutil.copy2(args.candidate, frozen_candidate)
        args.candidate = frozen_candidate
    state = {'status': 'running', 'config': config, 'cases': []}
    for repetition in range(1, args.repetitions+1):
        methods = args.methods if repetition % 2 else list(reversed(args.methods))
        for method in methods:
            case = args.output/f'rep_{repetition:02d}'/method
            timing = case/'timing.json'
            if timing.exists() and json.loads(timing.read_text())['status'] == 'completed':
                print(f'Reusing completed case {case}', flush=True)
            else:
                command = [sys.executable, str(HERE/'run_case_explore.py'), method, '--source', str(args.source), '--output', str(case), '--options', args.options]
                if args.candidate:
                    command += ['--candidate', str(args.candidate)]
                if args.sequence:
                    command += ['--sequence', str(args.sequence)]
                state['active_case'] = str(case)
                state['updated_utc'] = datetime.now(timezone.utc).isoformat()
                write(args.output/'suite_state.json', state)
                print(f'Starting {case}', flush=True)
                result = subprocess.run(command)
                if result.returncode:
                    state['status'] = 'failed'
                    write(args.output/'suite_state.json', state)
                    raise RuntimeError(f'Case failed: {case}')
            state['cases'].append(str(case))
    rows = []
    for method in args.methods:
        reports = [json.loads((args.output/f'rep_{i:02d}'/method/'timing.json').read_text()) for i in range(1,args.repetitions+1)]
        row = {'method': method, 'repetitions': len(reports)}
        for metric in ['encode_seconds', 'decode_seconds', 'encoded_bytes']:
            values = [r[metric] for r in reports]
            row[metric] = {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'values': values}
        row['decoded_identical_across_repetitions'] = all(r['decoded_frames'] == reports[0]['decoded_frames'] for r in reports)
        row['source_tree_sha256'] = reports[0]['source_tree_sha256']
        row['input_fingerprint_sha256'] = reports[0]['input_fingerprint_sha256']
        rows.append(row)
    write(args.output/'summary.json', {'rows': rows, 'config': config})
    state['status'] = 'completed'
    state.pop('active_case', None)
    write(args.output/'suite_state.json', state)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()

