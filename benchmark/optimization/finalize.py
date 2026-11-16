"""Post-timing integrity checks, screening evidence, and recoverable experiment state."""
from datetime import datetime,timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def finalize(output):
    if read(HERE/'STATE.json').get('requires_input_row_order') and output.name=='stage_latest':
        raise RuntimeError('stage_latest is frozen historical evidence; use a fresh source-order report directory')
    report=read(output/'comparison.json')
    preserved=read(HERE/'baselines/historical_20261116/manifest.json')
    modified=[]
    for item in preserved['historical_results']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            modified.append(item['path'])
    source_changed=[]
    for item in preserved['sources']:
        path=HERE/'baselines/historical_20261116/source'/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            source_changed.append(item['path'])
    input_changed=[]
    for item in preserved['inputs']:
        path=Path(item['path'])
        if not path.is_file() or sha(path)!=item['sha256']:
            input_changed.append(item['path'])
    latest_path=HERE/'references/latest_1607'
    latest_changed=[]
    for item in read(latest_path/'manifest.json'):
        if sha(latest_path/item['path'])!=item['sha256']:
            latest_changed.append(item['path'])
    reference_manifest_same_as_original=sha(latest_path/'encoded/manifest.json')==sha(ROOT/'benchmark/results_single_thread_v2/morton_av1_opt/encoded/manifest.json')
    if not reference_manifest_same_as_original:
        latest_changed.append('encoded/manifest.json differs from supplied original')
    integrity={'historical_files_checked':len(preserved['historical_results']),'historical_modified':modified,
               'source_files_checked':len(preserved['sources']),'frozen_source_modified':source_changed,
               'input_frames_checked':len(preserved['inputs']),'input_modified':input_changed,'latest_reference_modified':latest_changed,
               'latest_encoded_manifest_matches_supplied_original':reference_manifest_same_as_original,
               'note':'Only files present in the original manifest are compared; other tasks may have added reports. Current live benchmark REPORT was separately updated by another task; the initial report is preserved in the source snapshot.'}
    write(output/'preservation_check.json',integrity)
    if source_changed or input_changed or latest_changed:
        raise RuntimeError('Immutable evidence changed; inspect preservation_check.json before promotion')
    latest=read(HERE/'runs/latest_temporal_reference/rep_01/temporal/timing.json')
    cached=read(HERE/'runs/latest_cached_decode/rep_01/temporal/timing.json')
    cached_proof={'all_latest_v2_ply_hashes_identical':cached['decoded_frames']==latest['decoded_frames'],
                  'same_encoder_source':cached['encode_details']['encoder_source_sha256']==latest['codec_sha256'],
                  'frame_count':len(cached['decoded_frames']),'reference':str(HERE/'runs/latest_temporal_reference'),
                  'candidate':str(HERE/'runs/latest_cached_decode')}
    write(output/'cached_decoder_equivalence.json',cached_proof)
    if not cached_proof['all_latest_v2_ply_hashes_identical'] or not cached_proof['same_encoder_source']:
        raise RuntimeError('Cached decoder failed to preserve the latest v2 output or encoder')
    live_path=ROOT/'av1_codec.py'
    before_path=HERE/'provenance/av1_codec_before_cached_decode.py'
    before_ast=ast.parse(before_path.read_text(encoding='utf-8-sig'))
    live_ast=ast.parse(live_path.read_text(encoding='utf-8-sig'))
    before_functions={node.name:node for node in before_ast.body if isinstance(node,ast.FunctionDef)}
    live_functions={node.name:node for node in live_ast.body if isinstance(node,ast.FunctionDef)}
    integration={'entry':str(live_path),'before_sha256':sha(before_path),'after_sha256':sha(live_path),
                 'encoder_semantically_unchanged':ast.dump(before_functions['encode'])==ast.dump(live_functions['encode']),
                 'cli_semantically_unchanged':ast.dump(before_functions['main'])==ast.dump(live_functions['main'])}
    env=os.environ.copy()
    env['CHECK_LIVE_TEMPORAL']='1'
    completed=subprocess.run([sys.executable,str(HERE/'test_cached_decode.py')],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    integration['compatibility_tests_passed']=completed.returncode==0
    integration['test_stdout']=completed.stdout
    integration['test_stderr']=completed.stderr
    write(output/'canonical_entry_integration.json',integration)
    if not all(integration[key] for key in ['encoder_semantically_unchanged','cli_semantically_unchanged','compatibility_tests_passed']):
        raise RuntimeError('Canonical entry integration verification failed; inspect canonical_entry_integration.json')
    a=read(HERE/'runs/noncodebook_sh_color_screen/rep_01/predictive/timing.json')
    b=read(HERE/'runs/noncodebook_v1_control_screen/rep_01/morton_av1/timing.json')
    if a['frames']!=10 or b['frames']!=10 or a['input_fingerprint_sha256']!=b['input_fingerprint_sha256']:
        raise RuntimeError('Non-codebook screening did not use the same fixed ten frames')
    screening={'frames':10,'repetitions':1,'full_benchmark':False,
               'all_decoded_ply_hashes_identical':a['decoded_frames']==b['decoded_frames'],
               'predictive':{key:a[key] for key in ['encoded_bytes','encode_seconds','decode_seconds']},
               'control':{key:b[key] for key in ['encoded_bytes','encode_seconds','decode_seconds']},
               'ratios':{key:a[key]/b[key] for key in ['encoded_bytes','encode_seconds','decode_seconds']},
               'interpretation':'Single-run fixed-subset control tests a reversible non-codebook SH transform only; no full-scene rate-distortion or statistically stable speed claim.'}
    if not screening['all_decoded_ply_hashes_identical']:
        raise RuntimeError('Reversible non-codebook transform changed decoded attributes')
    screening['decision']='promising representation reduction; consider a combined ablation' if screening['ratios']['encoded_bytes']<.98 else 'not promoted; no substantial subset size reduction'
    write(output/'noncodebook_screening.json',screening)
    robust=[row for row in report['rows'][2:] if row['beats_tsog_outside_observed_time_ranges'] and row['equivalence'] and row['equivalence']['status']=='passed']
    state=read(HERE/'STATE.json')
    state.update({'status':'validated_target_reached' if robust and not modified else 'continue_research',
                  'stage':'latest_reference_stage_complete','active_suite':None,'latest_report':str(output.relative_to(ROOT)),
                  'best_candidate':read(output/'best.json') if (output/'best.json').exists() else None,
                  'historical_preservation_passed':not modified,'updated_utc':datetime.now(timezone.utc).isoformat(),
                  'next':'Review measured Pareto candidates and cross-scene limits. Goal completion requires the verified target, not a single finished experiment.' if robust else 'Inspect remaining time/size gap, then select a new family from research cards rather than repeatedly tuning an unsuccessful variant.'})
    write(HERE/'STATE.json',state)
    with (HERE/'EXPERIMENTS.jsonl').open('a',encoding='utf-8') as stream:
        for row in report['rows'][2:]:
            stream.write(json.dumps({'id':Path(row['suite']).name,'status':'fully_validated','method':row['method'],'label':row['label'],
                'bytes_median':row['encoded_bytes']['median'],'encode_median_seconds':row['encode_seconds']['median'],
                'decode_median_seconds':row['decode_seconds']['median'],'psnr_db':row['psnr_db'],
                'beats_tsog_outside_observed_time_ranges':row['beats_tsog_outside_observed_time_ranges'],
                'evidence':str(output.relative_to(ROOT)/'comparison.json')},ensure_ascii=False)+'\n')
        stream.write(json.dumps({'id':'C1-noncodebook-SH','status':'screened','result':screening},ensure_ascii=False)+'\n')
    plan_path=HERE/'PLAN.md'
    plan=plan_path.read_text(encoding='utf-8')
    plan=plan.replace('- [ ] SH 去重消融和候选完成。','- [x] SH 去重消融和候选完成；已结合用户最新 16.07 MB 参考调整主比较。')
    plan=plan.replace('- [ ] 多方向筛选及最佳组合完成。','- [x] 本阶段多方向筛选及候选完整验收完成；具体收益与未达标项见 reports/stage_latest，不能由阶段结束推断总体目标达成。')
    plan+='\n阶段最终证据：reports/stage_latest/comparison.json、preservation_check.json、cached_decoder_equivalence.json、noncodebook_screening.json。研究扩展与未实施方向保留在 literature/RESEARCH_CARDS.md。\n'
    plan_path.write_text(plan,encoding='utf-8')
    print(json.dumps({'post_timing_integrity':integrity,'cached_decoder_proof':cached_proof,
                      'canonical_entry_integration':{key:value for key,value in integration.items() if not key.startswith('test_')},
                      'noncodebook_screening':screening,'robust_target_candidates':[r['label'] for r in robust],
                      'state':state},indent=2,ensure_ascii=False),flush=True)
