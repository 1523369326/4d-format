"""Build an evidence-based report; never write the historical benchmark REPORT.md."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(suite,method):
    summary=read(suite/'summary.json')
    row=next(r for r in summary['rows'] if r['method']==method)
    repetitions=row['repetitions']
    if repetitions<3 or not row['decoded_identical_across_repetitions']:
        raise ValueError(f'Insufficient consistent repetitions: {suite}/{method}')
    reports=[read(suite/f'rep_{i:02d}'/method/'timing.json') for i in range(1,repetitions+1)]
    for report in reports:
        if report['status']!='completed' or report['frames']!=60 or report['affinity_mask'] != 1<<report['logical_cpu'] or report['cpu_math_threads']!=1:
            raise ValueError('Incomplete run or CPU execution policy violation')
        if report['input_fingerprint_sha256']!=row['input_fingerprint_sha256'] or report['source_tree_sha256']!=row['source_tree_sha256']:
            raise ValueError('Source/input fingerprint changed across repetitions')
        if any(report[key]!=reports[0][key] for key in ['codec_sha256','runner_sha256','runtime_sha256','logical_cpu','options']):
            raise ValueError('Codec, harness, options or CPU changed across repetitions')
        if any(e['affinity_mask']!=report['affinity_mask'] for e in report['child_process_audit']):
            raise ValueError('Codec child affinity mismatch')
    quality=read(suite/'rep_01'/method/'quality.json')
    if quality['images_compared']!=240 or quality['frames']!=60:
        raise ValueError('Incomplete quality evaluation')
    return {**row,'suite':str(suite.resolve()),'psnr_db':quality['pooled_psnr_db'],
            'per_view_psnr_db':quality['per_view_pooled_psnr_db'],'quality_images':quality['images_compared'],
            'logical_cpu':reports[0]['logical_cpu'],'codec_sha256':reports[0]['codec_sha256'],
            'encode_details':reports[0]['encode_details'],'equivalence':None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--tsog-baseline',type=Path,help='Optional newer isolated TSOG repetitions under the same CPU policy')
    p.add_argument('--candidate',type=Path,action='append',default=[],help='SUITE;METHOD;LABEL')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    baseline=summarize(args.baseline,'morton_av1')
    baseline['label']='AV1 v1 单 CPU 基线'
    tsog=summarize(args.tsog_baseline or args.baseline,'tsog')
    tsog['label']='TSOG / OpsiClear Python 单 CPU 基线'
    rows=[baseline,tsog]
    for item in args.candidate:
        suite,method,label=str(item).split(';',2)
        suite=Path(suite)
        candidate=summarize(suite,method)
        candidate['label']=label
        equivalence=suite/'rep_01'/method/'equivalence.json'
        if equivalence.exists():
            candidate['equivalence']=read(equivalence)
        rows.append(candidate)
    reference=(baseline['input_fingerprint_sha256'],baseline['logical_cpu'])
    for row in rows:
        if (row['input_fingerprint_sha256'],row['logical_cpu'])!=reference:
            raise ValueError('Methods did not use the same inputs and logical CPU')
        row['quality_pass_vs_tsog']=row['psnr_db']>=tsog['psnr_db']-.1 and all(row['per_view_psnr_db'][v]>=p-.1 for v,p in tsog['per_view_psnr_db'].items())
        row['tsog_ratios']={'bytes':row['encoded_bytes']['median']/tsog['encoded_bytes']['median'],
                            'encode_time':row['encode_seconds']['median']/tsog['encode_seconds']['median'],
                            'decode_time':row['decode_seconds']['median']/tsog['decode_seconds']['median']}
        row['beats_tsog_medians']=row['quality_pass_vs_tsog'] and all(value<=1 for value in row['tsog_ratios'].values())
        row['beats_tsog_outside_observed_time_ranges']=row['quality_pass_vs_tsog'] and row['encoded_bytes']['max']<=tsog['encoded_bytes']['min'] and all(row[metric]['max']<=tsog[metric]['min'] for metric in ['encode_seconds','decode_seconds'])
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,
            'quality_tolerance_db':.1,'cache_policy':'OS cache not flushed; input hashed before cases; independent serial processes; min/max reported',
            'limitations':['TSOG anchor is the existing OpsiClear independent Python implementation, not the paper authors official TypeScript implementation.','One scene, first 60 frames; no cross-scene generalization claim.','Three runs describe observed variation, not a statistical confidence interval.','Other user workloads can affect CPU clocks and storage; review contamination notes.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# 单 CPU AV1 优化实测','',
           '固定前 60 帧、完整 4 视角（240 图）。编解码限制到同一个逻辑 CPU，数学库/FFmpeg 单线程，无 GPU；CUDA 仅用于计时外质量评测。每个方法 3 次独立进程串行测量，表中为中位数。','',
           '| 方法 | MB | 编码 s | 解码 s | PSNR dB | 相对 TSOG 大小 / 编码 / 解码 |',
           '|---|---:|---:|---:|---:|---|']
    for row in rows:
        r=row['tsog_ratios']
        lines.append(f'| {row["label"]} | {row["encoded_bytes"]["median"]/1e6:.3f} | {row["encode_seconds"]["median"]:.3f} | {row["decode_seconds"]["median"]:.3f} | {row["psnr_db"]:.5f} | {r["bytes"]:.3f} / {r["encode_time"]:.3f} / {r["decode_time"]:.3f} |')
    lines+=['','## 观测波动','', '| 方法 | 编码 min–max s | 解码 min–max s |', '|---|---:|---:|']
    for row in rows:
        a,b=row['encode_seconds'],row['decode_seconds']
        lines.append(f'| {row["label"]} | {a["min"]:.3f}–{a["max"]:.3f} | {b["min"]:.3f}–{b["max"]:.3f} |')
    lines+=['','## 结论与边界','']
    for row in rows[2:]:
        result='大小、编码、解码和质量均达到 TSOG 门槛，且计时区间不重叠' if row['beats_tsog_outside_observed_time_ranges'] else ('中位数均达到 TSOG 门槛，但仍需关注计时波动' if row['beats_tsog_medians'] else '部分维度仍未达到 TSOG；不能称为全面超过')
        lines.append(f'- {row["label"]}：{result}。')
        if row['equivalence']:
            lines.append(f'  60 帧全部 62 列属性与 v1 逐值精确一致（按已验证排列比较），完整保留 SH3。')
    lines+=['','- 历史多线程数据保存在 benchmark/results，未覆盖；不能与本表直接作速度胜负比较。',
            '- TSOG 锚点沿用仓库现有 OpsiClear 独立 Python 实现，不是论文作者的官方 TypeScript 代码；本表结论只针对这一明确实现。',
            '- 文件大小包含视频及全部解码必需 sidecar/manifest；编码包含统计、排序、量化、码本、读写；解码包含标准 PLY 写盘。',
            '- 每个方法质量在 rep_01 测量，其他重复结果须逐帧 PLY SHA-256 相同才可复用质量结论。',
            '- 单场景实测不代表其他场景。三次 min/max 不是统计置信区间；机器温度与其他负载影响需结合日志解释。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    eligible=[r for r in rows[2:] if r['quality_pass_vs_tsog'] and r['equivalence'] and r['equivalence']['status']=='passed']
    if eligible:
        best=min(eligible,key=lambda row:row['encoded_bytes']['median'])
        (args.output/'best.json').write_text(json.dumps({'label':best['label'],'suite':best['suite'],'method':best['method'],
            'codec_sha256':best['codec_sha256'],'beats_tsog_medians':best['beats_tsog_medians'],
            'selection':'smallest fully validated v1-equivalent candidate above TSOG quality gate; report all time tradeoffs'},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('\n'.join(lines[:12]),flush=True)


if __name__=='__main__':
    main()
