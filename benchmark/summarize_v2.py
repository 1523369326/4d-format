"""Single-thread comparison, measured AV1 representation analysis and previews."""
import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_json,write_json,run_ffmpeg

NAMES={'texture_h264':'4DGS-compression / FFV1+H264',
       'gsd_v1':'GSD v1 / SH3','gsd_v2':'GSD v2 / SH0','tsog':'TSOG / SH3',
       'morton_av1':'AV1 v1 / 完整属性平面',
       'morton_av1_opt':'AV1 v2 / 时序+精确码本',
       'smv':'SMV 本地原型 / 显式时间线'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=ROOT/'benchmark/results_single_thread_v2')
    p.add_argument('--skip-previews',action='store_true')
    args=p.parse_args()
    rows=[]
    for method,name in NAMES.items():
        timing=read_json(args.results/method/'timing.json')
        quality=read_json(args.results/method/'quality.json')
        if timing['status']!='completed' or timing['cpu_math_threads']!=1 or quality['images_compared']!=240:
            raise ValueError(f'Incomplete single-thread result: {method}')
        rows.append({'method':method,'name':name,'threads':1,'sh_degree':timing['encode_details']['sh_degree'],
            'encoded_MB':timing['encoded_bytes']/1e6,'compression_ratio':timing['compression_ratio'],
            'encode_seconds':timing['encode_seconds'],'decode_seconds':timing['decode_seconds'],
            'psnr_db':quality['pooled_psnr_db'],
            **{f'psnr_{k}_db':v for k,v in quality['per_view_pooled_psnr_db'].items()}})
    by_method={r['method']:r for r in rows}
    old,new=by_method['morton_av1'],by_method['morton_av1_opt']
    if new['psnr_db'] < old['psnr_db']-.05:
        raise ValueError('Optimized AV1 failed the PSNR preservation target (allowed decrease 0.05 dB)')
    analysis=read_json(ROOT/'benchmark/av1_structure_analysis.json')
    cfg0=read_json(args.results/'morton_av1/encoded/manifest.json')
    cfg1=read_json(args.results/'morton_av1_opt/encoded/manifest.json')
    assert cfg0['mins']==cfg1['mins'] and cfg0['maxs']==cfg1['maxs']
    validation=read_json(args.results/'morton_av1_opt/timing.json')['decode_details']
    assert validation['all_quantized_attribute_checksums_match']
    improvement={'old_MB':old['encoded_MB'],'optimized_MB':new['encoded_MB'],
        'smaller_by_factor':old['encoded_MB']/new['encoded_MB'],
        'reduction_percent':100*(1-new['encoded_MB']/old['encoded_MB']),
        'old_psnr_db':old['psnr_db'],'optimized_psnr_db':new['psnr_db'],
        'psnr_delta_db':new['psnr_db']-old['psnr_db'],
        'all_60_quantized_attribute_tables_match_v1_quantizer':True,'threads':1}
    write_json(args.results/'summary.json',{'frames':60,'views':4,'rows':rows,'av1_improvement':improvement})
    write_json(args.results/'av1_improvement.json',improvement)
    with (args.results/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines=['# 前 60 帧单线程压缩基准与 AV1 优化','',
        '输入为 high-discussion 第 0～59 帧，共 1,483,526,368 字节，30 FPS，约 2 秒。',
        '**所有编解码重新实测，均为单线程 CPU；没有通过多线程或 GPU 加速。GPU 仅用于计时之外的统一画质渲染。**','',
        '| 方法 | MB | 压缩比 | 编码 s | 解码 s | 四视角 PSNR dB |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["name"]} | {row["encoded_MB"]:.2f} | {row["compression_ratio"]:.2f}x | '
                     f'{row["encode_seconds"]:.2f} | {row["decode_seconds"]:.2f} | {row["psnr_db"]:.2f} |')
    lines+=['','## 为什么旧 AV1 比 TSOG 大','',
        f'上轮旧 AV1 为 {analysis["old_av1_bytes"]/1e6:.2f} MB、{analysis["old_av1_psnr_db"]:.2f} dB；'
        f'TSOG 为 {analysis["tsog_bytes"]/1e6:.2f} MB、{analysis["tsog_psnr_db"]:.2f} dB，二者并不在同一画质点。','',
        f'1. 每帧独立 Morton 排序破坏像素槽与高斯点的时序对应。仅第 0→1 帧，'
        f'{analysis["frame0_to_frame1"]["fraction_sorted_slots_with_different_point_identity"]*100:.2f}% 的槽位换了点，'
        f'{analysis["frame0_to_frame1"]["fraction_changed_codes_independent_morton"]*100:.2f}% 的属性码改变；'
        f'固定同一 Morton 顺序后只有 {analysis["frame0_to_frame1"]["fraction_changed_codes_fixed_morton"]*100:.2f}% 的码改变。',
        '2. 将 45 个高阶 SH 分量逐点展开，占 59 个有效属性的 76.27%；每个点用 90 字节保存高阶 SH。'
        '三个 GOP 实际只有 1870/1848/1831 种量化 SH 向量，可改为两字节索引加一次性精确码本。'
        '上轮 TSOG 的 SH 码本与索引总共仅约 1.04 MB。',
        '3. 颜色、SH、透明度和缩放在一个 GOP 内不变。每组还有约 6.7 万个位置与旋转均静止的点。'
        '旧方案仍然每帧铺满全部属性，依赖视频编码器自行发现这些高维重复。',
        '4. 全部属性一律 16 位、高低字节分别铺平，产生大量难以预测的低位字节；TSOG 对旋转、缩放、DC、alpha 等使用更紧凑的量化表示。'
        '因此体积差距主要来自编码器之前的数据表示，而非 AV1 与 WebP 的名称。','',
        '这些结论由本地输入、码流和源码测量得出；分析原始数据见 `av1_structure_analysis.json`。','',
        '## 本次优化及画质保证','',
        '- 保持旧方案同一个全序列 min/max、同一套 16-bit 量化公式，未降低 SH 阶数，未删点，也未降低属性位数。',
        '- 按不可变属性变化划分 GOP；每组只在关键帧做一次 Morton 排序，组内复用点序。',
        '- SH 用精确唯一向量码本，避免有损聚类；不变属性与静态几何每组只存一次。',
        '- 视频仅编码动态点的位置与四元数：相对前一帧做模 65536 的有符号差分、zigzag，再拆高低字节。',
        f'- 灰度视频 atlas 从 3852×3210 缩至 {cfg1["width"]}×{cfg1["height"]}。'
        '仍然一个视频帧对应一个 PLY 时间戳，解码时用关键帧数据补全整个高斯场景。',
        '- 静态关键帧与码本为 NPZ+单线程 LZMA2，利用跨 GOP 的重复；运动仍为无损 AV1/MKV。'
        '报告体积包含视频、关键帧和全部播放必需元数据，不是只统计更小的视频。',
        '- 对视频 atlas 和重建后完整 59 属性量化码表分别逐帧 SHA-256 校验；全部 60 帧通过，证明没有引入新的属性量化损失。',
        f'- 实测从 {old["encoded_MB"]:.2f} MB 降至 {new["encoded_MB"]:.2f} MB，'
        f'减少 {improvement["reduction_percent"]:.2f}%（{improvement["smaller_by_factor"]:.2f} 倍）；'
        f'PSNR 从 {old["psnr_db"]:.4f} 到 {new["psnr_db"]:.4f} dB。',
        '- 恢复原始行序与旧方案每帧 Morton 行序可能影响深度完全相同的点的栅格排序；'
        '是否影响图像以这次四视角 PSNR 为准，而不是只凭量化码一致推断。','',
        '## SMV','',
        '`smv-codec` 已迁入 `third_party/smv-codec` 并修复引用。这是本地编写的 SMV 载荷编码原型，'
        '不是新下载的上游项目；按用户目录组织要求一并放置。',
        'SMV 使用 NPZ 关键帧、8-bit SH/DC/opacity/rotation 量化、16-bit 缩放、HEVC 无损运动视频。'
        '其显式、无淡入淡出的 GOP 时间线是扩展，原网页不能直接按原样播放该包。'
        '本次实际解码后以同一栅格器计算四视角 PSNR，不能用原网站格式兼容性代替画质验证。','',
        '## 测量规则','',
        '- 同一个 Python 3.12 压缩环境，方法串行；NumPy/BLAS/OpenMP、视频编解码、图像滤镜均限制一个工作线程。'
        'GSD 为一个工作进程；TSOG 的后台线程池替换为同步执行器，未并行生成图片。',
        '- libaom-av1 使用 cpu-used=8、CRF0、gray、threads=1、row-mt=0；SMV x265 禁用线程池，使用一个帧线程。',
        '- 编码包含读 PLY、分析、排序、量化、码本、压缩及写文件；解码包含读压缩文件、重建及标准 PLY 写入。'
        '外层进程启动、适配层导入和 GPU 渲染不计入，函数内按需导入计入。',
        '- 每种方法 60 帧×4 视角共 240 张图，1222×684，黑背景，三阶 SH 栅格器，关闭抗锯齿。'
        'GSD v2 本身只保留 SH0；其余方案保留 SH3。',
        '- 使用源相机和右移 0.2、左移 0.2、上移 0.15 场景单位的三个固定相机；参数与上轮相同，见 cameras.json。',
        '- PSNR 为浮点 RGB 裁剪 [0,1] 后的全图、全部通道平均 MSE，再取 -10log10；PNG/MP4 仅供预览，不参与数值计算。',
        '- 单次实测，未清空 OS 缓存；不是重复试验的统计均值。数据来自 SMV 解码后序列，组内静态/外观冗余较强，不能据此保证所有独立重建 PLY 都有同样收益。',
        '- 旧四线程结果完整保留在 benchmark/results；本表所有时间均来自新的一线程实测，未混用旧时间。',
        '- 编码参数参考 [FFmpeg 官方文档](https://ffmpeg.org/ffmpeg-codecs.html#libaom_002dav1)。','',
        '逐图指标见各方案 quality.json；CSV、时间日志、解码 PLY、量化校验、视频和预览均在本报告同目录。']
    report='\n'.join(lines)+'\n'
    (args.results/'REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'benchmark/REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'benchmark/AV1_OPTIMIZATION.md').write_text(report,encoding='utf-8')
    if not args.skip_previews:
        ffmpeg=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe'
        for name in ['reference','morton_av1','morton_av1_opt','smv']:
            run_ffmpeg(ffmpeg,['-y','-filter_threads','1','-framerate','30','-i',args.results/'renders'/f'playback_{name}'/'frame_%06d.png',
                '-frames:v','60','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','1',args.results/f'playback_{name}.mp4'])
        run_ffmpeg(ffmpeg,['-y','-threads','1','-i',args.results/'playback_reference.mp4','-i',args.results/'playback_morton_av1_opt.mp4',
            '-filter_complex_threads','1','-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264',
            '-crf','18','-pix_fmt','yuv420p','-threads','1',args.results/'playback_comparison_optimized.mp4'])
    print('\n'.join(lines[:16]),flush=True)


if __name__=='__main__':
    main()
