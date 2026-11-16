"""Combine measured codec timings and four-view image PSNR, plus preview videos."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapt.common import read_json,write_json,run_ffmpeg

NAMES={'texture_h264':'4DGS-compression / FFV1+H.264',
       'gsd_v1':'4dgs-converter / GSD v1 (SH3)',
       'gsd_v2':'4dgs-converter / GSD v2 (SH0)',
       'tsog':'TSOG / WebP (SH3)', 'morton_av1':'Morton + uint16 planes + AV1 (SH3)'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=ROOT/'benchmark/results')
    p.add_argument('--ffmpeg',type=Path,default=ROOT/'tools/ffmpeg-win-x86_64-v9.0.2.exe')
    p.add_argument('--skip-previews',action='store_true')
    args=p.parse_args()
    rows=[]
    for method in NAMES:
        timing=read_json(args.results/method/'timing.json')
        quality=read_json(args.results/method/'quality.json')
        if timing['status']!='completed' or quality['images_compared']!=240:
            raise ValueError(f'Incomplete benchmark: {method}')
        row={'method':method,'name':NAMES[method],'sh_degree':timing['encode_details']['sh_degree'],
            'encoded_MB':timing['encoded_bytes']/1e6,'compression_ratio':timing['compression_ratio'],
            'encode_seconds':timing['encode_seconds'],'decode_seconds':timing['decode_seconds'],
            'encode_ms_per_frame':timing['encode_ms_per_frame'],'decode_ms_per_frame':timing['decode_ms_per_frame'],
            'psnr_db':quality['pooled_psnr_db'],'mean_image_psnr_db':quality['mean_image_psnr_db'],
            **{f'psnr_{v}_db':q for v,q in quality['per_view_pooled_psnr_db'].items()}}
        rows.append(row)
    write_json(args.results/'summary.json',{'input_frames':60,'views':4,'rows':rows})
    output=io.StringIO(newline='')
    writer=csv.DictWriter(output,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    (args.results/'summary.csv').write_text(output.getvalue(),encoding='utf-8-sig')
    lines=['# high-discussion 前 60 帧压缩基准','',
        '输入 60 帧、30 FPS、约 2 秒；原始二进制 PLY 共 1,483,526,368 字节。',
        '所有压缩任务在同一个工作区 Python 3.12 环境中串行运行，GPU 渲染在编解码计时之外。','',
        '| 方法 | 大小 MB | 压缩比 | 编码 s | 解码 s | 四视角 PSNR dB |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["name"]} | {row["encoded_MB"]:.2f} | {row["compression_ratio"]:.2f}x | '
                     f'{row["encode_seconds"]:.2f} | {row["decode_seconds"]:.2f} | {row["psnr_db"]:.2f} |')
    lines+=['','## 四个视角','',
        '| 方法 | 源相机 | 右移相机 | 左移相机 | 上移相机 |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['name']+' | '+' | '.join(f'{row[f"psnr_{v}_db"]:.2f}' for v in ['source','right','left','up'])+' |')
    lines+=['','## 测量规则与限制','',
        '- PSNR 使用同一个 CUDA 3DGS 栅格器，分辨率 1222×684，黑色背景，三阶 SH，关闭抗锯齿。',
        '- 使用源相机，以及在相机局部坐标右移 0.2、左移 0.2、上移 0.15 场景单位的三个视角；偏移相机朝向源相机前方 2.5 单位处。',
        '- 每种方案比较 60×4=240 张图。先将浮点 RGB 裁剪到 [0,1]，再计算全图、全部 RGB 通道的 MSE；没有掩膜、没有 PNG 8-bit 量化参与指标。',
        '- 主表 PSNR = -10 log10(全部 240 张图的平均 MSE)。逐图 PSNR 的算术均值也保存在 summary.csv 和质量 JSON 中。',
        '- 编码时间包含读取 PLY、排序/量化/码本构建、格式编码、元数据和输出文件写入。解码时间包含读取压缩文件、重建以及标准二进制 PLY 写入。外层进程启动、适配层模块导入、GPU 渲染不计入；编解码函数内按需导入依赖的时间计入。',
        '- 采用一次完整实测；未清空操作系统文件缓存。结果受缓存、CPU/GPU 温度和后台负载影响，不代表重复试验的统计结论。',
        '- 文件大小含播放必需元数据，不含原始 PLY、重建 PLY、日志和测量报告。MB 为 1,000,000 字节。',
        '- 4DGS-compression 使用原属性分组、量化和解码逻辑；适配层固定 atlas 尺寸以支持点数变化，并将逐像素 Python 写入改为等价 NumPy 批量填充。该时间不是未经适配的上游 CLI 时间。',
        '- GSD v1 使用完整 SH3，位置 FP32，其余 FP16，无裁剪；GSD v2 使用原 VQ256 编码，仅保留 SH0。两者均调用原生转换 API，2 个工作进程。',
        '- TSOG 使用原生 encode/decode，按 10 帧分包，离散帧 ID；SH 码本由适配层用精确唯一向量构建，没有额外聚类损失。',
        '- Morton/AV1 使用固定全序列范围的 21-bit Morton 排序、全部 59 个有效属性的 16-bit 标量量化、高低字节灰度平面；一帧视频对应一个 PLY。libaom-av1 cpu-used=8、CRF0、4 线程，视频层逐帧 SHA-256 校验无损；属性量化仍有损。',
        '- Morton 排序改变点行顺序，不影响高斯场景本身；GSD 同样重排点。解码后点数和时间戳保持不变。',
        '- VQ 采样沿用上游随机采样实现。本次每个结果对应其实际生成文件，而不是多个随机结果中择优。',
        '- 原始截图、重建截图、每帧每视角 MSE/PSNR、相机矩阵、环境锁文件和源码 SHA-256 指纹均保留在 benchmark 中；原三方目录没有 Git 版本信息。',
        '- 本次 PSNR 衡量压缩重建相对于输入 PLY 的损失，不衡量输入 PLY 相对于真实拍摄视频的重建质量。','',
        '具体环境见 `environment.json`，相机见 `results/cameras.json`，逐图指标见各方案 `quality.json`。']
    (ROOT/'benchmark/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not args.skip_previews:
        for name in ['reference','morton_av1']:
            run_ffmpeg(args.ffmpeg,['-y','-framerate','30','-i',args.results/'renders'/f'playback_{name}'/'frame_%06d.png',
                '-frames:v','60','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','4',
                args.results/f'playback_{name}.mp4'])
        run_ffmpeg(args.ffmpeg,['-y','-i',args.results/'playback_reference.mp4','-i',args.results/'playback_morton_av1.mp4',
            '-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','18',
            '-pix_fmt','yuv420p','-threads','4',args.results/'playback_comparison.mp4'])
    print('\n'.join(lines[:14]),flush=True)


if __name__=='__main__':
    main()
