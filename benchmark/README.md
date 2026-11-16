# 压缩基准与 AV1 优化

最新完整结果见 [REPORT.md](REPORT.md) 和 `results_single_thread_v2/summary.csv`。
本轮全部七种方案都重新运行前 60 帧，编解码后端限制一个线程，CPU 编码，无 GPU 压缩。
GPU 只用于统一四视角渲染和播放，不计入编解码时间。

## 工作区

- `third_party/4DGS-compression`、`third_party/4dgs-converter`、`third_party/tsog`：三个原项目。
- `third_party/smv-codec`：本地编写的 SMV 原型，按用户要求迁入，并非下载的上游项目。
- `adapt/`：三方适配、原 AV1、优化 AV1、SMV 适配、播放器。
- 根目录 `av1_codec.py`：当前优化版的主实现与编码/解码命令入口。
- `benchmark/results/`：上轮结果，保留四线程等原始设置，没有覆盖。
- `benchmark/results_single_thread_v2/`：本轮七种方案完整资产、PLY、时间、画质及预览。
- `benchmark/experiments/av1_temporal_npz/`：优化中间实验，非最终版本。
- `benchmark/av1_structure_analysis.json`：旧 AV1 与 TSOG 的定量结构分析。
- `benchmark/environment.json`、`requirements-lock.txt`：输入校验和、源码指纹和环境。

压缩环境为 `.venv-compression/Scripts/python.exe`（Python 3.12.7、NumPy 2.2.6）。
渲染环境为 `E:/ProgramData/Anaconda/envs/cu121-3dgs/python.exe`，复用现有 CUDA 栅格器。

## 一键复现

从工作区根目录运行：

```powershell
pwsh -NoProfile -File .\benchmark\run_all.ps1
```

已完成、输入指纹匹配且线程数为 1 的结果会复用。重新测量时指定新根目录：

```powershell
pwsh -NoProfile -File .\benchmark\run_all.ps1 -Results benchmark/run3
```

单独测试：

```powershell
& '.\.venv-compression\Scripts\python.exe' .\benchmark\run_codec.py morton_av1_opt --threads 1 --results .\benchmark\run3
```

方法名为 `texture_h264`、`gsd_v1`、`gsd_v2`、`tsog`、`morton_av1`、`morton_av1_opt`、`smv`。
GSD 原生 API 适配要求输入是同一个目录自然排序的前 N 帧。

## 优化版 AV1

完整资产由 `morton_av1_opt/encoded/` 内三个文件组成：

- `motion.mkv`：动态点位置和旋转的高低字节残差平面，一帧对应一个 PLY 时间戳。
- `keyframes.npz.xz`：静态几何、不变属性、精确 SH 码本和索引、点序映射。
- `manifest.json`：量化范围、GOP、点数、时间戳、布局和完整性校验。

文件大小统计包含三者；没有把静态数据移到计费范围之外。保留原 AV1 v1 的 16-bit 量化，
没有减 SH 阶数或删点。全部视频字节和重建后的完整量化码表均逐帧校验。

解码成新的 PLY 目录：

```powershell
& '.\.venv-compression\Scripts\python.exe' .\av1_codec.py decode --encoded .\benchmark\results_single_thread_v2\morton_av1_opt\encoded --output .\benchmark\av1_v2_decoded
```

不落盘 PLY，直接从 MKV 和关键帧渲染播放：

```powershell
& 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe' -m adapt.play_temporal
```

按 1～4 切换相机，空格暂停，Esc 退出。无窗口验证可加 `--headless --limit 2`。
`play_av1.py` 保留为 v1 旧平面播放器；优化版使用 `play_temporal.py`。

结果目录内 `playback_comparison_optimized.mp4` 为左侧原始、右侧优化 AV1 的普通视频预览。
MP4 预览不参与 PSNR。SMV 的 `playback_smv.mp4` 也是解码后渲染预览。

## 验证与测量

- 全部 60 帧、4 个固定相机、1222×684；每种方案比较 240 张浮点 RGB 图像。
- 统一栅格器、SH3、黑背景、关闭抗锯齿；GSD v2 自身只保留 SH0，明确单列。
- 编码包含读取、分组、排序、量化、码本、压缩及写文件；解码包含重建和标准二进制 PLY 写入。
- 单次实测，OS 缓存未清空；没有把旧并行时间与新单线程时间混用。
- `test_adapters.py` 覆盖五种原方案及原纹理实现的等价向量化比较。
- `test_temporal.py` 覆盖模运算边界、点数变化、全静态/单帧 GOP，并与独立的 v1 编解码逐值比较。
- SMV 的 `test_roundtrip.py`、TSOG 上游测试也保留。

SMV 是 NPZ+HEVC 载荷加显式时间线扩展，原网页不能直接按原样播放。本轮用同一解码结果和相机测量，
不将“SMV 原型”表述为“原作者训练/压缩流程的完全复现”。
