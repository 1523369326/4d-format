# 压缩基准

结果见 [REPORT.md](REPORT.md)；机器可读表为 `results/summary.csv` 和 `results/summary.json`。
所有测量已实际运行。每种方案使用数据集前 60 帧、4 个固定相机视角，完整比较 240 张图。

目录：

```text
third_party/                 三个原项目，源码未作编辑
adapt/                       格式适配、AV1 平面编码器和直接播放入口
benchmark/input_first60.json  前 60 帧绝对路径和时间戳
benchmark/environment.json   环境、输入 SHA-256 和三方来源信息
benchmark/requirements-lock.txt
benchmark/build/             实际构建的 TSOG wheel
benchmark/results/<method>/encoded/  可解码压缩文件及必要元数据
benchmark/results/<method>/decoded/  完整重建 PLY 和简化 sequence.json
benchmark/results/<method>/timing.json
benchmark/results/<method>/quality.json  每帧、每视角 MSE/PSNR
benchmark/results/cameras.json
benchmark/results/renders/   原始与重建图片样例
```

## 环境

压缩环境：工作区内 `.venv-compression/Scripts/python.exe`，Python 3.12.7、NumPy 2.2.6。
TSOG 声明 NumPy 2 依赖，因此未改动已有的 Conda GPU 环境。
三种项目均为 Python 项目，无额外 C++ 编译入口；已完成 Python 编译检查，TSOG 已构建 wheel 并安装。
GPU 渲染/播放复用 `E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe`，
PyTorch 2.5.1、CUDA runtime 12.1、已编译的 `diff_gaussian_rasterization`。
依赖版本已冻结，未修改系统 Python 或现有 Conda 环境。

在工作目录根运行：

```powershell
pwsh -NoProfile -File .\benchmark\run_all.ps1
```

已完成的编解码结果会被复用；重新测量某个方案请使用新的结果根目录：

```powershell
& '.\.venv-compression\Scripts\python.exe' .\benchmark\run_codec.py morton_av1 --results .\benchmark\run2
```

支持方法：`texture_h264`、`gsd_v1`、`gsd_v2`、`tsog`、`morton_av1`。
`run_codec.py` 的 GSD 原生 API 适配要求输入是同一目录自然排序的前 N 帧。
完整复现时先运行 `prepare.py`，再依次执行编解码、`render_psnr.py` 和 `summarize.py`。
`render_psnr.py` 必须使用上述 GPU Python，其余脚本使用工作区压缩环境。

## AV1 解码与播放

压缩结果为 `results/morton_av1/encoded/attributes.mkv` 及 `manifest.json`。
MKV 中是属性字节平面，不是普通相机画面；需反量化后按高斯场景渲染。

独立解码回 PLY，输出新目录：

```powershell
& '.\.venv-compression\Scripts\python.exe' -m adapt.av1_planes decode --encoded .\benchmark\results\morton_av1\encoded --output .\benchmark\av1_decoded
```

直接从 MKV 解码并调用 CUDA 高斯栅格器播放，无需先保存 PLY：

```powershell
& 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe' -m adapt.play_av1
```

按 `1`～`4` 切换视角，空格暂停，Esc 退出。目标节奏为 30 FPS，实际速度取决于解码、CPU→GPU 传输和渲染。
无窗口验证命令：`-m adapt.play_av1 --headless --limit 2`。

另外提供 2 秒普通视频预览：

- `results/playback_reference.mp4`：源 PLY 渲染。
- `results/playback_morton_av1.mp4`：AV1 解码后的 PLY 渲染。
- `results/playback_comparison.mp4`：左侧源、右侧 AV1 重建。

预览 MP4 自身使用 H.264 展示，不参与 PSNR；PSNR 来自保存 PNG/MP4 之前的浮点图像。

## 验证

- `test_adapters.py`：全部五种方案的实际编解码冒烟测试，以及纹理批量填充对原逐像素实现的精确等价比较。
- TSOG 上游 22 项 Python 测试已通过。
- AV1 全量 60 帧的灰度字节 atlas 均与输入 SHA-256 一致。
- 每种方法解码点数、帧数和时间戳检查通过，质量测量覆盖全部前 60 帧。

这是一组参数的一次实测，没有在码率/画质曲线上调参，也没有重复测量取均值。
