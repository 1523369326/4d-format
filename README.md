# 4d-format

AV1 4DGS 序列压缩与编解码基准。

主实现为根目录的 [`av1_codec.py`](av1_codec.py)。它包含已讨论的 GOP 固定 Morton 顺序、精确 SH 码本、
静态/不变属性复用、动态 uint16 差分、字节平面 AV1 编码及逆向解码。
完整资产为 `motion.mkv`、`keyframes.npz.xz`、`manifest.json`，格式保持 `morton-av1-temporal` version 2。

## 运行环境

首次克隆时初始化三个固定版本的第三方子模块：

```powershell
git clone --recurse-submodules https://github.com/1523369326/4d-format.git
cd 4d-format
python -m venv .venv-compression
& '.\.venv-compression\Scripts\python.exe' -m pip install -r requirements.txt
```

已经克隆时可执行 `git submodule update --init --recursive`。
FFmpeg、PLY 数据集、论文、虚拟环境及大型 benchmark 运行产物保留在本地，不随 Git 上传。
请安装支持 `libaom-av1` 的 FFmpeg，将其放到 `tools/ffmpeg-win-x86_64-v9.0.2.exe`，
或通过编码/解码命令的 `--ffmpeg` 指定可执行文件路径。
输入可以是你自己的标准 3DGS PLY 序列与时间戳 JSON。

使用工作区 `.venv-compression/Scripts/python.exe`。主实现复用 `adapt` 的通用 PLY/JSON/Morton 工具、
`third_party/smv-codec` 的分组扫描函数，以及 `benchmark/optimization/runtime.py` 的单 CPU 审计策略。
因此运行时保留这些工作区目录。CLI 在导入数值库前限制线程、隐藏 CUDA，并固定一个逻辑 CPU；
FFmpeg 子进程继承相同亲和性并核验，视频编码仅用软件 AV1。

## 编码

从 `format` 目录运行；输出目录必须是新目录：

```powershell
& '.\.venv-compression\Scripts\python.exe' .\av1_codec.py encode --sequence .\benchmark\input_first60.json --output .\output\discussion_av1
```

输入 JSON 为 `{"frames":[{"file":"frame_000000.ply","timestamp":0.0}]}`，路径相对于 JSON。
该分组流程在点数或 DC/SH/opacity/scale 的有序数组变化时开启新 GOP，最多 30 帧；
组内复用输入点行的对应关系。相邻帧属性频繁变化会产生更多关键帧，压缩收益依赖数据。

## 解码

```powershell
& '.\.venv-compression\Scripts\python.exe' .\av1_codec.py decode --encoded .\output\discussion_av1 --output .\output\discussion_decoded
```

输出标准 3DGS PLY 序列及只含路径、时间戳的 `sequence.json`。
原来的 `python -m adapt.av1_temporal ...` 和 `import adapt.av1_temporal` 仍然可用，引用同一份根目录实现。

## 播放与测试

```powershell
& 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe' -m adapt.play_temporal --encoded .\output\discussion_av1
& '.\.venv-compression\Scripts\python.exe' .\benchmark\test_temporal.py
```

播放器使用 CUDA 渲染；编码和解码不使用 GPU。按 1～4 切换相机、空格暂停、Esc 退出。
benchmark 的 `morton_av1_opt` 也直接加载根目录实现。

这次变更只整理主代码位置和运行入口，未修改量化、码流、GOP 或重建公式。
历史 `benchmark/results`、REPORT、environment、input 及优化快照保持原样。
持续研究的三次单 CPU 正式验收状态以 `benchmark/optimization/STATE.json` 和实验日志为准。
