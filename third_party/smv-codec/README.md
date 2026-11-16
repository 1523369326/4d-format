# PLY 序列 → SMV 编码原型

本目录是本地编写的实验原型，按工作区组织要求移入 third_party，并非下载的上游仓库。

从标准 3DGS PLY 序列重新编码 NPZ 关键帧、球谐码本和 HEVC 运动视频，并能解码回 PLY。
编码器只读取输入 JSON 及其引用的 PLY，**不读取原始 discussion.zip 或运动视频**。

## 兼容范围

这是 **SMV 属性/运动载荷 + 显式时间线扩展**，不是原作者编码器，也不能保证还原原压缩包的字节。

- `quantized_gaussian_pruned.npz`、`gop_N_*.mkv`、运动量化 JSON 遵循本地保存的 SMV 网站解码公式。
- 解码和误差验证直接调用已有 `dataset/convert_smv_to_ply.py` 的属性、码本、运动解码函数。
- 新增 `manifest.json` 记录每个 GOP 的实际起始帧、长度、原点数和时间戳。无淡入淡出。
- **原网页目前不能直接正确播放这个包**：它忽略 manifest，假定固定 GOP 步长和重叠。
  对接播放器时可保留原来的属性/视频解码器，但需用 manifest 分派时间帧。
  文件名中的 `.smv.zip` 是本原型的命名，不代表一个已注册的官方文件扩展名。
- NPZ 额外保存 `gop_N_row_order`，本工具据此恢复原始点顺序。网站解码器会忽略该字段。

当前这批 PLY 的第一个稳定分组有 27 帧，后续多为 25 帧，末组为 23 帧。
这是前面导出时选择过渡区主导 GOP 的结果。显式时间线能够保留这 300 帧，避免错移时间。

## 输入假设

输入 JSON：

```json
{"frames":[
  {"file":"frames/frame_000000.ply","timestamp":0.0},
  {"file":"frames/frame_000001.ply","timestamp":0.03333333333333333}
]}
```

路径相对于输入 JSON；时间戳单位为秒，必须严格递增。支持二进制大小端标准 3DGS PLY、
degree-3 SH、logit opacity、log scale、wxyz quaternion；法线可省略。

SMV 在一个 GOP 中只能更新位置和旋转。编码器在点数或颜色/SH/透明度/缩放改变时自动开启新 GOP，
同时限制每组最多 30 帧。同组内假定同一行始终代表同一个高斯，不进行自动跟踪或最近邻匹配。
如果输入每帧重建、重排点或不断改变外观，可能产生大量小 GOP，压缩率会显著下降。
点数和不可变属性相同并不能证明点身份正确，需由输入生产流程保证行对应关系。

## 编码过程

1. 扫描 PLY，检测组内静态和动态点，并寻找跨所有 GOP 保持相同的静态行。
2. 每个 GOP 重排为静态点在前、动态点在后；共享静态点只写入 GOP 0。
3. 关键帧：位置 float16，旋转/透明度/DC 颜色 8-bit，线性缩放 16-bit。
4. SH：共享静态 1024 项码本和每 GOP 动态 1024 项码本，系数逐列 8-bit 量化。
   唯一向量不超过 1024 时直接建表；超过时使用确定随机种子的 MiniBatchKMeans。
5. 按**量化后的**关键帧计算位置偏移和相对四元数，避免把关键帧量化误差继续传到动态位置。
6. 位置偏移每帧分轴量化为 16-bit，分别保存高、低字节视频；四元数增量每分量 8-bit。
7. RGB 数据通过 libx265 `lossless=1`、`gbrp` 编码。视频字节无损不等于整个流程无损：
   属性量化、float16、码本聚类仍会带来误差。
8. 动态点补齐到正方形图像，补充点透明度约 1e-7；解码回 PLY 时去掉补充点并恢复原点序。
   全静态组和单帧组也生成可解码的占位运动视频，由 manifest 限定真实帧数。

## 命令

在 `E:\CloudPointTest\4dgs-view\format` 执行。依赖 NumPy、SciPy、scikit-learn，以及便携式 FFmpeg。
当前使用已验证的 `E:\ProgramData\anaconda3\python.exe`，无需安装或改动环境。

编码：

```powershell
& 'E:\ProgramData\anaconda3\python.exe' .\third_party\smv-codec\smv_codec.py encode --sequence .\dataset\high_discussion\sequence_simple.json --output .\third_party\smv-codec\output\discussion_reencoded.smv.zip --ffmpeg .\tools\ffmpeg-win-x86_64-v9.0.2.exe
```

重新解码并逐帧对比输入，生成数值误差报告（不另存一套 PLY）：

```powershell
& 'E:\ProgramData\anaconda3\python.exe' .\third_party\smv-codec\smv_codec.py verify --archive .\third_party\smv-codec\output\discussion_reencoded.smv.zip --sequence .\dataset\high_discussion\sequence_simple.json --ffmpeg .\tools\ffmpeg-win-x86_64-v9.0.2.exe
```

解码到新目录，生成 PLY 及简化时间戳 JSON：

```powershell
& 'E:\ProgramData\anaconda3\python.exe' .\third_party\smv-codec\smv_codec.py decode --archive .\third_party\smv-codec\output\discussion_reencoded.smv.zip --output .\third_party\smv-codec\output\decoded --ffmpeg .\tools\ffmpeg-win-x86_64-v9.0.2.exe
```

编码选项：`--max-gop 30`、`--position-bits 16`（默认）、`--preset fast`。
`--position-bits 8` 省略位置低字节流，更接近源场景的策略，体积更小但位置误差更大。
输出路径已存在时拒绝覆盖；重新运行请指定新输出文件。

小规模端到端测试：

```powershell
& 'E:\ProgramData\anaconda3\python.exe' .\third_party\smv-codec\test_roundtrip.py
```

测试覆盖实际 HEVC 编解码、非均匀时间线、单帧 GOP、全静态组、点数变化、补充点和点序恢复。
全量验证报告记录逐点位置、旋转角、相对缩放、线性透明度、DC 和 SH 误差；
不等同于画面 PSNR/SSIM 或视觉质量评测。

原始场景和网站代码的署名及许可说明仍见 `dataset/high_discussion/source/LICENSE`。
