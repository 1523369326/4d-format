# 使用已验证的 AV1 路径

所有示例在仓库根目录用 pwsh 执行，输出选择新目录。CLI 在 NumPy 导入前限制数学库线程，编解码固定到一个逻辑 CPU，FFmpeg 子进程继承并核验该限制；不使用 GPU。正式测量仍用快照执行器，不能把 CLI 偶然最快一次与报告中位数混比。

## 现有 v2 文件：兼容缓存解码

根 av1_codec.py 保留现有 v2 编码逻辑，解码已委托给 adapt/av1_cached_decode.py。它支持早期 NPZ 与当前 NPZ.XZ 两类侧信息，并保留原始行序、视频 atlas SHA 与逐帧全部量化属性 SHA。输出全部 62 列标准 PLY，包含完整 SH3。

```powershell
& '.\.venv-compression\Scripts\python.exe' av1_codec.py decode --encoded benchmark\results_single_thread_v2\morton_av1_opt\encoded --output benchmark\optimization\my_cached_v2_decode
```

这条路径不要求重编码已有 16.07 MB 文件。benchmark 的兼容别名 adapt.av1_temporal 仍指向统一根入口。接入前源码保留在 provenance/av1_codec_before_cached_decode.py；编码器与 CLI AST 的未变性、两类容器的统一入口兼容性测试在最终报告 canonical_entry_integration.json 中记录。

## GOP 布局与缓存：新产物恢复输入行序

```powershell
& '.\.venv-compression\Scripts\python.exe' -m adapt.av1_gop encode --sequence benchmark\input_first60.json --output benchmark\optimization\my_v5_encoded
& '.\.venv-compression\Scripts\python.exe' -m adapt.av1_gop decode --encoded benchmark\optimization\my_v5_encoded --output benchmark\optimization\my_v5_decoded
```

线性预测消融使用 encode 的 `--predictor linear`，关键帧预测用 `--predictor reference`；默认上一帧预测。当前正式最佳结果以 STATE.json 为准；reports/stage_latest/best.json 是旧 v4 历史选择。`--sidecar-codec deflate` 仅用于侧信息容器消融，默认使用单线程 LZMA；实际视频始终为软件 AV1。

2026-11-17 起，新编码产物使用 `morton-av1-gop` version 5：保存每个 GOP 的 Morton 排列，解码恢复每一帧输入 PLY 的原始行序。内部保留静态/动态 Morton 布局、SH 精确去重、预测、双字节平面和缓存解码。排列仅在 GOP 开始用于构建原行序模板，后续帧直接更新原行号对应的动态点。

历史 version 4 没有保存排列，仍支持读取，但其原始行号无法仅凭旧码流恢复，需要重新编码。此前约 15.41 MB 等数据对应省略排列的历史配置，新配置须重新测量，不能直接套用旧结果。

## 历史 v1

adapt/av1_planes.py 与最初源码副本仍保留 v1 解码能力；不可使用其旧的默认多线程 CLI 做新性能比较。用单核快照执行器测量 v1，命令见 README.md。视频字节无损不等于原始浮点属性无损：上述等价路线均保留 v1 的 16 位属性量化误差。

## 证据与边界

- runs/baseline_single_cpu：原版与 TSOG 的三次基线。
- references/latest_1607：用户提供最新参考的源码、编码文件、计时、质量与指纹。
- runs/latest_temporal_reference、latest_cached_decode、gop_cached_lzma、gop_linear_lzma：统一口径复测与候选。
- reports/stage_latest：完整比较、波动、量化等价、原 v2 输出等价、历史保留检查、非码本筛选与最佳候选。

TSOG 锚点是本仓库现有 OpsiClear Python 实现。结果仅针对 high_discussion 前 60 帧及固定四个视角，不代表所有序列或论文作者其他实现。
