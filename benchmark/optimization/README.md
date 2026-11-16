# 可恢复的单 CPU AV1 优化

开始前读 GOAL.md、PLAN.md、STATE.json、DIRECTIONS.md 与 EXPERIMENTS.jsonl。

历史 `benchmark/results` 不变。初始源码副本在 `baselines/historical_20261116/source`，完整文件指纹在对应 manifest.json；数据没有重复拷贝，原始数据和历史所有结果均有 SHA-256。这些文件不得改写。

## 执行
以下均在工作区根目录用 pwsh 运行：

```powershell
& '.\.venv-compression\Scripts\python.exe' benchmark\optimization\run_suite.py --source benchmark\optimization\baselines\historical_20261116\source --output benchmark\optimization\runs\baseline_single_cpu --methods morton_av1 tsog --repetitions 3
pwsh -NoProfile -File benchmark\optimization\run_stage.ps1 -WaitForBaseline
```

第二个命令等待第一个完整结束，再串行运行测试、PDF 读取、独立 CUDA 画质评测与候选编解码。等待器仅属于当前实验流程，不是定时自动化。失败即停止，保留现场；修正后复用已完成 suite 时，源码或配置不同必须选新输出目录。

用户随后提供了新的 16.07 MB 时序参考：旧流水线已在完成 SH 去重消融后按计划于套件边界停止，当前继续运行 `run_stage_latest.ps1`，显式冻结该参考并比较原 v2 格式的缓存解码、新 v4 布局、线性预测及无码本对照。不要重新启动旧阶段来运行已过时的时序源。

基线 runner 在三次测量间保持不变。扩展候选用 run_case_explore.py / run_exploration_suite.py；两者与基线执行策略相同，增加了非码本方法入口、冻结源码/输入指纹核查和候选套件源码副本。

单独候选运行示例：

```powershell
& '.\.venv-compression\Scripts\python.exe' benchmark\optimization\run_exploration_suite.py --source benchmark\optimization\baselines\historical_20261116\source --output benchmark\optimization\runs\new_dictionary_trial --methods shdict --candidate adapt\av1_shdict.py --repetitions 3
```

选择新目录；不要重新使用其他配置的输出目录。每个 case 的 timing.json 保留输入/源码/工具/环境指纹、CPU 亲和性、实际 FFmpeg 子命令、线程池状态、端到端时间和完整 PLY 重建哈希。阶段时间为包含式诊断，不可相加代替端到端时间。

## PDF
使用 Codex bundled artifact Python 执行 read_papers.py，生成 literature/index.json、逐页文本及可选页面 PNG；不安装或更新现有压缩/渲染环境。原始 PDF 不变。公式、图表、双栏文本阅读须对照渲染页面。

## 当前候选的含义
- av1_shdict.py：v3，逐帧精确 SH 去重，保留 v1 的 Morton 行序与量化值，作为消融。
- 快照内 av1_temporal.py：原有 v2 I/P 候选，新增任务不修改它，独立在相同执行限制下验收。
- av1_gop.py：v4，验证 GOP 内不变属性、只传动态运动残差；保留 GOP Morton 行序而不恢复输入行号，静态属性只反量化一次。
- av1_predictive.py：v5，独立非码本路线，SH 颜色通道/空间整数可逆预测。筛选结果不能充当全量胜负证据。

v1 解码实现保持原样，已有文件可继续解码。候选是研究路径，未经测试/完整质量/三次测量者不能发布为最佳版本。

v4 也提供独立单核 CLI（自动在导入数学库之前限制线程与隐藏 CUDA，并核验 FFmpeg 子进程）：

```powershell
& '.\.venv-compression\Scripts\python.exe' -m adapt.av1_gop encode --sequence benchmark\input_first60.json --output benchmark\optimization\manual_gop_encoded
& '.\.venv-compression\Scripts\python.exe' -m adapt.av1_gop decode --encoded benchmark\optimization\manual_gop_encoded --output benchmark\optimization\manual_gop_decoded
```

正式测量仍使用隔离快照执行器，以获得完整实验指纹和计时证据。

## 已观测的外部负载
首轮基线期间本机另有一个非本实验的 FFmpeg 进程，曾在全部逻辑 CPU 上运行。没有终止其他任务。该负载在第二轮 AV1 过程中已不再出现；保留全部原始时间与 min/max，只有明显超出波动的改进才作性能结论。如关键胜负接近波动，应在安静环境补测，不能只选最快一次。
