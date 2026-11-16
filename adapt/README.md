# 适配层

这些文件封装三方项目，原项目位于 `third_party`，未直接编辑其源码。

- `texture_video.py`：复用 4DGS-compression 的属性分组、归一化、PNG 和 PLY 解码；固定 atlas 尺寸，等价向量化像素填充；xyz FFV1，其他属性 H.264。
- `gsd.py`：调用 4dgs-converter 原生 GSD v1/v2 转换 API，并按实际输出格式读取 LZ4 纹理或 VQ 码本，逆变换坐标与四元数。
- `tsog_adapter.py`：原生 TSOG encode/decode；10 帧一包，离散 frame ID，唯一高阶 SH 向量码本。
- `av1_planes.py`：新增 Morton / 16-bit 属性量化 / 高低字节灰度平面 / AV1-MKV 编码与解码；可直接作为模块 CLI 运行。
- 根目录 `av1_codec.py`：主要优化实现，包含组内固定 Morton、精确 SH 码本、静态/不变属性复用、动态差分与 AV1 编解码。
- `av1_temporal.py`：指向根目录主实现的兼容入口，供原有导入和命令继续使用。
- `smv_adapter.py`：封装已迁移至 `third_party/smv-codec` 的本地 SMV 原型，记录其显式时间线扩展，加入统一编解码与 PSNR 基准。
- `play_temporal.py`：优化版 AV1/关键帧直接解码到高斯属性并播放，避免中间 PLY。
- `play_av1.py`：MKV → 属性平面 → 反量化高斯 → CUDA 渲染的本地播放器；支持四个相机视角。
- `common.py`：统一标准 3DGS PLY、属性顺序和时间戳 JSON。

统一表有 62 个 float32：xyz、占位法线、DC RGB、45 个通道优先 SH、logit opacity、log scale、wxyz 四元数。
AV1 使用其中 59 个有效属性，省略恒为 0 的法线。每属性是 uint16，低字节和高字节各一张 gray8 平面，
118 张平面按 12 列、10 行排成一张灰度 atlas，空余格为 0；每个 MKV 视频帧就是一个 PLY 时间帧。
atlas 为单色，不经过 RGB→YUV 矩阵或色度子采样。libaom-av1 CRF0 保证视频层字节无损，逐帧校验 SHA-256。
反量化仍有 16-bit 标量量化误差。元数据记录全序列 min/max、点数、时间戳、平面布局和校验和。

Morton 排序每帧独立计算，采用固定全序列位置范围、每轴 21 位、稳定排序；解码保持 Morton 点序，
不恢复原始行号。它保留高斯集合和点数，适用于渲染；若下游依赖跨帧点身份，需要另存原始点 ID。

以上 118 平面布局和逐帧 Morton 描述针对 v1。v2 每个 GOP 只排序一次，静态点和动态点分别排序；
视频只有动态 xyz+四元数共 7 属性、14 字节平面，4×4 排布，另两格为 0。当前数据是 756×756。
`keyframes.npz.xz` 保存静态几何、所有不变属性、精确 SH 码本/索引和恢复原始点序所需映射，
与 `motion.mkv`、`manifest.json` 共同构成完整可播放资产。体积必须把三个文件一起统计。
视频字节与完整 59 属性量化表都逐帧校验，已用独立的 v1 编解码函数验证数值完全相同。
默认单线程 CPU，无 GPU 压缩；GPU 只在画质测试和播放器中渲染。
