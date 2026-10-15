# GPU 加速调研汇总（GitHub 与 arXiv）

> 状态：调研完成（2026-10-14，四路并行检索：推理引擎 / 厂商 SDK 与基准 / arXiv 系统类 / arXiv 模型类）
> 本文是外部世界的事实清单；路线取舍决策记录在 `GPU加速候选路线.md`（已按本文结论修订）。
> 所有 arXiv ID 均经 arxiv.org 页面核实；GitHub 星数/活跃度为 2026-10-14 API 实测；查不到的项文末如实声明。

## 0. 一页结论（先看这个）

1. **V2509A = vivo X300 标准版 = 联发科天玑 9500（MT6990），GPU 为 Mali-G1-Ultra MC12**。
   多源印证（Geekbench 现身记录 2025-08 + 电信入网信息 + Arm 官网合作页）。推论：**QNN 路线关闭**（仅高通）；**NNAPI 路线关闭**（Android 15 弃用 + 天玑生态已绕开）。
2. **天玑 9500（MT6990）在 LiteRT MediaTek NPU 官方支持列表内**（Dimensity 7300–9500 全系）——我们的真机恰好有一条 Google 官方背书的 NPU 通路（CompiledModel API，要求 minSdk 31 / arm64 / .tflite 格式）。
3. **ONNX Runtime Android 官方包没有任何 GPU EP**（Java API 实测只有 `addNnapi/addXnnpack/addQnn/addCUDA`）——「在 ORT 里换个 EP 上 GPU」此路不通，想上 GPU 必须换推理引擎/格式。ORT 1.22.0（我们在用）→ 1.31.0（2026-10-09 发布）升级有免费的 CPU 优化红利。
4. **EdgeSAM 模型侧并不过时**：论文在 iPhone 14 上全流程 38.7 FPS（26ms，CoreML+ANE），我们 ORT CPU ~1s——**40 倍差距在推理栈而非模型**。且其 encoder（RepViT-M1）是**纯 CNN**（重参数化，非 ViT），对 LiteRT GPU delegate 的算子覆盖友好；decoder（two-way transformer）算子不在 GPU delegate 清单，但足够小（社区实测 CPU <100ms），可留 CPU。
5. **SAM 在 Android GPU 上没有成熟公开先例**：官方只有 ONNX(CPU)+CoreML(iPhone)；社区 3 个 Android 仓库全部停更/半成品。我们要做的是空白区，选择标准从「有无现成 example」降级为「转换链路 + 算子覆盖 + 社区厚度」。
6. **预期锚点（论文侧数字）**：移动 GPU fp16 vs CPU ≈ **2–6×**（CNN）；NPU 视觉 encoder 可到 **20–45×**；但**小模型 GPU 可能负优化**（ncnn 作者明言 + iPhone LLM 实证 0.75×），一切须真机实测。
7. 量化配套方法论现成：**fp16 基本免费**；**int8 有 SAM 专用 PTQ 方案**（PTQ4SAM/SAQ-SAM，W4 都可保精度；Q-TinySAM W8A8 = -21.8% 延迟 / -0.9 AP）。
8. ⚠️ **License 注意**：EdgeSAM 上游是 NTU S-Lab License 1.0（非标准开源，商用需审读；当前内部研究使用无碍）。

---

## 1. 背景事实（决定路线的三条硬事实）

### 1.1 V2509A SoC 确认（本次调研最重要结论）
- vivo X300 系列三型号：V2509A（标准版）/ V2502A（Pro）/ V2502DA（Pro 卫通版），中国电信入网信息确认标准版搭载天玑 9500。
- Geekbench 6（2025-08-15）：V2509A 单核 2352 / 多核 7129，超大核 4.21GHz。
- Arm 官网：X300 首发 Arm Lumex 平台 + 天玑 9500；GPU = Mali-G1-Ultra MC12。
- 天玑 9500 = MT6990 = **LiteRT MediaTek NPU 支持列表中的最新一代**（官方页列出 Dimensity 7300/8300/9000/9200/9300/9400/**9500 (MT6990)**）。
- 注：AI Benchmark 排行榜（2026-10 版）**尚未收录天玑 9500 成绩**（最新到 9600 系列，9600 为后继？存疑不引用）→ 我们芯片没有公开 NPU 跑分，必须自测。

### 1.2 NNAPI 弃用后的官方格局
- NNAPI 官方文档原文：可用区间 Android 8.1+，**deprecated in Android 15**。弃用≠删除（存量机仍能跑），但无人维护。
- Google 钦定继任者 = **LiteRT**（原 TFLite）：GPU 走 GPU delegate（OpenCL 优先→OpenGL ES 3.1 回退，不依赖 NNAPI）；NPU 走新 `CompiledModel` API（统一接口直达五家 NPU：Google Tensor / Intel / **Qualcomm** / **MediaTek** / Samsung）。
- LiteRT NPU 硬性要求：**Android API 31+、仅 arm64-v8a**；GPU 无此限制。
- NPU 分发方案 PODAI（Play for On-device AI）：按 SoC 编译模型 + Play Feature Delivery 分发厂商 runtime。JIT 编译缓存收益实测：resnet152 初始化 7465ms→198ms、内存 1525MB→355MB。
- 厂商轨迹佐证：三星 2021 年起冻结自研 Neural SDK 转接 LiteRT；AI Benchmark 榜上 2024-2026 旗舰成绩全部用厂商 delegate（`qh`=QNN HTP、`m`=MediaTek Neuron），不再走 NNAPI。

### 1.3 ONNX Runtime 在 Android 的真实能力边界
- 官方 AAR 的 Java API（`OrtSession.java` 源码核实）只有 4 个 EP 入口：`addNnapi / addXnnpack / addQnn / addCUDA`——**无 Vulkan/OpenCL GPU EP**；WebGPU EP 仅限 ORT Web。
- QNN EP：默认走 Hexagon NPU（HTP），可选 Adreno GPU/CPU；**Android 无预编译包，需自带 QNN SDK 源码构建**——且我们不是高通，双重出局。
- MTK 侧：ORT 上游无 NeuroPilot EP（MTK 自己的 ORT fork 仅限 Genio IoT 平台）。手机 Android 上经 ORT 访问天玑 NPU 的旧路（NNAPI）已弃用。
- 版本红利：`onnxruntime-android` 已发布 **1.31.0（2026-10-09）**，我们锁的 1.22.0（2025-05）落后一年半，升级零风险且可能白捡 CPU 算子/XNNPACK 优化。
- 结论：**「保住 ONNX 格式」与「上 GPU/NPU」在 2026 年的 Android 上基本互斥**——NPU 生态新入口统一是 .tflite（LiteRT），GPU 生态入口是 Vulkan/OpenCL 自带引擎（ncnn/MNN）或 LiteRT delegate。

---

## 2. GitHub 项目全景

### 2.1 推理引擎横评总表（星数/活跃度为 2026-10-14 API 实测）

| 引擎 | 星数 | 活跃度 | GPU 后端 | 吃 ONNX？ | 结论 |
|---|---|---|---|---|---|
| **ncnn**（腾讯） | 23,956 | ✅ 当天有推送，release 20260526 | **Vulkan**（OpenCL 分支已弃） | ✅ pnnx（官方主推，pip 直装） | **短名单 #1** |
| **MNN**（阿里） | 16,220 | ✅ 当天有推送，release 3.6.1（2026-07） | **OpenCL + Vulkan** + Metal | ✅ 转换器 158 个 ONNX 算子 | **短名单 #2** |
| **LiteRT**（Google） | 3,494（主仓） | ✅ 当天有推送，v2.3.0（2026-10-07） | GPU delegate：**OpenCL→GLES 回退**，fp16 | ❌ 无官方 ONNX 路径（LiteRT Torch 只吃 PyTorch，Beta） | **短名单 #3**（YOLO 稳 / SAM encoder 可行） |
| ONNX Runtime | 22,078 | ✅ v1.31.0（2026-10-09） | ❌ Android 无 GPU EP | 本体就是 ONNX | 基线 + 升级红利 |
| ExecuTorch（PyTorch） | 5,090 | ✅ v1.5.1（2026-09） | **Vulkan** + Qualcomm/MediaTek/Exynos 后端 | ❌ 只吃 torch export | 观察名单 |
| MLC-LLM / TVM | 23,244 / 13,832 | 活跃 | Vulkan / OpenCL（TVM 运行时） | 转换链路重 | 不进短名单（LLM 专用/工程量大） |
| llama.cpp | 131,149 | 极活跃 | Vulkan（Android 主力） | — | 仅作 Vulkan 生态成熟度证据 |
| TNN（腾讯优图） | 4,659 | ❌ 停滞 17 个月（最后推送 2025-05），release 还是 2021 年 | OpenCL | ✅ 100+ 算子 | 淘汰（孤儿框架） |
| MACE（小米） | 5,053 | ❌ 最后推送 2024-06，release 2022-01 | OpenCL + Hexagon DSP | ✅ | 淘汰（实质死亡） |
| Paddle Lite（百度） | 7,289 | ⚠️ 低强度（release 停在 2024-07） | OpenCL + Metal + NNAdapter | 绕经 Paddle 生态 | 淘汰（ONNX 迁移性价比低） |
| FeatherCNN（腾讯） | 1,229 | ❌ 2019 年死亡 | 无 | — | 淘汰 |

### 2.2 短名单引擎详情

**ncnn（https://github.com/Tencent/ncnn）**
- Vulkan-only 但方向正确（与 llama.cpp/ExecuTorch 同押 Vulkan；Vulkan 自 Android 7 起是 CTS 强制项，设备覆盖有制度保证）。
- 官方提供 `ncnn-20260526-android-vulkan.zip` 预编译包，集成成本低；BSD-3-Clause。
- YOLO 支持最好：官方 examples 含 yolov8/yolo11 全家桶，ncnn 作者 nihui 本人维护 `ncnn-android-yolov8`（229 星，2026-05 仍更新）。
- SAM 先例唯一存在：`FeiGeChuanShu/segment-anything-ncnn`（123 星，PyTorch→ONNX→ncnn 全管线，2023 停更）——可抄作业但作业很旧。
- ⚠️ 官方 Android demo README 明言：「**Most small models run slower on GPU than on CPU**」——对 YOLOv8n（3.2M 参数）这类小模型是直接警示。

**MNN（https://github.com/alibaba/MNN）**
- 双 GPU 后端（OpenCL 内核对 Adreno/Mali 调优业界最深 + Vulkan），Apache-2.0，3.6.1 持续发版，开放 issue 仅 48 个（维护力度极强）。
- ONNX 转换器支持 158 算子（官方 README），是「ONNX 资产保护」最好的换引擎选项。
- SAM 先例：`wangzhaode/mnn-segment-anything`（37 星，含编码器拆分优化）、`slz929/mobileSAM-Android-MNN`（34 星）——比 ncnn 侧略多但同样小众。
- ⚠️ 社区注意力已向 MNN-LLM 倾斜，视觉 demo 变薄；有 MLSys 2020 系统论文（见 §3.3）。

**LiteRT（https://github.com/google-ai-edge/LiteRT）**
- Android GPU 工程化天花板：GPU delegate 优先 OpenCL（TF 官方博客 2020 起）、回退 GLES 3.1 compute；**fp16/fp32**；支持序列化（初始化时间最多降 90%）；从 TFLite 迁移只改包名。
- **GPU delegate 只加速约 24 种 CNN 类算子**（CONV_2D/DEPTHWISE_CONV/FC/RESIZE_BILINEAR/TRANSPOSE_CONV 等，官方清单 2026-05-28 版），**无 attention/matmul** → EdgeSAM encoder（纯 CNN）覆盖良好，decoder（transformer）必回退 CPU（拆图执行有官方警告的陷阱，但 decoder 小、留 CPU 可接受）。
- 转换是主要门槛：无官方 ONNX→tflite；PyTorch→tflite 走 LiteRT Torch（https://github.com/google-ai-edge/litert-torch ，1,105 星，Beta，需 Linux+tf-nightly）。我们的 ONNX 均有 PyTorch 上游（EdgeSAM/MobileSAM 官方 repo），可从源头导出。
- NPU 路线（CompiledModel）官方页：Qualcomm 覆盖 SM8450–SM8850；**MediaTek 覆盖到天玑 9500（MT6990）**——对我们真机直接适用（详见 §2.3）。

### 2.3 厂商 SDK 与官方 NPU 通路

| SDK | 状态 | 与我们的关系 |
|---|---|---|
| Qualcomm QAIRT/QNN（含 SNPE 收编） | 活跃（QPM 分发；2026-05 发布首个 ORT Plugin EP） | ❌ 出局（非高通设备） |
| Qualcomm AIMET（量化工具，2,727 星，活跃） | 活跃 | 高通路线配套，出局；但量化方法论可借鉴 |
| Qualcomm AI Hub（云端真机编译，SAM3/MobileSam/YOLOv8 预优化模型） | 活跃 | ❌ 设备不符；但证明「SAM 家族上 NPU」已被工业验证 |
| **MediaTek NeuroPilot / Neuron SDK** | 8.0.11（2026-06），重心转 GenAI；手机 Android 正路 = **LiteRT MediaTek** | ✅ **我们的路线**：MT6990 在官方支持列表 |
| Samsung Neural SDK（ONE/ENN） | ❄️ 2021 年起冻结，转 Exynos AI Studio + LiteRT | 无关（仅三星设备）；其轨迹是行业风向标 |

### 2.4 SAM 手机端部署先例（全部检视过）

| 仓库 | 星数/状态 | 技术栈 | 关键信息 |
|---|---|---|---|
| **chongzhou96/EdgeSAM**（官方上游） | 1,180 / 2025-05 后不活跃 | ONNX + CoreML | **官方只有 ONNX + CoreML 导出，无 Android demo**；iPhone 14 38.7 FPS 靠 CoreML/ANE；NTU S-Lab License 1.0 |
| PrathamX595/Kortex-AdobeSubmission | 67 / 2025-12 | **ORT Android 1.17 + EdgeSAM ONNX** | 与我们栈最接近的先例；decoder CPU <100ms；encoder 慢（与我们互为印证） |
| FeiGeChuanShu/segment-anything-ncnn | 123 / 2023 停更 | ncnn | SAM→ncnn 唯一较完整管线 |
| slz929/SAM-Android-ncnn | 30 / 2023 停更 | ncnn | 同上，更小 |
| slz929/mobileSAM-Android-MNN | 34 / — | MNN | MNN 侧先例 |
| wangzhaode/mnn-segment-anything | 37 / — | MNN | 含编码器拆分优化 |
| shubham0204/Segment-Anything-Android | 65 / 2025-02 | ORT + ONNX fp16 | README 自述推理时间 "quite high even with float16" |

**定性结论：SAM×Android×GPU 是公开空白区**——没有任何一个「EdgeSAM + Android GPU/NPU」的现成实现，无论选哪条路都是自己蹚。官方 iPhone 30FPS 证明的是「模型+编译器+NPU」组合的上限，不是 Android 的现状。

### 2.5 YOLO 手机端部署（对照用，YOLO 是所有引擎的教科书模型）

- ultralytics/ultralytics：62,499 星；官方 export 支持 onnx/tflite/ncnn(PNNX)/coreml；官方 Android 途径 = ultralytics/yolo-flutter-app（505 星，2026-10 活跃）。
- nihui/ncnn-android-yolov8：229 星，作者维护，ultralytics→PNNX→ncnn 流水线。**README 警示（原文）："Most small models run slower on GPU than on CPU"**。
- hunglc007/tensorflow-yolov4-tflite：2,255 星，TFLite GPU delegate 老牌样例（2024 停更）。
- Qualcomm AI Hub YOLOv8-Detection：N 640² float 12.2MB / w8a8 3.25MB——量化体积参照。
- 结论：YOLOv8n 在哪都能跑，**不构成选型区分度**；反而要防「小模型 GPU 负优化」。

### 2.6 移动端 AI 基准项目

| 项目 | 位置 | 状态/价值 |
|---|---|---|
| MLPerf Mobile | github.com/mlcommons/mobile_app_open（72 星，活跃）+ mobile_results_v2.0–v6.0 | 正式轮次持续到 v6.0（2026-06）；App 化并上 Google Play；单流 90 百分位延迟为默认指标；**原始 JSON 含逐基准 delegate/加速器/精度**（如 SD 7+ Gen 2 HTP：MobileNet-EdgeTPU int8 2449.9 样本/s、分割 683.8/s）——但同提交内无 CPU 对照组 |
| AI Benchmark（ETH 系） | ai-benchmark.com + github.com/ai-benchmark | 78 项测试/180+ 指标，逐 SoC INT8+FP16 双精度实测；后端图例含 `g`(GPU delegate)/`qh`(QNN HTP)/`m`(MTK Neuron)/`nn`(NNAPI)/`c`(CPU)；**天玑 9500 尚未收录**；旗舰梯队参考：天玑 9600 Pro 22631 分 > 骁龙 8 Elite Gen5 17796 > 天玑 9400 5618（MTK 跑分适配波动大，引用需谨慎） |
| MobileAIBench（小米） | github.com/XiaoMi/mobile-ai-bench | 较老；芯片×框架端到端基准（速度+精度） |
| MLPerf Mobile 论文 | arXiv:2012.02328（MLSys 2022） | 方法学来源（run rules/单流延迟 vs 吞吐两种口径），见 §3.3 |

---

## 3. arXiv 论文（全部 ID 已核实）

### 3.1 SAM 高效模型家族（与我们模型直接相关）

| 论文 | arXiv ID | 年份/出处 | 手机端速度 | 精度代价 | 对我们的价值 |
|---|---|---|---|---|---|
| **EdgeSAM**（在用） | **2312.06660** | 2023-12 / IJCV 2025 | **iPhone 14 全流程 38.7 FPS（26ms；encoder 14ms@1024²）** | COCO box mIoU 76.7（SAM 77.3） | 见下方详注 |
| MobileSAM | 2306.14289 | 2023-06 | GPU 10ms/图；**iPhone 14 仅 4.9 FPS** | mIoU 74.4 | TinyViT 不被移动编译器优化——教训：服务器 ms 数不能外推手机 |
| TinySAM（华为） | 2312.13789 | 2023-12 / AAAI 2025 | T4+TensorRT 38.4ms；**Q-TinySAM(W8A8) 24.0ms（-21.8%）** | AP 42.3；量化仅 -0.9 AP | 「SAM+int8」最直接证据；但 TinyViT 系手机端有已知劣势 |
| EfficientSAM（Meta） | 2312.00863 | 2023-12 / CVPR 2024 | iPhone 14 仅 4.9 FPS | zero-shot 分割 +4 AP | 精度路线，手机端无速度优势，不适合 |
| **EfficientViT-SAM**（MIT） | **2402.05008** | 2024-02 / CVPR 2024 WS | **A100+TRT fp16：L0@512 8.2ms（对 SAM-ViT-H 48.9×）**；无手机数字 | mAP 45.7@512 | 线性注意力 GPU 友好；**512 输入是免费加速源**；若走 GPU 路线值得 A/B |
| RepViT-SAM（清华） | 2312.05760 | 2023-12 | 同 EdgeSAM（同 RepViT 骨干） | 比 EdgeSAM 低 1.0–2.9 mIoU | 被 EdgeSAM 严格支配，仅对照 |
| SqueezeSAM（Meta） | 2312.06736 | 2023-12 | 比 SAM 快 62.5× | 域特化（手机摄影抠图） | 专用小模型路线的存在性证明 |
| FastSAM | 2306.12156 | 2023-06 | T4 153.6ms | AP 37.9 | **非 promptable 接口**，与我们点提示用法不符 |
| MobileSAM2 | 2607.12297 | 2026-07 / ECCV 2026 | 延迟数字未取得（不编造） | SAM2 蒸馏+NAS | 关注项：若做视频跟踪（BoardTracker 方向契合）是下一个替换池 |
| SAM 家族综述 | 2410.04960 | 2024-10 | — | — | 追新的入口 |
| SAMfast | 无 arXiv（PyTorch 官方博客 2023-11 + github.com/meta-pytorch/segment-anything-fast） | 2023-11 | 纯工程优化（SDPA/fp16/kernel compile）**8×、零精度损失** | — | 「部署栈收益」的独立佐证 |

**EdgeSAM 详注（我们的模型）**：
- 方法：SAM 的 ViT 编码器蒸馏为 **RepViT-M1 纯 CNN**（9.6M 参数 / 22.1 GFLOPs）+ **prompt-in-the-loop 蒸馏**（把 prompt encoder/decoder 纳入蒸馏，从师生分歧区动态采点提示；论文证明 encoder-only 蒸馏有天花板）。仅用 1% SA-1B。
- 关键消融：**三种同量级 backbone 在 2080 Ti 上 FPS 只差 1.4×，在 iPhone 14 上差 14×**——架构×编译器匹配度决定端侧速度，模型大小不是。
- 论文明确把 quantization/pruning/mixed-precision 列为遗留方向——**模型还有加速空间可榨**。
- 对我们的推论：V2509A CPU ~1s vs iPhone ANE 26ms 的 40× 差距在推理栈；**先换执行后端，模型不换**。

### 3.2 SAM 量化专题（部署侧必读）

| 论文 | arXiv ID | 出处 | 核心内容 |
|---|---|---|---|
| **PTQ4SAM** | **2405.03144** | CVPR 2024 | SAM 专用 PTQ（W4/W8）：定位两个量化坑——post-GELU 激活**双峰分布**（Bimodal Separation 解法）+ prompt-patch attention 离群值（Temporal Information Patch）；ViT 型和 CNN 型 SAM 都适用 |
| **SAQ-SAM** | **2503.06515** | AAAI 2026 | 语义级对齐量化：W4 下实例分割比 baseline 高 11.7% mAP（baseline 崩、它保住）；利用 SAM 可提示接口重构校准目标 |
| Q-TinySAM（上表） | 2312.13789 | AAAI 2025 | W8A8：延迟 -21.8%、AP -0.9；6bit 以下开始明显劣化 |
| Mobile AI 2021 量化挑战报告 | 2105.07825 | Workshop | INT8 部署工程现实：量化模型要**设计时就迁就目标加速器算子集** |
| NCNet | 2208.11609 | AIM 2022 | **「改写等价算子去迁就加速器」比「求驱动支持怪算子」更可行**（最近邻上采样改成卷积实现以过 NNAPI）——对我们 decoder 算子适配直接启发 |
| AWQ/TinyChat | 2306.00978 | MLSys 2024 最佳论文 | fp16 kernel 融合 + 平台感知权重排布在**移动 GPU** >3×（LLM 侧旁证） |

### 3.3 系统与基准评测

| 论文 | arXiv ID | 出处 | 核心结论 | 对我们的价值 |
|---|---|---|---|---|
| **AI Benchmark 2019** | **1910.06663** | 系列（排行持续更新在 ai-benchmark.com） | 全 SoC 横评 CPU/GPU(fp16)/NPU(int8)；**移动 GPU fp16 vs CPU fp32 典型 2–6×** | 横评方法学鼻祖 + 加速比先验 |
| **MLPerf Mobile** | **2012.02328** | MLSys 2022 | 首个工业标准移动 ML 基准；run rules；**六个月内纯软件栈改进：吞吐 3×、延迟最多 12×** | 软件栈版本敏感——横评必须锁版本；区分单流延迟/离线吞吐两种口径 |
| MNN 引擎 | 2002.12418 | MLSys 2020 | pre-inference（加载期优化）+ 算子调优 + 多后端混合调度 | 引擎设计参考架构 |
| MNN-LLM | 2506.10443 | MMAsia'24 WS | 按**移动 GPU 特性重排权重/输入布局**；对主流 LLM 框架最高 8.6× | 小模型 GPU 部署通用手法 |
| EPAM | 2303.01509 | IEEE ICC 2023 | **NNAPI 作为被测后端与 CPU/GPU 同台实测**（延迟+能耗+内存）；GP 回归能耗预测 | 「四后端同台横评」结构模板 |
| HG-Caffe | 1901.00858 | 2019 | OpenCL+FP16 移动 GPU 引擎，对 Caffe CPU 最高 20×（**基线太老，不能当现代预期**） | OpenCL+FP16 可行性历史文献 |
| Phase Matters | 2606.27906 | 2026-06 | **视觉 encoder 上 NPU vs CPU 20–45×；均温低 10.47°C、能耗 1/2.52** | NPU 上限的参照数字 |

### 3.4 热节流与能效（横评方法学的关键输入）

| 论文 | arXiv ID | 出处 | 核心结论 |
|---|---|---|---|
| **Play It Cool** | **2206.10849** | ICML 2022 WS Spotlight | 持续推理→热节流→延迟**急剧增大且不一致**；对策=按热状态动态切换大/小模型。**不控温的 A/B 对比无效**（先跑者吃亏） |
| **LLM Inference at the Edge** | **2603.23640** | 2026-03 | 最完整的持续负载测量协议（20 次预热迭代+固定负载+逐迭代记录热状态）；**iPhone 16 Pro 两轮迭代吞吐近腰斩；S24 Ultra 被 OS 强制 GPU 频率地板直接终止推理**；结论应声明为「平台级软硬一体表征」 |
| Lotus | 2410.10847 | DAC'24 | 检测器逐帧延迟方差大（proposal 数动态变化×DVFS）→ 用 RL 在线调频；小米 11 Lite 实测。**报告延迟分布而非单值** |
| DVFS Governors | 2507.02135 | 2025 | CPU/GPU/内存三路 governor 互不协调 → 同能耗下延迟多损 40.4% |
| PELM | 2609.09662 | SenSys'26 | DVFS+投机解码，能耗 -52.4%（功耗优化储备文献） |

### 3.5 GPU 算子与结构启示

| 论文 | arXiv ID | 出处 | 核心结论 |
|---|---|---|---|
| FCM/FusePlanner | 2404.19331 | 2024 | DW+PW 卷积融合（桌面 GPU）：最高省 83% 访存、对 cuDNN 3.7×；融合决策靠成本模型非全融合。**arXiv 无移动 GPU 特化版本（如实声明）** |
| FasterNet | 2303.03667 | CVPR 2023 | 「FLOPs 降≠延迟降」；DWConv 频繁访存是根因；**逐设备逐算子延迟表是方法学范文** |
| InceptionNeXt | 2303.16900 | CVPR 2024 | ConvNeXt-T 与 ResNet-50 FLOPs 相当，A100 吞吐仅 ~60%——大核 DWConv 的 GPU 代价量化 |
| Diagonalwise Refactorization | 1803.09926 | 2018 | DWConv 对角化转标准卷积，5 框架 1.4–15.4×——「DWConv 是 GPU 二等公民」的经典支撑 |
| MobileOne | 2206.04040 | CVPR 2023 | 重参数化（训练多分支→推理单分支）；**FLOPs/参数量与手机实测延迟不相关，必须按 on-device latency 设计**；iPhone 12 <1ms |

### 3.6 端侧 LLM 的 GPU 生态旁证（Vulkan/OpenCL 可行性证据）

| 论文 | arXiv ID | 核心结论 |
|---|---|---|
| **Transformer-Lite**（荣耀） | **2403.20041** | 手机 GPU 跑 LLM 代表作：动态 shape 符号表达式 + FP4 M0E4（**动机就是降低反量化开销**）+ sub-tensor 消 KV 拷贝；对 GPU 版 MLC-LLM 2–3×、CPU 版 FastLLM prefill >10×；**兼容高通+MTK**——手机 GPU 上 transformer 类负载可行且有引擎间 2–3× 优化空间 |
| **Challenging GPU Dominance** | **2505.06461** | 反直觉实证：iPhone 15 Pro 1B LLM，**CPU 双线程 fp16 17 tok/s > GPU 12.8 tok/s（0.75×）**——GPU 内存搬运税；「上 GPU 不是免费收益」 |
| HeteroInfer | 2501.14794 | GPU+NPU 异构并行，比单后端 SOTA 快 1.34–6.02× |
| HybridInfer | 2609.30270 | **骁龙 8 Elite 上持续负载令 OpenCL kernel 编译/长 prefill 把 GPU runtime 搞崩或挂死**——移动 GPU 路线稳定性第一手警讯 |
| PowerBench（Is Your NPU Ready?） | 2607.05475 | 五框架×CPU/GPU/NPU：**框架间差距在 NPU 上放大到 10×**（offloading/量化策略不同→小 kernel 回退 CPU）；NPU 偏好大而形状固定的负载，小 kernel 动态负载反而 CPU 最优 |

---

## 4. 公开「加速比」数据汇总（预期锚点表）

| 场景 | 数字 | 出处 |
|---|---|---|
| 移动 GPU fp16 vs CPU fp32（CNN，同引擎） | **2–6×** | AI Benchmark arXiv:1910.06663 |
| 视觉 encoder NPU vs CPU | **20–45×**（且均温 -10.5°C、能耗 ÷2.5） | arXiv:2606.27906 |
| 引擎级优化差距（LLM 端到端，GPU 路线） | 2–3×（vs MLC）/ >10×（vs CPU 引擎 prefill） | arXiv:2403.20041 |
| 量化+融合内核 vs fp16 | >3× | AWQ/TinyChat 2306.00978 |
| W8A8 量化（SAM 家族） | 延迟 -21.8%、精度 -0.9 AP | Q-TinySAM 2312.13789 |
| 纯工程优化（SDPA/fp16/compile，零精度损失） | 8× | SAMfast（PyTorch 博客） |
| LiteRT 官方口径（营销上限） | "up to 25×" | Google 开发者博客 |
| **GPU 负优化（小模型/访存受限）** | **0.75×（GPU 输给 CPU）** | 2505.06461；ncnn-android-yolov8 README |
| 持续负载惩罚 | 吞吐腰斩（两轮迭代内）/ GPU 被 OS 封顶 | 2603.23640 |
| 软件栈版本差异（六个月） | 吞吐 3×、延迟 12× | MLPerf Mobile 2012.02328 |

**对我们两个模型的预期**：EdgeSAM encoder（RepViT 纯 CNN、22 GFLOPs、大算量）GPU 合理预期 2–5×，NPU 上限更高；YOLOv8n（3.2M 参数、大量 DWConv、访存受限）**必须防负优化**；SAM decoder（动态小算子流水线）是回退重灾区，可能留 CPU 最优。

---

## 5. 对候选路线的修订（调研直接产出）

原路线表（`GPU加速候选路线.md` v0）按本次调研修订为：

| 路线 | 原状态 | 修订后 | 依据 |
|---|---|---|---|
| A. ORT CPU/XNNPACK | 待跑 | ✅ **保留，且升级 ORT 1.22→1.31** | §1.3：无 GPU EP 但有版本红利；零风险基线 |
| B. NNAPI | 待跑 | ❌ **关闭** | §1.2：Android 15 弃用；天玑生态已绕开；不再投入 |
| C. LiteRT GPU delegate | 待跑 | ✅ **升为主线** | Mali-G1-Ultra（OpenCL/GLES）；EdgeSAM encoder 纯 CNN 算子覆盖良好；YOLO 稳 |
| C+. **LiteRT MediaTek NPU（CompiledModel）** | （新增） | ✅ **新增上限路线** | MT6990 在官方支持列表；minSdk 31/量化/AOT 门槛高，放 G2 |
| D. QNN | 待跑 | ❌ **关闭** | V2509A 非高通 |
| E. ncnn(Vulkan)/MNN(OpenCL+Vulkan) | 备选 | ✅ 保留备选（短名单 #1/#2） | ONNX 资产保护最好（pnnx/158 算子）；LiteRT 转换不顺时接棒 |
| F. 原生 OpenCL/Vulkan | 兜底 | ⚠️ 保留但降权 | Mali 的 OpenCL 通常不开放给普通 App；HybridInfer 警示 OpenCL 稳定性 |

**顺带修订的方法学**（论文侧输入，并入设计文档 §4）：
- 对比顺序控温/交错（Play It Cool：不控温的 A/B 无效）；逐迭代记录热状态入元数据。
- 报告延迟分布（median/p90/p99/**方差**），不只单值（Lotus）。
- 同引擎内切后端对比，锁版本入报告（MLPerf Mobile：软件栈半年 3–12×）。
- 整模型加速比之前先做 per-op 分解，定位回退算子（FasterNet/FCM）。
- encoder/decoder 分开计量，防 decoder 绕行吃光 encoder 收益。
- 精度评估口径：与 CPU fp32 输出的 IoU/mAP 差（PTQ4SAM 的评估姿势）。

---

## 6. 风险警示清单（论文+社区实证过的坑）

1. **小模型 GPU 负优化**：ncnn 作者明言 + iPhone LLM 0.75× 实证——YOLOv8n 上 GPU 前必须实测。
2. **内存搬运税**：GPU 路线隐性成本；HybridInfer 实证骁龙上 OpenCL 编译可致 runtime 崩溃/挂死。
3. **算子回退/图分区**：NPU 上框架间差距放大到 10×（PowerBench）；LiteRT GPU delegate 无 attention（decoder 必回退，需拆图策略）。
4. **量化≠更快**：反量化开销可能吃掉收益（Transformer-Lite 为此做 FP4 M0E4）；int8 真正主场是 NPU。
5. **热节流**：持续负载吞吐腰斩/OS 封顶 GPU（2603.23640）；我们象棋 App 是 1Hz 持续负载型，迟早进节流区。
6. **DWConv 是 GPU 二等公民**：FLOPs 与延迟脱钩（YOLOv8n 主力算子恰是 DWConv）。
7. **License**：EdgeSAM = NTU S-Lab License 1.0（非标准，商用需审）。
8. **天玑 NPU 跑分波动**：AI Benchmark 上天玑 9400 成绩异常低（疑似 delegate 适配问题）——MTK 侧性能不确定性高于高通，自测前不设过高预期。

---

## 7. 诚实声明（查不到/未核实的项）

- MACE/FeatherCNN/TNN/ncnn **无 arXiv 论文**（站内搜索零命中，已验证）；引用只能用 GitHub/技术报告。
- **NNAPI 碎片化专题论文**与 **Adreno/Mali 专属 int8/fp16 系统评测**：arXiv 无专门文献，已用最接近文献补位。
- MobileSAM2 具体延迟数字、SlimSAM 的 arXiv ID、RepViT 本体 arXiv ID：未核实/未取得，不写。
- 专门在**安卓手机 GPU** 上横评 YOLO 家族的 arXiv 论文：未检索到（最接近=Jetson 评测 2409.16808）。
- AI Benchmark 逐设备的 CPU/GPU/NPU 三列明细在动态详情页，静态抓取未取得。
- TNN/ExecuTorch/EdgeSAM 的许可证为自定义文本（GitHub API NOASSERTION），未逐条审读。
- 一次搜索建议的 "2311.18247=MLPerf Mobile" 经核实为引力波论文，已剔除（严格核实 ID 的必要性示例）。

## 8. 主要来源索引

- 引擎/仓库：github.com/{Tencent/ncnn, alibaba/MNN, Tencent/TNN, XiaoMi/mace, PaddlePaddle/Paddle-Lite, google-ai-edge/LiteRT, google-ai-edge/litert-torch, microsoft/onnxruntime, pytorch/executorch, mlc-ai/mlc-llm, apache/tvm, ggml-org/llama.cpp, chongzhou96/EdgeSAM, xinghaochen/TinySAM, mit-han-lab/efficientvit, THU-MIG/RepViT, yformer95/efficient-sam, meta-pytorch/segment-anything-fast, ultralytics/ultralytics, nihui/ncnn-android-yolov8, FeiGeChuanShu/segment-anything-ncnn, slz929/SAM-Android-ncnn, slz929/mobileSAM-Android-MNN, wangzhaode/mnn-segment-anything, shubham0204/Segment-Anything-Android, PrathamX595/Kortex-AdobeSubmission, qualcomm/aimet, qualcomm/ai-hub-models, mlcommons/mobile_app_open, mlcommons/mobile_models, XiaoMi/mobile-ai-bench, samsung/ONE}
- 官方文档：developer.android.com/ndk/guides/neuralnetworks（NNAPI 弃用）｜developers.google.com/edge/litert/performance/gpu（GPU delegate 算子清单）｜developers.google.com/edge/litert/next/{npu,qualcomm,mediatek,samsung}｜onnxruntime.ai/docs/execution-providers/｜qualcomm.com/developer/software/*｜neuropilot.mediatek.com｜ai-benchmark.com/{ranking_processors,tests}.html
- 论文：arXiv ID 见 §3 各表（全部经 arxiv.org 核实）。
- 设备：mobile.pconline.com.cn/1964/19641513.html（V2509A Geekbench 现身）｜arm.com/zh-CN/company/success-library/vivo-x300-smartphones。
