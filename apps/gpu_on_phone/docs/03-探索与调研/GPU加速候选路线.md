# GPU 加速候选路线（决策跟踪表）

> 状态：路线修订版（2026-10-14 晚）——外部调研完成后修订状态与前置事实；**均未经本机实测**。
> 详细证据（GitHub 项目全景 + arXiv 论文 + 加速比数据表）见 `GPU加速调研汇总（GitHub与arXiv）.md`；各路线实验设计与裁决门见 `../01-现行设计/实验验证计划.md`。
> 用法：每条路线实测后在此更新「实测结论」；死路结论保留（防重走弯路）。

## 0. 前置事实（2026-10-14 调研确认，真机 getprop 待复核）

| 项 | 结论 | 影响 |
|---|---|---|
| V2509A SoC | **联发科天玑 9500（MT6990）**= vivo X300 标准版（入网信息 + Geekbench 现身 + Arm 官网三源印证） | ❌ D QNN 路线关闭（仅高通） |
| GPU | **Mali-G1-Ultra MC12** | F 原生 OpenCL 降权（Mali OpenCL 通常不开放）；C LiteRT GPU（OpenCL→GLES 回退）适用 |
| Android 版本 | 新机（2025-10 发布）→ Android 15/16 | NNAPI 已弃用 → ❌ B 路线关闭 |
| NPU 通路 | **天玑 9500（MT6990）在 LiteRT MediaTek 官方支持列表内** | ✅ 新增 C+ 路线（CompiledModel API，minSdk 31/arm64/.tflite/量化） |
| ORT 能力边界 | **ONNX Runtime Android 官方包无任何 GPU EP**（实测 Java API 仅 NNAPI/XNNPACK/QNN/CUDA） | 「ORT 内换 EP 上 GPU」不通；ORT 1.22→1.31 升级有免费 CPU 红利 |

## 1. 路线 A：ONNX Runtime CPU EP / XNNPACK EP（基线）——✅ 保留，先做

- **原理**：XNNPACK 为 ARM NEON 优化的 CPU 算子库，ORT 经 EP 挂载；官方 AAR 自带。
- **行动**：开 XNNPACK EP + 模型转 fp16 + **ORT 升级 1.22.0→1.31.0**（2026-10-09 发布，一年半的优化红利）。零迁移风险，作一切对比的分母。
- **验证点**：三件套（EdgeSAM enc/dec、YOLO）在 CPU vs XNNPACK×fp32/fp16 的延迟差。

## 2. 路线 B：NNAPI——❌ 关闭（2026-10-14）

- 关闭依据：Android 15 起 NNAPI 官方弃用；LiteRT Play Services 版已无 NNAPI delegate；AI Benchmark 榜上 2024–2026 旗舰（含天玑）全部改用厂商 delegate（QNN HTP / MTK Neuron）。存量机或仍能跑，但属无人维护的遗产路径，不再投入。
- ~~原文（建项时）保留于 git 历史：NNAPI EP 接入门槛低但算子覆盖不全→图分区回退 CPU，负优化是常见结局。~~
- 残余价值：象棋 M1 期间的 NNAPI 初步 bench 结论仍值得搬运归档（历史对照）。

## 3. 路线 C：LiteRT（TFLite）GPU delegate——✅ 升为主线

- **原理**：GPU delegate 优先 OpenCL、无则回退 GLES 3.1 compute；fp16/fp32；支持序列化（初始化最多降 90%）。
- **调研修正（重要）**：EdgeSAM encoder 是 **RepViT-M1 纯 CNN**（非 ViT）——GPU delegate 的 ~24 种 CNN 算子覆盖良好；decoder（two-way transformer）不在算子清单 → 留 CPU（社区实测 <100ms，可接受）。**encoder 占 ~90% 耗时，吃下它就是主要收益**。
- **风险**：无官方 ONNX→tflite 路径（需从 EdgeSAM/YOLO 的 PyTorch 上游经 LiteRT Torch 导出，Beta）；小模型 GPU 负优化警示（ncnn 作者 + iPhone LLM 0.75× 实证）——YOLOv8n 必须实测。
- **验证点**：① 从 PyTorch 上游导出 tflite 的可行性；② encoder GPU vs CPU 延迟；③ decoder 留 CPU 的拆图开销；④ YOLOv8n 是否负优化。

## 3+. 路线 C+：LiteRT MediaTek NPU（CompiledModel）——✅ 新增（上限路线，放 G2）

- **依据**：MT6990 在官方支持列表；NPU 对视觉 encoder 公开参照 20–45×（arXiv:2606.27906）。
- **门槛**：minSdk 31（我们 App 是 24，需运行时分支）；模型须量化（int8/int16 是 NPU 满速格式）；AOT（litert CLI）或端上 JIT 编译（有缓存，初始化 7465ms→198ms 实测参照）。
- **风险**：天玑 NPU 跑分波动大（9400 代成绩异常低）；只覆盖天玑 9500 一档设备；量化精度需过 PTQ4SAM 类方案。
- **验证点**：G2 阶段对 encoder 单独做 NPU 化 A/B（decoder 与其余留 GPU/CPU，LiteRT 支持子图回退）。

## 4. 路线 D：Qualcomm QNN——❌ 关闭（2026-10-14）

- 关闭依据：V2509A = 天玑 9500，非高通。若未来购入高通真机再开（届时 ORT QNN EP 需源码构建，LiteRT QNN delegate 覆盖 SM8450–SM8850）。

## 5. 路线 E：ncnn（Vulkan）/ MNN（OpenCL+Vulkan）——✅ 保留备选（短名单 #1/#2）

- **ncnn**：23.9k 星、当天活跃、官方 Vulkan 预编译包、pnnx 直吃 ONNX、官方 YOLO example 全、SAM 有旧社区先例（2023 停更）。BSD-3。
- **MNN**：16.2k 星、OpenCL 内核调优最深、ONNX 转换器 158 算子（**ONNX 资产保护最好**）、SAM 有 2 个小型先例。Apache-2.0。
- **定位**：若 LiteRT 转换链路（无 ONNX 路径）受阻，E 接棒为「保 ONNX 的 GPU 路线」；ncnn 押 Vulkan（跨芯片稳），MNN 双后端对冲。
- **验证点**：pnnx/MNN 转换 EdgeSAM 三件套的算子通过率；decoder 动态 shape（[1,N,2] 点提示）在静态图引擎的表达（定长 padding 绕行）。

## 6. 路线 F：原生 OpenCL / Vulkan compute——⚠️ 保留但降权

- Mali 的 OpenCL 通常不开放给普通 App；HybridInfer（arXiv:2609.30270）实证移动 OpenCL 有 runtime 崩溃案例。仅当所有框架路线对关键算子失效时再评估。

## 7. 非推理侧的旁路优化（记入但非本项目主线）

- 模型侧：降输入分辨率 1024→512（EfficientViT-SAM 验证 512 路线，棋盘场景大概率够用——**这可能是单笔最大的免费收益之一，可与 GPU 加速叠加**）。
- 流水线侧：象棋 BoardTracker 帧间跟踪（少调 SAM）与 GPU 加速正交，双管齐下。

## 验证状态汇总

| 路线 | 状态 | 结论落点 |
|---|---|---|
| A CPU/XNNPACK + ORT 升级 | ⬜ 待跑（G1 首项） | 本文 §1 |
| B NNAPI | ❌ 关闭（弃用） | 本文 §2 |
| C LiteRT GPU delegate | ⬜ 待跑（G1 主线） | 本文 §3 |
| C+ LiteRT MTK NPU | ⬜ 待跑（G2，上限路线） | 本文 §3+ |
| D QNN | ❌ 关闭（非高通） | 本文 §4 |
| E ncnn / MNN | ⬜ 备选（转换链路受阻时接棒） | 本文 §5 |
| F 原生 | ⚠️ 兜底降权 | 本文 §6 |
| 旁路：512 分辨率 | ⬜ 待验证（与 GPU 叠加） | 本文 §7 |
