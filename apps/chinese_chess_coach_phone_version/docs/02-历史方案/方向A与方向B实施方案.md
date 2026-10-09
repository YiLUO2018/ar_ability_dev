# 方向 A 与方向 B 实施方案

> ⚠️ **已归档（2026-10-09）**：方向 A（EdgeSAM 轻量分割）已于 2026-10-04 落地并大幅演进（候选仲裁/角点外推复原等，远超本文范围）；方向 B（重训角点模型）已放弃（合成→真实域差不可行）。现状见 `00-现状/里程碑M1总结.md`。

**日期**：2026-10-02
**状态**：已归档（A 落地、B 放弃）
**前置结论**：见 `棋盘定位方案探索记录.md`——SAM 分割是唯一稳定定位方案（~3–4px），角点模型与 Hough 均已验证失效。

---

## 一、当前状态摘要

- **已跑通**：相机预览 → 角点定位 → 透视校正 → 整盘分类（TFLite）→ 引擎走子建议，整条代码链路已在真机跑过。
- **卡点**：角点定位（`KeypointBoardDetector`，合成训练的回归模型）在真实斜拍图上输出垃圾角点，导致后续全链路失效。
- **已验证可行**：SAM（ViT-B）分割粗定位，角点误差 ~3–4px。
- **下一步**：把 SAM 换成端上能跑的轻量模型（EdgeSAM），实现 `SegmentationBoardDetector`，打通链路。

---

## 二、方向 A：EdgeSAM 端上分割定位（具体实现）

### 目标
用轻量分割模型（EdgeSAM）替换角点回归模型，实现稳定的棋盘定位，让「定位 → 校正 → 识别 → 建议」整链路可用。

### 技术选型
- **模型**：EdgeSAM（官方提供 TFLite，专为边缘设备）。
  - `edge_sam_3x`：量化，~9MB，最快；`edge_sam`：~40MB，精度更高。
  - 备选：MobileSAM（需自转 TFLite，链路长，不优先）。
- **提示策略（初版）**：中心点提示（假设棋盘接近画面中心）。后续升级：多候选点 + 按掩码面积/宽高比（≈8:9）选最优。

### 实现步骤

1. **获取模型**：下载 EdgeSAM TFLite（`edge_sam_3x` 或 `edge_sam`），放进 `app/src/main/assets/models/`。
2. **确认契约**：查 EdgeSAM TFLite 的输入/输出张量——输入图像尺寸（通常 1024×1024）、RGB 归一化方式、point prompt 张量格式；输出为掩码 logits 或二值掩码。
3. **实现 `chess/SegmentationBoardDetector.kt`**（实现现有 `BoardDetector` 接口）：
   ```
   detect(bitmap) 流程：
   ① 预处理：缩放 → 模型输入尺寸，RGB [0,1]
   ② 构造 prompt（中心点）
   ③ 推理 → 掩码
   ④ 后处理：掩码上采样回原图 → findContours → 取最大轮廓
      → approxPolyDP 拟合 4 顶点 → 排序 TL,TR,BR,BL
   ⑤ warp 到 canonical（见下方「坐标定义」）
   ⑥ 返回 BoardInfo
   ```
4. **接入 `chess/BoardDetectorFactory.kt`**：分割检测器优先，`KeypointBoardDetector` / `GridBoardDetector` 作兜底。
5. **真机验证**：跑通全链路，看识别/建议是否正常。

### 关键接口
- `BoardDetector`（`chess/BoardDetector.kt`）：`detect(Bitmap): BoardInfo?` + `close()`。
- `BoardInfo`（`chess/BoardInfo.kt`）：`corners`、`warped`（450×500 canonical 位图）、`cellSize`、`originX/Y`。
- `Homography`（`chess/Homography.kt`）：`compute(dst, src)` + `warp(...)`。
- `BoardGeometry`（`chess/BoardGeometry.kt`）：CELL=50、MARGIN=25、BOARD_W=450、BOARD_H=500。

### ⚠️ 关键坑：坐标定义（物理边 vs 网格线）

SAM 掩码给出的是**物理棋盘边**，而 `BoardGeometry.corners` 是**最外层网格线交点**（canonical 内缩 MARGIN=25px）。两者差 25px，必须分开处理：

- SAM 物理边四角 → 应 warp 到 canonical **物理边界** `[(0,0),(450,0),(450,500),(0,500)]`（不是 `BoardGeometry.corners`）。
- 这样网格交点自然落在 `(MARGIN + file*CELL, MARGIN + rank*CELL)`，与分类器预期一致。
- 因此需要新增一组「物理边界」canonical 常量（区别于 `BoardGeometry.corners` 网格角点）。
- `BoardInfo.originX/Y` 仍为 MARGIN(25)、`cellSize` 仍为 CELL(50)（分类器/叠加层按网格交点读）。

### 风险
- EdgeSAM TFLite 接口（尤其 prompt 张量）需实测确认。
- 中心点提示对「棋盘偏离画面中心」不鲁棒 → 后续加多候选点策略。
- 掩码边缘模糊/角点圆钝 → `approxPolyDP` 误差几 px，对识别影响需真机评估。

---

## 三、方向 B：SAM 伪标签重训角点模型（优化）

### 目标
保留端上快速小模型（角点回归），用真实数据重训解决域差，兼顾精度与速度。

### 核心思路
SAM（离线，可跑大模型）给真实帧打**伪标签** → 混合合成数据微调角点模型 → 重新导出 TFLite。

### ⚠️ 关键前提：统一坐标定义
- 当前角点模型学的是「网格线交点」，SAM 给的是「物理棋盘边」，差 25px 边框。
- **建议：让角点模型改学「物理棋盘边四角」**（与 SAM 掩码对齐），这样伪标签最干净、无系统偏差。
- 配套改动：`render.py` 合成数据的角点标签、`CornerDataset`、真机 warp 坐标定义都要统一到物理边。

### 数据方案
1. **真实采集**（主）：真机多角度/多光照/多背景拍棋盘，覆盖「斜拍 + 杂乱」这类当前失败场景。
2. **网络视频**（辅，多样性）：象棋视频抽帧，注意：
   - 抽帧去重（时间间隔 + 帧间相似度），避免相邻帧冗余。
   - 过质量闸门：mask 分数、面积占比、四边形拟合误差、宽高比≈8:9，不合格丢弃。
   - 与真实采集按比例混合，别被某类视频棋盘带偏。
3. **合成数据**（保留，提多样性/正则）：与真实伪标签混合微调。

### 伪标签生成
- SAM 分割 → 掩码 → `approxPolyDP` 拟合四角 → 作角点伪标签。
- 附质量分（mask score / 面积 / 拟合误差），低分样本人工抽检。

### 训练流程
1. 小规模验证：`debug_frames` 现有真实帧 + SAM 伪标签 → 微调 `CornerModel` → 看能否收敛到 ~3–4px。
2. 扩大规模：真机批量采集 + 视频抽帧 + SAM 打伪标签 → 混合合成数据微调。
3. 重新导出：`tools/train/export.py` 导出 TFLite → 替换 `assets/models/board_corner.tflite`。

### 可选：知识蒸馏
把 SAM 当 teacher、角点模型当 student，直接拟合 SAM 输出的四角（本质即伪标签蒸馏），并让角点模型的置信度学「分割质量」信号，使「无棋盘」门控更可靠。

### 验证指标
- 角点误差（vs 人工标注真值）< 5px（理想 <2px）。
- 置信度门控：真实棋盘高分、无棋盘低分，可分离。
- 端到端：识别准确率、走子建议正确率。

---

## 四、下次快速开始指南

### 立即要做的第一步
> **查 EdgeSAM TFLite 接口 + 下载模型**，然后实现 `SegmentationBoardDetector.kt`。

### 关键文档
- `docs/棋盘定位方案探索记录.md` — 6 方法对比与验证结论。
- `docs/方向A与方向B实施方案.md` — 本文件（实施细节）。
- `docs/真实数据自动标注方案.md` — 阶段 1.5 伪标签/几何精修计划。
- `docs/开发日志.md` — 历史开发记录。

### 关键代码路径
- 定位接口：`app/src/main/java/.../chess/BoardDetector.kt`、`BoardDetectorFactory.kt`、`BoardGeometry.kt`、`Homography.kt`
- 现有定位实现：`KeypointBoardDetector.kt`（角点模型，失效）、`GridBoardDetector.kt`（Hough，失效）
- 识别/引擎：`TFLitePieceRecognizer.kt`、`MainActivity.kt`
- 训练/导出：`tools/train/`（models.py、train_corner.py、export.py、datasets.py）
- 真实数据工具：`tools/label_real/`（refine.py、lines.py、geometry.py、test_sam*.py）
- SAM 资源：`L:\Reinforcement_Learning\segment\segment-anything-main`（权重在 `models/`）

### 环境命令（bash，Windows）
```bash
export JAVA_HOME="/l/Program Files/AndroidStudio/jbr"
ADB="/l/AndroidStudioSdk/platform-tools/adb.exe"   # 用引号包路径，必要时 MSYS_NO_PATHCONV=1
cd /l/projects/ar_ability_dev/apps/chinese_chess_coach_phone_version
./gradlew :app:assembleDebug          # 构建
"$ADB" install -r app/build/outputs/apk/debug/app-debug.apk
"$ADB" shell am start -n com.arability.chinese_chess_coach_phone_version/.MainActivity
"$ADB" logcat -s KeypointBoardDetector:I TFLitePieceRecognizer:I ChessCoach:I   # 看日志
```

### 调试帧
- `debug_frames/board_21.png` — 斜拍真实棋盘（480×640），真实角点 TL(116,77) TR(346,72) BR(376,297) BL(96,306)。
- `debug_frames/board_21_sam_result.png` — SAM 分割可视化。

### 待办清单（按优先级）
1. [ ] 查 EdgeSAM TFLite 接口、下载模型（edge_sam_3x 优先）。
2. [ ] 实现 `SegmentationBoardDetector.kt`（含物理边→canonical 边界的坐标映射）。
3. [ ] 接入 `BoardDetectorFactory`，真机验证全链路。
4. [ ] （后续）方向 B：统一坐标定义 → SAM 伪标签 → 重训角点模型。
