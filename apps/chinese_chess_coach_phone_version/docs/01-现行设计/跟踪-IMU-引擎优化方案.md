# 跟踪 / IMU 补偿 / 引擎优化方案

本文记录「定位稳定性增强」「转动补偿」「下一步走法建议」三条线的改动项与实施状态。
背景：方向 A（EdgeSAM ONNX 分割定位）已在真机跑通，误差 ~3–4px。

## 关键发现与重排优先级（2026-10-04 真机）

**真机日志确认：棋子识别失效**。`piece_classifier.tflite`（合成数据训练）在真实照片上输出垃圾：
`grid=..kkrkkk./...`（随机散落）、`grid=.........`（全空）。与之前角点模型同类「合成→真实域差」问题。
因此「下一步走法建议」当前**被棋子识别阻塞**——棋盘定位（SAM）已稳定，但送入引擎的局面是错的，
引擎在垃圾局面上算不出有意义的结果。

**重排后的执行顺序**：
1. **修棋子识别**（阻塞项）——逐子检测 或 伪标签重训（见「棋子识别方案调研」）。
2. **引擎 + 走子文字解释**——识别正确后才有意义（把走法转成「炮二平五」等记谱 + 原因）。
3. **IMU 旋转补偿**——改善移动跟随，独立于识别。

## 棋子识别方案调研：OCR / 逐子检测 vs 整盘分类

**结论先行**：文字识别模型存在，且「识别文字 + 给出框位置」**可以替代**整盘分类，而且通常更鲁棒。
但通用 OCR 对这个任务是「大材小用且不适配」——象棋棋子是**孤立单字 + 360° 任意旋转 + 小目标**，
属于**闭集单字识别**（只有 11 个不同字：帅/仕/相/兵/将/士/象/卒/马/车/炮），一个微型 CNN 分类器
就能做到 99%+，远好于整盘分类（Swin Transformer2 整盘 full-correct 仅 0.83）。

**业界主流路线（逐子检测，优于整盘分类）**：
| 方案 | 要点 | 精度/性能 |
| :-- | :-- | :-- |
| Hough 圆检测 + HSV 判色 + CNN 认字（专利 CN109726721A） | 圆检测定位棋子 → HSV 分红黑 → CNN 认字 | 99.995%，20–22ms/子 |
| SSD-MobileNet（Edge AI 论文） | 14 类目标检测，ONNX/TensorRT 部署 | 0.022s/帧，实时 |
| YOLOv5 | 检测每颗棋子 + 置信度分级 | 批量高效 |

**两个关键子问题**：
1. **颜色**：马/车/炮三种字红黑共用，OCR 只能给字、不能给色；需额外 HSV 阈值（红 0–20°/150–180°，
   黑 30–130°）或检测器直接输出颜色类。
2. **位置**：框中心 → 映射到最近交叉点（仍需棋盘几何/单应，但比整盘分类容错高，因为逐子独立）。

**两条落地路径**：
- **路径 A（端上替换）**：用一个小检测器（YOLO-nano / SSD-MobileNet，14 类 + 颜色）替代现有
  `TFLitePieceRecognizer`。不依赖完美 warp，逐子独立检测，最鲁棒。
- **路径 B（离线伪标签 → 重训）**：用强模型（多模态大模型如 Gemini/GPT-4o，或训练好的检测器）
  给真实照片打标 → 重训现有分类器/检测器。大模型只适合离线标注，不适合端上实时。

**参考**：
- PaddleOCR 端侧（PaddleLite/ncnn/ONNX）：`github.com/equationl/paddleocr4android`、`deepwiki.com/PaddlePaddle/PaddleOCR/6.6-mobile-and-edge-deployment`
- 逐子检测论文：`jurnal.polibatam.ac.id/index.php/JAIC/article/download/10056/3009/33493`（SSD-MobileNet Edge AI）
- 整盘分类开源：`github.com/TheOne1006/chinese-chess-recognition`（Swin Transformer2）
- 大模型 OCR 对比实测：`linux.do/t/topic/2089854/2`

## 变更优先级总览

| 级别 | 编号 | 改动项 | 状态 |
| ---- | ---- | ------ | ---- |
| P0 | Q1① | 提示点回灌（用上次四角质心作为下次 SAM 提示点） | ✅ 已实现 |
| P0 | Q1② | 锁定 + 迟滞（短暂漏检时保持上次结果） | ✅ 已实现 |
| P0 | Q3-A | 修复 `BoardOverlayView.intersectionX/Y` 映射 bug | ✅ 已实现 |
| P1 | Q3-A | 局面时间投票（最近 N 帧逐格多数投票） | ✅ 已实现 |
| P1 | Q2 | 四角 EMA 平滑（无 IMU 的低成本抖动抑制） | ✅ 已实现 |
| P0 | Q3 | 修棋子识别（逐子检测 或 伪标签重训） | ⏳ 待实施（阻塞项） |
| P1 | Q3 | 引擎 + 走子文字解释（记谱 + 原因） | ⏳ 待实施 |
| P2 | Q2 | 旋转矢量 IMU 补偿（rotation-only v1） | ⏳ 待实施 |

---

## P0-① 提示点回灌

**问题**：SAM 每次都用画面中心 `(0.5, 0.5)` 作点提示。棋盘偏移时，中心点可能落在棋盘外或
边缘，导致掩码不稳定或选中邻近大物体。

**方案**：`SegmentationBoardDetector` 的 `promptX/promptY` 从 `val` 改为 `var`；每次检测成功后，
用四个物理角的质心（归一化到 `[0,1]`，钳制在 `[0.02, 0.98]`）回灌为下一次提示点。这样提示点
会「粘」在棋盘上，显著提升连续帧的掩码稳定性。

**文件**：`chess/SegmentationBoardDetector.kt`

---

## P0-② 锁定 + 迟滞

**问题**：`MainActivity.processFrame` 每 `DETECT_EVERY_N=5` 帧检测一次；一旦 `detect` 返回
`null`，立即清空 `latestBoardInfo/latestBoard`，叠加层闪烁。

**方案**：引入 `lockMisses` 计数器。检测成功时清零并更新结果；检测失败时计数，只有连续失败
超过 `LOCK_MAX_MISSES`（默认 3，即约 15 帧）才真正清空。期间保持上一次棋盘与走法，配合
提示点回灌，能桥接绝大多数瞬时漏检。

**文件**：`MainActivity.kt`

---

## P0-③ 修复 `BoardOverlayView` 映射 bug

**问题**：`intersectionX(info, file)` 内部把 `fy` 硬编码为 0，`intersectionY(info, rank)` 把 `fx`
硬编码为 0，导致 `(file, rank)` 的 X 永远取顶边、Y 永远取左边，非边界交叉点全部错位。

**方案**：合并为单一 `intersection(info, file, rank)`，X/Y 同时依赖 `file` 与 `rank`，做真正的
四角双线性插值；并按 `originX/originY` 的 MARGIN 内缩，使网格交叉点映射到物理棋盘边内部，
而非物理角本身。

**注意（坐标定义坑）**：`BoardInfo.corners` 语义因检测器而异——
- `SegmentationBoardDetector`（当前主路径）：`corners` 是**物理棋盘四角**，warp 到
  `BoardGeometry.physicalCorners`（整张 canonical 图 0..450 / 0..500）。
- `KeypointBoardDetector` / `GridBoardDetector`（已失效降级路径）：`corners` 是**网格最外层交点**，
  warp 到 `BoardGeometry.corners`（内缩 MARGIN）。

本修复按主路径（物理角）实现，用 `(origin + file*cell) / BOARD_W|BOARD_H` 归一化后插值。
若后续要统一降级路径的语义，需单独对齐（见 `棋盘定位方案探索记录.md`）。

**文件**：`overlay/BoardOverlayView.kt`

---

## P1 局面时间投票

**问题**：棋子分类器单帧输出有噪声，导致识别出的局面（及建议走法）跳变。

**方案**：`MainActivity` 维护最近 `HISTORY_SIZE=5` 帧识别网格（`ArrayDeque`）。每次成功识别后
入队、淘汰最旧；对每个 `(rank,file)` 格在历史中做多数投票（`Piece` 是 data class，可直接作 key）。
平票时保留更近期的一票（按时间倒序首次出现者）。用投票结果构建 `latestBoard`。

**文件**：`MainActivity.kt`

---

## P1 四角 EMA 平滑

**问题**：即便定位稳定，逐帧四角仍有亚像素抖动，传导到 warp 与棋子采样。

**方案**：`SegmentationBoardDetector` 对输出四角做 EMA（`EMA_ALPHA=0.3`）：
`out = α·raw + (1-α)·prev`。当四角质心相对上一帧位移超过画面对角线 15% 时判定为「新棋盘」，
直接采用 raw 并重置平滑，避免跨场景拖影。

**文件**：`chess/SegmentationBoardDetector.kt`

---

## P2 IMU 旋转补偿（待实施）

**目标**：在维持当前检测帧率（每 5 帧检测一次）的前提下，用手机姿态在两帧之间外推四角，
输出「实时」的框位置。

**方案（rotation-only v1）**：
1. `SensorManager.TYPE_ROTATION_VECTOR`（或游戏旋转矢量）采样设备姿态。
2. 结合 `CameraCharacteristics.LENS_INTRINSIC_CALIBRATION` 内参，把姿态旋转增量投影到像平面。
3. 单应矩阵分解（Zhang）得到平面法向量，用旋转增量更新棋盘四角的透视变换。
4. 仅补偿「绕光轴的旋转」，平移/尺度留给下一次真实检测刷新，避免累积漂移。

**依赖/难点**：需要相机内参、时间戳对齐、旋转增量→像平面单应的正确推导；建议先在 Python 端
用 debug_frames 验证后再落 Android。

**文件**：`MainActivity.kt`（新增 sensor 监听）+ 新的 `RotationCompensator.kt`。

---

## P2 引擎增强：开源引擎选型评估（2026-10-04 更新，仅评估未实施）

**决策变更**：原方案是「先 Kotlin 增强、必要时上 XQWLight JNI」。经讨论改为**直接接入成熟开源引擎**——
象棋策略早已被攻破，自研/增强性价比低。本节记录选型评估结果与实施前的准备工作。

### 候选对比

| 引擎 | 棋力 | 语言 | 协议 | 许可 | Android 集成 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **XQWLight（象眼轻量版）** | 中强（PVS+置换表+杀手+历史+空着裁剪，够教学） | **C++/Java/JS/AS** | UCCI | **GPL-2.0** ⚠️ | **Java 直译，最轻** |
| Pikafish（皮卡鱼） | 最强（Stockfish+NNUE，超人类） | C++ | UCI | GPL-3.0 + NNUE 权重禁商用 | NDK+JNI+UCI+nnue，最重 |
| ElephantEye（象眼） | 强 | C++ | UCCI 3.0 | LGPL-2.1 | NDK+JNI+UCCI，重 |

### 推荐：XQWLight Java 版（理由 + 关键发现）

1. **有 Java 版，无需 NDK/JNI**（纠正之前「XQWLight 需 JNI」的错误认知）：Java 是 Android 一等公民，
   可直接以 `.java` 源码引入，零移植风险。
2. **FEN 即接口**：xqwlight 的 `Position.fromFen()/toFen()` 与我们 `Board.toFen()` 同用标准象棋 FEN，
   集成只需 FEN 往返，无需深耦合。
3. **API 已确认**（多来源核实）：
   - `Position`：`fromFen(str)` / `toFen()` / `generateAllMoves(mvs)` / `makeMove(mv)` / `undoMakeMove()` / `legalMove(mv)`；着法用 `int`（`Position.MOVE(src,dst)`）编码。
   - `Search`：`new Search(pos, depth)` → `searchMain(mills)`（按时间）/ `searchMain(depth, mills)`；`getKNPS()`。
4. **移植先例**：Dart 包 `cchess_engine` 已从 xqwlight.java 翻译成功（`XQIsoSearch.getMove(fen)`），
   证明代码可移植、逻辑稳定。

### 源码核实结论（2026-10-04 已通读全部源码）

已通过 `gh-proxy.com` 镜像下载并通读 `Java ME` 版全部源码，结论如下（此前「待确认」项全部落实）：

- **引擎 = 3 个文件**：`Position.java`(1133 行) / `Search.java`(455 行) / `Util.java`(128 行)，纯 Java，无第三方依赖。
- **许可 = GPL-2.0**（`LICENSE` 文件 + 每个源文件头均明确 GPL v2）。
- **FEN 完全兼容（关键）**：xqwlight 的 `FEN_PIECE = "KABNRCP kabnrcp"`，字母 `k/a/b/n/r/c/p`
  与 `Board.toFen()` **逐字符一致**；开局 FEN 两者完全相同：
  `rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w`。
  `fromFen` 解析到空格后只读一个 `'w'/'b'` 定执方，多余后缀（` - - 0 1`）自动忽略——**零 FEN 映射成本**。
- **着法编码**：`MOVE(src,dst) = src + (dst << 8)`；解码 `SRC(mv)=mv&255`、`DST(mv)=mv>>8`；
  格坐标 `sq = (file+3) + ((rank+3) << 4)`（file 0..8、rank 0..9）。映射公式：
  `fromFile=(src&15)-3`，`fromRank=(src>>4)-3`（dst 同理）。双向无损。
- **搜索接口**：`new Position()` → `fromFen(fen)` → `new Search(pos, hashLevel)` → `searchMain(millis)`
  → 返回 `int mv`（0 = 无着法/被将死）。`hashLevel` 建议 16（`2^16` 个 HashItem ≈ 786KB）。
- **搜索算法**：PVS（principal variation search）+ 静态搜索(quiescence) + 置换表 + 杀手着法 +
  历史启发 + 空着裁剪 + 迭代加深（时间受限）。
- **评估函数**：带位置表的子力评估（`PIECE_VALUE` 7×256 位置分值），**远优于**当前 `KotlinChessEngine` 的纯子力。
- **开局库可选**：`static{}` 尝试从 classpath 读 `/book/BOOK.DAT`，缺失则 `bookSize=0`、`bookMove()` 返回 0、
  自动转入搜索——**无需携带 BOOK.DAT**。
- **Android 兼容**：只用 `java.io.InputStream` / `java.util.Random` / `StringBuffer`，无 Android 专用 API，
  直接可编译（`package xqwlight;` 保留即可）。

### ⚠️ 许可风险（需决策）

XQWLight 是 **GPL-2.0**（传染性 copyleft）：若将 App **闭源分发/商业化**，需整体以 GPL-2.0 开源，否则违规。
- 若当前仅内部验证 → GPL-2.0 无碍。
- 若未来商业化闭源 → 需改用 **ElephantEye（LGPL-2.1，可闭源链接，但 C++ 要 NDK+JNI）**，或自研。
- **建议**：验证阶段用 XQWLight；正式商业化前评估 ElephantEye 或自研替代。

### 集成方案（实施时）

```
Board.toFen() → XQWLight Position.fromFen() → Search.searchMain(mills)
→ 解码 int move → src/dst → Board.applyMove() → Move
```
- 新增 `chess/XqwlightEngine.kt`（或直接引入 xqwlight 的 .java）+ 实现现有 `ChessEngine` 接口。
- `ChessEngineFactory` 改一行：`KotlinChessEngine` → `XqwlightEngine`。
- 引擎搜索放后台线程（沿用现有异步处理）。

### 准备工作清单（实施前待办）

| # | 事项 | 状态 |
| :-- | :-- | :-- |
| 1 | 获取源码 | ✅ 已完成（gh-proxy.com 镜像，3 文件 + LICENSE 已通读） |
| 2 | 确认 FEN 字母映射与 `Board.toFen()` 一致 | ✅ 已确认（逐字符一致） |
| 3 | 确认 move int 解码 | ✅ 已确认（公式见「源码核实结论」） |
| 4 | 棋力验证：跑 benchmark/简单对局，确认「够教学」 | ⏳ 待验证（算法已足够，需实测） |
| 5 | 耗时验证：手机端 `searchMain(mills)` 每步耗时 | ⏳ 待验证（按时间上限，传 500–1000ms 即受控） |
| 6 | 许可决策：GPL-2.0 vs 商业化（见上） | ⏳ 待决策 |

### 源码获取方式（供实施时复用）

GitHub 本机直连失败（connection reset），**可用代理 `gh-proxy.com` 下载**：

```bash
# 整仓 zip（约 3MB，会因连接不稳而中断，建议逐文件下载）
curl -L "https://gh-proxy.com/https://github.com/xqbase/xqwlight/archive/refs/heads/master.zip" -o xqwlight.zip
# 逐文件（推荐，仅引擎核心 3 文件 + LICENSE，稳定）
curl -L "https://gh-proxy.com/https://raw.githubusercontent.com/xqbase/xqwlight/master/Java%20ME/src/xqwlight/Position.java" -o Position.java
curl -L "https://gh-proxy.com/https://raw.githubusercontent.com/xqbase/xqwlight/master/Java%20ME/src/xqwlight/Search.java" -o Search.java
curl -L "https://gh-proxy.com/https://raw.githubusercontent.com/xqbase/xqwlight/master/Java%20ME/src/xqwlight/Util.java" -o Util.java
```

引擎核心文件（`Java ME/src/xqwlight/` 下）在 `Ajax` / `Applet` / `Java ME` 三个版本中内容一致，取 `Java ME` 版最干净（无 Web 依赖）。

### 参考

- xqwlight 官方：`github.com/xqbase/xqwlight`（镜像 `droiter/xqwlight`、`littleflute/xqwlight`）
- 象眼：`github.com/xqbase/eleeye`（LGPL-2.1）
- 皮卡鱼：`github.com/official-pikafish/Pikafish`
- Dart 移植先例：`pub.dev/packages/cchess_engine`

---

## 实施记录

- 2026-10-04：P0①②③、P1（时间投票 + EMA）已实现，`assembleDebug` 通过；另补了「最大面积校验」
  与「棋盘裁切校验」两处鲁棒性修复（真机调试发现背景误分割 / 棋盘部分出画）。
- 2026-10-04：真机日志确认棋子识别失效（合成数据泛化失败），成为「下一步走法」阻塞项；据此重排
  优先级为「修棋子识别 → 引擎+文字解释 → IMU」，并完成棋子识别方案调研（OCR/逐子检测 vs 整盘分类）。
