# 架构修订方案 — 保留棋盘定位 + YOLO 棋子识别

**日期**：2026-10-02（方案）；2026-10-06（实施更新）
**状态**：已实施（YOLO-nano 自训 + ONNX 端侧部署，见 M5–M7）
**触发原因**：后续需在画面中用 UI（箭头/高亮）向使用者说明下一步走法，而非仅文字说明。

---

## 1. 核心结论

要在画面中画出「下一步怎么走」，必须先知道**棋盘网格坐标 ↔ 图像像素**的映射关系。这个映射由**棋盘定位**（四角 + 单应矩阵）提供，因此棋盘定位是 UI 叠加的**几何基础，必须保留**。

据此把识别链路拆成两层，职责清晰、互不重叠：

| 模块 | 角色 | 提供 | 处理 |
| :-- | :-- | :-- | :-- |
| `BoardDetector`（棋盘定位） | **几何层** | 棋盘四角 + 单应矩阵（grid↔image） | 保留并升级（已用 `SegmentationBoardDetector`） |
| `PieceRecognizer`（棋子识别） | **语义层** | 每颗棋子的类别与位置 | 换成 YOLO 检测（`YoloPieceDetector`，自训 yolov8n） |
| `BoardOverlayView`（叠加渲染） | **展示层** | 画黄色外框 + 走法箭头/高亮 | 保留并升级 |

> 「定位」回答**在哪**，「识别」回答**是什么**。两者互补，不冗余。

---

## 2. 修订后数据流

```
相机帧 Bitmap
   │
   ├─ BoardDetector.detect(bitmap) ──► BoardInfo(四角 + 单应矩阵 grid↔image)
   │                                        │
   │                                        ├──► 供 UI 叠加：grid → image（画箭头/高亮）
   │                                        │
   │                                        └──► YoloPieceDetector.detect(bitmap, boardInfo)
   │                                                  │  YOLO 检测棋子（框 + 类别）
   │                                                  │  每个 bbox 中心 ──单应──► 最近交叉点
   │                                                  ▼
   │                                          Array<Array<Piece?>>  (9×10 网格)
   │                                                  ▼
   │                                          Board(局面) ──► ChessEngine.bestMove ──► Move(from,to)
   │                                                                                   │
   └───────────────────────────────────────────────────────────────────────────────────┤
                                                                                       ▼
                                  BoardOverlayView.draw(move, boardInfo)  ← 画面内箭头/高亮
```

---

## 3. 关键改造点（对应代码）

### 3.1 `BoardInfo` 增加单应矩阵
现状：`GridBoardDetector` 计算了 `hInv`（grid→image）但用后即弃，只把四角存进 `BoardInfo`。

改为：`BoardInfo` 同时保存 `gridToImage` 与 `imageToGrid` 两个方向的 3×3 单应矩阵，供 Overlay 与 YOLO 复用；并提供便捷方法：
- `gridPointToImage(file, rank): (x, y)` — 供 UI 叠加
- `imagePointToGrid(x, y): (file, rank)` — 供 YOLO 结果回映射

### 3.2 修正 `BoardOverlayView` 的近似映射
现状 `BoardOverlayView.intersectionX/Y` 用四角双线性插值，**只在无透视（仿射）时精确**；实拍棋盘有透视，箭头落点会偏。

改为：直接用单应矩阵 `gridToImage` 做 grid→image 映射，再经 `toView` 转视图坐标。黄色棋盘外框绘制保留。

### 3.3 `PieceRecognizer` 接口签名加 bitmap
`recognize(boardInfo)` → `recognize(bitmap, boardInfo)`，因 YOLO 要看**原帧**而非 warped 图。

`TemplatePieceRecognizer` / `TFLitePieceRecognizer` 同步改签名（忽略 bitmap 参数，作为降级实现保留）。

### 3.4 新增 `YoloPieceDetector`（替代原 `YoloPieceRecognizer`）
- 加载**自训 yolov8n** 导出的 ONNX（`yolov8n_chess.onnx`，14 类），**放弃 chess-dike / VinXiangQi 的 GPL-3.0 模型**。
- 预处理：letterbox 缩放到 640×640 + 归一化。
- 推理 + 解码（YOLOv8 输出 `[1,18,8400]` = 前 4 通道 box(xywh) + 后 14 通道类分数，无 objectness）+ NMS。
- 每颗棋 bbox 中心 → `imageToGrid` → 就近交叉点 → 填 9×10 网格。
- 运行时用 ONNX Runtime（`onnxruntime-android`，与棋盘定位 `SegmentationBoardDetector` 共用同一运行时）。

### 3.5 工厂
`PieceRecognizerFactory`：优先 YOLO（模型存在且加载成功），否则降级模板匹配。

---

## 4. 已敲定结论（原风险项）

1. ~~模型契约未知~~ → **已确认**：自训 yolov8n 输出 `[1,18,8400]`（前 4 通道 box + 14 类分数），无 anchors/NMS（YOLOv8 头无 objectness），解码可完全自控。
2. ~~实拍泛化存疑~~ → **改用自训模型**：单视频（debug_video_01）训练对同场景检测到位（mAP50=0.97、mAP50-95=0.91），但**跨场景（board_21）仍有域差**（29/38），由「端侧采集 → 自动标注 → 重训」数据飞轮逐步覆盖。
3. ~~许可~~ → **已规避**：完全自训 + 自导出 ONNX，无 GPL-3.0 传染风险。
4. **走法正确性依赖几何精度**：YOLO 框中心经单应映射到交叉点，若检测框偏移（倾斜视角、遮挡）会错位到相邻交叉点，需靠「就近交叉点 + 置信度 + 历史帧稳定」兜底。

---

## 5. 实施结论（2026-10-06）

> 原「调研结论」见 `docs/调研笔记-既有象棋软件实现.md`。这里记录最终落地路线。

1. **范式确认**：本项目属物理棋盘范式（棋盘定位单应 + 逐子 YOLO），与屏幕识别（chess-dike）不同，直接复用其模型不成立。
2. **模型选型**：`yolov8n`（Ultralytics），14 类中文名（红：帅/仕/相/车/马/炮/兵；黑：将/士/象/车/马/炮/卒）。
3. **训练数据**：`tools/label_real/` M0–M5 自动标注管线产出 `m5_samples_v3.json`（5636 框 / 184 关键帧），导出 `m5_yolo_v3` Ultralytics 数据集。
4. **训练结果**：val mAP50=0.9706、mAP50-95=0.9074（1114 实例，诚实划分）；同视频漏检已修，跨场景仍待数据飞轮。
5. **部署**：`best.pt` → ONNX 导出 → `YoloPieceDetector.kt` + ONNX Runtime 推理。
6. 识别结果加**走法合法性校验**（数量/将帅/上限）兜底。

---

**文档结束**
