# 架构修订方案 — 保留棋盘定位 + YOLO 棋子识别

**日期**：2026-10-02
**状态**：方案待定（调研中，尚未实施）
**触发原因**：后续需在画面中用 UI（箭头/高亮）向使用者说明下一步走法，而非仅文字说明。

---

## 1. 核心结论

要在画面中画出「下一步怎么走」，必须先知道**棋盘网格坐标 ↔ 图像像素**的映射关系。这个映射由**棋盘定位**（四角 + 单应矩阵）提供，因此棋盘定位是 UI 叠加的**几何基础，必须保留**。

据此把识别链路拆成两层，职责清晰、互不重叠：

| 模块 | 角色 | 提供 | 处理 |
| :-- | :-- | :-- | :-- |
| `BoardDetector`（棋盘定位） | **几何层** | 棋盘四角 + 单应矩阵（grid↔image） | 保留并升级 |
| `PieceRecognizer`（棋子识别） | **语义层** | 每颗棋子的类别与位置 | 换成 YOLO 检测 |
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
   │                                        └──► YoloPieceRecognizer.recognize(bitmap, boardInfo)
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

### 3.4 新增 `YoloPieceRecognizer`
- 加载 `yolov5n_xq_fp16.tflite`（chess-dike / VinXiangQi 权重）。
- 预处理：letterbox 缩放到模型输入 + 归一化。
- 推理 + 解码（YOLOv5 输出解码）+ NMS。
- 每颗棋 bbox 中心 → `imageToGrid` → 就近交叉点 → 填 9×10 网格。

### 3.5 工厂
`PieceRecognizerFactory`：优先 YOLO（模型存在且加载成功），否则降级模板匹配。

---

## 4. 待敲定的风险 / 未知项

1. **模型契约未知**：`yolov5n_xq_fp16.tflite` 的输入尺寸、YOLOv5 输出张量布局（是否含 anchors/NMS）、类别顺序，需拿到模型或源码确认后再写解码。
2. **实拍泛化存疑**：chess-dike 是「屏幕识别」（识别别的象棋 App 渲染画面），对「相机实拍棋盘」泛化能力未知，需真机验证，必要时用实拍数据微调。
3. **许可**：chess-dike / VinXiangQi 相关模型与代码为 GPL-3.0，直接内嵌有传染风险，商用/闭源需评估。
4. **走法正确性依赖几何精度**：YOLO 框中心经单应映射到交叉点，若检测框偏移（倾斜视角、遮挡）会错位到相邻交叉点，需靠「就近交叉点 + 置信度 + 历史帧稳定」兜底。

---

## 5. 调研结论

详见 `docs/调研笔记-既有象棋软件实现.md`。要点：

1. 存在两大范式：**屏幕识别**（chess-dike/VinXiangQi，YOLO 检测棋盘框+棋子）与**物理棋盘**（OpenCV 四角 / ArUco / 神经网络四角 + 单应 + 逐格或 YOLO）。本项目属物理棋盘范式。
2. chess-dike 的 `yolov5n_xq_fp16.tflite` 契约（据公开资料）：输入 640×640 letterbox RGB；输出 **15 类 = 14 棋子 + 1 棋盘框**；解码 + 类别感知 NMS；用棋盘框宽/8、高/9 反推交叉点。
3. **风险确认**：该模型为屏幕数据训练，对实拍棋盘泛化存疑，需真机验证；许可 GPL-3.0。
4. 若泛化差，可回退到 xq-vision 的「四角检测 + 透视校正 + 逐格分类」两阶段（与我们原架构同思路，改 ONNX/自训），或 chinese-chess-helper 的 OpenCV 传统路线。
5. 识别结果应加**走法合法性校验**（数量/将帅/上限）兜底。

---

**文档结束**
