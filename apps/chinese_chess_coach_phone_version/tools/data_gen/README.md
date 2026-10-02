# 合成数据渲染器（方案 v2 · 阶段 0）

程序化渲染中国象棋棋盘，标签天生已知，实现「零人工标注」。

## 环境

- Python 3.9+，依赖见 `requirements.txt`（numpy / opencv-python / Pillow）。
- 汉字字形依赖系统字体（Windows 下 `simhei.ttf` / `simkai.ttf` / `simsun.ttc` 等，缺失则退化为默认字体）。

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 生成 1000 角点 + 1000 分类训练样本，各 100 验证样本
python render.py --out data --corner 1000 --classify 1000 --corner-val 100 --classify-val 100 --seed 0

# 少量预览（生成预览图到 out/preview/，供人工核对）
python render.py --out data --corner 8 --classify 8 --visualize 8
```

## 输出结构

```
out/
├─ train/
│  ├─ corner/    images/*.jpg + labels/*.txt   # 角点数据集
│  └─ classify/  images/*.jpg + labels/*.txt   # 分类数据集
├─ val/          （同上）
├─ preview/      corner_*.png / classify_*.png  # 可视化预览
└─ meta.json     几何与类别说明
```

## 标签格式

### corner（棋盘角点）
- 图像：含透视棋盘 + 随机背景的场景图，默认 640×640。
- `labels/*.txt`：4 行 `x y`（归一化 [0,1]），顺序 **TL, TR, BR, BL**。

### classify（棋子分类）
- 图像：透视校正后的 canonical 棋盘，默认 **450×500**。
- `labels/*.txt`：10 行 × 9 列整数（rank-major，即 `grid[rank][file]`）。

## 类别映射（与 `TFLitePieceRecognizer` 对齐）

| label | 含义 |
| :---: | :--- |
| 0 | 空位 |
| 1–7 | 红方 帅/仕/相/马/车/炮/兵（`PieceType.entries` 顺序 GENERAL..PAWN） |
| 8–14 | 黑方 将/士/象/马/车/炮/卒 |

换算：`label = 1 + type_index + (7 if black else 0)`。

## 几何

- `CELL=50`、`MARGIN=25`，canonical 棋盘 **450×500**。
- 交叉点 `(file, rank)` 像素 = `(25 + file*50, 25 + rank*50)`。
- 网格四角（TL,TR,BR,BL）= `(25,25) (425,25) (425,475) (25,475)`。

## 实现要点

- **布局**：标准开局（约 25–30%）+ 随机中局（必有双方将/帅在各自九宫，其余按颜色偏置随机摆放）。
- **透视**：3D 针孔投影（roll/pitch/yaw + 焦距 + 距离）生成真实梯形畸变，经单应变换贴入场景。
- **领域随机化**：桌面背景（纯色/渐变/木纹/噪声）、阴影、亮度/对比度/gamma、色偏、高斯/椒盐噪声、高斯/运动/中值模糊、径向镜头畸变、JPEG 压缩。
- **分类样本抖动**：对 canonical 棋盘施加 ±4px 角点抖动 + 轻微径向畸变，模拟真实 warp 残差，训练分类器对透视校正误差鲁棒。
