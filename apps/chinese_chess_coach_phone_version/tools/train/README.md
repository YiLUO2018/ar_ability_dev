# 训练与导出（方案 v2 · 阶段 1）

用 `tools/data_gen` 生成的合成数据训练两个模型，并导出 TFLite 供 Android 端使用。

## 环境

训练/导出均在 WSL 的 conda 环境 `le_wm`（torch 2.12 + CUDA 13.0 + RTX 4060 + litert-torch）中运行。
依赖见 `requirements.txt`（torch / torchvision / numpy / opencv-python / litert-torch）。

> `litert-torch` 是 `ai-edge-torch` 的更名（已弃用），导入名为 `litert_torch`。

## 文件

| 文件 | 说明 |
|:--|:--|
| `models.py` | `CornerModel`（角点）+ `PieceClassifier`（整盘分类）+ DSNT + 高斯热力图 |
| `datasets.py` | `CornerDataset`（含 `no_board` 负样本）+ `ClassifyDataset` |
| `train_corner.py` | 训练角点模型 |
| `train_classify.py` | 训练分类模型 |
| `export.py` | 导出 TFLite |

## 步骤

```bash
# 1. 生成数据（在 tools/data_gen 下）
cd ../data_gen
python render.py --out out --corner 4000 --corner-val 400 \
    --corner-neg 800 --corner-neg-val 80 \
    --classify 4000 --classify-val 400 --seed 0

# 2. 训练角点模型
cd ../train
python train_corner.py --data ../data_gen/out --epochs 30 --batch-size 32

# 3. 训练分类模型
python train_classify.py --data ../data_gen/out --epochs 30 --batch-size 32

# 4. 导出 TFLite
python export.py --model corner   --checkpoint checkpoints/corner/best.pt   --out board_corner.tflite
python export.py --model classify --checkpoint checkpoints/classify/best.pt --out piece_classifier.tflite
```

## 模型契约（导出后 TFLite）

### board_corner.tflite
- 输入 float32 `[1,320,320,3]`，RGB，`[0,1]`。
- 输出 float32 `[1,12]`：`x0,y0,x1,y1,x2,y2,x3,y3,c0,c1,c2,c3`
  —— 四角 **TL,TR,BR,BL** 归一化 `[0,1]` 坐标 + 各自置信度（低即视为「无棋盘」）。

### piece_classifier.tflite
- 输入 float32 `[1,400,360,3]`，RGB，`[0,1]`（canonical 棋盘 resize 到 360×400）。
- 输出 float32 `[1,15,10,9]`：rank-major，`[0,:,r,f]` 为交叉点 `(r,f)` 的 15 类 logits，
  argmax 取类别。类别映射同 `data_gen/labels.py`（0=空，1–7=红，8–14=黑）。

## 设计说明

- **角点模型**：轻量 CNN 骨干（stride 4）+ 1×1 头输出 4 通道热力图；训练用高斯热力图 MSE +
  坐标 L1；解码用 DSNT 软 argmax。置信度取热力图 sigmoid 峰值，用于端上「无棋盘」门控。
  采用自定义轻量骨干（而非预训练 MobileNetV3），保证离线可复现且导出算子全兼容。
- **分类模型**：交叉点像素在 stride-4 特征图后每 10px 对齐，`avg_pool2d(10)` 与 90 个
  交叉点严格对齐，无需插值近似；输出 10×9×15。
