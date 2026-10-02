# tools/label_real — 真实数据自动标注（阶段 1.5）

把真实照片的**棋盘角点**自动打上伪标签，主来源是「网格几何精修」（方法 B），
零人工、零模型自举。详见 `../../docs/真实数据自动标注方案.md`。

## 文件

| 文件 | 说明 |
| :-- | :-- |
| `geometry.py` | 棋盘 canonical 几何（CELL/MARGIN/角点/90 节点）+ 单应与正交/宽高比度量 |
| `lines.py` | Canny + HoughLines → 水平/垂直直线分族 + rho 聚类 + 交点提取 |
| `refine.py` | **核心**：`refine_corners(img, corners_init_norm)` 几何精修 |
| `validate_synthetic.py` | 合成数据量化验证（扰动粗角点 → 精修 → 对比 GT） |

## 依赖

- `opencv-python`、`numpy`（与 `tools/data_gen` 相同）。
- 验证脚本额外需要 `tools/data_gen` 的 `render.py`/`labels.py`（仅导入，无需权重）。

## 用法

### 1. 精修单张图（核心 API）

```python
from refine import refine_corners

# corners_init_norm: 粗 4 角，归一化 [0,1]，顺序 TL,TR,BR,BL
res = refine_corners(img_bgr, corners_init_norm)
corners = res["corners_norm"]       # 精修后归一化 4 角
print(res["grid_rms"])              # 匹配网格线 RMS（canonical px，质量分）
print(res["n_nodes"])               # 匹配网格节点数
print(res["ortho_deg"])             # 偏离 90° 的角度
print(res["aspect"])                # 估计宽高比（理想 8/9）
```

### 2. 合成数据验证精度

```bash
python validate_synthetic.py --n 50 --noise 10,20,30 --seed 0
```

输出不同扰动噪声下，精修前后相对 GT 的角点误差（median/mean/p90）。

## 方法（rectify → 网格拟合 → 单应校正）

1. 用粗 4 角算初始单应 H（canonical → image）。
2. **rectify**：把灰度图 warp 回正对棋盘（横平竖直），此时棋盘任意倾斜都能处理。
3. **直线检测**：Canny + HoughLines（高阈值 `~140` 以滤掉棋子/文字伪直线），分水平/
   垂直两族，按 rho 聚类，剔除外框线。
4. **匹配**：把检测到的横线/竖线一对一匹配到 canonical 的 rank/file 位置，交叉得到
   「canonical 节点 → 检测节点」的对应点。
5. **单应校正**：`cv2.findHomography(..., RANSAC)` 拟合残差单应 D，`H ← H @ D`，迭代。
6. **质量闸门**：若精修后网格对齐（grid_rms）没变好，回退到粗角点，保证不比输入差。

## 验证结果（合成数据，640×640，扰动粗角点）

| 粗角点噪声 | 精修前中位误差 | 精修后中位误差 | 说明 |
| :-- | :-- | :-- | :-- |
| 0px（理想） | 0.0px | ~1.6px | 失真 + 检测噪声的下限 |
| 10px（接近模型实际） | 11.7px | 8.7px | **降约 26%** |
| 20px | 24.5px | 24.7px | 基本持平 |
| 30px | 35.7px | 37.4px | 略变差（闸门已尽量回退） |

**结论**：在粗角点误差 ≤ ~10px（对应阶段 1 角点模型的实际输出）时，几何精修能稳定
把角点误差降低约 1/4；误差更大时收益有限。

**限制（后续改进方向）**：
- 合成数据**镜头畸变**（k1 到 ±0.15）使网格线弯曲，单应只能近似，约 1.6px 下限。
- **棋子密集 + 楚河汉界**遮挡/打断网格线，Hough 检测精度 ~2~5px（非亚像素）。
- 后续可用 LSD / 投影剖面亚像素拟合、镜头畸变标定、SAM 去棋子等进一步提升。

## 质量分 → 伪标签接受标准

- `grid_rms < 2px`（且正交/宽高比偏差小）→ 直接当角点伪标签。
- `2~5px` → 候选，人工微调。
- `> 5px` → 丢弃或人工重标。

## 后续（按方案）

- 方法 A：SAM/MobileSAM 粗定位 → 缩小 ROI → 交给 `refine_corners`。
- 方法 C：棋子时序投票 + 合法性校验（阶段 2 接入）。
