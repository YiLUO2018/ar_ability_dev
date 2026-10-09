"""逐格裁剪 + 分类原型（路径 A 的最轻量实现：方法 1）。

核心思想：象棋棋子必然落在 9x10 网格交点上，位置离散且已知。用已有棋盘定位
（SAM 分割 -> 物理四角）算单应，把棋盘 warp 成正视图，再在每个交叉点裁固定大小的
patch，逐格送分类器（15 类：空 + 红7 + 黑7）。位置问题用几何解决，只剩分类。

本脚本分两部分：
1. 逐格裁剪：`crop_cells` 用物理四角 + 单应，输出 90 个 patch + 各自在源图坐标。
2. 分类：`classify_patch` 目前是「颜色启发式」占位（红/黑/空），仅用于验证对齐；
   正式分类器（15 类 CNN）训练后替换此函数即可，接口不变。

运行：
    python per_cell_classify.py [image_path]
默认处理 debug_frames/board_21.png，输出 warped 正视图、90 patch 拼图、节点叠加图，
并在终端打印 10x9 的文本网格（. = 空, R = 红, B = 黑）用于快速核对对齐。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# ---- canonical 几何（与 geometry.py / BoardGeometry.kt 保持一致）----
CELL = 50
MARGIN = 25
BOARD_W = MARGIN * 2 + CELL * 8   # 450
BOARD_H = MARGIN * 2 + CELL * 9   # 500

# 物理棋盘四角（canonical，整张图四角），顺序 TL,TR,BR,BL
PHYSICAL_CORNERS = np.array(
    [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float32
)

# board_21.png 的物理四角（SAM 拟合值，旧 TL(166,77) 偏右 ~50px 已弃），顺序 TL,TR,BR,BL
GT_CORNERS = np.array(
    [[116, 76], [350, 76], [375, 299], [102, 306]], dtype=np.float32
)

PATCH = 48          # 每格裁剪边长（canonical px，略小于 CELL 避免邻居混入）
PATCH_MARGIN = 4    # 从节点向四周扩展的量


def canonical_nodes():
    """90 个网格交点 (file, rank) 的 canonical 坐标，rank-major，shape (90, 2)。"""
    pts = [
        (MARGIN + f * CELL, MARGIN + r * CELL)
        for r in range(10)
        for f in range(9)
    ]
    return np.array(pts, dtype=np.float32)


NODES = canonical_nodes()


def crop_cells(img_bgr: np.ndarray, corners_px: np.ndarray):
    """逐格裁剪。

    Args:
        img_bgr: 源图（BGR）。
        corners_px: 物理棋盘四角（像素，TL,TR,BR,BL）。

    Returns:
        warped: warp 后的 canonical 正视图（BOARD_H x BOARD_W）。
        patches: list[90]，每个是 PATCH x PATCH 的 BGR patch，rank-major。
        node_img: 90 个交点在源图的像素坐标 (N,2)，rank-major。
    """
    # H: canonical -> image（物理四角 -> 像素四角）
    H = cv2.getPerspectiveTransform(
        PHYSICAL_CORNERS.astype(np.float32), corners_px.astype(np.float32)
    )
    # warp：dst=canonical，src=image，M 需把 dst 映射回 src，即 canonical->image 的 H
    warped = cv2.warpPerspective(img_bgr, H, (BOARD_W, BOARD_H))

    # 90 个交点在源图的坐标（供叠加可视化 / 后续按需在源图采样）
    node_img = cv2.perspectiveTransform(
        NODES.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    half = PATCH // 2
    patches = []
    for (cx, cy) in NODES:
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        # 越界时用边缘填充（棋盘边角节点 crop 会超出 canonical 图）
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, (x0 + PATCH) - BOARD_W)
        pad_bottom = max(0, (y0 + PATCH) - BOARD_H)
        region = warped[
            max(0, y0):y0 + PATCH,
            max(0, x0):x0 + PATCH,
        ]
        if pad_left or pad_top or pad_right or pad_bottom:
            region = cv2.copyMakeBorder(
                region, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        patches.append(region)

    return warped, patches, node_img


def classify_patch(patch_bgr: np.ndarray) -> str:
    """颜色启发式占位分类：返回 'R'（红）/ 'B'（黑）/ '.'（空）。

    仅用于验证对齐；正式方案替换为 15 类 CNN（空 + 红7 + 黑7）。
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 红：饱和度较高且 hue 落在红区间（OpenCV H 0-179：红在 0-15 或 160-179）
    red_mask = (s > 70) & ((h < 15) | (h > 160))
    # 黑：明度很低（黑字/黑子）
    black_mask = v < 70
    total = patch_bgr.shape[0] * patch_bgr.shape[1]
    if red_mask.mean() > 0.12:
        return "R"
    if black_mask.mean() > 0.10:
        return "B"
    return "."


def make_montage(patches, cols=9, rows=10, gap=2):
    """把 90 个 patch 拼成 9 列 x 10 行大图（行间/列间留缝）。"""
    p = PATCH
    out = np.full(
        (rows * (p + gap) + gap, cols * (p + gap) + gap, 3),
        fill_value=60, dtype=np.uint8,
    )
    for i, patch in enumerate(patches):
        r, c = divmod(i, cols)
        y = gap + r * (p + gap)
        x = gap + c * (p + gap)
        out[y:y + p, x:x + p] = patch
    return out


def print_grid(labels):
    """labels: 90 个 'R'/'B'/'.'，rank-major。打印 10 行 x 9 列棋盘文本。"""
    print("\n== 逐格颜色启发式结果（rank 0 黑方在上） ==")
    for r in range(10):
        row = labels[r * 9:(r + 1) * 9]
        print(f"rank{r}  " + " ".join(row))
    print()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(
            base, "..", "..", "debug_frames", "board_21.png"
        )
    img_path = os.path.abspath(img_path)
    img = cv2.imread(img_path)
    assert img is not None, f"无法读取 {img_path}"
    print(f"源图 {img_path} 尺寸 {img.shape[1]}x{img.shape[0]}")

    # TODO: 此处四角应来自 SAM 分割输出；本原型用 board_21 的已知物理四角代替。
    corners = GT_CORNERS
    warped, patches, node_img = crop_cells(img, corners)

    # 1) 正视图
    out_dir = os.path.dirname(img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_warped.png"), warped)

    # 2) 90 patch 拼图
    montage = make_montage(patches)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_patches.png"), montage)

    # 3) 节点叠加到源图
    vis = img.copy()
    for (x, y) in node_img:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    cv2.polylines(vis, [corners.astype(int)], True, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_nodes.png"), vis)

    # 4) 文本网格 + 统计
    labels = [classify_patch(p) for p in patches]
    print_grid(labels)
    n_r = labels.count("R")
    n_b = labels.count("B")
    print(f"统计：红 {n_r}、黑 {n_b}、空 {90 - n_r - n_b}")

    print("\n已保存：")
    print(f"  {stem}_warped.png  （canonical 正视图 {BOARD_W}x{BOARD_H}）")
    print(f"  {stem}_patches.png （90 patch 拼图 {PATCH}px）")
    print(f"  {stem}_nodes.png   （90 交点叠加 + 物理四角红框）")


if __name__ == "__main__":
    main()
