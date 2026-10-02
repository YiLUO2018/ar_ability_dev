"""棋盘 canonical 几何与单应/约束工具。

与 ``tools/data_gen/render.py`` 保持同一套几何常量：CELL=50、MARGIN=25，
网格 9 列（file 0..8）× 10 行（rank 0..9）。网格最外层交点即角点。
"""
from __future__ import annotations

import cv2
import numpy as np

CELL = 50
MARGIN = 25
BOARD_W = MARGIN * 2 + CELL * 8   # 450
BOARD_H = MARGIN * 2 + CELL * 9   # 500

# canonical 四角（网格最外层交点），顺序 TL, TR, BR, BL
CANON_CORNERS = np.array(
    [
        [MARGIN, MARGIN],
        [MARGIN + 8 * CELL, MARGIN],
        [MARGIN + 8 * CELL, MARGIN + 9 * CELL],
        [MARGIN, MARGIN + 9 * CELL],
    ],
    dtype=np.float32,
)

# 网格宽高比（最外层网格线围成的矩形）8:9
GRID_ASPECT = 8.0 / 9.0


def canonical_nodes() -> np.ndarray:
    """90 个网格交点 (file, rank) 的 canonical 坐标，rank-major，shape (90, 2)。"""
    pts = [
        (MARGIN + f * CELL, MARGIN + r * CELL)
        for r in range(10)
        for f in range(9)
    ]
    return np.array(pts, dtype=np.float32)


CANON_NODES = canonical_nodes()


def homography_from_corners(corners_px: np.ndarray) -> np.ndarray:
    """4 角（像素，TL,TR,BR,BL）→ 单应 H（canonical → image），3×3。"""
    return cv2.getPerspectiveTransform(
        CANON_CORNERS.astype(np.float32), corners_px.astype(np.float32)
    )


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """用 H（canonical→image）变换点集，``pts`` (N,2) → (N,2)。"""
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H)
    return out.reshape(-1, 2)


def corners_from_H(H: np.ndarray) -> np.ndarray:
    """单应 H → 4 角（像素，TL,TR,BR,BL）。"""
    return warp_points(CANON_CORNERS, H)


def rectify_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """用 H 的逆（image→canonical）把图像点变换回 canonical。"""
    Hinv = np.linalg.inv(H)
    return warp_points(pts, Hinv)


def axes_orthogonality(H: np.ndarray) -> float:
    """度量 rectified 网格两条轴的正交性偏差（度）。

    用 H 把 canonical 轴方向 (1,0)、(0,1) 映射回图像，取与无穷远点的方向；
    在 canonical 里两轴本应正交，这里用 H 的线性部分度量其被投影后的夹角偏差。
    返回偏离 90° 的角度（越小越好）。
    """
    # H 的线性部分（忽略平移）
    A = H[:2, :2]
    # canonical 水平方向 (1,0) 与竖直方向 (0,1) 在图像中的方向
    u = A @ np.array([1.0, 0.0])
    v = A @ np.array([0.0, 1.0])
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 90.0
    cosang = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    ang = np.degrees(np.arccos(cosang))
    return abs(90.0 - ang)


def aspect_ratio(H: np.ndarray) -> float:
    """rectified 网格宽高比。用 H 线性部分作用在 (8,9) 轴向上，估计图像中两方向长度比。

    返回 image 中「一列宽 / 一行高」的比值；理想上等于 GRID_ASPECT（8/9）。
    """
    A = H[:2, :2]
    u = np.linalg.norm(A @ np.array([1.0, 0.0]))   # 每格宽的图像长度
    v = np.linalg.norm(A @ np.array([0.0, 1.0]))   # 每格高的图像长度
    if v < 1e-9:
        return 0.0
    return float(u / v)
