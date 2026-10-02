"""棋盘角点几何精修（核心）。

给定一张含棋盘的场景图 + 粗 4 角（模型 / SAM / 任意来源），用「rectify 到正对 →
直线检测 → 匹配 canonical 网格节点 → 单应校正」迭代精修 4 角（向亚像素逼近，受镜头
失真与直线检测精度限制）。直角 / 宽高比 / 等距约束由「拟合到 canonical 网格」隐式施加
（canonical 节点本身正交、8:9、等距）。

关键点：
1. 棋盘在透视下可能任意倾斜，先 rectify（正对）再检测直线，横平竖直才适用 Hough 的
   0°/90° 分族。
2. 粗 4 角带噪声时，残差是「投影变换」（非均匀间距），因此把检测到的横/竖线交叉出
   的节点拟合到 canonical 节点的**完整单应**（而非等距网格），RANSAC 迭代收敛。
"""
from __future__ import annotations

import cv2
import numpy as np

from geometry import (
    BOARD_H,
    BOARD_W,
    CANON_CORNERS,
    CELL,
    GRID_ASPECT,
    MARGIN,
    aspect_ratio,
    axes_orthogonality,
    corners_from_H,
    homography_from_corners,
)
from lines import grid_lines

CANON_X = [MARGIN + f * CELL for f in range(9)]    # 竖线（file 0..8）x 位置
CANON_Y = [MARGIN + r * CELL for r in range(10)]   # 横线（rank 0..9）y 位置

# 物理棋盘外框矩形在 ~6px 处，网格线在 25px 处；用 20px 门槛剔除外框线
_BORDER = 20.0


def _rectify(gray, Hmat):
    """把灰度图 rectify 到正对棋盘坐标（BOARD_W×BOARD_H）。"""
    Hinv = np.linalg.inv(Hmat)
    return cv2.warpPerspective(gray, Hinv, (BOARD_W, BOARD_H))


def _detect_lines_rectified(rect_gray, canny_low, canny_high, hough_threshold):
    """在 rectify（正对）棋盘灰度图上检测水平/垂直网格线，返回剔除外框后的 rho 列表。"""
    h_lines, v_lines = grid_lines(rect_gray, canny_low, canny_high, hough_threshold)
    h_rhos = [float(r) for r, _ in h_lines]
    v_rhos = [float(r) for r, _ in v_lines]
    v_rhos = [r for r in v_rhos if _BORDER <= r <= BOARD_W - _BORDER]
    h_rhos = [r for r in h_rhos if _BORDER <= r <= BOARD_H - _BORDER]
    return h_rhos, v_rhos


def _match_1d(detected, canon, max_dist):
    """把检测到的线位置（一维 rho）匹配到 canonical 位置，一对一、取最近。

    返回 dict ``{canonical_pos: detected_rho}``。
    """
    matches = {}
    for rho in sorted(detected):
        d = np.abs(np.asarray(canon, dtype=np.float32) - rho)
        j = int(np.argmin(d))
        if d[j] > max_dist:
            continue
        cpos = canon[j]
        if cpos not in matches or abs(matches[cpos] - cpos) > d[j]:
            matches[cpos] = rho
    return matches


def _node_correspondences(rect_gray, canny_low, canny_high, hough_threshold, max_dist):
    """在 rectify 棋盘上匹配「横线×竖线」节点到 canonical 网格节点。

    返回 ``(src, dst)``：src 为 canonical 节点坐标 (N,2)，dst 为检测节点（rectify 系）(N,2)。
    """
    h_rhos, v_rhos = _detect_lines_rectified(rect_gray, canny_low, canny_high, hough_threshold)
    hm = _match_1d(h_rhos, CANON_Y, max_dist)
    vm = _match_1d(v_rhos, CANON_X, max_dist)

    src, dst = [], []
    for cy, rho_y in hm.items():
        for cx, rho_x in vm.items():
            src.append((cx, cy))            # canonical 节点 (x, y)
            dst.append((rho_x, rho_y))      # 检测节点（rectify 系）
    if not src:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.array(src, np.float32), np.array(dst, np.float32)


def _grid_rms(rect_gray, canny_low, canny_high, hough_threshold, max_dist):
    """用当前 rectify 图，统计匹配网格线相对 canonical 的 RMS 偏差（canonical px）。"""
    h_rhos, v_rhos = _detect_lines_rectified(rect_gray, canny_low, canny_high, hough_threshold)
    hm = _match_1d(h_rhos, CANON_Y, max_dist)
    vm = _match_1d(v_rhos, CANON_X, max_dist)
    errs = [abs(rho - cpos) for cpos, rho in list(hm.items()) + list(vm.items())]
    if not errs:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errs))))


def refine_corners(
    img_bgr,
    corners_init_norm,
    *,
    iters=6,
    match_radius=30.0,
    ransac_reproj=6.0,
    canny_low=40,
    canny_high=150,
    hough_threshold=140,
):
    """精修 4 角。

    ``corners_init_norm``: 粗 4 角，归一化 [0,1]，顺序 TL,TR,BR,BL。

    返回 dict: ``corners_norm``, ``H``, ``grid_rms``(匹配网格线 RMS，canonical px),
    ``n_nodes``(匹配网格节点数), ``ortho_deg``, ``aspect``, ``aspect_target``。
    """
    H, W = img_bgr.shape[:2]
    corners_px = (corners_init_norm.astype(np.float64) * np.array([W, H])).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    H_init = homography_from_corners(corners_px)
    rms_init = _grid_rms(_rectify(gray, H_init), canny_low, canny_high, hough_threshold, match_radius)

    Hmat = H_init.copy()
    n_nodes = 0
    for _ in range(iters):
        rect = _rectify(gray, Hmat)
        src, dst = _node_correspondences(rect, canny_low, canny_high, hough_threshold, match_radius)
        if len(src) < 4:
            break
        D, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj)
        if D is None:
            break
        n_nodes = int(np.count_nonzero(inliers)) if inliers is not None else len(src)
        Hmat = Hmat @ D

    rms_ref = _grid_rms(_rectify(gray, Hmat), canny_low, canny_high, hough_threshold, match_radius)
    # 质量闸门：精修后网格对齐若没变好，回退到粗角点（避免更差）
    if rms_ref >= rms_init:
        Hmat = H_init
        grid_rms = rms_init
    else:
        grid_rms = rms_ref

    corners_norm = (corners_from_H(Hmat) / np.array([W, H], np.float32)).astype(np.float32)
    return {
        "corners_norm": corners_norm,
        "H": Hmat,
        "grid_rms": grid_rms,
        "n_nodes": n_nodes,
        "ortho_deg": axes_orthogonality(Hmat),
        "aspect": aspect_ratio(Hmat),
        "aspect_target": GRID_ASPECT,
    }
