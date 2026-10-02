"""棋盘网格直线检测与交点提取（纯几何，无监督）。

用 Canny + HoughLines 提取直线，按方向分水平/垂直两族，按 rho 聚类得到
代表性网格线，再求「水平 × 垂直」交点作为候选网格节点。
"""
from __future__ import annotations

import cv2
import numpy as np

_HALF_PI = np.pi / 2.0


def _cluster_1d(values: np.ndarray, gap: float):
    """对一维值做 gap-based 聚类，返回每个簇的均值列表（升序）。"""
    if len(values) == 0:
        return []
    values = np.sort(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def detect_hough_lines(gray: np.ndarray, canny_low=50, canny_high=150, threshold=100):
    """Canny + HoughLines，返回 (N,2) 的 (rho, theta)，theta∈[0,π)。"""
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return np.zeros((0, 2), dtype=np.float32)
    lines = lines.reshape(-1, 2).astype(np.float32)
    # 归一化：rho>=0，theta∈[0,π)
    neg = lines[:, 0] < 0
    lines[neg, 0] = -lines[neg, 0]
    lines[neg, 1] = lines[neg, 1] - np.pi
    lines[:, 1] = np.where(lines[:, 1] < 0, lines[:, 1] + np.pi, lines[:, 1])
    return lines


def split_hv(lines: np.ndarray, tol_deg=15.0):
    """把 (rho,theta) 直线分成水平（θ≈π/2）与垂直（θ≈0）两族。"""
    tol = np.radians(tol_deg)
    horiz, vert = [], []
    for r, t in lines:
        if abs(t - _HALF_PI) < tol:
            horiz.append((r, t))
        elif abs(t) < tol or abs(t - np.pi) < tol:
            vert.append((r, t))
    return (
        np.array(horiz, dtype=np.float32).reshape(-1, 2),
        np.array(vert, dtype=np.float32).reshape(-1, 2),
    )


def grid_lines(
    gray: np.ndarray,
    canny_low=50,
    canny_high=150,
    threshold=100,
    tol_deg=15.0,
    rho_gap=12.0,
):
    """提取水平/垂直代表性网格线。返回 (h_lines, v_lines)，各为 (M,2) 的 (rho,theta)。"""
    lines = detect_hough_lines(gray, canny_low, canny_high, threshold)
    horiz, vert = split_hv(lines, tol_deg)

    h_rhos = _cluster_1d(horiz[:, 0] if len(horiz) else np.array([]), rho_gap)
    v_rhos = _cluster_1d(vert[:, 0] if len(vert) else np.array([]), rho_gap)

    h_lines = np.array([(r, _HALF_PI) for r in h_rhos], dtype=np.float32).reshape(-1, 2)
    v_lines = np.array([(r, 0.0) for r in v_rhos], dtype=np.float32).reshape(-1, 2)
    return h_lines, v_lines


def intersect_rho_theta(r1, t1, r2, t2):
    """两条 (rho,theta) 直线的交点 (x,y)。"""
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    det = c1 * s2 - s1 * c2
    if abs(det) < 1e-9:
        return None
    x = (r1 * s2 - r2 * s1) / det
    y = (c1 * r2 - c2 * r1) / det
    return (x, y)


def detect_grid_intersections(
    gray: np.ndarray,
    canny_low=50,
    canny_high=150,
    threshold=100,
    tol_deg=15.0,
    rho_gap=12.0,
):
    """检测棋盘网格交点，返回 (M,2) 候选点（无标签）。"""
    h_lines, v_lines = grid_lines(gray, canny_low, canny_high, threshold, tol_deg, rho_gap)
    pts = []
    for rh, th in h_lines:
        for rv, tv in v_lines:
            p = intersect_rho_theta(rh, th, rv, tv)
            if p is not None:
                pts.append(p)
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)
