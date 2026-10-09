"""M3 前置：视频帧棋盘角点精修（物理角 → 网格角 → refine_corners）。

M0/M1 用的「SAM 物理四角 → PHYSICAL_CORNERS」单应，在视频帧上会让网格线整体偏移
~13 canonical px、压缩 ~8%（SAM 拟合四边形略大于真实棋盘边）。M3 的 from→to 需要
90 交点精确落到网格线上，否则占位采样会偏半格。

本脚本：给定一帧的 SAM 物理四角（输入），
  1. 物理角 → 网格最外层角（内缩 MARGIN，用物理单应 H_phys 反投 CANON_CORNERS）；
  2. refine_corners 用「rectify → 网格线匹配 → 单应校正」迭代精修；
  3. 输出 grid_rms / n_nodes / ortho_deg / aspect 质量分 + 90 交点可视化。

纯几何、秒级（SAM 只负责给初始角，不在此脚本内）。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (
    BOARD_H,
    BOARD_W,
    CANON_CORNERS,
    CELL,
    MARGIN,
    canonical_nodes,
    warp_points,
)
from refine import refine_corners

VIDEO = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\debug_video_01.mp4"
OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\m3_refine_vis.png"

# 帧 0 的 SAM 物理四角（TL,TR,BR,BL）——上一轮 SAM 分割拟合值，本脚本把它当输入。
SAM_PHYS_CORNERS = np.array(
    [[58, 145], [956, 123], [1029, 1071], [22, 1090]], dtype=np.float32
)

PHYSICAL_CORNERS = np.array(
    [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float32
)


def phys_to_grid_corners(phys_img):
    """物理四角（图像）→ 网格最外层四角（图像）。

    物理单应 H_phys: canonical 物理角 → 图像；CANON_CORNERS 是 canonical 里的
    网格最外层角（内缩 MARGIN），用它反投回图像即得网格角。
    """
    H_phys = cv2.getPerspectiveTransform(PHYSICAL_CORNERS.astype(np.float32),
                                         phys_img.astype(np.float32))
    return warp_points(CANON_CORNERS, H_phys)


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    frame = read_frame(VIDEO, 0)
    if frame is None:
        print("读帧失败"); return
    H, W = frame.shape[:2]
    print(f"帧 0: {W}x{H}")

    grid_init = phys_to_grid_corners(SAM_PHYS_CORNERS)
    print(f"物理角 → 网格角（图像 px）:\n{np.round(grid_init, 1)}")

    corners_norm = (grid_init.astype(np.float64) * np.array([W, H])).astype(np.float32)
    corners_norm = grid_init / np.array([W, H], dtype=np.float32)

    res = refine_corners(frame, corners_norm)
    print(f"\n[refine] grid_rms={res['grid_rms']:.2f} px  n_nodes={res['n_nodes']}  "
          f"ortho_deg={res['ortho_deg']:.2f}  aspect={res['aspect']:.3f} (目标 {res['aspect_target']:.3f})")

    corners_ref = res["corners_norm"] * np.array([W, H])
    H_ref = res["H"]

    # 90 交点反投
    nodes_img = warp_points(canonical_nodes(), H_ref)

    vis = frame.copy()
    cv2.polylines(vis, [corners_ref.astype(int)], True, (0, 0, 255), 2)
    for p in nodes_img:
        cv2.drawMarker(vis, tuple(np.round(p).astype(int)), (0, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 5, 1)
    cv2.imwrite(OUT, vis)
    print(f"\n[vis] 已保存 {OUT}（红框=精修网格角，黄十字=90 交点）")

    # 网格线对齐明细（rectify 后检测 vs canonical）
    from refine import _rectify, _detect_lines_rectified
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = _rectify(gray, H_ref)
    h_rhos, v_rhos = _detect_lines_rectified(rect, 40, 150, 140)
    canon_x = [MARGIN + f * CELL for f in range(9)]
    canon_y = [MARGIN + r * CELL for r in range(10)]
    print(f"竖线: 检测 {sorted(round(r,1) for r in v_rhos)}")
    print(f"      canon {canon_x}")
    print(f"横线: 检测 {sorted(round(r,1) for r in h_rhos)}")
    print(f"      canon {canon_y}")


if __name__ == "__main__":
    main()
