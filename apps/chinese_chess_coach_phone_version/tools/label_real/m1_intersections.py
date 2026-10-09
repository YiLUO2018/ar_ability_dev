"""M1：交点检测 + 占位（单应 H + MARGIN → 90 交点；warp 投影找线校验 MARGIN）。

棋盘四角是「棋盘定位」阶段的输出（端上 `SegmentationBoardDetector`，SAM 分割 → 拟合
物理四角），本脚本把它当作**输入**，聚焦 M1 的增量工作：
  1. 单应 H（image 物理四角 → canonical 物理角）；
  2. warp 到 canonical，投影找网格线 → 估计 MARGIN / CELL（未知棋盘时用它，校验 =25/50）；
  3. 90 交点反投影回原图 + 可视化。

占位（哪个交点有棋子）复用 M0 的 `sam_piece_spike.predict_and_classify`，不在本脚本内。

投影找线用 HoughLines（高阈值滤棋子），不用朴素 Sobel 梯度投影——后者会被棋子边缘淹没
（棋子是强梯度大圆，棋盘线是细弱线）。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lines import grid_lines

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
OUT_NODES = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_m1_nodes.png"
OUT_WARP = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_warped_m1.png"

# canonical 物理棋盘（§3.3）
CELL = 50
MARGIN = 25
BOARD_W = 450
BOARD_H = 500
PHYSICAL_CORNERS = np.array(
    [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float32
)

# board_21 的物理四角（SAM 拟合值，TL,TR,BR,BL）——棋盘定位阶段的输出，作为本脚本输入
CORNERS = np.array([[116, 76], [350, 76], [375, 299], [102, 306]], dtype=np.float32)


def project_lines(warped_gray):
    """在 rectified 棋盘上用 HoughLines 找网格线，返回 (竖线 x 列表, 横线 y 列表)。

    高阈值(140)滤掉棋子/文字伪直线；剔除贴近图像边缘的外框线(20px 内)。
    """
    h_lines, v_lines = grid_lines(
        warped_gray, canny_low=40, canny_high=150, threshold=140, tol_deg=8.0, rho_gap=12.0
    )
    vx = sorted(r for r, _ in v_lines if 20.0 <= r <= BOARD_W - 20.0)
    hy = sorted(r for r, _ in h_lines if 20.0 <= r <= BOARD_H - 20.0)
    return vx, hy


def main():
    img = cv2.imread(IMG)

    # 1) 单应：image 物理四角 → canonical 物理角
    M = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), PHYSICAL_CORNERS)
    warped = cv2.warpPerspective(img, M, (BOARD_W, BOARD_H))
    cv2.imwrite(OUT_WARP, warped)

    # 2) warp 投影找线 → 估计 MARGIN / CELL（未知棋盘时用它）
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    vx, hy = project_lines(warped_gray)
    print(f"投影找线：竖线 x={vx}")
    print(f"          横线 y={hy}")
    print(f"  期望：竖线 {[MARGIN + f * CELL for f in range(9)]}")
    print(f"        横线 {[MARGIN + r * CELL for r in range(10)]}")
    if len(vx) >= 2:
        print(f"  竖线估计 MARGIN={vx[0]:.1f} CELL={np.median(np.diff(vx)):.1f}（期望 25 / 50）")
    if len(hy) >= 2:
        print(f"  横线估计 MARGIN={hy[0]:.1f} CELL={np.median(np.diff(hy)):.1f}（期望 25 / 50）")

    # 3) 90 交点反投影回原图
    nodes_canon = np.array(
        [(MARGIN + f * CELL, MARGIN + r * CELL) for r in range(10) for f in range(9)],
        dtype=np.float32,
    )
    nodes_img = cv2.perspectiveTransform(
        nodes_canon.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(M)
    ).reshape(-1, 2)

    # 4) 可视化：物理四角红框 + 90 交点黄色小十字
    vis = img.copy()
    cv2.polylines(vis, [CORNERS.astype(int)], True, (0, 0, 255), 2)
    for p in nodes_img:
        cv2.drawMarker(vis, tuple(np.round(p).astype(int)), (255, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 5, 1)
    cv2.imwrite(OUT_NODES, vis)
    print(f"\n[vis] 已保存 {OUT_NODES}（红框=物理四角，黄十字=90 交点）")
    print(f"[warp] 已保存 {OUT_WARP}")


if __name__ == "__main__":
    main()
