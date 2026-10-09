"""M3 检测结果 ASCII 报告：逐帧打印检测到的 (file,rank) 网格，对照标准开局标注。

对 frame 71 用标准开局参照，标出 正确/误报(FP)/漏检(miss)。
其余帧只打印检测网格 + 棋子数（无 GT，供人工判断）。

用法：python m3_debug_report.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_bbox_track import detect_pieces, cell_metrics, CHECKPOINT, SAM_ROOT

sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

STANDARD_OPENING = set()
for f in range(9):
    STANDARD_OPENING.add((f, 0))
    STANDARD_OPENING.add((f, 9))
for f in (1, 7):
    STANDARD_OPENING.add((f, 2))
    STANDARD_OPENING.add((f, 7))
for f in (0, 2, 4, 6, 8):
    STANDARD_OPENING.add((f, 3))
    STANDARD_OPENING.add((f, 6))

FRAMES = [71, 148, 212, 583]


def grid_str(det):
    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in det:
        occ[r][f] = 1
    rows = []
    for r in range(10):
        rows.append(f"rank{r}: " + "".join("X" if occ[r][f] else "." for f in range(9)))
    return "\n".join(rows)


def main():
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    for fi in FRAMES:
        frame = read_frame(VIDEO, fi)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = detect_pieces(predictor, nodes_img, cw, ch, ca)
        print(f"\n===== frame {fi} : {len(det)} pieces =====")
        print(grid_str(det))
        if fi == 71:
            fp = sorted(set(det.keys()) - STANDARD_OPENING)
            miss = sorted(STANDARD_OPENING - set(det.keys()))
            print(f"  误报 FP (检测到但标准开局无): {fp if fp else '无'}")
            print(f"  漏检 miss (标准开局有但未检出): {miss if miss else '无'}")


if __name__ == "__main__":
    main()
