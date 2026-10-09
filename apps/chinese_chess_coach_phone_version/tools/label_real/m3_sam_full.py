"""M3 占位真值：视频帧 0 的 SAM 全量逐子检测（M5 同款管线）。

复用 sam_detect_pieces.py 的「基线中心提示 + 8 邻域探针补漏 + IoU 防重 +
全局最近空交点分配」，给视频帧 0 产出最可靠的 90 交点占位真值，用来校准
便宜占位（边缘密度）的阈值与歧义格。

SAM 只跑一帧；生产管线逐帧用便宜边缘占位。
"""
import os
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from sam_piece_spike import predict_and_classify, DIST_CELL, CELL, MARGIN
from sam_detect_pieces import largest_component_mask, mask_iou
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO, edge_scores

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"

PROBES = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
          (0.35, 0.35), (0.35, -0.35), (-0.35, 0.35), (-0.35, -0.35)]
FILL_DIST = 0.75
IOU_DUP = 0.3


def main():
    t0 = time.time()
    frame = read_frame(VIDEO, 0)
    H, res = board_homography(frame, SAM_PHYS)
    print(f"refine grid_rms={res['grid_rms']:.2f}")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("加载 SAM ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    print(f"就绪 {time.time()-t0:.1f}s")

    nodes_img = warp_points(canonical_nodes(), H)
    cell_areas = []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]
        fr = f + 1 if f < 8 else f - 1
        rd = r + 1 if r < 9 else r - 1
        cw = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
        ch = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
        cell_areas.append(cw * ch)

    detected = {}
    claimed = []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]; ca = cell_areas[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, ca)
        if is_piece and centroid is not None:
            dist = float(np.hypot(centroid[0]-p[0], centroid[1]-p[1]))
            if dist < DIST_CELL * np.sqrt(ca):
                clean = largest_component_mask(mask)
                detected[(f, r)] = clean
                claimed.append(clean)
    print(f"[基线] {len(detected)} 子")

    empty_ids = [i for i in range(90) if (i % 9, i // 9) not in detected]
    candidates = []
    for i in empty_ids:
        f, r = i % 9, i // 9
        p = nodes_img[i]; ca = cell_areas[i]
        cw = float(np.linalg.norm(p - nodes_img[r * 9 + (f + 1 if f < 8 else f - 1)]))
        ch = float(np.linalg.norm(p - nodes_img[(r + 1 if r < 9 else r - 1) * 9 + f]))
        for ox, oy in PROBES:
            pt = (p[0] + ox * cw, p[1] + oy * ch)
            mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, pt, ca)
            if not (is_piece and centroid is not None):
                continue
            d = float(np.hypot(centroid[0]-p[0], centroid[1]-p[1]))
            if d > FILL_DIST * np.sqrt(ca):
                continue
            clean = largest_component_mask(mask)
            if any(mask_iou(clean, m) > IOU_DUP for m in claimed):
                continue
            candidates.append((centroid, clean))
            claimed.append(clean)
            break

    for centroid, mask in candidates:
        best_i = min(empty_ids, key=lambda j: float(np.hypot(
            centroid[0]-nodes_img[j][0], centroid[1]-nodes_img[j][1])))
        d_best = float(np.hypot(centroid[0]-nodes_img[best_i][0],
                                centroid[1]-nodes_img[best_i][1]))
        if d_best > FILL_DIST * np.sqrt(cell_areas[best_i]):
            continue
        f, r = best_i % 9, best_i // 9
        if (f, r) in detected:
            continue
        detected[(f, r)] = mask
        empty_ids.remove(best_i)
    print(f"[补漏后] {len(detected)} 子")

    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in detected:
        occ[r][f] = 1
    print("SAM 全量占位：")
    for r in range(10):
        print(f"rank{r}: " + "".join("X" if occ[r][f] else "." for f in range(9)))
    print(f"总数 {int(occ.sum())}")

    # 对比边缘占位
    edge_occ = (edge_scores(frame, H) > 65).astype(int)
    print("\n边缘占位（阈值65）：")
    for r in range(10):
        print(f"rank{r}: " + "".join("X" if edge_occ[r][f] else "." for f in range(9)))
    print(f"总数 {int(edge_occ.sum())}")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
