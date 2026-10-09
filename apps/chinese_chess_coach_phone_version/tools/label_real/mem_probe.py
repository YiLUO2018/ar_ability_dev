"""内存探针：逐帧跑 SAM 单点推理，打印进程 RSS 与系统可用内存，定位 ~20 帧崩溃是否内存泄漏。"""
import os
import sys
import time

import cv2
import numpy as np
import psutil

SAM_ROOT = os.environ.get(
    "SAM_ROOT", r"L:\Reinforcement_Learning\segment\segment-anything-main")
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from sam_piece_spike import largest_component, AREA_LO, AREA_HI, CIRC_MIN, DIST_CELL
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO

CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 ** 2


def avail_gb():
    return psutil.virtual_memory().available / 1024 ** 3


def cell_areas_for(nodes_img):
    areas = []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]
        fr = f + 1 if f < 8 else f - 1
        rd = r + 1 if r < 9 else r - 1
        cw = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
        ch = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
        areas.append(cw * ch)
    return areas


def sam_occupancy(predictor, nodes_img, cell_areas):
    occ = np.zeros(90, dtype=np.uint8)
    for i in range(90):
        p = nodes_img[i]
        ca = cell_areas[i]
        masks, _, _ = predictor.predict(
            point_coords=np.array([p], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        comp = largest_component(mask)
        if comp is None:
            continue
        area, peri, (cx, cy) = comp
        circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        if not (AREA_LO * ca <= area <= AREA_HI * ca) or not (circ > CIRC_MIN):
            continue
        d = float(np.hypot(cx - p[0], cy - p[1]))
        if d < DIST_CELL * np.sqrt(ca):
            occ[i] = 1
    return occ


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cell_areas = cell_areas_for(nodes_img)
    print(f"grid_rms={res['grid_rms']:.2f}  rss={rss_mb():.0f}MB  avail={avail_gb():.1f}GB", flush=True)

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu")
    sam.eval()
    predictor = SamPredictor(sam)
    print(f"model-loaded  rss={rss_mb():.0f}MB  avail={avail_gb():.1f}GB", flush=True)

    for k in range(n):
        fi = 1000 + k  # 固定偏移，避免与已有结果重叠
        frame = read_frame(VIDEO, fi)
        if frame is None:
            print("read None", flush=True)
            break
        t0 = time.time()
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        occ = sam_occupancy(predictor, nodes_img, cell_areas)
        dt = time.time() - t0
        print(f"frame {fi:5d}  pieces {int(occ.sum()):2d}  {dt:5.1f}s  "
              f"rss={rss_mb():.0f}MB  avail={avail_gb():.1f}GB", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
