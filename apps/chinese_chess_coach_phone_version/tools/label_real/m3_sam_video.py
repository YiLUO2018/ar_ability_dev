"""M3：逐帧 SAM 分割占位 + 可视化（离线标注，质量优先，不追求便宜）。

固定视角：帧 0 精修棋盘单应 H 后，后续所有帧复用 H（只在棋盘 90 交点范围内检测）。
每帧 SAM set_image 一次 → 90 交点逐一单点提示 → 判「该交点是否有棋子」，
画可视化帧图（绿=棋子 mask，红叉=空交点，黄十字=交点）+ 导出每帧 90bit 占位 CSV。

用法：
  python m3_sam_video.py --start 0 --end 90 --step 30   # 先预览（每 30 帧取 1）
  python m3_sam_video.py --start 0 --end 3384 --step 1  # 全量逐帧
"""
import os
import sys
import time

import cv2
import numpy as np

SAM_ROOT = os.environ.get(
    "SAM_ROOT", r"L:\Reinforcement_Learning\segment\segment-anything-main")
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from sam_piece_spike import largest_component, AREA_LO, AREA_HI, CIRC_MIN, DIST_CELL
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO

CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT", os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth"))
OUT_DIR = os.path.join(os.path.dirname(VIDEO), "sam_video_frames")
OUT_CSV = os.path.join(os.path.dirname(VIDEO), "sam_video_occupancy.csv")

def cell_areas_for(nodes_img):
    """每个交点的局部格面积（原图像素，右/下邻居距离之积）。"""
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
    """90 交点逐一单点提示 → 90bit 占位 + (mask, centroid) 列表。

    最简单方式：每个交点单独调 predictor.predict()，不做任何 batch/分块，
    避免 repeat_interleave + 全分辨率 postprocess 的内存峰值。
    """
    occ = np.zeros(90, dtype=np.uint8)
    out_masks = [None] * 90
    cents = [None] * 90
    for i in range(90):
        p = nodes_img[i]
        ca = cell_areas[i]
        masks, _, _ = predictor.predict(
            point_coords=np.array([p], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]  # bool (H, W)
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
            out_masks[i] = mask
            cents[i] = (cx, cy)
    return occ, out_masks, cents


def draw_overlay(frame, occ, masks, nodes_img, fi):
    """在原始帧上画可视化：绿=棋子，红叉=空，黄十字=交点。"""
    vis = frame.copy()
    overlay = vis.copy()
    for i in range(90):
        if occ[i]:
            overlay[masks[i]] = (0, 220, 0)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    for i in range(90):
        p = tuple(np.round(nodes_img[i]).astype(int))
        if occ[i]:
            cnts, _ = cv2.findContours((masks[i].astype(np.uint8)) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            cv2.drawMarker(vis, p, (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        else:
            cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
    cv2.putText(vis, f"frame {fi}  pieces {int(occ.sum())}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def main():
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    end = int(sys.argv[sys.argv.index("--end") + 1]) if "--end" in sys.argv else 3384
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 1

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cell_areas = cell_areas_for(nodes_img)
    print(f"refine grid_rms={res['grid_rms']:.2f}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SAM (device={device}) ...")
    t0 = time.time()
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to(device); sam.eval()
    predictor = SamPredictor(sam)
    print(f"model ready {time.time()-t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = list(range(start, end, step))
    print(f"frames to process: {len(frames)} ({start}..{end} step {step})")

    # 断点续跑：已存在帧图则跳过；CSV 边跑边 append（崩溃也不丢已完成结果）
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w") as fh:
            fh.write("frame," + ",".join(str(i) for i in range(90)) + "\n")

    done = 0
    skipped = 0
    for k, fi in enumerate(frames):
        out_png = os.path.join(OUT_DIR, f"frame_{fi:05d}.png")
        if os.path.exists(out_png):
            skipped += 1
            continue
        t1 = time.time()
        frame = read_frame(VIDEO, fi)
        if frame is None:
            continue
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        occ, masks, cents = sam_occupancy(predictor, nodes_img, cell_areas)
        vis = draw_overlay(frame, occ, masks, nodes_img, fi)
        cv2.imwrite(out_png, vis)
        with open(OUT_CSV, "a") as fh:
            fh.write(f"{fi}," + ",".join(str(int(o)) for o in occ) + "\n")
        done += 1
        dt = time.time() - t1
        print(f"frame {fi:5d}  pieces {int(occ.sum()):2d}  {dt:5.1f}s/帧"
              f"  已算 {done}  跳过 {skipped}", flush=True)

    print(f"\n[vis] frames -> {OUT_DIR}")
    print(f"[csv] occupancy -> {OUT_CSV}（本次新算 {done}，跳过已存在 {skipped}）")


if __name__ == "__main__":
    main()
