"""M4 第 2 步：对每个稳态段代表帧跑 SAM bbox 逐子检测，保存结果供身份传播。

用 m3_bbox_track.detect_pieces（基线 90 交点 + 8 邻域补漏 + IoU 防重 + 最近空交点分配，
对偏移棋子鲁棒），对每个稳态段选「干净代表帧」检测，导出 JSON：
  {reps:[...], segs:[[s,e],...], cells:[[[f,r],...],...]}

用法：python m4_detect_segments.py [--limit N] [--start K]
"""
import os
import sys
import time
import json

import cv2
import numpy as np

SAM_ROOT = os.environ.get(
    "SAM_ROOT", r"L:\Reinforcement_Learning\segment\segment-anything-main")
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_motion import board_diff_series, find_bursts
from m3_bbox_track import detect_pieces, cell_metrics, load_occ

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_CSV = os.path.join(HERE, "..", "..", "debug_frames", "sam_video_occupancy.csv")
OCC_CSV = os.path.normpath(OCC_CSV)
OUT_JSON = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames", "m3_debug", "m4_detect_segments.json"))
CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")


def main():
    start_k = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    print(f"grid_rms={res['grid_rms']:.2f}", flush=True)

    idxs, diffs = board_diff_series(VIDEO, Hinv, step=3)
    base = float(np.median(diffs))
    thresh = max(base * 3.0, base + 8.0)
    bursts = find_bursts(idxs, diffs, thresh, min_gap=2)
    segs = []
    prev = 0
    for s, e, pk in bursts:
        if s - prev >= 6:
            segs.append((prev, s))
        prev = e
    segs.append((prev, int(idxs[-1])))
    print(f"steady segments={len(segs)}", flush=True)

    frames, occs = load_occ(OCC_CSV)
    reps = []
    for s, e in segs:
        m = (frames >= s) & (frames <= e)
        sub = occs[m]
        if len(sub) == 0:
            reps.append((s + e) // 2)
            continue
        j = int(np.argmin(np.abs(sub.sum(axis=1) - 32)))
        reps.append(int(frames[m][j]))
    if limit:
        segs = segs[:limit]
        reps = reps[:limit]
    print(f"representative frames ({len(reps)}): {reps}", flush=True)

    print("loading SAM (cpu) ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    out = {"reps": [], "segs": [], "cells": []}
    for k in range(start_k, len(reps)):
        rep = reps[k]
        t0 = time.time()
        frame = read_frame(VIDEO, rep)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = detect_pieces(predictor, nodes_img, cw, ch, ca)
        cells = sorted((int(f), int(r)) for (f, r) in det.keys())
        out["reps"].append(rep)
        out["segs"].append([int(segs[k][0]), int(segs[k][1])])
        out["cells"].append(cells)
        print(f"[{k+1}/{len(reps)}] frame {rep:5d}: {len(cells):2d} pieces  ({time.time()-t0:4.1f}s)",
              flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh)
    print(f"\n[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
