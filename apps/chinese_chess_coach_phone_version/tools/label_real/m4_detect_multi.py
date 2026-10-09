"""M4 第 2 步（多帧版）：每稳态段采样多帧跑 detect_pieces + 多数投票，压掉闪烁误报/漏检。

单帧 detect_pieces 有 ~6 漏检 + 5-8 误报（低对比棋子 + 反射闪烁）。跨帧多数投票
（真实棋子稳定出现、反射闪烁）能大幅净化。输出每段「投票占用」JSON。

用法：python m4_detect_multi.py [--frames N] [--limit K]
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
OCC_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames", "sam_video_occupancy.csv"))
OUT_JSON = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames", "m3_debug", "m4_detect_multi.json"))
CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")


def sample_frames(seg, n):
    s, e = seg
    if e - s + 1 <= n:
        return list(range(s, e + 1))
    return [int(round(s + (e - s) * k / (n - 1))) for k in range(n)]


def main():
    n_frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 3
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
    if limit:
        segs = segs[:limit]
    print(f"steady segments={len(segs)}  frames/seg={n_frames}", flush=True)

    # 用 CSV 挑每段的候选帧（occ sum 接近 32 优先），再在其附近采样
    frames, occs = load_occ(OCC_CSV)
    seg_frames = []
    for s, e in segs:
        m = (frames >= s) & (frames <= e)
        sub = occs[m]
        if len(sub) == 0:
            seg_frames.append([(s + e) // 2])
            continue
        order = np.argsort(np.abs(sub.sum(axis=1) - 32))   # 越接近 32 越靠前
        best_frames = [int(frames[m][j]) for j in order[:n_frames]]
        best_frames = sorted(best_frames)                   # 保持时间顺序
        seg_frames.append(best_frames)

    print("loading SAM (cpu) ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    out = {"segs": [], "reps": [], "cells": [], "counts": []}
    for k, (seg, frs) in enumerate(zip(segs, seg_frames)):
        votes = {}
        for fi in frs:
            t0 = time.time()
            frame = read_frame(VIDEO, fi)
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            det = detect_pieces(predictor, nodes_img, cw, ch, ca)
            for (f, r) in det:
                votes[(int(f), int(r))] = votes.get((int(f), int(r)), 0) + 1
            print(f"  seg{k} frame {fi}: {len(det)} pieces ({time.time()-t0:.0f}s)", flush=True)
        th = len(frs) // 2 + 1   # 多数
        cells = sorted(c for c, v in votes.items() if v >= th)
        out["segs"].append([int(seg[0]), int(seg[1])])
        out["reps"].append(frs)
        out["cells"].append(cells)
        out["counts"].append({f"{c[0]},{c[1]}": v for c, v in votes.items()})
        print(f"seg{k} [{seg[0]},{seg[1]}]: voted {len(cells)} pieces", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"))
    print(f"\n[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
