"""M4 身份传播 第 1 步：稳态段占位 + from→to 诊断。

读 sam_video_occupancy.csv（3384×90 SAM 占位），用运动 burst 切稳态段，
段内多数投票得干净占位，逐段打印 + 相邻段 from→to 对比，评估信号干净度。

不跑 SAM（纯 CSV + 帧间 diff），便宜。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import BOARD_W, BOARD_H
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_motion import board_diff_series, find_bursts

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames", "sam_video_occupancy.csv"))

# 用户确认的持续误报（首段已知为 FP；后续段可能被真实走子占据，仅首段强制清零）
FP_CONFIRMED = {(3, 7), (6, 7), (7, 7)}


def load_occ():
    data = np.loadtxt(CSV, delimiter=",", skiprows=1)
    frames = data[:, 0].astype(int)
    occ = data[:, 1:].astype(np.uint8)
    return frames, occ


def seg_occupancy(occ, frames, seg):
    """段内多数投票。返回 90bit。"""
    mask = (frames >= seg[0]) & (frames <= seg[1])
    sub = occ[mask]
    if len(sub) == 0:
        return np.zeros(90, dtype=np.uint8)
    return (sub.sum(axis=0) * 2 > len(sub)).astype(np.uint8)


def grid_str(occ):
    rows = []
    for r in range(10):
        rows.append(f"rank{r}: " + "".join("X" if occ[r * 9 + f] else "." for f in range(9)))
    return "\n".join(rows)


def cells(occ):
    return {(i % 9, i // 9) for i in range(90) if occ[i]}


def main():
    frames, occ = load_occ()

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    idxs, diffs = board_diff_series(VIDEO, Hinv, step=3)
    base = float(np.median(diffs))
    thresh = max(base * 3.0, base + 8.0)
    bursts = find_bursts(idxs, diffs, thresh, min_gap=2)

    segs = []
    prev_end = 0
    for s, e, pk in bursts:
        if s - prev_end >= 6:
            segs.append((prev_end, s))
        prev_end = e
    segs.append((prev_end, int(idxs[-1])))
    print(f"稳态段 {len(segs)} 个（阈值 {thresh:.1f}）\n")

    occs = []
    for k, (s, e) in enumerate(segs):
        v = seg_occupancy(occ, frames, (s, e))
        if k == 0:
            for (f, r) in FP_CONFIRMED:
                v[r * 9 + f] = 0
        occs.append(v)
        print(f"=== 段{k} 帧[{s},{e}] t=[{s/30:.1f},{e/30:.1f}]s  子数 {int(v.sum())} ===")
        print(grid_str(v))
        print()

    print("=== 相邻段 from→to ===")
    for k in range(len(occs) - 1):
        a, b = occs[k], occs[k + 1]
        d = b.astype(int) - a.astype(int)
        frm = [(i % 9, i // 9) for i in range(90) if d[i] == -1]
        to = [(i % 9, i // 9) for i in range(90) if d[i] == 1]
        n_a, n_b = int(a.sum()), int(b.sum())
        print(f"段{k}→段{k+1}: 子数 {n_a}→{n_b}  from={frm}  to={to}")


if __name__ == "__main__":
    main()
