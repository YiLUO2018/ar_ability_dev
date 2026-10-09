"""M3：全量逐帧 SAM 占位 → 稳态段综合（段内多数投票，非滑窗）→ from→to。

输入：
  - debug_frames/sam_video_occupancy.csv（m3_sam_video.py 全量输出：frame,bit0..bit89）
  - 视频（算运动分段，复用帧 0 单应）

流程：
  1. 读视频 diff → 运动 burst → 稳态段（find_steady_segments）
  2. 读 CSV 逐帧 SAM 占位
  3. 每个稳态段内多数投票（非滑窗）→ 稳定 90bit 占位
  4. 相邻有效稳态段对比 → from→to

用法：python m3_sam_aggregate.py
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_pipeline import diff_series, find_steady_segments, fromto, MAX_PIECES
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO, print_grid

OCC_CSV = os.path.join(os.path.dirname(VIDEO), "sam_video_occupancy.csv")


def load_occupancy(csv_path):
    """读 CSV → (frames N, occ N×90 uint8)。"""
    frames, occs = [], []
    with open(csv_path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            frames.append(int(row[0]))
            occs.append([int(x) for x in row[1:]])
    return np.array(frames), np.array(occs, dtype=np.uint8)


def vote_occupancy(occs, frames, seg):
    """稳态段内多数投票（非滑窗）。"""
    mask = (frames >= seg[0]) & (frames <= seg[1])
    sub = occs[mask]
    if len(sub) == 0:
        return np.zeros(90, dtype=np.uint8)
    return (sub.sum(axis=0) > len(sub) / 2).astype(np.uint8)


def main():
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    print(f"refine grid_rms={res['grid_rms']:.2f}")

    idxs, diffs = diff_series(VIDEO, Hinv)
    segs = find_steady_segments(idxs, diffs)
    print(f"稳态段 {len(segs)} 个：")
    for k, (s, e) in enumerate(segs):
        print(f"  段{k}: 帧 [{s:5d},{e:5d}] 时间 [{s/30:6.2f}s,{e/30:6.2f}s]")

    frames, occs = load_occupancy(OCC_CSV)
    print(f"\nSAM 占位 {len(frames)} 帧（帧 {frames.min()}..{frames.max()}）")

    voted, valid = [], []
    for k, (s, e) in enumerate(segs):
        v = vote_occupancy(occs, frames, (s, e))
        voted.append(v)
        n = int(v.sum())
        ok = n <= MAX_PIECES
        if ok:
            valid.append(k)
        tag = "" if ok else "  [污染]"
        print(f"\n=== 段{k} (帧 {s}-{e}, t={s/30:.1f}-{e/30:.1f}s) 投票 {n} 子{tag} ===")
        print_grid(["X" if v[i] else "." for i in range(90)])

    print("\n=== from→to（相邻有效稳态段，跳过污染段）===")
    for a, b in zip(valid, valid[1:]):
        frm, to = fromto(voted[a], voted[b])
        sa, ea = segs[a]; sb, eb = segs[b]
        if not frm and not to:
            print(f"段{a}→段{b}: 无变化")
        else:
            print(f"段{a}(t≈{ea/30:.1f}s)→段{b}(t≈{sb/30:.1f}s): from={frm} to={to}")


if __name__ == "__main__":
    main()
