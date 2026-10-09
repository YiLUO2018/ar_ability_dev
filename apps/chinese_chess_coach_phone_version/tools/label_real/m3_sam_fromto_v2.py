"""M3 v2: fine-grained steady segments (adaptive threshold) + SAM occupancy from->to.

Diagnosis: m3_sam_aggregate.py used fixed MOTION_THRESH=20 (too coarse, 10 segs),
missing small moves (peaks 11-19). This uses m3_motion's adaptive threshold
(base*3) to cut ~39 fine segments, then majority-vote SAM occupancy per segment
and diff adjacent segments to extract from->to.

Usage: python m3_sam_fromto_v2.py
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_motion import board_diff_series, find_bursts
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO

OCC_CSV = os.path.join(os.path.dirname(VIDEO), "sam_video_occupancy.csv")


def load_occupancy(csv_path):
    frames, occs = [], []
    with open(csv_path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            frames.append(int(row[0]))
            occs.append([int(x) for x in row[1:]])
    return np.array(frames), np.array(occs, dtype=np.uint8)


def vote(occs, frames, s, e):
    m = (frames >= s) & (frames <= e)
    sub = occs[m]
    if len(sub) == 0:
        return np.zeros(90, np.uint8)
    return (sub.sum(0) > len(sub) / 2).astype(np.uint8)


def main():
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    print(f"grid_rms={res['grid_rms']:.2f}")

    idxs, diffs = board_diff_series(VIDEO, Hinv, step=3)
    base = float(np.median(diffs))
    thresh = max(base * 3.0, base + 8.0)
    print(f"diff base={base:.1f}  thresh={thresh:.1f}")
    bursts = find_bursts(idxs, diffs, thresh, min_gap=2)
    print(f"bursts={len(bursts)}")

    # steady segments = gaps between bursts + head/tail
    segs = []
    prev = 0
    for s, e, pk in bursts:
        if s - prev >= 6:
            segs.append((prev, s))
        prev = e
    segs.append((prev, int(idxs[-1])))
    print(f"steady segments={len(segs)}")

    frames, occs = load_occupancy(OCC_CSV)

    voted = [vote(occs, frames, s, e) for s, e in segs]
    for k, (s, e) in enumerate(segs):
        print(f"seg{k}: [{s:5d},{e:5d}] t={s/30:5.1f}-{e/30:5.1f}s  {int(voted[k].sum()):2d} pieces")

    print("\n=== from->to (adjacent segs) ===")
    for a in range(len(segs) - 1):
        d = voted[a + 1].astype(int) - voted[a].astype(int)
        frm = [(i % 9, i // 9) for i in range(90) if d[i] == -1]
        to = [(i % 9, i // 9) for i in range(90) if d[i] == 1]
        if frm or to:
            print(f"seg{a}->seg{a+1}: from={frm} to={to}   (d_from={len(frm)} d_to={len(to)})")
        else:
            print(f"seg{a}->seg{a+1}: no change")


if __name__ == "__main__":
    main()
