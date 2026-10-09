"""M3：from→to 检测全管线（运动分段 + 占位时间投票 + 稳态对比）。

流水线：
  1. 帧 0 精修棋盘单应 H；
  2. 逐 3 帧 warp→canonical 帧间 diff → 运动 burst → 稳态段（走子事件）；
  3. 逐 3 帧算 std+红 占位（便宜）→ 每个稳态段内**多数投票**得到稳定 90bit 占位；
  4. 相邻稳态段占位差 → from（1→0）/ to（0→1）。

只跑便宜占位（std+红），不逐帧 SAM。运动 burst 用来切稳态段，占位投票用来压掉
右下角反光区的帧间抖动。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import BOARD_W, BOARD_H
from m3_occupancy import (
    read_frame, board_homography, SAM_PHYS, VIDEO,
    classify_occupancy, print_grid,
)

STEP = 3
MOTION_THRESH = 20.0
MIN_SEG = 30      # 稳态段最短帧数（1s）
MERGE_GAP = 30    # burst 合并窗口（1s）
MAX_PIECES = 34   # 合法局面最多 32 子 + 2 容差；远超（如 45+）= 手遮挡/阴影污染，段作废


def diff_series(video, Hinv):
    cap = cv2.VideoCapture(video)
    idxs, diffs = [], []
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STEP != 0:
            i += 1
            continue
        warped = cv2.warpPerspective(frame, Hinv, (BOARD_W, BOARD_H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            idxs.append(i)
            diffs.append(float(np.abs(gray - prev).mean()))
        prev = gray
        i += 1
    cap.release()
    return np.array(idxs), np.array(diffs)


def occupancy_series(video, H):
    """逐 STEP 帧算 std+红 占位，返回 (idxs, occ 矩阵 N×90)。"""
    cap = cv2.VideoCapture(video)
    idxs, occs = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STEP != 0:
            i += 1
            continue
        occ, _, _ = classify_occupancy(frame, H)
        idxs.append(i)
        occs.append(occ)
        i += 1
    cap.release()
    return np.array(idxs), np.array(occs)


def find_steady_segments(idxs, diffs):
    """运动 burst 之间为稳态段。返回 [(start_idx, end_idx)]。"""
    active = diffs > MOTION_THRESH
    bursts = []
    k = 0
    while k < len(active):
        if active[k]:
            s = k
            while k < len(active) and active[k]:
                k += 1
            e = k - 1
            if bursts and (s - bursts[-1][1]) <= MERGE_GAP:
                bursts[-1] = (bursts[-1][0], e)
            else:
                bursts.append((s, e))
        else:
            k += 1
    segs = []
    prev = 0
    for s, e in bursts:
        if idxs[s] - idxs[prev] >= MIN_SEG:
            segs.append((int(idxs[prev]), int(idxs[s])))
        prev = e
    if idxs[-1] - idxs[prev] >= MIN_SEG:
        segs.append((int(idxs[prev]), int(idxs[-1])))
    return segs


def vote_occupancy(occs, idxs, seg):
    """稳态段内多数投票（占位 0/1 逐格取众数），返回 90bit。"""
    mask = (idxs >= seg[0]) & (idxs <= seg[1])
    sub = occs[mask]
    if len(sub) == 0:
        return np.zeros(90, dtype=np.uint8)
    voted = (sub.sum(axis=0) > len(sub) / 2).astype(np.uint8)
    return voted


def fromto(a, b):
    d = b.astype(int) - a.astype(int)
    frm = [(i % 9, i // 9) for i in range(90) if d[i] == -1]
    to = [(i % 9, i // 9) for i in range(90) if d[i] == 1]
    return frm, to


def main():
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    print(f"refine grid_rms={res['grid_rms']:.2f}")

    idxs, diffs = diff_series(VIDEO, Hinv)
    print(f"diff: 中位数 {np.median(diffs):.1f} 峰值 {diffs.max():.1f}")

    segs = find_steady_segments(idxs, diffs)
    print(f"\n稳态段 {len(segs)} 个：")
    for k, (s, e) in enumerate(segs):
        print(f"  段{k}: 帧 [{s:5d},{e:5d}] 时间 [{s/30:6.2f}s,{e/30:6.2f}s]")

    oidx, occs = occupancy_series(VIDEO, H)
    print(f"\n占位采样 {len(oidx)} 帧")

    voted = []
    valid = []
    for k, (s, e) in enumerate(segs):
        v = vote_occupancy(occs, oidx, (s, e))
        voted.append(v)
        n = int(v.sum())
        ok = n <= MAX_PIECES
        if ok:
            valid.append(k)
        tag = "" if ok else "  [污染，作废]"
        print(f"\n=== 稳态段{k} (帧 {s}–{e}, t={s/30:.1f}–{e/30:.1f}s) 投票占位 {n} 子{tag} ===")
        print_grid(["X" if v[i] else "." for i in range(90)])

    print("\n=== from→to（相邻有效稳态段对比，跳过污染段）===")
    for a, b in zip(valid, valid[1:]):
        frm, to = fromto(voted[a], voted[b])
        sa, ea = segs[a]; sb, eb = segs[b]
        if not frm and not to:
            print(f"段{a}→段{b}: 无变化")
        else:
            print(f"段{a}(t≈{ea/30:.1f}s)→段{b}(t≈{sb/30:.1f}s): from={frm} to={to}")


if __name__ == "__main__":
    main()
