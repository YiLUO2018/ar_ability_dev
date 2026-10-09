"""M3：运动剖面 + 稳态分段（走子事件检测）。

走子 = 两个稳态段之间的跳变。逐帧把棋盘 warp 到 canonical（用帧 0 单应），
对棋盘区域做帧间绝对差，找运动尖峰（burst），切成稳态段。

只依赖帧间差（便宜），不受浅色棋子占位难的影响。稳态段代表帧后续交给 SAM 占位。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import BOARD_W, BOARD_H
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO


def board_diff_series(video, H_inv, step=3):
    """逐 step 帧 warp→canonical 并帧间 diff，返回 (idx, diff)。"""
    cap = cv2.VideoCapture(video)
    idxs, diffs = [], []
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step != 0:
            i += 1
            continue
        warped = cv2.warpPerspective(frame, H_inv, (BOARD_W, BOARD_H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            idxs.append(i)
            diffs.append(float(np.abs(gray - prev).mean()))
        prev = gray
        i += 1
    cap.release()
    return np.array(idxs), np.array(diffs)


def find_bursts(idxs, diffs, thresh, min_gap=2):
    """返回运动 burst 列表 [(start_idx, end_idx, peak)]，min_gap 帧内合并为同一 burst。"""
    active = diffs > thresh
    bursts = []
    k = 0
    n = len(idxs)
    while k < n:
        if active[k]:
            s = k
            while k < n and active[k]:
                k += 1
            e = k - 1
            # 与上一 burst 若间隔 <= min_gap 则合并
            if bursts and (s - bursts[-1][1]) <= min_gap:
                bursts[-1] = (bursts[-1][0], e, max(bursts[-1][2], diffs[s:e+1].max()))
            else:
                bursts.append((s, e, diffs[s:e+1].max()))
        else:
            k += 1
    return [(int(idxs[s]), int(idxs[e]), float(pk)) for s, e, pk in bursts]


def main():
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)

    idxs, diffs = board_diff_series(VIDEO, Hinv, step=3)
    base = float(np.median(diffs))
    print(f"采样 {len(idxs)} 帧  diff 中位数 {base:.1f} 均值 {diffs.mean():.1f} "
          f"p90 {np.percentile(diffs,90):.1f} p99 {np.percentile(diffs,99):.1f} 峰值 {diffs.max():.1f}")

    thresh = max(base * 3.0, base + 8.0)
    print(f"走子阈值 {thresh:.1f}")

    bursts = find_bursts(idxs, diffs, thresh, min_gap=2)
    print(f"\n运动 burst {len(bursts)} 个：")
    for s, e, pk in bursts:
        print(f"  帧 [{s:5d}, {e:5d}]  时间 [{s/30:6.2f}s, {e/30:6.2f}s]  峰值 {pk:.1f}")

    # 稳态段 = burst 之间的区间
    segs = []
    prev_end = 0
    for s, e, pk in bursts:
        if s - prev_end >= 6:  # 至少 6 帧（0.2s）才算稳态
            segs.append((prev_end, s, (prev_end + s) // 2))
        prev_end = e
    segs.append((prev_end, int(idxs[-1]), (prev_end + int(idxs[-1])) // 2))

    print(f"\n稳态段 {len(segs)} 个：")
    for k, (s, e, rep) in enumerate(segs):
        print(f"  段{k}: 帧 [{s:5d}, {e:5d}]  时间 [{s/30:6.2f}s, {e/30:6.2f}s]  代表帧 {rep}")


if __name__ == "__main__":
    main()
