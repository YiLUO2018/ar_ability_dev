"""M4 前奏：分析 sam_video_occupancy.csv 的噪声水平，为身份传播定策略。

输出：
  1. 每格占用频率（0~1，跨全部 3384 帧），10×9 网格；
  2. 每帧棋子数时间线（找手遮挡段）；
  3. 对照已知 TP/FP/miss 结论。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "debug_frames", "sam_video_occupancy.csv")
CSV = os.path.normpath(CSV)

# 已知结论（前 500 帧 TP 确认）
TP_CORRECT = {(f, 0) for f in range(9)} | {(1, 2), (7, 2)} | \
             {(f, 3) for f in (0, 2, 4, 6, 8)} | {(f, 6) for f in (0, 2, 4, 6, 8)} | \
             {(1, 7), (4, 7)} | {(f, 9) for f in range(9)}
FP_CONFIRMED = {(3, 7), (6, 7), (7, 7)}
MISS_CONFIRMED = {(1, 0), (8, 3), (8, 6)}


def load():
    data = np.loadtxt(CSV, delimiter=",", skiprows=1)
    frames = data[:, 0].astype(int)
    occ = data[:, 1:].astype(np.uint8)   # N x 90
    return frames, occ


def main():
    frames, occ = load()
    N = len(frames)
    print(f"frames: {N}  (frame {frames[0]}..{frames[-1]})")
    print(f"per-frame piece count: min={occ.sum(1).min()} max={occ.sum(1).max()} "
          f"median={int(np.median(occ.sum(1)))} mean={occ.sum(1).mean():.1f}\n")

    # 1) 每格频率
    freq = occ.mean(axis=0)
    print("=== 每格占用频率（0~1，跨全帧）===")
    for r in range(10):
        row = "".join(("1" if freq[r * 9 + f] >= 0.8 else
                       ("." if freq[r * 9 + f] < 0.2 else str(int(round(freq[r * 9 + f] * 10)))))
                      for f in range(9))
        print(f"rank{r}: " + " ".join(f"{freq[r*9+f]:.2f}" for f in range(9)))

    # 2) 对照已知 TP/FP/miss
    print("\n=== 对照已知结论（前500帧 TP 确认）===")
    tp_missed = [(f, r) for f, r in TP_CORRECT if freq[r * 9 + f] < 0.5]
    fp_present = [(f, r) for f, r in FP_CONFIRMED if freq[r * 9 + f] >= 0.5]
    miss_present = [(f, r) for f, r in MISS_CONFIRMED if freq[r * 9 + f] >= 0.5]
    print(f"TP 正确子在全帧频率<0.5（会被多数投票漏掉）: {tp_missed}")
    print(f"FP 误报在全帧频率>=0.5（会被多数投票保留）: {fp_present}")
    print(f"已知漏检在全帧频率>=0.5: {miss_present}")

    # 3) 棋子数时间线（粗看手遮挡段）
    counts = occ.sum(1)
    print("\n=== 棋子数时间线（每 100 帧的 min/median/max）===")
    for s in range(0, N, 100):
        c = counts[s:s + 100]
        print(f"frame {frames[s]:5d}-{frames[min(s+99,N-1)]:5d}: "
              f"min={c.min():2d} med={int(np.median(c)):2d} max={c.max():2d}")

    # 低子数段（手遮挡）
    low = counts < 28
    segs = []
    k = 0
    while k < N:
        if low[k]:
            s = k
            while k < N and low[k]:
                k += 1
            segs.append((int(frames[s]), int(frames[k - 1])))
        else:
            k += 1
    print(f"\n=== 棋子数<28 的段（疑似手遮挡）: {len(segs)} 段 ===")
    for s, e in segs:
        print(f"  frame {s}-{e}  (t={s/30:.1f}-{e/30:.1f}s, {e-s+1}帧)")


if __name__ == "__main__":
    main()
