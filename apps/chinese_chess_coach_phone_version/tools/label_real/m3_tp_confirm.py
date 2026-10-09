"""M3 第一步：前 500 帧时间一致性投票，确定哪些检测框是 TP。

方法：真实棋子跨帧稳定出现（位置不变），反光/纹理误报闪烁。前 [start,end) 帧均匀采样
N 帧跑 SAM bbox 检测，按 (file,rank) 统计出现频率。高频=TP，低频=FP，中频=移动棋子/临界。

输出：
  - 频率表（每格出现次数/频率 + TP/FP/mid 分类）
  - 可视化（debug_frames/m3_debug/m3_tp_confirm.png）

用法：python m3_tp_confirm.py [--sample N] [--end 500] [--freq_hi 0.7] [--freq_lo 0.3]
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_bbox_track import detect_pieces, cell_metrics, CHECKPOINT, SAM_ROOT

sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

OUT_DIR = os.path.join(os.path.dirname(VIDEO), "m3_debug")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    sample = int(arg("--sample", 10))
    end = int(arg("--end", 500))
    freq_hi = float(arg("--freq_hi", 0.7))
    freq_lo = float(arg("--freq_lo", 0.3))

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    print("loading SAM (cpu) ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    sample_frames = np.linspace(0, end - 1, sample, dtype=int)
    print(f"sample frames: {sample_frames.tolist()}")

    count = np.zeros(90, dtype=int)
    det_sets = {}
    for fi in sample_frames:
        t0 = time.time()
        frame = read_frame(VIDEO, int(fi))
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = detect_pieces(predictor, nodes_img, cw, ch, ca)
        det_sets[int(fi)] = det
        for (f, r) in det:
            count[r * 9 + f] += 1
        print(f"frame {fi:5d}: {len(det):2d} pieces  ({time.time()-t0:4.1f}s)")

    freq = count / sample
    tp = [(i % 9, i // 9) for i in range(90) if freq[i] >= freq_hi]
    fp = [(i % 9, i // 9) for i in range(90) if freq[i] < freq_lo]
    mid = [(i % 9, i // 9) for i in range(90) if freq_lo <= freq[i] < freq_hi]

    print(f"\n===== 时间一致性（采样 {sample} 帧，前 {end} 帧）=====")
    print(f"TP(>= {freq_hi}) {len(tp)} 个 | mid({freq_lo}~{freq_hi}) {len(mid)} 个 | FP(< {freq_lo}) {len(fp)} 个\n")
    for r in range(10):
        row = []
        for f in range(9):
            i = r * 9 + f
            if freq[i] >= freq_hi:
                row.append(f"{int(count[i])}")  # TP 显示次数
            elif freq[i] >= freq_lo:
                row.append(".")                  # mid 不显示
            else:
                row.append("_")                  # FP 下划线
        print(f"rank{r}: " + " ".join(f"{c:>1}" for c in row))

    print("\n频率明细（出现次数/sample）:")
    for r in range(10):
        print(f"rank{r}: " + " ".join(f"{freq[r*9+f]:.1f}" for f in range(9)))

    # 保存数据供第二步复用
    rep = min(det_sets, key=lambda k: abs(len(det_sets[k]) - 32))
    np.savez(os.path.join(OUT_DIR, "m3_tp_confirm_data.npz"),
             count=count, freq=freq, nodes_img=nodes_img, rep=rep,
             sample_frames=sample_frames)
    print(f"\n[data] -> {os.path.join(OUT_DIR, 'm3_tp_confirm_data.npz')}")

    # 可视化：裁剪棋盘区域 + 大字号 (file,rank) 编号
    rep_frame = read_frame(VIDEO, rep)
    x0 = max(0, int(SAM_PHYS[:, 0].min()) - 30)
    x1 = min(rep_frame.shape[1], int(SAM_PHYS[:, 0].max()) + 30)
    y0 = max(0, int(SAM_PHYS[:, 1].min()) - 30)
    y1 = min(rep_frame.shape[0], int(SAM_PHYS[:, 1].max()) + 30)
    vis = rep_frame[y0:y1, x0:x1].copy()
    off = np.array([x0, y0])
    nd = nodes_img - off
    for p in nd:
        cv2.drawMarker(vis, tuple(np.round(p).astype(int)),
                       (255, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
    det_rep = det_sets[rep]
    for i in range(90):
        f, r = i % 9, i // 9
        p = tuple(np.round(nd[i]).astype(int))
        if freq[i] >= freq_hi:
            color = (0, 255, 0)
            label = f"{f},{r}"
        elif freq[i] >= freq_lo:
            color = (0, 255, 255)
            label = f"{f},{r}"
        else:
            color = (0, 0, 255)
            label = f"{f},{r}"
        cv2.drawMarker(vis, p, color, cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(vis, label, (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for (f, r), (cx, cy, mask) in det_rep.items():
        if freq[r * 9 + f] >= freq_hi:
            ys, xs = np.nonzero(mask)
            if len(xs):
                cv2.rectangle(vis, (int(xs.min()) - x0, int(ys.min()) - y0),
                              (int(xs.max()) - x0, int(ys.max()) - y0), (0, 255, 0), 3)
    # 放大棋盘（宽 1400px）
    scale = 1400 / vis.shape[1]
    vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                     interpolation=cv2.INTER_CUBIC)
    out = os.path.join(OUT_DIR, "m3_tp_confirm.png")
    cv2.imwrite(out, vis)
    print(f"[vis] 代表帧 {rep} -> {out}")


if __name__ == "__main__":
    main()
