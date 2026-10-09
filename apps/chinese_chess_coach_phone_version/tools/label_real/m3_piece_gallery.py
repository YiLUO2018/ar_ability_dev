"""M3 第二步：棋子图鉴——抠出 TP 框棋子 ROI 放大排列，供人工确认类别。

读 m3_tp_confirm_data.npz（代表帧 + 频率），重跑代表帧 bbox 检测拿 mask，
对 29 个正确 TP（剔除用户确认的误报）抠棋子 ROI（含 margin），缩放拼成
按 (file,rank) 排列的图鉴，标注坐标。

用法：python m3_piece_gallery.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_bbox_track import detect_pieces, cell_metrics, CHECKPOINT, SAM_ROOT

sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

OUT_DIR = os.path.join(os.path.dirname(VIDEO), "m3_debug")
FP_CONFIRMED = {(3, 7), (6, 7), (7, 7)}   # 用户确认的误报
FREQ_HI = 0.7
CELL = 112   # 图鉴每个格子像素
MARGIN = 8   # 抠图 margin


def main():
    data = np.load(os.path.join(OUT_DIR, "m3_tp_confirm_data.npz"))
    rep = int(data["rep"])
    nodes_img = data["nodes_img"]
    freq = data["freq"]

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    cw, ch, ca = cell_metrics(nodes_img)

    print("loading SAM (cpu) ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    frame = read_frame(VIDEO, rep)
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    det = detect_pieces(predictor, nodes_img, cw, ch, ca)
    print(f"代表帧 {rep}: {len(det)} pieces")

    tp = [(f, r) for r in range(10) for f in range(9)
          if freq[r * 9 + f] >= FREQ_HI and (f, r) not in FP_CONFIRMED]
    print(f"TP {len(tp)} 个（剔除误报后）")

    # 图鉴画布：10 行 × 9 列
    H_pad, W_pad = 24, 24
    canvas = np.full((10 * (CELL + H_pad) + H_pad, 9 * (CELL + W_pad) + W_pad, 3),
                     50, dtype=np.uint8)
    for (f, r) in tp:
        if (f, r) not in det:
            print(f"  ! ({f},{r}) 代表帧未检出，跳过")
            continue
        cx, cy, mask = det[(f, r)]
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0, x1 = max(0, xs.min() - MARGIN), min(frame.shape[1], xs.max() + MARGIN)
        y0, y1 = max(0, ys.min() - MARGIN), min(frame.shape[0], ys.max() + MARGIN)
        roi = frame[y0:y1, x0:x1]
        # 等比缩放进 CELL×CELL
        rh, rw = roi.shape[:2]
        s = min(CELL / rw, CELL / rh)
        roi = cv2.resize(roi, (max(1, int(rw * s)), max(1, int(rh * s))))
        th, tw = roi.shape[:2]
        y_off = H_pad + r * (CELL + H_pad) + (CELL - th) // 2
        x_off = W_pad + f * (CELL + W_pad) + (CELL - tw) // 2
        canvas[y_off:y_off + th, x_off:x_off + tw] = roi
        cv2.rectangle(canvas, (W_pad + f * (CELL + W_pad), H_pad + r * (CELL + H_pad)),
                      (W_pad + f * (CELL + W_pad) + CELL, H_pad + r * (CELL + H_pad) + CELL),
                      (0, 200, 0), 1)
        cv2.putText(canvas, f"{f},{r}",
                    (W_pad + f * (CELL + W_pad) + 4, H_pad + r * (CELL + H_pad) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    out = os.path.join(OUT_DIR, "m3_piece_gallery.png")
    cv2.imwrite(out, canvas)
    print(f"[vis] 图鉴 -> {out}")

    # 同时输出 TP 坐标清单（供后续类别标注对齐）
    print("\nTP 坐标清单:")
    print(", ".join(f"({f},{r})" for f, r in tp))


if __name__ == "__main__":
    main()
