"""M3 可视化诊断：展示 SAM bbox 检测在视频帧上的结果，判断识别对错。

对若干代表帧跑 detect_pieces，画：
  - 绿框 = 检测到的棋子（标注 (file,rank)）
  - 红框 = 开局参照下疑似误报（检测到但标准开局布局无此子）
  - 红叉+「漏」 = 开局参照下疑似漏检（标准开局应有但未检出）
  - 黄十字 = 90 交点

输出到 debug_frames/m3_debug/。

用法：python m3_debug_vis.py
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

# 标准开局 32 子布局（rank-major，(file,rank)）
STANDARD_OPENING = set()
for f in range(9):
    STANDARD_OPENING.add((f, 0))   # rank0 黑底线 9 子
    STANDARD_OPENING.add((f, 9))   # rank9 红底线 9 子
for f in (1, 7):
    STANDARD_OPENING.add((f, 2))   # rank2 炮
    STANDARD_OPENING.add((f, 7))   # rank7 炮
for f in (0, 2, 4, 6, 8):
    STANDARD_OPENING.add((f, 3))   # rank3 兵
    STANDARD_OPENING.add((f, 6))   # rank6 卒

# 展示帧：开局(71) / 误报(148,212) / 手遮挡(583)
FRAMES = [71, 148, 212, 583]
OUT_DIR = os.path.join(os.path.dirname(VIDEO), "m3_debug")


def draw_detections(frame, nodes_img, det, opening_ref=False):
    vis = frame.copy()
    for p in nodes_img:
        cv2.drawMarker(vis, tuple(np.round(p).astype(int)),
                       (0, 255, 255), cv2.MARKER_TILTED_CROSS, 6, 1)
    for (f, r), (cx, cy, mask) in det.items():
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
        if opening_ref:
            correct = (f, r) in STANDARD_OPENING
            color = (0, 255, 0) if correct else (0, 0, 255)
            label = f"{f},{r}" if correct else f"{f},{r} FP"
        else:
            color = (0, 255, 0)
            label = f"{f},{r}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if opening_ref:
        for (f, r) in STANDARD_OPENING - set(det.keys()):
            p = tuple(np.round(nodes_img[r * 9 + f]).astype(int))
            cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"miss {f},{r}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    print("loading SAM (cpu) ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    for fi in FRAMES:
        frame = read_frame(VIDEO, fi)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = detect_pieces(predictor, nodes_img, cw, ch, ca)
        opening_ref = (fi == 71)
        vis = draw_detections(frame, nodes_img, det, opening_ref)
        out = os.path.join(OUT_DIR, f"m3_detect_{fi:05d}.png")
        cv2.imwrite(out, vis)
        n = len(det)
        note = " [开局参照]" if opening_ref else ""
        print(f"frame {fi:5d}: {n:2d} pieces -> {os.path.basename(out)}{note}")

    print(f"\n[done] 可视化 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
