"""诊断难帧（如 1045）：粗糙四角/格尺寸 + SAM (f,r) 分配占位图 vs 标准开局对照。"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_detect_pieces import detect_pieces
from fit_std_homography import rough_nodes, CHECKPOINT
from m2_opening import STD_OCC

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_02.mp4")


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1045
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    H, W = img.shape[:2]

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    rn = rough_nodes(pred, img)
    assert rn, "无合法掩码"
    nodes, cw, ch, ca = rn
    print(f"帧{idx} {W}x{H}")
    print(f"粗糙节点 spread: x {nodes[:,0].min():.0f}-{nodes[:,0].max():.0f}"
          f" y {nodes[:,1].min():.0f}-{nodes[:,1].max():.0f}")
    print(f"cell: x方向 {np.mean(cw):.1f}px  y方向 {np.mean(ch):.1f}px  sqrt(面积) {np.mean(np.sqrt(ca)):.1f}px")

    detected = detect_pieces(pred, nodes, cw, ch, ca)
    print(f"SAM 检出 {len(detected)}，粗糙 (f,r) 分配（* = 有子）vs 标准开局（S=std 有子）：")
    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in detected:
        occ[r][f] = 1
    for r in range(10):
        row = " ".join(("*" if occ[r][f] else ".") for f in range(9))
        std = " ".join(("S" if STD_OCC[r][f] else " ") for f in range(9))
        print(f"  r{r}: {row}   | {std}")

    # 每行的 x 间距（看粗糙四角透视是否畸变：正常的棋盘线 x 间距应单调缓变）
    print("行内节点 x 坐标（r0/r4/r9）：")
    for r in (0, 4, 9):
        print(f"  r{r}: " + " ".join(f"{nodes[r*9+f][0]:6.0f}" for f in range(9)))
    print("列内节点 y 坐标（f0/f4/f8）：")
    for f in (0, 4, 8):
        print(f"  f{f}: " + " ".join(f"{nodes[r*9+f][1]:6.0f}" for r in range(10)))


if __name__ == "__main__":
    main()
