"""把视频帧 + warp 后棋盘渲染成 ASCII 灰度画，直接在终端「看」布局。

用法：python diag_ascii.py [frame_idx]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sam_piece_spike import PHYSICAL_CORNERS, BOARD_W, BOARD_H

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_02.mp4")
CHARS = "@%#*+=-:. "  # 暗 → 亮


def ascii_img(gray, w_cols):
    h, w = gray.shape
    rows = max(1, int(h * w_cols / w / 2))  # 字符高宽比 ~2:1
    small = cv2.resize(gray, (w_cols, rows), interpolation=cv2.INTER_AREA)
    out = []
    for r in range(rows):
        line = "".join(CHARS[int(v) * (len(CHARS) - 1) // 255] for v in small[r])
        out.append(line)
    return "\n".join(out)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1797
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    assert ok
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    print(f"== 原始帧 {idx}（{W}x{H}，每字符≈{W//54}px）==")
    print(ascii_img(gray, 54))

    # 用 diag_warp_grid 已存的 warp（若刚跑过）；否则直接把整帧下采样即可，这里重新 warp 一次
    from segment_anything import sam_model_registry, SamPredictor
    SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
    sys.path.insert(0, SAM_ROOT)
    pred = SamPredictor(sam_model_registry["vit_b"](checkpoint=SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"))
    sam = pred.model
    sam.to("cpu"); sam.eval()
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    from diag_warp_grid import order_quad, quad_from_mask
    masks, _, _ = pred.predict(
        point_coords=np.array([[W / 2, H / 2]]), point_labels=np.array([1]),
        multimask_output=True)
    best = None
    for k in range(masks.shape[0]):
        frac = masks[k].mean()
        if not (0.10 <= frac <= 0.60):
            continue
        q = quad_from_mask(masks[k])
        if q is None or (q[:, 0].min() < 0 or q[:, 0].max() > W or q[:, 1].min() < 0 or q[:, 1].max() > H):
            continue
        a = cv2.contourArea(q)
        if best is None or a > best[0]:
            best = (a, q, frac)
    if best:
        _, quad, frac = best
        print(f"\n== SAM 四角 {np.round(quad).astype(int).tolist()} 掩码{frac*100:.0f}% ==")
        M = cv2.getPerspectiveTransform(quad, PHYSICAL_CORNERS)
        warp = cv2.warpPerspective(img, M, (BOARD_W, BOARD_H))
        wg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        print(f"== warp 棋盘 450x500 → ASCII（每字符≈10px；红子看不清时用原帧）==")
        print(ascii_img(wg, 45))
    else:
        print("无合法掩码")


if __name__ == "__main__":
    main()
