"""分步定位坍缩环节：quad → nodes → detect 质心 → fit H → corners 投影。"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_piece_spike import canonical_nodes, project, PHYSICAL_CORNERS
from sam_detect_pieces import detect_pieces
from fit_std_homography import rough_nodes, fit_h, putative_from_detected, std_points, CHECKPOINT

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_02.mp4")


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 116
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
    nodes0, cw, ch, ca = rn
    print("1) 粗节点 spread: x", nodes0[:, 0].min().round(1), "-", nodes0[:, 0].max().round(1),
          " y", nodes0[:, 1].min().round(1), "-", nodes0[:, 1].max().round(1))
    print("   cell 均值:", round(float(np.mean(np.sqrt(ca))), 1), "px")

    detected = detect_pieces(pred, nodes0, cw, ch, ca)
    det_items = list(detected.items())
    det_xy = np.array([v["centroid"] for _, v in det_items], np.float32)
    print(f"2) 检出 {len(det_xy)} 子, 质心 x {det_xy[:,0].min():.0f}-{det_xy[:,0].max():.0f}"
          f" y {det_xy[:,1].min():.0f}-{det_xy[:,1].max():.0f}")
    uniq = np.unique(np.round(det_xy, 1), axis=0)
    print("   唯一质心数:", len(uniq))
    if len(uniq) < 10:
        print("   ", uniq)

    stds = std_points()
    det_keys = {k: i for i, (k, _) in enumerate(det_items)}
    putative, identity = putative_from_detected(stds, det_keys, det_xy)
    Hfit, inl = fit_h([p[0] for p in stds], det_xy, float(np.mean(np.sqrt(ca))), putative, identity)
    print(f"3) fit 内点 {int(inl.sum()) if inl is not None else 'None'}")
    print("   H=", np.round(Hfit, 3).tolist() if Hfit is not None else None)
    if Hfit is not None:
        corners = project(PHYSICAL_CORNERS.astype(np.float32), Hfit)
        print("4) corners=", np.round(corners, 1).tolist())
        nodes_img = project(canonical_nodes(), Hfit)
        print("   nodes x", nodes_img[:, 0].min().round(1), "-", nodes_img[:, 0].max().round(1),
              " y", nodes_img[:, 1].min().round(1), "-", nodes_img[:, 1].max().round(1))


if __name__ == "__main__":
    main()
