"""诊断单帧几何对齐：warp 后打印 90 交点 grayMin 网格 + 检测占位，与标准开局对照。

若 warp 对齐正确，标准开局 32 子会在 grayMin 网格上呈现暗斑图案（数值低），
与 STD_OCC 吻合；若图案错乱 → 几何（SAM 四角/refine）问题；若图案对但检测占位乱
→ detect_pieces 阈值/分配问题。

用法：python diag_warp_grid.py [frame_idx]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_piece_spike import PHYSICAL_CORNERS, BOARD_W, BOARD_H, canonical_nodes, project
from geometry import CANON_CORNERS
from refine import refine_corners
from sam_detect_pieces import detect_pieces
from m2_opening import STD_OCC

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_02.mp4")

AREA_MIN, AREA_MAX = 0.10, 0.60


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_from_mask(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in [0.01, 0.015, 0.02, 0.03, 0.05, 0.08]:
        a = cv2.approxPolyDP(c, eps * peri, True)
        if len(a) == 4:
            return order_quad(a.reshape(4, 2))
    rect = cv2.minAreaRect(c)
    return order_quad(cv2.boxPoints(rect))


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1797
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    assert ok
    H, W = img.shape[:2]

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    masks, _, _ = pred.predict(
        point_coords=np.array([[W / 2, H / 2]]), point_labels=np.array([1]),
        multimask_output=True)
    best = None
    for k in range(masks.shape[0]):
        frac = masks[k].mean()
        if not (AREA_MIN <= frac <= AREA_MAX):
            continue
        q = quad_from_mask(masks[k])
        if q is None:
            continue
        if (q[:, 0].min() < 0 or q[:, 0].max() > W or q[:, 1].min() < 0 or q[:, 1].max() > H):
            continue
        a = cv2.contourArea(q)
        if best is None or a > best[0]:
            best = (a, q, frac)
    assert best, "无合法掩码"
    _, quad, frac = best
    print(f"帧{idx} 掩码占比={frac*100:.0f}% SAM物理四角={np.round(quad).astype(int).tolist()}")

    # 两种几何：A=仅 SAM 粗四角；B=SAM→refine 精修
    geoms = {}
    M_phys = cv2.getPerspectiveTransform(quad, PHYSICAL_CORNERS)  # image→canonical(物理框)
    geoms["A_sam"] = M_phys
    grid_corners = project(CANON_CORNERS.astype(np.float32), np.linalg.inv(M_phys))
    res = refine_corners(img, grid_corners / np.array([W, H], np.float32))
    print(f"refine: rms={res['grid_rms']:.2f} n_nodes={res['n_nodes']}")
    geoms["B_refine"] = np.linalg.inv(res["H"])  # image→canonical

    nodes_can = canonical_nodes()
    for tag, Mimg2can in geoms.items():
        warp = cv2.warpPerspective(img, Mimg2can, (BOARD_W, BOARD_H))
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(DBG, f"diag_warp_{tag}.png"), warp)
        print(f"\n== {tag} grayMin 网格（每格 15x15 patch 最小灰度；标准开局应在 STD_OCC 处呈低值）==")
        print("     " + " ".join(f"f{f}  " for f in range(9)))
        for r in range(10):
            row = []
            for f in range(9):
                x, y = int(round(nodes_can[r*9+f][0])), int(round(nodes_can[r*9+f][1]))
                p = gray[y-15:y+16, x-15:x+16]
                row.append(f"{p.min():3d}")
            print(f"r{r}:  " + " ".join(row))
        print("STD:  " + " ".join(" *" if STD_OCC[r][f] else " ." for f in range(9)))

    # B_refine 下的 detect_pieces 占位
    Hg = res["H"]
    nodes_img = project(nodes_can, Hg)
    cell_w, cell_h, cell_areas = [], [], []
    for r in range(10):
        for f in range(9):
            i = r*9+f
            p = nodes_img[i]
            fr = f+1 if f < 8 else f-1
            rd = r+1 if r < 9 else r-1
            cw = float(np.linalg.norm(p - nodes_img[r*9+fr]))
            ch = float(np.linalg.norm(p - nodes_img[rd*9+f]))
            cell_w.append(cw); cell_h.append(ch); cell_areas.append(cw*ch)
    detected = detect_pieces(pred, nodes_img, cell_w, cell_h, cell_areas)
    print(f"\ndetect_pieces → {len(detected)} 子，占位：")
    for r in range(10):
        print(f"r{r}:  " + " ".join(" *" if (f, r) in detected else " ." for f in range(9)))


if __name__ == "__main__":
    main()
