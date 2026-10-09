"""验证 debug_video_02 摆位（多帧共识版）。

先验（用户提供）：整段视频棋子零变化、只有视角变。因此：
- 每帧只做「SAM 棋盘四角 + refine_corners 精修 + detect_pieces 占位」；
- 跨帧按交点多数投票出共识占位 → match_standard 一次定身份；
- 与共识偏差大的帧 = 几何/检测失败帧，被投票自然过滤。

用法：python verify_std02.py [video] [n_frames]
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_piece_spike import (
    PHYSICAL_CORNERS, BOARD_W, BOARD_H, canonical_nodes, project,
)
from geometry import CANON_CORNERS
from refine import refine_corners
from sam_detect_pieces import detect_pieces
from m2_opening import match_standard, STD_OCC

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DBG, "debug_video_02.mp4")
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 5

AREA_MIN, AREA_MAX = 0.10, 0.60   # 棋盘掩码合法面积区间（占整帧比例）
RMS_GATE = 3.0                    # refine 后 grid_rms 超过此值判几何失败
OUT_TAG = os.path.splitext(os.path.basename(VIDEO))[0]


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


def board_quad(pred, img):
    """SAM 中心点提示 → 合法面积区间内最大掩码 → 物理四角。"""
    H, W = img.shape[:2]
    masks, scores, _ = pred.predict(
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
        if (q[:, 0].min() < 0 or q[:, 0].max() > W or
                q[:, 1].min() < 0 or q[:, 1].max() > H):
            continue
        area = cv2.contourArea(q)
        if best is None or area > best[0]:
            best = (area, q, frac)
    if best is None:
        return None, None
    return best[1], best[2]


def frame_occupancy(pred, img):
    """单帧：四角 → 精修 → detect_pieces → (occ, info)。失败返回 (None, 原因)。"""
    H, W = img.shape[:2]
    quad, frac = board_quad(pred, img)
    if quad is None:
        return None, "无合法掩码"
    M_phys = cv2.getPerspectiveTransform(quad, PHYSICAL_CORNERS)
    grid_corners = project(CANON_CORNERS.astype(np.float32), np.linalg.inv(M_phys))
    res = refine_corners(img, grid_corners / np.array([W, H], np.float32))
    Hg, rms = res["H"], res["grid_rms"]
    if not np.isfinite(rms) or rms > RMS_GATE:
        return None, f"grid_rms={rms:.1f}"

    nodes = project(canonical_nodes(), Hg)
    cell_w, cell_h, cell_areas = [], [], []
    for r in range(10):
        for f in range(9):
            i = r * 9 + f
            p = nodes[i]
            fr = f + 1 if f < 8 else f - 1
            rd = r + 1 if r < 9 else r - 1
            cw = float(np.linalg.norm(p - nodes[r * 9 + fr]))
            ch = float(np.linalg.norm(p - nodes[rd * 9 + f]))
            cell_w.append(cw); cell_h.append(ch); cell_areas.append(cw * ch)

    detected = detect_pieces(pred, nodes, cell_w, cell_h, cell_areas)
    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in detected:
        occ[r][f] = 1
    info = {"frac": frac, "rms": rms, "n": len(detected),
            "nodes": nodes, "H": Hg, "detected": detected}
    return occ, info


def main():
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratios = [0.5 - 0.42 * (1 - 2 * i / (N_FRAMES - 1)) for i in range(N_FRAMES)]
    t0 = time.time()
    print(f"加载 SAM ViT-B ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)

    frames = []   # (ratio, occ 或 None, info/原因)
    for ratio in ratios:
        idx = int(n * ratio)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            frames.append((ratio, None, "读帧失败"))
            continue
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        occ, info = frame_occupancy(pred, img)
        tag = "ok" if occ is not None else info
        print(f"  帧{idx:5d} ({ratio:.2f}) {tag}  "
              f"{'掩码%.0f%% rms=%.2f 子=%d' % (info['frac']*100, info['rms'], info['n']) if occ is not None else ''}"
              f"  {time.time()-t1:.0f}s", flush=True)
        frames.append((ratio, occ, info))
    cap.release()

    occs = [o for _, o, _ in frames if o is not None]
    if not occs:
        print("[fail] 无有效帧")
        return
    votes = np.sum(occs, axis=0)
    k = len(occs)
    consensus = (votes * 2 > k).astype(int)   # 严格多数
    print(f"\n有效帧 {k}/{N_FRAMES}")
    print("共识占位（* = 有子）：")
    for r in range(10):
        print("  " + " ".join("*" if consensus[r][f] else "." for f in range(9)))

    extra = (consensus == 1) & (STD_OCC == 0)
    missing = (consensus == 0) & (STD_OCC == 1)
    print(f"共识 vs 标准开局：子 {consensus.sum()}/32  多={int(extra.sum())} 缺={int(missing.sum())}")

    # 每帧与共识的一致率
    for ratio, occ, info in frames:
        if occ is None:
            continue
        agree = float((occ == consensus).mean())
        print(f"  帧({ratio:.2f}) 与共识一致率 {agree*100:.0f}%")

    is_std, grid = match_standard(consensus)
    print(f"\n[判定] match_standard(共识) = {is_std}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
