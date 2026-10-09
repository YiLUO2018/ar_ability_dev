"""M3 v3: bbox 逐子跟踪 from->to（替代占位 diff）。

占位 diff 在 debug_video_01.mp4 上被「低对比棋子 + 手遮挡」卡死（每步 3-14 个
from/to 且不守恒）。改用 M5 已验证的 SAM bbox 逐子检测（38/38、0 误报）做跟踪：

  1. 运动分段（自适应阈值）得稳态段；
  2. 每段选「干净代表帧」= 段内已有 CSV 占位 sum 最接近 32 的帧（避开手遮挡）；
  3. 代表帧跑 SAM bbox 检测（基线 90 交点 + 8 邻域补漏 + IoU 防重 + 最近空交点分配），
     得到逐子质心 (cx,cy) 连续坐标；
  4. 相邻代表帧棋子质心做贪心最近邻匹配（阈值 ~0.5 格），未匹配 = from(消失)/to(新增)。

吃子语义与占位 diff 一致：吃子时 to 格 1→1 不变，故 bbox 匹配会把「被吃子」与
「走子」在同格配对，to 为空，只有 from=走子原位置——正确。

用法：python m3_bbox_track.py [--limit N]   # --limit 只跑前 N 个代表帧（快速验证）
"""
import os
import sys
import time
import csv

import cv2
import numpy as np

SAM_ROOT = os.environ.get(
    "SAM_ROOT", r"L:\Reinforcement_Learning\segment\segment-anything-main")
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from sam_piece_spike import predict_and_classify, DIST_CELL
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_motion import board_diff_series, find_bursts

CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")
OCC_CSV = os.path.join(os.path.dirname(VIDEO), "sam_video_occupancy.csv")

PROBES = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
          (0.35, 0.35), (0.35, -0.35), (-0.35, 0.35), (-0.35, -0.35)]
FILL_DIST = 0.75   # 邻域探针质心到空交点最大距离（格）
IOU_DUP = 0.3      # IoU 防重阈值
MATCH_CELL = 0.5   # 匹配距离阈值（格边长）


def largest_component_mask(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(mask.shape, dtype=bool)
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros(mask.shape, dtype=bool)
    cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def cell_metrics(nodes_img):
    cw, ch, ca = [], [], []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]
        fr = f + 1 if f < 8 else f - 1
        rd = r + 1 if r < 9 else r - 1
        w = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
        h = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
        cw.append(w); ch.append(h); ca.append(w * h)
    return cw, ch, ca


def detect_pieces(predictor, nodes_img, cw, ch, ca):
    """SAM bbox 逐子检测 → { (f,r): (cx, cy, mask) }."""
    detected = {}
    claimed = []
    # 1) 基线：90 交点中心点提示
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]; a = ca[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, a)
        if is_piece and centroid is not None:
            d = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if d < DIST_CELL * np.sqrt(a):
                clean = largest_component_mask(mask)
                detected[(f, r)] = (centroid[0], centroid[1], clean)
                claimed.append(clean)
    base = len(detected)
    # 2) 补漏：空交点 8 邻域探针
    empty_ids = [i for i in range(90) if (i % 9, i // 9) not in detected]
    cand = []
    for i in empty_ids:
        f, r = i % 9, i // 9
        p = nodes_img[i]; a = ca[i]
        for ox, oy in PROBES:
            pt = (p[0] + ox * cw[i], p[1] + oy * ch[i])
            mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, pt, a)
            if not (is_piece and centroid is not None):
                continue
            d = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if d > FILL_DIST * np.sqrt(a):
                continue
            clean = largest_component_mask(mask)
            if any(mask_iou(clean, m) > IOU_DUP for m in claimed):
                continue
            cand.append((centroid[0], centroid[1], clean))
            claimed.append(clean)
            break
    # 3) 全局最近空交点分配
    for cx, cy, mask in cand:
        best = min(empty_ids, key=lambda j: float(np.hypot(
            cx - nodes_img[j][0], cy - nodes_img[j][1])))
        db = float(np.hypot(cx - nodes_img[best][0], cy - nodes_img[best][1]))
        if db > FILL_DIST * np.sqrt(ca[best]):
            continue
        f, r = best % 9, best // 9
        if (f, r) in detected:
            continue
        detected[(f, r)] = (cx, cy, mask)
        empty_ids.remove(best)
    return detected


def load_occ(csv_path):
    frames, occs = [], []
    with open(csv_path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            frames.append(int(row[0]))
            occs.append([int(x) for x in row[1:]])
    return np.array(frames), np.array(occs, dtype=np.uint8)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    Hinv = np.linalg.inv(H)
    print(f"grid_rms={res['grid_rms']:.2f}")

    idxs, diffs = board_diff_series(VIDEO, Hinv, step=3)
    base = float(np.median(diffs))
    thresh = max(base * 3.0, base + 8.0)
    bursts = find_bursts(idxs, diffs, thresh, min_gap=2)
    segs = []
    prev = 0
    for s, e, pk in bursts:
        if s - prev >= 6:
            segs.append((prev, s))
        prev = e
    segs.append((prev, int(idxs[-1])))
    print(f"steady segments={len(segs)}")

    frames, occs = load_occ(OCC_CSV)
    reps = []
    for s, e in segs:
        m = (frames >= s) & (frames <= e)
        sub = occs[m]
        if len(sub) == 0:
            reps.append((s + e) // 2)
            continue
        j = int(np.argmin(np.abs(sub.sum(axis=1) - 32)))
        reps.append(int(frames[m][j]))
    if limit:
        reps = reps[:limit]
    print(f"representative frames ({len(reps)}): {reps}")

    print("loading SAM (cpu) ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)
    match_px = MATCH_CELL * float(np.mean(cw))

    piece_sets = []
    for rep in reps:
        t0 = time.time()
        frame = read_frame(VIDEO, rep)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = detect_pieces(predictor, nodes_img, cw, ch, ca)
        pieces = sorted((f, r, cx, cy) for (f, r), (cx, cy, _) in det.items())
        piece_sets.append(pieces)
        print(f"frame {rep:5d}: {len(pieces):2d} pieces  ({time.time()-t0:4.1f}s)")

    print(f"\n=== from->to (match_px={match_px:.0f}) ===")
    for a in range(len(reps) - 1):
        A = piece_sets[a]; B = piece_sets[a + 1]
        usedB = set()
        matchedA = set()
        for ia, (fa, ra, cxa, cya) in enumerate(A):
            best = -1; bd = 1e9
            for ib, (fb, rb, cxb, cyb) in enumerate(B):
                if ib in usedB:
                    continue
                d = float(np.hypot(cxa - cxb, cya - cyb))
                if d < bd:
                    bd = d; best = ib
            if best >= 0 and bd < match_px:
                matchedA.add(ia)
                usedB.add(best)
        frm = [(fa, ra) for ia, (fa, ra, _, _) in enumerate(A) if ia not in matchedA]
        to = [(fb, rb) for ib, (fb, rb, _, _) in enumerate(B) if ib not in usedB]
        tag = ""
        if len(frm) > 1 or len(to) > 1 or abs(len(A) - len(B)) > 1:
            tag = "  [dirty]"
        print(f"seg{a}->seg{a+1} (frame {reps[a]}->{reps[a+1]}): "
              f"from={frm} to={to}  (nA={len(A)} nB={len(B)}){tag}")


if __name__ == "__main__":
    main()
