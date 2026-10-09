"""飞轮真机帧标注（标准开局先验版，P1-a）：设备域 480×640 帧 → 棋子锚定拟合 → std 身份标签。

与 video_label_std.py 的区别：
- 输入 = 真机 pull 的 flywheel/（frame_*.jpg + metadata.jsonl），非视频抽帧；
- 粗糙节点来自 metadata 的设备四角（SegmentationBoardDetector 物理四角，TL,TR,BR,BL），
  无需离线再跑 SAM 棋盘掩码；
- 空交点**不写框**——这正是压假阳的负样本信号（v3 的固定框遗留 + v4 的 file4 假阳）。

用法：python flywheel_label_std.py [in_dir] [out_dir] [--max 80]
"""
import os
import sys
import time
import json

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_piece_spike import canonical_nodes, project, PHYSICAL_CORNERS
from sam_detect_pieces import detect_pieces
from m2_opening import STD_GRID
from flywheel_label import write_frame, class_id, mask_bbox
from fit_std_homography import (
    fit_h, putative_from_detected, redness, std_points, INLIER_CELL, CHECKPOINT,
)

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
GAP_WEAK = 0.05
CELL_BOX = 0.85


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def label_frame(pred, img, corners_meta, prev_gap_sign):
    """单帧：metadata 四角 → 粗糙节点 → detect → 拟合 → std 标签。"""
    corners = np.array(corners_meta, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(corners, PHYSICAL_CORNERS)
    nodes0 = project(canonical_nodes(), np.linalg.inv(M))
    cw, ch, ca = [], [], []
    for r in range(10):
        for f in range(9):
            i = r * 9 + f
            p = nodes0[i]
            fr = f + 1 if f < 8 else f - 1
            rd = r + 1 if r < 9 else r - 1
            cw.append(float(np.linalg.norm(p - nodes0[r * 9 + fr])))
            ch.append(float(np.linalg.norm(p - nodes0[rd * 9 + f])))
            ca.append(cw[-1] * ch[-1])

    detected = detect_pieces(pred, nodes0, cw, ch, ca)
    if len(detected) < 24:
        return None, f"检出过少 {len(detected)}"
    det_items = list(detected.items())
    det = [v for _, v in det_items]
    det_xy = np.array([d["centroid"] for d in det], np.float32)
    det_keys = {k: i for i, (k, _) in enumerate(det_items)}
    cell = float(np.mean(np.sqrt(ca)))

    stds = std_points()
    std_xy = np.array([p[0] for p in stds], np.float32)
    putative, identity = putative_from_detected(stds, det_keys, det_xy)
    H, inl = fit_h(std_xy, det_xy, cell, putative, identity)
    if H is None or inl is None or inl.sum() < int(opt("--min-inliers", "28")):
        return None, f"内点不足 {0 if inl is None else int(inl.sum())}"

    def match(Htry):
        proj = project(std_xy, Htry)
        d = np.linalg.norm(proj[:, None, :] - det_xy[None, :, :], axis=2)
        j = d.argmin(axis=1)
        ok = d.min(axis=1) < INLIER_CELL * cell
        return j, ok

    flip = np.array([[-1, 0, 450.0], [0, -1, 500.0], [0, 0, 1]], np.float32)

    def gap_of(Htry):
        j, ok = match(Htry)
        reds, blacks = [], []
        for k, p in enumerate(stds):
            if not ok[k]:
                continue
            v = redness(img, det_xy[j[k]][0], det_xy[j[k]][1], cell)
            (reds if p[1] == "r" else blacks).append(v)
        if not reds or not blacks:
            return 0.0
        return float(np.mean(reds) - np.mean(blacks))

    g1 = gap_of(H)
    g2 = gap_of(H @ flip)
    if abs(g1) < GAP_WEAK and abs(g2) < GAP_WEAK:
        if prev_gap_sign is None:
            return None, "红度弱且无已验证方向"
        sign = prev_gap_sign
    else:
        sign = 1 if g1 >= g2 else -1
    if sign < 0:
        H = H @ flip
    j, ok = match(H)

    nodes_img = project(canonical_nodes(), H)
    cell_w, cell_h = [], []
    for r in range(10):
        for f in range(9):
            i = r * 9 + f
            p = nodes_img[i]
            fr = f + 1 if f < 8 else f - 1
            rd = r + 1 if r < 9 else r - 1
            cell_w.append(float(np.linalg.norm(p - nodes_img[r * 9 + fr])))
            cell_h.append(float(np.linalg.norm(p - nodes_img[rd * 9 + f])))

    boxes = []
    for k, p in enumerate(stds):
        _, color, (f, r) = p
        i = r * 9 + f
        t = STD_GRID[r][f][1]
        cls = class_id(color, t)
        b = mask_bbox(det[j[k]]["mask"]) if ok[k] else None
        if b is None:
            # SAM 漏检：交点心合成框（位置由拟合验证过，占位可信）
            px, py = nodes_img[i]
            bw, bh = CELL_BOX * cell_w[i], CELL_BOX * cell_h[i]
            b = (px - bw / 2.0, py - bh / 2.0, bw, bh)
        boxes.append((cls,) + tuple(b))

    info = {"inliers": int(ok.sum()), "gap": (g1 if sign > 0 else g2),
            "corners": project(PHYSICAL_CORNERS.astype(np.float32), H)}
    return boxes, info


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else os.path.join(DBG, "flywheel_std")
    out_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
        else os.path.join(DBG, "flywheel_std_yolo")
    max_frames = int(opt("--max", "80"))

    metas = []
    with open(os.path.join(in_dir, "metadata.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metas.append(json.loads(line))

    # 去重：FEN 相同且四角相近（<20px）的连续帧只留 1 张（2s 节流下静态视角高度重复）。
    picked = []
    for m in metas:
        c = np.array(m["corners"], dtype=np.float32)
        dup = False
        for p in picked[-8:]:
            if p["fen"] == m["fen"] and np.abs(
                    np.array(p["corners"], np.float32) - c).max() < 20:
                dup = True
                break
        if not dup:
            picked.append(m)
    print(f"元数据 {len(metas)} 帧 → 去重后 {len(picked)} 帧，采样 {min(max_frames, len(picked))}", flush=True)
    step = max(1, len(picked) // max_frames)
    picked = picked[::step][:max_frames]

    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    t0 = time.time()
    print("加载 SAM ViT-B ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)

    written, skipped = [], 0
    prev_sign = None
    for m in picked:
        name = m["file"]
        jpeg = os.path.join(in_dir, name + ".jpg")
        img = cv2.imread(jpeg)
        if img is None:
            skipped += 1
            continue
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        boxes, info = label_frame(pred, img, m["corners"], prev_sign)
        if boxes is None:
            print(f"  [skip] {name}: {info}  {time.time()-t1:.0f}s", flush=True)
            skipped += 1
            continue
        prev_sign = 1 if info["gap"] >= 0 else -1
        out_name = "fw_" + name
        write_frame(img, info["corners"], boxes, img_dir, lbl_dir, out_name)
        written.append(out_name)
        print(f"  {out_name}: {info['inliers']}/32 内点 gap={info['gap']:+.3f}  {time.time()-t1:.0f}s", flush=True)
    print(f"\n标注 {len(written)} 跳过 {skipped}  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
