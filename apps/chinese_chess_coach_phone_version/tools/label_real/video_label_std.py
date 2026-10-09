"""视频标注管线（标准开局先验版，debug_video_02）：棋子锚定单应 + 32 子身份免费出标签。

链路（每关键帧）：
  SAM 粗四角（允许溢出/偏 1 格）→ detect_pieces 质心（与节点分配无关、可靠）
  → 对应无关 RANSAC 把 32 个标准开局位置拟合到质心（内点数即摆位验证）
  → 180° 翻转歧义用墨色红度差判别（弱信号则继承上一帧）
  → 身份 = 标准开局网格；bbox = SAM mask（漏检子用 0.85 格固定框补）
  → 裁剪棋盘区（H 投影物理四角外扩 PAD）写 YOLO images/labels

按采样顺序（=时间序）切 train/val（80/20），写 train.txt/val.txt 防相邻帧泄漏
（train_yolo.py 检测到这两个文件就不随机切）。

用法：python video_label_std.py [video] [out_dir] [--n 80] [--min-inliers 28]
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

from sam_piece_spike import canonical_nodes, project, PHYSICAL_CORNERS
from sam_detect_pieces import detect_pieces
from m2_opening import STD_GRID
from flywheel_label import write_frame, class_id, mask_bbox
from fit_std_homography import (
    rough_nodes, fit_h, putative_from_detected, redness, std_points,
    INLIER_CELL, CHECKPOINT,
)

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
GAP_WEAK = 0.05    # 红度差低于此值视为弱信号 → 继承上一帧方向
CELL_BOX = 0.85    # SAM 漏检子的合成 bbox 边长（格）


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def label_frame(pred, img, prev_gap_sign):
    """单帧全链路。返回 (boxes, info dict) 或 (None, 原因)。"""
    rn = rough_nodes(pred, img)
    if rn is None:
        return None, "无合法掩码"
    nodes0, cw, ch, ca = rn
    detected = detect_pieces(pred, nodes0, cw, ch, ca)
    if len(detected) < 24:
        return None, f"检出过少 {len(detected)}"
    det_items = list(detected.items())            # [((f,r), {...}), ...]
    det = [v for _, v in det_items]               # [{"mask", "centroid"}]
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

    # 180° 翻转歧义：墨色红度差
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
        sign = prev_gap_sign  # 弱信号继承上一帧
    else:
        sign = 1 if g1 >= g2 else -1
    if sign < 0:
        H = H @ flip
    j, ok = match(H)

    # 重复检出口径：n_det - 被标准位认领的唯一质心数（粗节点偏移时同一棋子会被多个探针重复检出）
    extras = len(det_xy) - len(set(int(x) for x in j[ok])) if ok.any() else len(det_xy)

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
            px, py = nodes_img[i]
            bw, bh = CELL_BOX * cell_w[i], CELL_BOX * cell_h[i]
            b = (px - bw / 2.0, py - bh / 2.0, bw, bh)
        boxes.append((cls,) + tuple(b))

    corners = project(PHYSICAL_CORNERS.astype(np.float32), H)
    info = {"inliers": int(ok.sum()), "extras": extras,
            "gap": (g1 if sign > 0 else g2), "n_det": len(det_xy),
            "corners": corners}
    return boxes, info


def main():
    video = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else os.path.join(DBG, "debug_video_02.mp4")
    out_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
        else os.path.join(DBG, "std02_yolo")
    n_samples = int(opt("--n", "80"))

    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    t0 = time.time()
    print(f"加载 SAM ViT-B ...（{n_samples} 采样帧）", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)

    written, skipped = [], 0
    prev_sign = None  # 首个红度 decisive 帧验证方向后才有值
    for s in range(n_samples):
        idx = int(n_frames * (s + 0.5) / n_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        okf, img = cap.read()
        if not okf:
            print(f"  [warn] 读帧失败 {idx}", flush=True)
            skipped += 1
            continue
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        boxes, info = label_frame(pred, img, prev_sign)
        if boxes is None:
            print(f"  [skip] 帧{idx}: {info}  {time.time()-t1:.0f}s", flush=True)
            skipped += 1
            continue
        prev_sign = 1 if info["gap"] >= 0 else -1
        name = f"std02_f{idx:05d}"
        write_frame(img, info["corners"], boxes, img_dir, lbl_dir, name)
        written.append(name)
        print(f"  {name}: {info['inliers']}/32 内点 {len(boxes)} 框 "
              f"extras={info['extras']} gap={info['gap']:+.3f}  {time.time()-t1:.0f}s", flush=True)
    cap.release()

    # 按时间序（采样序）切 train/val，防相邻帧泄漏
    n_val = max(1, len(written) // 5)
    train, val = written[:-n_val], written[-n_val:]
    with open(os.path.join(out_dir, "train.txt"), "w") as fh:
        for nm in train:
            fh.write(os.path.join(img_dir, nm + ".jpg") + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as fh:
        for nm in val:
            fh.write(os.path.join(img_dir, nm + ".jpg") + "\n")
    print(f"\ntrain={len(train)} val={len(val)}（时间序切分）skip={skipped}")
    print(f"总耗时 {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
