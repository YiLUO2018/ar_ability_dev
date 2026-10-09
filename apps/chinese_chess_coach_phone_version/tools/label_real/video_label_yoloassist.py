"""YOLO 助力标注（飞轮模式离线验证）：用 v5 检出质心补 SAM 检出的不足，标注此前跳过的难帧。

之前 debug_video_02 有 50/80 帧因「SAM 检出过少 / 拟合内点不足」跳过（斜视角遮挡段）。
本脚本验证飞轮核心环节：YOLO 提议几何锚点（框中心）+ 标准开局先验定身份 + RANSAC
唯一性内点闸裁决——身份不来自 YOLO（无自证循环），YOLO 只贡献几何。

每帧流程：
  SAM 棋盘粗四角 → 粗糙节点 → SAM detect_pieces（质心+mask）
  + v5 ONNX 推理（棋盘裁剪区，手写解码，conf≥0.30）→ YOLO 质心
  → 合并质心 RANSAC 拟合（唯一性计分）→ ≥28 内点 → 180° 墨色判别 → std 身份标签
  → 同时输出评估：YOLO 预测类别 vs std 身份的一致率（v5 在难帧上的真实识别率）

用法：python video_label_yoloassist.py
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
    rough_nodes, fit_h, redness, std_points, INLIER_CELL, CHECKPOINT,
)

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ONNX = os.path.join(ROOT, "app", "src", "main", "assets", "models", "yolov8n_chess.onnx")
VIDEO = os.path.join(DBG, "debug_video_02.mp4")
OUT_DIR = os.path.join(DBG, "std02_yolo_assist")

CONF = 0.30
IOU = 0.45
GAP_WEAK = 0.05
CELL_BOX = 0.85
YOLO_PAD_RATIO = 0.06   # 裁剪区四角外扩比例（与端上 YoloPieceDetector 一致）

# 2026-10-07 video_label_std 跳过的 50 帧（来自运行日志）
SKIPPED = [890, 968, 1045, 1123, 1200, 1278, 1355, 1433, 1510, 1587,
           1665, 1742, 1820, 1897, 1975, 2052, 2130, 2207, 2285, 2362,
           2440, 2517, 2594, 2672, 2749, 2827, 2904, 2982, 3059, 3292,
           3524, 3834, 3911, 3989, 4066, 4144, 4221, 4299, 4376, 4454,
           4531, 4609, 4686, 4918, 4996, 5073, 5151, 5228, 5306, 5693]


# ------------------------------------------------------------- YOLO 推理（复用 verify_onnx_decode 的解码）

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r, (nh, nw)


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area + (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) - inter)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloOnnx:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def detect(self, img_bgr):
        """整图推理 → [(cx, cy, cls, conf)]（原图坐标）。"""
        H, W = img_bgr.shape[:2]
        canvas, r, _ = letterbox(img_bgr, 640)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None]
        pred = self.sess.run(None, {"images": x})[0][0]   # [18,8400]
        boxes, scores = pred[:4].T, pred[4:].T
        cls_ids, cls_scores = scores.argmax(1), scores.max(1)
        m = cls_scores > CONF
        if not m.any():
            return []
        kb, kc, ks = boxes[m], cls_ids[m], cls_scores[m]
        xyxy = np.zeros_like(kb)
        xyxy[:, 0] = kb[:, 0] - kb[:, 2] / 2
        xyxy[:, 1] = kb[:, 1] - kb[:, 3] / 2
        xyxy[:, 2] = kb[:, 0] + kb[:, 2] / 2
        xyxy[:, 3] = kb[:, 1] + kb[:, 3] / 2
        xyxy /= r
        out = []
        for c in np.unique(kc):
            idx = np.where(kc == c)[0]
            for i in nms(xyxy[idx], ks[idx], IOU):
                b = xyxy[idx][i]
                out.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, int(kc[idx][i]), float(ks[idx][i]), b))
        return out


def pca_putative(std_xy, det_xy, cell):
    """PCA 粗对齐构造候选对：对旋转/平移/尺度鲁棒，不依赖粗糙四角（斜拍/旋转帧的
    粗糙节点分配会错乱 ±2~3 格，±1 邻域候选对救不回来）。

    用两团点的主轴（特征向量）估计旋转（8 种符号/换轴组合全试，自动覆盖 180° 翻转
    视角），尺度用特征值比。对齐后按预测位置吸附质心 → 候选对。
    返回 (putative, identity)。
    """
    s = np.asarray(std_xy, np.float64)
    d = np.asarray(det_xy, np.float64)

    def pca(x):
        m = x.mean(0)
        c = (x - m).T @ (x - m) / len(x)
        w, v = np.linalg.eigh(c)
        o = np.argsort(w)[::-1]
        return m, v[:, o], w[o]

    ms, vs, ws = pca(s)
    md, vd, wd = pca(d)
    best = (0, [], [])
    for swap in (False, True):
        vs2 = vs[:, ::-1] if swap else vs
        ws2 = ws[::-1] if swap else ws
        for f0 in (1.0, -1.0):
            for f1 in (1.0, -1.0):
                R = vd @ np.diag([f0, f1]) @ vs2.T
                scale = np.sqrt(np.clip(wd, 1e-6, None) / np.clip(ws2, 1e-6, None))
                proj = md + (s - ms) @ R.T * scale
                pairs, ident = [], []
                for k in range(len(s)):
                    dist = np.linalg.norm(d - proj[k], axis=1)
                    j = int(dist.argmin())
                    if dist[j] < 1.2 * cell:
                        pairs.append((k, j))
                        if dist[j] < 0.5 * cell:
                            ident.append((k, j))
                if len(pairs) > best[0]:
                    best = (len(pairs), pairs, ident)
    return best[1], best[2]


def main():
    yolo = YoloOnnx(ONNX)
    img_dir = os.path.join(OUT_DIR, "images")
    lbl_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    t0 = time.time()
    print(f"加载 SAM ViT-B ...（{len(SKIPPED)} 个此前跳过的帧）", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)

    cap = cv2.VideoCapture(VIDEO)
    written = 0
    prev_sign = None
    stats = []   # (idx, inliers, yolo_total, yolo_correct, yolo_matched)
    for idx in SKIPPED:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        t1 = time.time()
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        rn = rough_nodes(pred, img)
        if rn is None:
            print(f"  [skip] 帧{idx}: 无合法掩码", flush=True)
            continue
        nodes0, cw, ch, ca = rn
        detected = detect_pieces(pred, nodes0, cw, ch, ca)
        n_sam = len(detected)
        cell = float(np.mean(np.sqrt(ca)))

        # YOLO：棋盘裁剪区推理（与端上部署一致），质心换回全图坐标。
        # 粗糙节点 0/8/80/89 号即网格四角，外扩 pad 近似物理裁剪区。
        tl, tr = nodes0[0], nodes0[8]
        br, bl = nodes0[89], nodes0[80]
        pad = YOLO_PAD_RATIO
        x0 = max(0, int(min(tl[0], tr[0], br[0], bl[0]) * (1 - pad)))
        x1 = min(img.shape[1], int(max(tl[0], tr[0], br[0], bl[0]) * (1 + pad)))
        y0 = max(0, int(min(tl[1], tr[1], br[1], bl[1]) * (1 - pad)))
        y1 = min(img.shape[0], int(max(tl[1], tr[1], br[1], bl[1]) * (1 + pad)))
        crop = img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img
        yolo_dets = yolo.detect(crop)
        yolo_pts = [((cx + x0, cy + y0), cls, conf, box, box + 0)
                    for (cx, cy, cls, conf, box) in yolo_dets]

        # 合并质心：SAM 项（含 mask）在前，YOLO 项在后
        det_items = list(detected.items())
        det = [v for _, v in det_items] + [{"mask": None, "centroid": np.array([p[0][0], p[0][1]], np.float32)}
                                           for p in yolo_pts]
        det_xy = np.array([d["centroid"] for d in det], np.float32)
        if len(det_xy) < 16:
            print(f"  [skip] 帧{idx}: 合并质心过少 {len(det_xy)}"
                  f"（SAM={n_sam} YOLO={len(yolo_pts)}）", flush=True)
            continue

        stds = std_points()
        std_xy = np.array([p[0] for p in stds], np.float32)

        # PCA 粗对齐候选对（不依赖粗糙节点分配；斜拍/旋转/反向视角全覆盖）
        putative, identity = pca_putative(std_xy, det_xy, cell)

        H, inl = fit_h(std_xy, det_xy, cell, putative, identity)
        if H is None or inl is None or inl.sum() < 28:
            print(f"  [skip] 帧{idx}: 内点不足 {0 if inl is None else int(inl.sum())}"
                  f"（SAM={n_sam} YOLO={len(yolo_pts)} 候选对={len(putative)}）  {time.time()-t1:.0f}s", flush=True)
            continue
        is_flip_fit = False  # PCA 已在 8 种朝向组合里择优，无需单独翻转标记

        def match(Htry):
            proj = project(std_xy, Htry)
            d = np.linalg.norm(proj[:, None, :] - det_xy[None, :, :], axis=2)
            j = d.argmin(axis=1)
            okk = d.min(axis=1) < INLIER_CELL * cell
            return j, okk

        flip = np.array([[-1, 0, 450.0], [0, -1, 500.0], [0, 0, 1]], np.float32)

        def gap_of(Htry):
            j, okk = match(Htry)
            reds, blacks = [], []
            for k, p in enumerate(stds):
                if not okk[k]:
                    continue
                v = redness(img, det_xy[j[k]][0], det_xy[j[k]][1], cell)
                (reds if p[1] == "r" else blacks).append(v)
            if not reds or not blacks:
                return 0.0
            return float(np.mean(reds) - np.mean(blacks))

        g1, g2 = gap_of(H), gap_of(H @ flip)
        if abs(g1) < GAP_WEAK and abs(g2) < GAP_WEAK:
            if prev_sign is None:
                print(f"  [skip] 帧{idx}: 红度弱且无已验证方向（内点 {int(inl.sum())}）", flush=True)
                continue
            sign = prev_sign
        else:
            sign = 1 if g1 >= g2 else -1
        if sign < 0:
            H = H @ flip
        prev_sign = sign
        j, okk = match(H)

        # 评估：YOLO 框中心吸附拟合后节点，比对类别 vs std 身份
        yolo_total = yolo_correct = yolo_matched = 0
        nodes_img = project(canonical_nodes(), H)
        for (cxy, cls, conf, box, _) in yolo_pts:
            d = np.linalg.norm(nodes_img - np.array(cxy), axis=1)
            i = int(d.argmin())
            if d[i] > 0.5 * cell:
                continue
            f, r = i % 9, i // 9
            ident = STD_GRID[r][f]
            if ident is None:
                continue
            yolo_total += 1
            yolo_matched += 1
            if cls == class_id(*ident):
                yolo_correct += 1

        # 出框
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
            cls = class_id(color, STD_GRID[r][f][1])
            b = mask_bbox(det[j[k]]["mask"]) if (okk[k] and det[j[k]]["mask"] is not None) else None
            if b is None:
                px, py = nodes_img[i]
                bw, bh = CELL_BOX * cell_w[i], CELL_BOX * cell_h[i]
                b = (px - bw / 2.0, py - bh / 2.0, bw, bh)
            boxes.append((cls,) + tuple(b))

        corners = project(PHYSICAL_CORNERS.astype(np.float32), H)
        name = f"std02_f{idx:05d}"
        write_frame(img, corners, boxes, img_dir, lbl_dir, name)
        written += 1
        stats.append((idx, int(okk.sum()), yolo_total, yolo_correct, yolo_matched))
        acc = (yolo_correct / yolo_matched * 100) if yolo_matched else 0
        print(f"  {name}: 内点{int(okk.sum())}/32{'(反向视角)' if is_flip_fit else ''} "
              f"SAM={n_sam} YOLO={len(yolo_pts)} "
              f"识别率 {yolo_correct}/{yolo_matched}={acc:.0f}%  {time.time()-t1:.0f}s", flush=True)
    cap.release()

    tot_m = sum(s[3] for s in stats)
    tot_c = sum(s[2] for s in stats)
    print(f"\n[done] 新标注 {written}/{len(SKIPPED)} 帧")
    if tot_m:
        print(f"v5 在难帧上的身份识别率（吸附到 std 位后）：{tot_c}/{tot_m} = {tot_c/tot_m*100:.1f}%")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
