"""棋子锚定的标准开局单应拟合（对应无关 RANSAC）。

动机：暗桌面场景下 SAM 棋盘掩码溢出、refine_corners 锁错线组，几何链路全挂；
但 SAM 逐子检测的质心与节点分配无关、始终可靠。标准开局 32 子位置是已知先验，
直接把「32 个 canonical 位置」RANSAC 拟合到「检测质心」，同时解出：
  ① 单应 H（canonical 网格 → 图像）；② 摆位是否标准开局（内点数 ≈ 子数）。
180° 翻转歧义（开局占位旋转对称）用棋子红黑外观色判别；镜像因开局左右对称、
每子落在同身份棋子上，对标注无害。

用法：python fit_std_homography.py [frame_idx]
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
from sam_detect_pieces import detect_pieces
from m2_opening import STD_GRID

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_02.mp4")

AREA_MIN, AREA_MAX = 0.10, 0.60
PUTATIVE_CELL = 2.0     # 粗四角误差容限（格）：质心 ↔ 标准位置的候选配对半径
INLIER_CELL = 0.35      # 内点判据（格）：重投影误差阈值
RANSAC_ITERS = 4000
RED_DG, RED_S = 25.0, 50.0


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


def rough_nodes(pred, img):
    """SAM 粗四角 → 粗略节点 + 格尺寸（允许偏 1~2 格，只要探针还能打到棋子）。"""
    H, W = img.shape[:2]
    masks, _, _ = pred.predict(
        point_coords=np.array([[W / 2, H / 2]]), point_labels=np.array([1]),
        multimask_output=True)
    best = None
    for k in range(masks.shape[0]):
        frac = masks[k].mean()
        if not (AREA_MIN <= frac <= AREA_MAX):
            continue
        q = quad_from_mask(masks[k])
        if q is None or (q[:, 0].min() < 0 or q[:, 0].max() > W or
                         q[:, 1].min() < 0 or q[:, 1].max() > H):
            continue
        a = cv2.contourArea(q)
        if best is None or a > best[0]:
            best = (a, q)
    if best is None:
        return None
    quad = best[1]
    M = cv2.getPerspectiveTransform(quad, PHYSICAL_CORNERS)
    nodes = project(canonical_nodes(), np.linalg.inv(M))
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
    return nodes, cell_w, cell_h, cell_areas


def piece_color(img, cx, cy, cell):
    """棋子 patch 外观色 → 'r'/'b'/'?'。"""
    x0 = max(0, int(cx - cell * 0.4)); x1 = min(img.shape[1], int(cx + cell * 0.4))
    y0 = max(0, int(cy - cell * 0.4)); y1 = min(img.shape[0], int(cy + cell * 0.4))
    if x1 <= x0 or y1 <= y0:
        return "?"
    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    dg = float(patch[:, :, 2].mean() - patch[:, :, 1].mean())
    sat = float(hsv[:, :, 1].mean())
    if dg > RED_DG and sat > RED_S:
        return "r"
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if float(g.min()) < 50:
        return "b"
    return "?"


def redness(img, cx, cy, cell):
    """棋子墨色红度：patch 内最暗 25% 像素（墨字笔画）的平均 R-G。

    红墨 R-G 显著 > 黑墨（≈0），相对判别、不依赖绝对阈值（这块棋盘红字偏暗，
    绝对阈值 R-G>25&S>50 一个像素都不过）。
    """
    x0 = max(0, int(cx - cell * 0.45)); x1 = min(img.shape[1], int(cx + cell * 0.45))
    y0 = max(0, int(cy - cell * 0.45)); y1 = min(img.shape[0], int(cy + cell * 0.45))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = img[y0:y1, x0:x1].astype(np.float32)
    gray = patch.mean(axis=2)
    thr = np.quantile(gray, 0.25)
    ink = gray <= thr
    if ink.sum() < 10:
        return 0.0
    return float((patch[:, :, 2] - patch[:, :, 1])[ink].mean())


def std_points():
    """标准开局 32 子 (canonical 坐标, color, (f,r))。"""
    nodes = canonical_nodes()
    pts = []
    for r in range(10):
        for f in range(9):
            ident = STD_GRID[r][f]
            if ident is None:
                continue
            color, _ = ident
            pts.append((nodes[r * 9 + f].copy(), color, (f, r)))
    return pts


def valid_h(Htry, cell):
    """H 有效性：有限、非退化（四角投影围成面积 ≥ 16 格²、宽高各 ≥ 4 格）。"""
    if Htry is None or not np.all(np.isfinite(Htry)):
        return False
    c = project(PHYSICAL_CORNERS.astype(np.float32), Htry)
    if not np.all(np.isfinite(c)):
        return False
    w = float(c[:, 0].max() - c[:, 0].min())
    h = float(c[:, 1].max() - c[:, 1].min())
    if w < 4 * cell or h < 4 * cell:
        return False
    return cv2.contourArea(c.astype(np.float32)) > (4 * cell) ** 2


def fit_h(std_xy, det_xy, cell, putative, identity, iters=RANSAC_ITERS, seed=0):
    """标准位置 → 检测质心的 RANSAC。

    putative: [(std_idx, det_idx)] ±1 格邻域全候选；identity: 恒等对子集
    （std (f,r) ↔ 同 key 检测，粗误差 <0.5 格时多数正确）。采样偏置：70% 从
    identity 抽（正确率 ~p⁴），30% 从全候选抽（容忍系统性偏移）。
    退化线解被「唯一性计分」与 [valid_h] 双闸拒绝。

    返回 (H, 内点掩码)；H 将 canonical 坐标映射到图像。
    """
    rng = np.random.default_rng(seed)
    S = np.array(std_xy, np.float32)
    D = np.array(det_xy, np.float32)
    thr = INLIER_CELL * cell

    def evaluate(Htry):
        proj = project(S, Htry)
        d = np.linalg.norm(proj[:, None, :] - D[None, :, :], axis=2)
        j = d.argmin(axis=1)
        inl = d.min(axis=1) < thr
        uniq = len(set(int(x) for x in j[inl]))
        return uniq, inl, j

    best_H, best_score, best_inl, best_j = None, -1, None, None
    pools = [(identity, 0.7), (putative, 0.3)] if identity else [(putative, 1.0)]
    for _ in range(iters):
        pool = pools[0] if (len(pools) < 2 or rng.random() < pools[0][1]) else pools[1]
        if len(pool[0]) < 4:
            continue
        sel = rng.choice(len(pool[0]), 4, replace=False)
        idx = [pool[0][k] for k in sel]
        if len({i for i, _ in idx}) < 4 or len({j for _, j in idx}) < 4:
            continue
        H = cv2.getPerspectiveTransform(S[[i for i, _ in idx]], D[[j for _, j in idx]])
        if not valid_h(H, cell):
            continue
        score, inl, j = evaluate(H)
        if score > best_score:
            best_H, best_score, best_inl, best_j = H, score, inl, j

    if best_H is None:
        return None, None

    # 精化：内点一对一重估，只在变好时接受
    for _ in range(3):
        if best_inl.sum() < 4:
            break
        # 一对一：每个被认领的检测只留一个标准位
        claim = {}
        for k in np.where(best_inl)[0]:
            dj = int(best_j[k])
            if dj not in claim:
                claim[dj] = k
        H2, _ = cv2.findHomography(S[list(claim.values())], D[list(claim.keys())])
        if H2 is None or not valid_h(H2, cell):
            break
        score2, inl2, j2 = evaluate(H2)
        if score2 <= best_score:
            break
        best_H, best_score, best_inl, best_j = H2, score2, inl2, j2

    return best_H, best_inl


def putative_from_detected(stds, detected_keys, detected_list):
    """粗糙节点分配 → (全候选对 ±1 格邻域, 恒等对子集)。

    stds: [(canonical_xy, color, (f,r))]；detected_keys: (f,r)→index 映射。
    返回 (putative, identity)，均为 [(std_idx, det_idx)]。
    """
    pairs, identity = [], []
    for k, p in enumerate(stds):
        f0, r0 = p[2]
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                key = (f0 + df, r0 + dr)
                if key in detected_keys:
                    pairs.append((k, detected_keys[key]))
                    if df == 0 and dr == 0:
                        identity.append((k, detected_keys[key]))
    return pairs, identity


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1797
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    assert ok

    t0 = time.time()
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    pred = SamPredictor(sam)
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(f"SAM 就绪 {time.time()-t0:.1f}s", flush=True)

    rn = rough_nodes(pred, img)
    assert rn, "无合法掩码"
    nodes, cw, ch, ca = rn
    detected = detect_pieces(pred, nodes, cw, ch, ca)
    print(f"detect_pieces → {len(detected)} 子  {time.time()-t0:.1f}s")
    cell = float(np.mean(np.sqrt(ca)))

    det_items = list(detected.items())            # [((f,r), {...}), ...]
    det_xy = np.array([v["centroid"] for _, v in det_items], np.float32)
    det_keys = {k: i for i, (k, _) in enumerate(det_items)}
    stds = std_points()
    std_xy = [p[0] for p in stds]

    t1 = time.time()
    putative, identity = putative_from_detected(stds, det_keys, det_xy)
    H, inl = fit_h(std_xy, det_xy, cell, putative, identity)
    print(f"RANSAC {time.time()-t1:.1f}s  内点 {int(inl.sum()) if inl is not None else 0}/{len(std_xy)}"
          f"（唯一性计分）")

    if H is None or inl is None or inl.sum() < 20:
        print("[fail] 拟合失败或内点过少（摆位非标准开局？）")
        return

    # 180° 翻转歧义：聚合红度差。正解 → 红方标准位的棋子红度显著高于黑方位；
    # 翻转解 → 差值为负。比逐子硬投票（大量 '?'）稳健。
    flip = np.array([[-1, 0, BOARD_W], [0, -1, BOARD_H], [0, 0, 1]], np.float32)  # canonical 180°
    Hf = H @ flip

    def red_gap(Htry):
        proj = project(np.array(std_xy, np.float32), Htry)
        reds, blacks = [], []
        for k, p in enumerate(stds):
            d = np.linalg.norm(proj[k] - det_xy, axis=1)
            j = int(d.argmin())
            if d[j] > INLIER_CELL * cell:
                continue
            v = redness(img, det_xy[j][0], det_xy[j][1], cell)
            (reds if p[1] == "r" else blacks).append(v)
        r = float(np.mean(reds)) if reds else 0.0
        b = float(np.mean(blacks)) if blacks else 0.0
        return r - b, r, b

    gap1, r1, b1 = red_gap(H)
    gap2, r2, b2 = red_gap(Hf)
    print(f"红度差 正解={gap1:+.3f} (红{r1:.3f}/黑{b1:.3f})  翻转解={gap2:+.3f} (红{r2:.3f}/黑{b2:.3f})")
    if gap2 > gap1:
        H = Hf
        proj = project(np.array(std_xy, np.float32), H)
        d = np.linalg.norm(proj[:, None, :] - det_xy[None, :, :], axis=2)
        inl = d.min(axis=1) < INLIER_CELL * cell
        print("采用 180° 翻转解")
    else:
        print("保持正解")

    print(f"\n[结果] 内点 {int(inl.sum())}/32  → 摆位{'是' if inl.sum() >= 28 else '存疑'}标准开局")
    missing = [p[2] for k, p in enumerate(stds) if not inl[k]]
    if missing:
        print(f"未匹配（SAM 漏检/偏移）：{missing}")
    # 可视化：H warp
    warp = cv2.warpPerspective(img, np.linalg.inv(H), (BOARD_W, BOARD_H))
    cv2.imwrite(os.path.join(DBG, "fit_std_warp.png"), warp)
    nodes_can = canonical_nodes()
    nodes_img = project(nodes_can, H)
    for p in std_xy:
        pass
    print(f"[vis] {DBG}/fit_std_warp.png")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
