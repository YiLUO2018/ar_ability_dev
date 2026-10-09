"""数据生成侧：SAM 逐子鲁棒检测（M5 的 bbox 标注组件）。

M0 的「交点单点提示」对偏离交点 ~0.5 格的棋子会漏检（提示点落在空棋盘 → SAM 返回整块棋盘）。
数据生成侧必须把每颗棋子都标出 bbox（否则训练数据缺漏、YOLO 会继承缺口）。

干净解法（不引入全局质心聚类的桥接/合并坑）：
  1. 基线 = M0 中心点提示：90 交点各提示一次，面积比 + 圆度 + 质心距三重判棋子（可靠 35/38、0 误报）；
  2. 补漏 = 对「判空」的交点，向 8 邻域（0.5 格）探针，收集偏离交点的棋子候选；
  3. 防重 = 邻域探针找到的 mask 若与已检出棋子 mask 高 IoU（同一颗），丢弃；
  4. 分配 = 每个候选按「离它最近的仍空交点」一对一分配（而非第一个扫到的空交点）。

实现细节：SAM 单点提示在相邻棋子密集处（rank8/rank9 底线）会返回「桥接」掩码（把两颗子连成
一大块），分类只认最大连通域（正确），但 bbox 必须用 `largest_component_mask` 抠出单颗，否则
bbox 会变成 100×70 的大块。

注：单张静态图上，偏离 ~0.5 格的棋子，其 (file, rank) 归属在几何上只能按「最近交点」给个
近似，可能偏到相邻交点；这类歧义由方法 D 的视频规则传播（M3/M4，运动功能监督）消解，不
阻塞 M5 的 bbox 标注。

输出：逐子 (file, rank, bbox) 的 14 类标注原料 + 可视化。bbox 用原图像素坐标。
"""
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\tools\label_real")
from sam_piece_spike import (
    predict_and_classify, DIST_CELL,
    CELL, MARGIN, BOARD_W, BOARD_H, PHYSICAL_CORNERS, BOARD_CORNERS,
    canonical_nodes, project,
)

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_sam_detect.png"
OUT_CSV = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\m5_bboxes.csv"

# 邻域探针（单位：格）。8 方向半径 0.5 格，能抓到偏移 ≤~0.6 格的棋子。
PROBES = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
          (0.35, 0.35), (0.35, -0.35), (-0.35, 0.35), (-0.35, -0.35)]
FILL_DIST = 0.75   # 邻域探针质心到空交点的最大距离（格）
IOU_DUP = 0.3      # 与已检出棋子 mask 的 IoU 超过此值视为同一颗 → 丢弃


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def largest_component_mask(mask):
    """返回 mask 最大连通域的二值掩码。

    SAM 单点提示在相邻棋子密集处（如 rank8/rank9 底线）会返回「桥接」掩码（把两颗子连成
    一大块），`predict_and_classify` 的分类只认最大连通域（正确），但返回的仍是全掩码。
    这里把最大连通域单独抠出来，保证后续 bbox / IoU 只作用在单颗子上。
    """
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(mask.shape, dtype=bool)
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros(mask.shape, dtype=bool)
    cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out


def detect_pieces(predictor, nodes_img, cell_w, cell_h, cell_areas):
    """SAM 逐子鲁棒检测（可复用）：中心点提示 + 8 邻域探针 + 全局最近空交点分配。

    ``nodes_img`` 为 90 交点原图坐标（rank-major，i = r*9+f）；``cell_w/cell_h/cell_areas``
    与之对齐。返回 ``{(f, r): {"mask": bool_mask, "centroid": (cx, cy)}}``，
    ``mask`` 为最大连通域二值掩码（bbox 可据此 ``np.nonzero`` 得到）。
    """
    detected = {}   # (f, r) -> {"mask", "centroid"}
    claimed = []    # 已认领的 mask（防重）

    # 1) 基线：中心点提示（与 M0 一致）
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]; ca = cell_areas[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, ca)
        if is_piece and centroid is not None:
            dist = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if dist < DIST_CELL * np.sqrt(ca):
                clean = largest_component_mask(mask)
                detected[(f, r)] = {"mask": clean, "centroid": centroid}
                claimed.append(clean)

    # 2) 补漏：对空交点做 8 邻域探针，收集偏移子候选。
    # 注意：找到候选后立即加入 claimed，避免同一颗偏移子被相邻多个空交点重复发现。
    empty_ids = [i for i in range(90) if (i % 9, i // 9) not in detected]
    candidates = []  # (centroid, mask)
    for i in empty_ids:
        f, r = i % 9, i // 9
        p = nodes_img[i]; ca = cell_areas[i]
        for ox, oy in PROBES:
            pt = (p[0] + ox * cell_w[i], p[1] + oy * cell_h[i])
            mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, pt, ca)
            if not (is_piece and centroid is not None):
                continue
            d = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if d > FILL_DIST * np.sqrt(ca):
                continue
            clean = largest_component_mask(mask)
            if any(mask_iou(clean, m) > IOU_DUP for m in claimed):
                continue
            candidates.append((centroid, clean))
            claimed.append(clean)  # 防重：后续探针不再重复发现同一颗
            break

    # 3) 全局分配：每个候选 → 离它最近的仍空交点（一对一）
    for centroid, mask in candidates:
        best_i = min(empty_ids, key=lambda j: float(np.hypot(
            centroid[0] - nodes_img[j][0], centroid[1] - nodes_img[j][1])))
        d_best = float(np.hypot(centroid[0] - nodes_img[best_i][0],
                                centroid[1] - nodes_img[best_i][1]))
        if d_best > FILL_DIST * np.sqrt(cell_areas[best_i]):
            continue
        f, r = best_i % 9, best_i // 9
        if (f, r) in detected:
            continue  # 已被别的候选占用（应罕见）
        detected[(f, r)] = {"mask": mask, "centroid": centroid}
        empty_ids.remove(best_i)
    return detected


def main():
    t0 = time.time()
    img = cv2.imread(IMG)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("加载 SAM ViT-B ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    print(f"模型就绪 {time.time()-t0:.1f}s")

    M = cv2.getPerspectiveTransform(BOARD_CORNERS.astype(np.float32), PHYSICAL_CORNERS)
    nodes_img = project(canonical_nodes(), np.linalg.inv(M))

    cell_w, cell_h, cell_areas = [], [], []
    for r in range(10):
        for f in range(9):
            p = nodes_img[r * 9 + f]
            fr = f + 1 if f < 8 else f - 1
            rd = r + 1 if r < 9 else r - 1
            cw = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
            ch = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
            cell_w.append(cw); cell_h.append(ch); cell_areas.append(cw * ch)

    # SAM 逐子鲁棒检测：中心点提示 + 8 邻域探针 + 全局最近空交点分配。
    detected = detect_pieces(predictor, nodes_img, cell_w, cell_h, cell_areas)

    n = len(detected)
    print(f"\n检出唯一棋子 {n} / 38")

    # 4) 逐 rank 分布
    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in detected:
        occ[r][f] = 1
    print("按 rank 分布:")
    for r in range(10):
        print(f"  rank{r}: " + "".join("X" if occ[r][f] else "." for f in range(9)))

    # 5) 可视化 + bbox（image 坐标）
    boxes = {}
    for (f, r), v in detected.items():
        ys, xs = np.nonzero(v["mask"])
        boxes[(f, r)] = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    vis = img.copy()
    overlay = vis.copy()
    for (f, r), v in detected.items():
        overlay[v["mask"]] = (0, 220, 0)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    for (f, r), (x, y, w, h) in boxes.items():
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"{f},{r}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(OUT, vis)
    print(f"\n[vis] 已保存 {OUT}")

    with open(OUT_CSV, "w", encoding="utf-8") as fh:
        fh.write("file,rank,x,y,w,h\n")
        for (f, r), (x, y, w, h) in sorted(boxes.items()):
            fh.write(f"{f},{r},{x},{y},{w},{h}\n")
    print(f"[csv] 已保存 {OUT_CSV}")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
