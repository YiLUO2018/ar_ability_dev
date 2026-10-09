"""M0 spike：验证 SAM 能在真实棋盘上把单个棋子干净分割出来（方法 D 的地基）。

在 board_21.png 上对 90 个网格交点逐一做点提示，判「该交点是否有棋子」，
可视化并输出检出统计。验收：≥28/32 棋子检出、空交点无误报。

自包含执行手册见 docs/项目全景梳理与实施方案.md §6.4。
"""
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_sam_pieces.png"
WARP_OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_warped_m0.png"
CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"

# canonical 物理棋盘（§3.3）：BOARD_W=450, BOARD_H=500, CELL=50, MARGIN=25
CELL = 50
MARGIN = 25
BOARD_W = 450
BOARD_H = 500
PHYSICAL_CORNERS = np.array(
    [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float32
)

# 棋盘物理四角（原图像素，顺序 TL,TR,BR,BL）——来自 SAM 分割拟合（见 test_sam.py 输出）。
# 注意：旧 GT 的 TL(166,77) 是错的（差 ~50px），SAM 拟合的 [[116,76],[350,76],[375,299],[102,306]] 才是对的。
# M0 spike 聚焦「逐子分割」，棋盘定位用已确认的 SAM 四角，不做自动拟合（自动定位留给端上/M1）。
BOARD_CORNERS = np.array([[116, 76], [350, 76], [375, 299], [102, 306]], dtype=np.float32)

# 棋子判定阈值（经验调优，见 M0 执行记录）
# 空交点 → SAM 分割整块棋盘（面积 ~60–160 倍格面积）；棋子 → ~0.5–1.3 倍格面积。
# 斜拍下远处棋子比格略大（格面积估计偏小），故上界放宽到 1.6。
AREA_LO = 0.25    # 格面积下界（拒绝 SAM 误分割的小碎块 ~0.2）
AREA_HI = 1.6     # 格面积上界（拒绝整棋盘 ~60+）
CIRC_MIN = 0.75   # 圆度（棋子 0.84–0.91，棋盘 0.38–0.70，碎块 0.5–0.66）
DIST_CELL = 1.0   # 质心到交点的最大距离（单位：格边长 sqrt(格面积)）


def canonical_nodes():
    """90 交点 (file, rank) 的 canonical 坐标，rank-major，shape (90, 2)。"""
    pts = [(MARGIN + f * CELL, MARGIN + r * CELL)
           for r in range(10) for f in range(9)]
    return np.array(pts, dtype=np.float32)


def project(pts, H):
    """用 H 变换点集 (N,2) → (N,2)。"""
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H)
    return out.reshape(-1, 2)


def largest_component(mask):
    """返回二值 mask 最大连通域的面积 / 周长 / 质心。无连通域时返回 None。"""
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 1:
        return None
    peri = cv2.arcLength(c, True)
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return area, peri, (cx, cy)


def predict_and_classify(predictor, point, cell_area):
    """在 point 处做单点提示，返回 (mask, is_piece, area, circ, centroid, score)。

    is_piece 只判面积比 + 圆度；质心距约束由调用方统一对「原交点」判定，
    这样邻域搜索的候选也能用同一质心距阈值。
    """
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[point[0], point[1]]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    mask = masks[0]
    comp = largest_component(mask)
    if comp is None:
        return mask, False, 0.0, 0.0, None, float(scores[0])
    area, peri, (cx, cy) = comp
    circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
    ok_area = AREA_LO * cell_area <= area <= AREA_HI * cell_area
    ok_circ = circ > CIRC_MIN
    return mask, (ok_area and ok_circ), area, circ, (cx, cy), float(scores[0])


def main():
    t0 = time.time()
    img = cv2.imread(IMG)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 1) 加载 SAM（image embedding 缓存后，棋盘分割 + 逐子分割共用）
    print("加载 SAM ViT-B ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu")
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    print(f"模型就绪 {time.time()-t0:.1f}s")

    # 2) 单应 M：image → canonical（物理棋盘角，用 SAM 已确认四角）
    M = cv2.getPerspectiveTransform(BOARD_CORNERS, PHYSICAL_CORNERS)

    # 4) 校验 warp：棋盘应正对填满、无黑边
    warped = cv2.warpPerspective(img, M, (BOARD_W, BOARD_H))
    cv2.imwrite(WARP_OUT, warped)
    print(f"[warp] 已保存 {WARP_OUT}（应为正对填满的 450×500 棋盘）")

    # 5) 90 交点反投影回原图坐标
    nodes_canon = canonical_nodes()
    nodes_img = project(nodes_canon, np.linalg.inv(M))

    # 每个交点的局部格面积（按原图像素）：右/下邻居距离之积
    cell_areas = []
    for i, (f, r) in enumerate([(f, r) for r in range(10) for f in range(9)]):
        p = nodes_img[i]
        f_r = f + 1 if f < 8 else f - 1
        r_d = r + 1 if r < 9 else r - 1
        pr = nodes_img[r * 9 + f_r]
        pd = nodes_img[r_d * 9 + f]
        cw = float(np.linalg.norm(p - pr))
        ch = float(np.linalg.norm(p - pd))
        cell_areas.append(cw * ch)

    # 6) 逐交点提示 + 判定
    detections = []  # (idx, file, rank, area, circularity, centroid_dist, score)
    all_masks = []   # 与 detections 对齐，保存每个交点的 SAM mask
    for i, (f, r) in enumerate([(f, r) for r in range(10) for f in range(9)]):
        p = nodes_img[i]
        ca = cell_areas[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, ca)

        # 面积比 + 圆度 + 质心距（质心须贴近交点，排除偏移/邻格误检）
        dist = 0.0
        if is_piece and centroid is not None:
            dist = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if dist >= DIST_CELL * np.sqrt(ca):
                is_piece = False

        all_masks.append(mask)
        detections.append((i, f, r, area, circ, dist, score, is_piece))

    n_piece = sum(1 for d in detections if d[7])
    n_empty = 90 - n_piece
    print(f"\n检出棋子 {n_piece} / 90 交点，判空 {n_empty}")

    # 6) 可视化：棋子 mask 绿色半透明填充 + 轮廓；空交点红色小叉；交点黄色小十字
    vis = img.copy()
    overlay = vis.copy()
    for d in detections:
        i, f, r, area, circ, dist, score, is_piece = d
        if is_piece:
            overlay[all_masks[i]] = (0, 220, 0)   # 棋子 mask 绿色填充
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    for d in detections:
        i, f, r, area, circ, dist, score, is_piece = d
        p = tuple(np.round(nodes_img[i]).astype(int))
        if is_piece:
            cnts, _ = cv2.findContours((all_masks[i].astype(np.uint8)) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            cv2.putText(vis, f"{f},{r}", (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

    for p in nodes_img:
        p = tuple(np.round(p).astype(int))
        cv2.drawMarker(vis, p, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 5, 1)

    cv2.imwrite(OUT, vis)
    print(f"[vis] 已保存 {OUT}（绿色填充+轮廓=棋子 mask，红叉=空交点，黄十字=90 交点）")

    # 7) 逐 rank 打印棋子分布（X=棋子 .=空，rank0=黑方底线；标准开局应 rank0/9 各 9、rank2/7 各 2、rank3/6 各 5）
    print("\n按 rank 的棋子分布:")
    for r in range(10):
        row = "".join("X" if detections[r * 9 + f][7] else "." for f in range(9))
        print(f"  rank{r}: {row}")

    # 8) 全量指标 dump（便于调阈值）
    dump = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\m0_metrics.csv"
    with open(dump, "w", encoding="utf-8") as fh:
        fh.write("file,rank,is_piece,area,cell_area,area_ratio,circularity,centroid_dist,score\n")
        for d in detections:
            i, f, r, area, circ, dist, score, is_piece = d
            ca = cell_areas[i]
            fh.write(f"{f},{r},{int(is_piece)},{area:.1f},{ca:.1f},{area/ca:.3f},"
                     f"{circ:.3f},{dist:.1f},{score:.3f}\n")
    print(f"[metrics] 已保存 {dump}")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
