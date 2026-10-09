"""M3 校验：SAM 逐子真值 vs 便宜占位分类。

帧 0 上用 SAM 在 90 交点逐点提示（M0 同款），得到可靠的占位真值；再与
`m3_occupancy.classify_occupancy` 的便宜颜色分类对比，量化便宜分类的漏检/误报，
用来调阈值。

SAM 只跑一帧（校验用），生产管线每帧用便宜分类。
"""
import os
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import BOARD_W, BOARD_H, canonical_nodes, warp_points
from sam_piece_spike import predict_and_classify, DIST_CELL
from m3_occupancy import read_frame, board_homography, classify_occupancy, SAM_PHYS, VIDEO

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\m3_sam_occupancy.png"


def main():
    t0 = time.time()
    frame = read_frame(VIDEO, 0)
    H, res = board_homography(frame, SAM_PHYS)
    print(f"refine: grid_rms={res['grid_rms']:.2f}")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("加载 SAM ViT-B ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    print(f"模型就绪 {time.time()-t0:.1f}s")

    nodes_img = warp_points(canonical_nodes(), H)

    cell_areas = []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]
        fr = f + 1 if f < 8 else f - 1
        rd = r + 1 if r < 9 else r - 1
        pr = nodes_img[r * 9 + fr]; pd = nodes_img[rd * 9 + f]
        cw = float(np.linalg.norm(p - pr)); ch = float(np.linalg.norm(p - pd))
        cell_areas.append(cw * ch)

    sam_occ = np.zeros(90, dtype=np.uint8)
    sam_tags = []
    for i in range(90):
        p = nodes_img[i]; ca = cell_areas[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, ca)
        if is_piece and centroid is not None:
            d = float(np.hypot(centroid[0] - p[0], centroid[1] - p[1]))
            if d < DIST_CELL * np.sqrt(ca):
                sam_occ[i] = 1
        sam_tags.append("X" if sam_occ[i] else ".")

    print(f"\nSAM 真值检出 {int(sam_occ.sum())} 子：")
    for r in range(10):
        print(f"rank{r}: " + "".join(sam_tags[r * 9 + f] for f in range(9)))

    cheap_occ, cheap_tags = classify_occupancy(frame, H)
    print(f"\n便宜占位检出 {int(cheap_occ.sum())} 子：")
    for r in range(10):
        print(f"rank{r}: " + "".join(cheap_tags[r * 9 + f] for f in range(9)))

    # 对比
    agree = (sam_occ == cheap_occ).sum()
    fn = ((sam_occ == 1) & (cheap_occ == 0)).sum()   # SAM 有、便宜漏
    fp = ((sam_occ == 0) & (cheap_occ == 1)).sum()   # SAM 无、便宜误报
    print(f"\n一致 {agree}/90；便宜漏检 {fn}，便宜误报 {fp}")
    if fn:
        miss = [(i % 9, i // 9) for i in range(90) if sam_occ[i] and not cheap_occ[i]]
        print(f"漏检位置 {miss}")
    if fp:
        falsep = [(i % 9, i // 9) for i in range(90) if cheap_occ[i] and not sam_occ[i]]
        print(f"误报位置 {falsep}")

    # 可视化：SAM 真值绿 mask + 便宜分类标签
    vis = frame.copy()
    for i in range(90):
        f, r = i % 9, i // 9
        p = tuple(np.round(nodes_img[i]).astype(int))
        color = (0, 255, 0) if sam_occ[i] else (0, 0, 255)
        cv2.drawMarker(vis, p, color, cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(OUT, vis)
    print(f"[vis] {OUT}（绿叉=SAM 真值棋子，红叉=空）")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
