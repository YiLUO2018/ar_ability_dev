"""M3：from→to 检测（SAM 关键帧占位 → 相邻稳态段对比）。

占位分类在本视频上难（浅色棋子），SAM 基线（90 中心点提示）最可靠（~29–31 子，
系统性误差稳定）。from→to 用「相邻稳态段代表帧的占位差」：源格 1→0 + 目标格 0→1。

SAM 只跑关键帧（每个稳态段 1 帧），生产管线不逐帧跑 SAM。

用法：python m3_fromto.py  [--frames 0,360,690,...]
默认帧列表 = 手动挑选的稳态代表帧（配合 m3_motion 的输出）。
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
from geometry import canonical_nodes, warp_points
from sam_piece_spike import predict_and_classify, DIST_CELL
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"

# 手动挑选的稳态代表帧（帧 idx）；由 m3_motion 的稳态段而来
DEFAULT_FRAMES = [0, 360, 690, 1050, 1350, 2400, 3300]


def sam_occupancy(predictor, frame, H):
    """SAM 基线占位（90 中心点提示）→ 90bit。"""
    nodes_img = warp_points(canonical_nodes(), H)
    cell_areas = []
    for i in range(90):
        f, r = i % 9, i // 9
        p = nodes_img[i]
        fr = f + 1 if f < 8 else f - 1
        rd = r + 1 if r < 9 else r - 1
        cw = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
        ch = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
        cell_areas.append(cw * ch)

    occ = np.zeros(90, dtype=np.uint8)
    for i in range(90):
        p = nodes_img[i]; ca = cell_areas[i]
        mask, is_piece, area, circ, centroid, score = predict_and_classify(predictor, p, ca)
        if is_piece and centroid is not None:
            d = float(np.hypot(centroid[0]-p[0], centroid[1]-p[1]))
            if d < DIST_CELL * np.sqrt(ca):
                occ[i] = 1
    return occ


def occ_grid(occ):
    return ["".join("X" if occ[r * 9 + f] else "." for f in range(9)) for r in range(10)]


def diff_fromto(a, b):
    """a,b 为 90bit 占位。返回 (from_list, to_list)：from=1→0，to=0→1。"""
    d = b.astype(int) - a.astype(int)
    frm = [(i % 9, i // 9) for i in range(90) if d[i] == -1]
    to = [(i % 9, i // 9) for i in range(90) if d[i] == 1]
    return frm, to


def main():
    frames = DEFAULT_FRAMES
    if "--frames" in sys.argv:
        frames = [int(x) for x in sys.argv[sys.argv.index("--frames") + 1].split(",")]

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    print(f"refine grid_rms={res['grid_rms']:.2f}")

    print("加载 SAM ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    occs = {}
    for fi in frames:
        t0 = time.time()
        frame = read_frame(VIDEO, fi)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        occ = sam_occupancy(predictor, frame, H)
        occs[fi] = occ
        print(f"\n=== 帧 {fi} (t={fi/30:.1f}s) 占位 {int(occ.sum())} 子 ===")
        for row in occ_grid(occ):
            print("  " + row)
        print(f"  ({time.time()-t0:.1f}s)")

    # 相邻对比 → from→to
    print("\n=== from→to ===")
    ks = sorted(occs.keys())
    for a, b in zip(ks, ks[1:]):
        frm, to = diff_fromto(occs[a], occs[b])
        if not frm and not to:
            print(f"帧 {a}→{b}: 无变化")
            continue
        print(f"帧 {a}({a/30:.1f}s)→{b}({b/30:.1f}s): from={frm} to={to}")


if __name__ == "__main__":
    main()
