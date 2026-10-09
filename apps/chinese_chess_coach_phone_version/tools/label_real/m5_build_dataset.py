"""M5：从 M4 传播结果 + SAM 检测，生成高置信 YOLO 训练样本（bbox + 14 类身份）。

核心筛选 = 双重确认：格子 (f,r) 同时满足
  1) 传播身份非空（states[k] 有该格身份）—— 身份有象棋规则链支撑
  2) SAM detect_pieces 在代表帧检出该格 —— bbox 有独立检测支撑
误报（检测有/身份无）与漏检（身份有/检测无）都被排除，宁缺毋滥。

每个样本附带置信度标签：
  - stable_segs：该格被「双重确认」过的稳态段数（越大越稳，长时间驻留 + 每次都被检出）
  - is_move_boundary：该格是否出现在某步走子的 from/to（走子边界格，身份刚变化）

输出 m5_samples.json + 类别分布统计。

用法：python m5_build_dataset.py [--limit K]   # --limit 只跑前 K 段（快速验证）
"""
import os
import sys
import time
import json
from collections import Counter

import cv2
import numpy as np

SAM_ROOT = os.environ.get(
    "SAM_ROOT", r"L:\Reinforcement_Learning\segment\segment-anything-main")
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_bbox_track import detect_pieces, cell_metrics, load_occ

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
M3 = os.path.join(DBG, "m3_debug")
PROP_JSON = os.path.join(M3, "m4_propagate_result.json")
DET_JSON = os.path.join(M3, "m4_detect_multi.json")
OCC_CSV = os.path.join(DBG, "sam_video_occupancy.csv")
OUT_JSON = os.path.join(DBG, "m5_samples.json")
CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")

# 14 类：红 0-6，黑 7-13（与 YOLO 训练类别顺序对齐）
CLASS_ID = {
    ('R', 'K'): 0, ('R', 'A'): 1, ('R', 'E'): 2, ('R', 'R'): 3,
    ('R', 'H'): 4, ('R', 'C'): 5, ('R', 'P'): 6,
    ('B', 'K'): 7, ('B', 'A'): 8, ('B', 'E'): 9, ('B', 'R'): 10,
    ('B', 'H'): 11, ('B', 'C'): 12, ('B', 'P'): 13,
}
CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']


def mask_to_bbox(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [int(x), int(y), int(w), int(h)]


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    prop = json.load(open(PROP_JSON))
    det = json.load(open(DET_JSON))
    states = prop["states"]          # states[k]["board"] = {"f,r":[c,t]}
    segs = det["segs"]
    reps = det["reps"]
    moves = prop.get("moves", [])

    # 每格被双重确认过的段数（跨段稳定度）
    stable = Counter()
    for k in range(len(segs)):
        P = {tuple(map(int, key.split(','))) for key in states[k]["board"]}
        for c in det["cells"][k]:
            c = tuple(c)
            if c in P:
                stable[c] += 1

    # 走子边界格集合
    move_cells = set()
    for m in moves:
        move_cells.add(tuple(m["from"]))
        move_cells.add(tuple(m["to"]))

    # 选每段代表帧：段内 CSV 占位 sum 最接近 32 的帧
    frames, occs = load_occ(OCC_CSV)
    rep_frames = []
    for s, e in segs:
        m = (frames >= s) & (frames <= e)
        sub = occs[m]
        if len(sub) == 0:
            rep_frames.append((s + e) // 2)
            continue
        j = int(np.argmin(np.abs(sub.sum(axis=1) - 32)))
        rep_frames.append(int(frames[m][j]))
    if limit:
        segs = segs[:limit]
        rep_frames = rep_frames[:limit]

    print(f"segments={len(segs)}", flush=True)
    print("loading SAM (cpu) ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)
    print(f"grid_rms={res['grid_rms']:.2f}", flush=True)

    H0, W0 = frame0.shape[:2]

    samples = []
    for k, (seg, rep) in enumerate(zip(segs, rep_frames)):
        P = {tuple(map(int, key.split(','))): tuple(v)
             for key, v in states[k]["board"].items()}
        t0 = time.time()
        frame = read_frame(VIDEO, rep)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = detect_pieces(predictor, nodes_img, cw, ch, ca)
        n_keep = 0
        for (f, r), (cx, cy, mask) in d.items():
            if (f, r) not in P:
                continue          # 检测误报（身份无此格）→ 排除
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            piece = P[(f, r)]
            samples.append({
                "frame": int(rep), "f": int(f), "r": int(r),
                "class_id": CLASS_ID[piece], "class_name": CLASS_NAMES[CLASS_ID[piece]],
                "bbox": bbox, "cx": float(cx), "cy": float(cy),
                "stable_segs": stable.get((f, r), 0),
                "is_move_boundary": (f, r) in move_cells,
            })
            n_keep += 1
        print(f"seg{k:2d} frame {rep:5d}: det={len(d)} keep={n_keep}  ({time.time()-t0:.0f}s)",
              flush=True)

    # 类别分布
    dist = Counter(s["class_id"] for s in samples)
    print(f"\n===== 高置信样本总数: {len(samples)} =====", flush=True)
    for cid in range(14):
        if dist.get(cid, 0) > 0:
            print(f"  {CLASS_NAMES[cid]:>2} (id {cid:2d}): {dist[cid]:3d}", flush=True)

    out = {
        "class_names": CLASS_NAMES,
        "frame_size": [W0, H0],
        "samples": samples,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"\n[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
