"""M5 扩量版：每段多帧采样 + 双重确认 → YOLO 训练样本。

与单帧版（m5_build_dataset.py）的区别：每段均匀采 N 帧（默认 5），不再只选
「最干净代表帧」，从而覆盖反光/遮挡/角度微变等多样性。每帧仍用双重确认
（身份有规则支撑 + SAM detect_pieces 独立检出）筛选，保证标签正确。

输出 m5_samples_v2.json。支持断点续跑（--start K 从第 K 段继续，读已有样本）。

用法：python m5_build_v2.py [--frames N] [--start K] [--limit K]
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
from m3_bbox_track import detect_pieces, cell_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
M3 = os.path.join(DBG, "m3_debug")
PROP_JSON = os.path.join(M3, "m4_propagate_result.json")
DET_JSON = os.path.join(M3, "m4_detect_multi.json")
OUT_JSON = os.path.join(DBG, "m5_samples_v2.json")
CHECKPOINT = os.path.join(SAM_ROOT, "models", "sam_vit_b_01ec64.pth")

CLASS_ID = {
    ('R', 'K'): 0, ('R', 'A'): 1, ('R', 'E'): 2, ('R', 'R'): 3,
    ('R', 'H'): 4, ('R', 'C'): 5, ('R', 'P'): 6,
    ('B', 'K'): 7, ('B', 'A'): 8, ('B', 'E'): 9, ('B', 'R'): 10,
    ('B', 'H'): 11, ('B', 'C'): 12, ('B', 'P'): 13,
}
CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']


def sample_frames(seg, n):
    s, e = seg
    if e - s + 1 <= n:
        return list(range(s, e + 1))
    return [int(round(s + (e - s) * k / (n - 1))) for k in range(n)]


def mask_to_bbox(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [int(x), int(y), int(w), int(h)]


def save(samples, frame_size):
    out = {"class_names": CLASS_NAMES, "frame_size": frame_size, "samples": samples}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)


def main():
    n_frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 5
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    prop = json.load(open(PROP_JSON))
    det = json.load(open(DET_JSON))
    states = prop["states"]
    segs = det["segs"]
    moves = prop.get("moves", [])

    # 段级置信度标签（与采样帧数无关）
    stable = Counter()
    for k in range(len(segs)):
        P = {tuple(map(int, key.split(','))) for key in states[k]["board"]}
        for c in det["cells"][k]:
            c = tuple(c)
            if c in P:
                stable[c] += 1
    move_cells = set()
    for m in moves:
        move_cells.add(tuple(m["from"]))
        move_cells.add(tuple(m["to"]))

    if limit:
        segs = segs[:limit]

    # 断点续跑：读已有样本
    samples = []
    if start > 0 and os.path.exists(OUT_JSON):
        samples = json.load(open(OUT_JSON))["samples"]
        print(f"[resume] 已有 {len(samples)} 样本，从 seg{start} 继续", flush=True)

    print(f"segments={len(segs)}  frames/seg={n_frames}", flush=True)
    print("loading SAM (cpu) ...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    frame0 = read_frame(VIDEO, 0)
    H, res = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)
    H0, W0 = frame0.shape[:2]
    print(f"grid_rms={res['grid_rms']:.2f}", flush=True)

    total_frames = 0
    for k in range(start, len(segs)):
        P = {tuple(map(int, key.split(','))): tuple(v)
             for key, v in states[k]["board"].items()}
        frs = sample_frames(segs[k], n_frames)
        seg_kept = 0
        for rep in frs:
            t0 = time.time()
            frame = read_frame(VIDEO, rep)
            if frame is None:
                continue
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = detect_pieces(predictor, nodes_img, cw, ch, ca)
            n_keep = 0
            for (f, r), (cx, cy, mask) in d.items():
                if (f, r) not in P:
                    continue
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
            seg_kept += n_keep
            total_frames += 1
            print(f"  seg{k:2d} frame {rep:5d}: det={len(d)} keep={n_keep}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        save(samples, [W0, H0])
        print(f"seg{k:2d} 完成：本段 {seg_kept} 样本，累计 {len(samples)}", flush=True)

    dist = Counter(s["class_id"] for s in samples)
    print(f"\n===== 高置信样本总数: {len(samples)}（{total_frames} 帧）=====", flush=True)
    for cid in range(14):
        if dist.get(cid, 0) > 0:
            print(f"  {CLASS_NAMES[cid]:>2} (id {cid:2d}): {dist[cid]:3d}", flush=True)
    print(f"[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
