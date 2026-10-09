"""M5 补全版：为「身份已知但 SAM 未检出」的格子补固定框。

问题：SAM detect_pieces 对低对比黑方棋子（后线 r=0 的将/士/象/马 + 黑炮等）漏检，
导致 m5_samples_v2.json 标签不完整（如 frame 500 缺 5 个黑子）。YOLO 训练继承该缺口。

修复：对每个已采样帧，凡是「传播身份 P 里有、但 SAM 没检出」的格子，用
「交点为中心、0.9 格为边长」的固定框补一个样本。这些漏检子几乎都在后线交点正上方
（偏移 ~5px，p90 58px ≈ 0.6 格），固定框足以覆盖；而棋子直径实测 ~0.9 格。

不改身份（身份来自规则传播，已验证自洽），只补 bbox 来源。

输出 m5_samples_v3.json（无需重跑 SAM，纯后处理，秒级）。
"""
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import canonical_nodes, warp_points
from m3_occupancy import read_frame, board_homography, SAM_PHYS, VIDEO
from m3_bbox_track import cell_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
M3 = os.path.join(DBG, "m3_debug")
PROP_JSON = os.path.join(M3, "m4_propagate_result.json")
DET_JSON = os.path.join(M3, "m4_detect_multi.json")
SRC_JSON = os.path.join(DBG, "m5_samples_v2.json")
OUT_JSON = os.path.join(DBG, "m5_samples_v3.json")

BOX_FRAC = 0.9  # 固定框边长 / 格子边长（棋子直径实测 ~0.9 格）

CLASS_ID = {
    ('R', 'K'): 0, ('R', 'A'): 1, ('R', 'E'): 2, ('R', 'R'): 3,
    ('R', 'H'): 4, ('R', 'C'): 5, ('R', 'P'): 6,
    ('B', 'K'): 7, ('B', 'A'): 8, ('B', 'E'): 9, ('B', 'R'): 10,
    ('B', 'H'): 11, ('B', 'C'): 12, ('B', 'P'): 13,
}
CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']


def main():
    prop = json.load(open(PROP_JSON))
    det = json.load(open(DET_JSON))
    src = json.load(open(SRC_JSON))
    states = prop["states"]
    segs = det["segs"]

    # 去重：段边界帧被相邻两段各采一次（如 786/1767/1932/.../3381），
    # 按 (frame, f, r) 保留首条（首条来自较早段，标签自洽）。
    seen = {}
    samples = []
    for s in src["samples"]:
        key = (s["frame"], s["f"], s["r"])
        if key in seen:
            continue
        seen[key] = True
        samples.append(s)
    print(f"去重: {len(src['samples'])} -> {len(samples)}")

    frame0 = read_frame(VIDEO, 0)
    H, _ = board_homography(frame0, SAM_PHYS)
    nodes_img = warp_points(canonical_nodes(), H)
    cw, ch, ca = cell_metrics(nodes_img)

    # 已有样本按帧分组
    have = defaultdict(set)
    for s in samples:
        have[s["frame"]].add((s["f"], s["r"]))

    # 元数据（与 m5_build_v2.py 一致，仅用于 stable_segs / is_move_boundary）
    stable = Counter()
    for k in range(len(segs)):
        P = {tuple(map(int, key.split(','))) for key in states[k]["board"]}
        for c in det["cells"][k]:
            c = tuple(c)
            if c in P:
                stable[c] += 1
    move_cells = set()
    for m in prop.get("moves", []):
        move_cells.add(tuple(m["from"]))
        move_cells.add(tuple(m["to"]))

    # 补漏
    added = 0
    missing_color = Counter()
    missing_piece = Counter()
    for frame_idx, cells in sorted(have.items()):
        # 定位该帧所属段
        k = None
        for ki, (s, e) in enumerate(segs):
            if s <= frame_idx <= e:
                k = ki
                break
        if k is None:
            continue
        P = {tuple(map(int, key.split(','))): tuple(v)
             for key, v in states[k]["board"].items()}
        for (f, r), piece in P.items():
            if (f, r) in cells:
                continue
            i = r * 9 + f
            cx, cy = nodes_img[i]
            w = BOX_FRAC * cw[i]
            h = BOX_FRAC * ch[i]
            bbox = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]
            samples.append({
                "frame": int(frame_idx), "f": int(f), "r": int(r),
                "class_id": CLASS_ID[piece],
                "class_name": CLASS_NAMES[CLASS_ID[piece]],
                "bbox": bbox, "cx": float(cx), "cy": float(cy),
                "stable_segs": stable.get((f, r), 0),
                "is_move_boundary": (f, r) in move_cells,
                "source": "fallback",
            })
            added += 1
            missing_color[piece[0]] += 1
            missing_piece[piece] += 1

    out = {"class_names": CLASS_NAMES, "frame_size": src["frame_size"],
           "samples": samples}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)

    print(f"补漏样本: {added}（红 {missing_color.get('R', 0)} / 黑 {missing_color.get('B', 0)}）")
    print("按棋子身份分布:")
    for piece, n in sorted(missing_piece.items(), key=lambda x: -x[1]):
        print(f"  {piece[0]}{piece[1]}: {n}")
    print(f"总样本: {len(samples)}（原 {len(src['samples'])}）")
    print(f"[done] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
