"""清洗 m5_yolo_v3 标签里的「空交点固定框」（video_01 假阳根因）。

背景：m5_fill_gaps 对「传播身份 P 有、SAM 未检出」的格子补了交点心固定框，其中一部分
其实是传播判错（格子真空）→ 模型学会「空交点也报子」→ 端上/回放稳定假阳。
本脚本用 m3_occupancy 的占位判别（grayMin 墨色，反光免疫）逐框复核：
  保守规则——仅当「确信为空」（grayMin≥60 且 std<45 且非红）才删框；边界样本保留。

输出 m5_yolo_v3_clean/（images 复制自 v3，labels 只含保留框；同款固定 crop 归一化）。

用法：python clean_m5_labels.py
"""
import json
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3_occupancy import read_frame, board_homography, occupancy_features, SAM_PHYS, VIDEO

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
SRC_JSON = os.path.join(DBG, "m5_samples_v3.json")
SRC = os.path.join(DBG, "m5_yolo_v3")
OUT = os.path.join(DBG, "m5_yolo_v3_clean")

PAD = 40
GRAY_EMPTY = 60.0   # grayMin ≥ 此值且 std 低且非红 → 确信空
STD_EMPTY = 45.0
RED_DG, RED_S = 25.0, 50.0


def main():
    data = json.load(open(SRC_JSON, encoding="gbk"))
    samples = data["samples"]
    W, H = data["frame_size"]
    x0 = max(0, int(SAM_PHYS[:, 0].min()) - PAD)
    x1 = min(W, int(SAM_PHYS[:, 0].max()) + PAD)
    y0 = max(0, int(SAM_PHYS[:, 1].min()) - PAD)
    y1 = min(H, int(SAM_PHYS[:, 1].max()) + PAD)
    cw, ch = x1 - x0, y1 - y0

    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

    by_frame = defaultdict(list)
    for s in samples:
        by_frame[s["frame"]].append(s)

    kept_total = dropped_total = 0
    drop_log = []
    for frame_idx, ss in sorted(by_frame.items()):
        img = read_frame(VIDEO, frame_idx)
        if img is None:
            continue
        # m3 已验证的精修单应 + 占位特征（rank-major，i = r*9+f）
        H, _ = board_homography(img, SAM_PHYS)
        dark, std, red, sat = occupancy_features(img, H)
        lines = []
        for s in ss:
            f, r = s["f"], s["r"]
            i = r * 9 + f
            d, sd = float(dark[i]), float(std[i])
            is_red = red[i] > RED_DG and sat[i] > RED_S
            # 占位判别说空 且 dark 高（确信空，避开 50–60 边界带）才删
            confidently_empty = (not (d < 50.0 or sd > 50.0 or is_red)) and d >= GRAY_EMPTY
            if confidently_empty:
                dropped_total += 1
                drop_log.append((frame_idx, f, r, s["class_name"],
                                 round(d, 1), round(sd, 1), round(float(red[i]), 1)))
            else:
                lines.append(s); kept_total += 1
        # 写标签（同 m5_export_yolo 的归一化）
        name = f"seg_frame{frame_idx}"
        shutil.copy2(os.path.join(SRC, "images", name + ".jpg"),
                     os.path.join(OUT, "images", name + ".jpg"))
        with open(os.path.join(OUT, "labels", name + ".txt"), "w") as fh:
            for s in lines:
                bx, by_, bw, bh = s["bbox"]
                cx = (bx - x0 + bw / 2.0) / cw
                cy = (by_ - y0 + bh / 2.0) / ch
                fh.write(f"{s['class_id']} {cx:.6f} {cy:.6f} {bw / cw:.6f} {bh / ch:.6f}\n")

    print(f"[done] 保留 {kept_total} 框，删除 {dropped_total} 框"
          f"（{dropped_total * 100.0 / (kept_total + dropped_total):.1f}%）-> {OUT}")
    for d in drop_log[:40]:
        print("  drop:", d)


if __name__ == "__main__":
    main()
