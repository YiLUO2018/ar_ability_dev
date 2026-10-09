"""清洗 v2：m5 幽灵框复核改用「框中心 patch 直读」（不依赖 m3 单应节点）。

v1 教训（2026-10-09 定位）：m3 单应节点有半格级竖向偏移 + 红墨亮（grayMin 59~63），
节点 patch 采样不到笔画 → 真实红兵被误判「确信空」删除（frame0/18/36/54 每帧 1~2 个，
连续帧重复）→ v7 模型在 video_01 开局视角漏检红兵（f5 检出 0 颗）。

v2 规则（diag3 实测标定）——对每个框以其中心 40×40 patch 直读原图：
  删除，当且仅当 grayMin5% ≥ 60 且 std < 45 且 最暗25%像素 R-G < 20
  （真实棋子：墨色 std 63+ 或 R-G +116；真空交点：grayMin 82~95, std 28, R-G≈-6）

用法：python clean_m5_labels_v2.py
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
from m3_occupancy import read_frame, SAM_PHYS, VIDEO

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
SRC_JSON = os.path.join(DBG, "m5_samples_v3.json")
OUT = os.path.join(DBG, "m5_yolo_v3_clean2")

PATCH = 20          # 半宽 → 40×40
GRAY_EMPTY = 60.0   # grayMin5% ≥ 此 且 std 低 且 非红 → 确信空
STD_EMPTY = 45.0
RED_RG = 20.0

CN = ["帅", "仕", "相", "车", "马", "炮", "兵", "将", "士", "象", "车", "马", "炮", "卒"]


def patch_verdict(img, cx, cy):
    """框中心 40×40 patch：返回 (确信空?, grayMin5%, std, 墨R-G)。"""
    xi, yi = int(round(cx)), int(round(cy))
    patch = img[max(0, yi - PATCH):yi + PATCH, max(0, xi - PATCH):xi + PATCH]
    if patch.size == 0:
        return True, 255.0, 0.0, 0.0
    b, g, r = patch[..., 0].astype(np.float32), patch[..., 1].astype(np.float32), patch[..., 2].astype(np.float32)
    gray = r * 0.299 + g * 0.587 + b * 0.114
    k = max(1, int(gray.size * 0.05))
    dark = float(np.sort(gray, axis=None)[:k].mean())
    sd = float(gray.std())
    m25 = gray < np.percentile(gray, 75)          # 最暗 25%（含墨）
    rg = float((r - g)[m25].mean())
    empty = (dark >= GRAY_EMPTY) and (sd < STD_EMPTY) and (rg < RED_RG)
    return empty, dark, sd, rg


def main():
    data = json.load(open(SRC_JSON, encoding="gbk"))
    samples = data["samples"]
    W, H = data["frame_size"]
    x0 = max(0, int(SAM_PHYS[:, 0].min()) - 40)
    x1 = min(W, int(SAM_PHYS[:, 0].max()) + 40)
    y0 = max(0, int(SAM_PHYS[:, 1].min()) - 40)
    y1 = min(H, int(SAM_PHYS[:, 1].max()) + 40)
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
        lines = []
        for s in ss:
            bx, by_, bw, bh = s["bbox"]
            empty, dark, sd, rg = patch_verdict(img, bx + bw / 2.0, by_ + bh / 2.0)
            if empty:
                dropped_total += 1
                drop_log.append((frame_idx, s["f"], s["r"], CN[s["class_id"]],
                                 round(dark, 1), round(sd, 1), round(rg, 1)))
            else:
                lines.append(s)
                kept_total += 1
        name = f"seg_frame{frame_idx}"
        shutil.copy2(os.path.join(DBG, "m5_yolo_v3", "images", name + ".jpg"),
                     os.path.join(OUT, "images", name + ".jpg"))
        with open(os.path.join(OUT, "labels", name + ".txt"), "w") as fh:
            for s in lines:
                bx, by_, bw, bh = s["bbox"]
                cx = (bx - x0 + bw / 2.0) / cw
                cy = (by_ - y0 + bh / 2.0) / ch
                fh.write(f"{s['class_id']} {cx:.6f} {cy:.6f} {bw / cw:.6f} {bh / ch:.6f}\n")

    print(f"[done] 保留 {kept_total} 框，删除 {dropped_total} 框"
          f"（{dropped_total * 100.0 / (kept_total + dropped_total):.1f}%，v1 为 20.9%）-> {OUT}")
    print("早期帧删除明细（v1 曾误删红兵，v2 应只剩真空格）：")
    for d in drop_log:
        if d[0] <= 120:
            print("  drop:", d)


if __name__ == "__main__":
    main()
