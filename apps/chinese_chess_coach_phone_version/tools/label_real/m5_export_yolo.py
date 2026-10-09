"""M5 导出：把 m5_samples.json 转成 YOLO 检测训练集（Ultralytics 格式）。

输出 debug_frames/m5_yolo/：
  images/  seg{k}_frame{rep}.jpg   每段代表帧（裁剪棋盘区域，减少背景）
  labels/  seg{k}_frame{rep}.txt   每行 class_id cx cy w h（归一化到 0-1）
  data.yaml                         14 类中文名 + 训练路径

bbox 坐标系：原图像素 → 减 crop 原点 → 归一化到 crop 尺寸。
（端上若在 warp 正对棋盘上检测，可后续用 board_homography 把框 warp 到物理坐标。）
"""
import os
import sys
import json
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_occupancy import read_frame, SAM_PHYS, VIDEO

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
SRC_JSON = os.path.join(DBG, "m5_samples_v3.json")
OUT_DIR = os.path.join(DBG, "m5_yolo_v3")
IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")

PAD = 40  # 棋盘外 padding（像素）


def main():
    data = json.load(open(SRC_JSON))
    samples = data["samples"]
    names = data["class_names"]
    W, H = data["frame_size"]

    # crop 边界：棋盘物理四角外扩 PAD（棋盘位置固定，全视频共用）
    x0 = max(0, int(SAM_PHYS[:, 0].min()) - PAD)
    x1 = min(W, int(SAM_PHYS[:, 0].max()) + PAD)
    y0 = max(0, int(SAM_PHYS[:, 1].min()) - PAD)
    y1 = min(H, int(SAM_PHYS[:, 1].max()) + PAD)
    cw, ch = x1 - x0, y1 - y0

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    # 按帧分组
    by_frame = defaultdict(list)
    for s in samples:
        by_frame[s["frame"]].append(s)

    written = 0
    for frame_idx, ss in sorted(by_frame.items()):
        img = read_frame(VIDEO, frame_idx)
        if img is None:
            print(f"  [warn] frame {frame_idx} 读取失败，跳过", flush=True)
            continue
        crop = img[y0:y1, x0:x1]
        name = f"seg_frame{frame_idx}"
        cv2.imwrite(os.path.join(IMG_DIR, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lines = []
        for s in ss:
            bx, by_, bw, bh = s["bbox"]
            # 转 crop 坐标 → 中心点归一化
            cx = (bx - x0 + bw / 2.0) / cw
            cy = (by_ - y0 + bh / 2.0) / ch
            wn = bw / cw
            hn = bh / ch
            # 钳制到 [0,1] 防越界
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)
            lines.append(f"{s['class_id']} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        with open(os.path.join(LBL_DIR, name + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        written += 1
        print(f"  {name}: {len(lines)} boxes", flush=True)

    # data.yaml
    yaml_lines = [
        f"# 象棋棋子检测数据集（{written} 张图，{len(samples)} 个标注框）",
        f"path: {OUT_DIR.replace(os.sep, '/')}",
        "train: images",
        "val: images",
        "",
        "nc: 14",
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
    ]
    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(yaml_lines) + "\n")

    print(f"\n[done] {written} 张图 -> {OUT_DIR}", flush=True)
    print(f"crop 区域: x[{x0},{x1}] y[{y0},{y1}] 尺寸 {cw}x{ch}", flush=True)


if __name__ == "__main__":
    main()
