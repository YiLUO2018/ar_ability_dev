"""用训练好的 best.pt 在「未见过的帧」上预测，做泛化抽查。

测试帧 = 非代表帧（不在 39 个代表帧里），但落在稳态段内，棋盘状态已知。
输出每帧检测到的棋子（类别 + 置信度），并与该帧真实身份（m4_identity_per_frame.csv）对比数量。
"""
import os
import sys
import csv
from collections import Counter

import cv2
from ultralytics import YOLO

DBG = "/mnt/l/projects/ar_ability_dev/apps/chinese_chess_coach_phone_version/debug_frames"
VIDEO = os.path.join(DBG, "debug_video_01.mp4")
BEST = os.path.join(DBG, "m5_yolo_v3", "runs", "train", "weights", "best.pt")
ID_CSV = os.path.join(DBG, "m4_identity_per_frame.csv")

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']

# crop 边界（与 m5_export_yolo 一致）
SAM_PHYS_x = [58, 956, 1029, 22]
SAM_PHYS_y = [145, 123, 1071, 1090]
PAD = 40
x0 = max(0, min(SAM_PHYS_x) - PAD)
y0 = max(0, min(SAM_PHYS_y) - PAD)


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def ground_truth_count(frame_idx):
    """从 m4_identity_per_frame.csv 取该帧非空/非? 的格子数。"""
    with open(ID_CSV, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        # header: frame, 0..89
        for row in rd:
            if int(row[0]) == frame_idx:
                vals = row[1:]
                return sum(1 for v in vals if v not in ('.', '?'))
    return None


def main():
    test_frames = [int(x) for x in sys.argv[1:]] or [100, 300, 500, 1000, 2000, 3000]
    model = YOLO(BEST)

    for fi in test_frames:
        frame = read_frame(VIDEO, fi)
        if frame is None:
            print(f"frame {fi}: 读取失败", flush=True)
            continue
        # 模型在 crop 图上训练，所以同样 crop 再预测
        crop = frame[y0:y0 + 1047, x0:x0 + 1069]
        res = model.predict(crop, imgsz=640, conf=0.25, verbose=False)[0]
        names = res.names
        counts = Counter()
        for c in res.boxes.cls.tolist():
            counts[int(c)] += 1
        gt = ground_truth_count(fi)
        pred_total = sum(counts.values())
        detail = " ".join(f"{CLASS_NAMES[c]}x{n}" for c, n in sorted(counts.items()))
        print(f"frame {fi:4d}: 预测 {pred_total:2d} 子 | GT {gt:2d} 子 | {detail}", flush=True)


if __name__ == "__main__":
    main()
