"""跨场景泛化抽查：用 v3 best.pt 在真实斜拍照片（非训练视频）上检测棋子。

训练数据来自 debug_video_01.mp4（浅色木圆盘 + 薄墨字，1080×1920）。
这里用 board_21.png（另一套真实棋盘照片，480×640）直接测，回答「换场景还能不能检出」。

board_21 GT：非标准开局 38 子；SAM 四角 [[116,76],[350,76],[375,299],[102,306]]。
输出检测总数 + 类别分布，并存可视化。
"""
import os
import sys
from collections import Counter

import cv2
from ultralytics import YOLO

DBG = "/mnt/l/projects/ar_ability_dev/apps/chinese_chess_coach_phone_version/debug_frames"
BEST = os.path.join(DBG, "m5_yolo_v3", "runs", "train", "weights", "best.pt")

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']

# board_21 的棋盘四角（网格最外层交点）
BOARD_21_CORNERS = [[116, 76], [350, 76], [375, 299], [102, 306]]
PAD = 40


def main():
    model = YOLO(BEST)
    for name, corners in [("board_21", BOARD_21_CORNERS)]:
        path = os.path.join(DBG, name + ".png")
        img = cv2.imread(path)
        H, W = img.shape[:2]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x0 = max(0, min(xs) - PAD); x1 = min(W, max(xs) + PAD)
        y0 = max(0, min(ys) - PAD); y1 = min(H, max(ys) + PAD)
        crop = img[y0:y1, x0:x1]

        res = model.predict(crop, imgsz=640, conf=0.25, verbose=False)[0]
        counts = Counter(int(c) for c in res.boxes.cls.tolist())
        total = len(res.boxes)
        detail = " ".join(f"{CLASS_NAMES[c]}x{n}" for c, n in sorted(counts.items()))
        print(f"{name}: 检出 {total} 子 | {detail}", flush=True)

        vis = res.plot()
        cv2.imwrite(os.path.join(DBG, name + "_yolo_v3.png"), vis)
        print(f"  可视化 -> {name}_yolo_v3.png", flush=True)


if __name__ == "__main__":
    main()
