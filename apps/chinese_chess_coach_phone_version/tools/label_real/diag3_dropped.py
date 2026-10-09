"""验证 clean_m5_labels 在 video_01 早期帧删除的框是否为真实棋子（原图 patch 直读，不经单应）。"""
import io
import json
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
from m3_occupancy import SAM_PHYS, read_frame

data = json.load(open(os.path.join(DBG, "m5_samples_v3.json"), encoding="gbk"))
W, H = data["frame_size"]
x0 = max(0, int(SAM_PHYS[:, 0].min()) - 40)
x1 = min(W, int(SAM_PHYS[:, 0].max()) + 40)
y0 = max(0, int(SAM_PHYS[:, 1].min()) - 40)
y1 = min(H, int(SAM_PHYS[:, 1].max()) + 40)
cw, ch = x1 - x0, y1 - y0


def load(p):
    out = []
    for ln in open(p):
        c, cx, cy, w, h = ln.split()
        out.append((int(c), float(cx) * cw + x0, float(cy) * ch + y0))
    return out


CN = ["帅", "仕", "相", "车", "马", "炮", "兵", "将", "士", "象", "车", "马", "炮", "卒"]


def h_compute(src, dst):
    a = np.zeros((8, 9), np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[i * 2] = [x, y, 1, 0, 0, 0, -x * u, -y * u, u]
        a[i * 2 + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v, v]
    for col in range(8):
        piv = col + int(np.argmax(np.abs(a[col:, col])))
        a[[col, piv]] = a[[piv, col]]
        if abs(a[col, col]) < 1e-9:
            continue
        a[col] /= a[col, col]
        for r in range(8):
            if r != col:
                a[r] -= a[r, col] * a[col]
    return np.append(a[:, 8], 1.0).reshape(3, 3)


S = np.array([[0, 0], [450, 0], [450, 500], [0, 500]], np.float32)
Hm = h_compute(SAM_PHYS.astype(np.float32), S)  # 图像→规范

img = read_frame(os.path.join(DBG, "debug_video_01.mp4"), 18)
a = load(os.path.join(DBG, "m5_yolo_v3", "labels", "seg_frame18.txt"))
b = load(os.path.join(DBG, "m5_yolo_v3_clean", "labels", "seg_frame18.txt"))
used = [False] * len(b)
dropped = []
for c, px, py in a:
    hit = False
    for i, (c2, px2, py2) in enumerate(b):
        if not used[i] and c2 == c and abs(px - px2) < 3 and abs(py - py2) < 3:
            used[i] = True
            hit = True
            break
    if not hit:
        dropped.append((c, px, py))

print("frame18 删除框原图 patch 判读（40×40，不依赖单应）：")
for c, px, py in dropped:
    p = Hm @ np.array([px, py, 1.0])
    xx, yy = p[0] / p[2], p[1] / p[2]
    f_, r_ = round((xx - 25) / 50), round((yy - 25) / 50)
    xi, yi = int(px), int(py)
    patch = img[max(0, yi - 20):yi + 20, max(0, xi - 20):xi + 20].astype(np.float32)
    b_, g_, r2 = patch[..., 0], patch[..., 1], patch[..., 2]
    gray = r2 * 0.299 + g_ * 0.587 + b_ * 0.114
    k = max(1, int(gray.size * 0.05))
    dark = float(np.sort(gray, axis=None)[:k].mean())
    sd = float(gray.std())
    rg = float((r2 - g_)[gray < np.percentile(gray, 25)].mean())
    if dark < 50 or sd > 50:
        verdict = "有墨棋子（黑/深墨）"
    elif rg > 20:
        verdict = "红子"
    else:
        verdict = "疑似空"
    print(f"  {CN[c]} cell≈({f_},{r_}) img=({xi},{yi}) grayMin5%={dark:.0f} "
          f"std={sd:.0f} 墨R-G={rg:+.0f} => {verdict}")
