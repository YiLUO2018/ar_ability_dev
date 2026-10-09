"""补充诊断：① m5 早期帧覆盖与清洗删框影响 ② video_01 f95 假阳格位真值 ③ video_02 f195 红子像素红度/亮度"""
import io
import json
import os
import re
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))

# ---------------- ① m5 帧覆盖 + 清洗删框（video_01 前 200 帧）
data = json.load(open(os.path.join(DBG, "m5_samples_v3.json"), encoding="gbk"))
frames = sorted({s["frame"] for s in data["samples"]})
early = [f for f in frames if f <= 200]
print(f"m5 采样帧总数 {len(frames)}，最小 {frames[0]} 最大 {frames[-1]}")
print(f"前 200 帧内的采样帧: {early}")

# 原始 vs 清洗 每帧框数
def count_boxes(d):
    out = {}
    lbl = os.path.join(DBG, d, "labels")
    for fn in os.listdir(lbl):
        m = re.search(r"(\d+)", fn)
        if m:
            out[int(m.group(1))] = sum(1 for _ in open(os.path.join(lbl, fn)))
    return out

orig, clean = count_boxes("m5_yolo_v3"), count_boxes("m5_yolo_v3_clean")
for f in early:
    o, c = orig.get(f, 0), clean.get(f, 0)
    flag = " ←删了框" if c < o else ""
    print(f"  frame{f}: 原{o}框 → 清洗后{c}框{flag}")

# ---------------- ② video_01 f94(0基) 假阳格位真值
from m3_occupancy import read_frame, board_homography, occupancy_features, SAM_PHYS

V1 = os.path.join(DBG, "debug_video_01.mp4")
img = read_frame(V1, 94)
H, _ = board_homography(img, SAM_PHYS)
dark, std, red, sat = occupancy_features(img, H)
print("\nvideo_01 f95 占位真值（.空 B墨 R红）：")
for r in range(10):
    row = ""
    for f_ in range(9):
        i = r * 9 + f_
        is_red = red[i] > 25.0 and sat[i] > 50.0
        row += "." if not (dark[i] < 50.0 or std[i] > 50.0 or is_red) else "R" if is_red else "B"
    print("   ", row, f" r{r}")
for (f_, r) in [(2, 5), (8, 6), (4, 7), (1, 2), (6, 2), (6, 3), (0, 3)]:
    i = r * 9 + f_
    print(f"    假阳候选格({f_},{r}): dark={dark[i]:.0f} std={std[i]:.0f} "
          f"red={red[i]:.0f} sat={sat[i]:.0f}")

# ---------------- ③ video_02 f194 vs f9：红方底线墨色红度/亮度
V2 = os.path.join(DBG, "debug_video_02.mp4")
PHYS = np.array([[0, 0], [450, 0], [450, 500], [0, 500]], np.float32)

def h_compute(src, dst):
    a = np.zeros((8, 9), np.float64)
    for i in range(4):
        x, y = src[i]; u, v = dst[i]
        a[i*2] = [x, y, 1, 0, 0, 0, -x*u, -y*u, u]
        a[i*2+1] = [0, 0, 0, x, y, 1, -x*v, -y*v, v]
    for col in range(8):
        piv = col + int(np.argmax(np.abs(a[col:, col])))
        a[[col, piv]] = a[[piv, col]]
        if abs(a[col, col]) < 1e-9: continue
        a[col] /= a[col, col]
        for r in range(8):
            if r != col: a[r] -= a[r, col] * a[col]
    return np.append(a[:, 8], 1.0).reshape(3, 3)

def ink_redness(img, quad, r):
    """r 行 9 个节点的最暗25%像素平均 R-G 与亮度。"""
    H = h_compute(PHYS, np.asarray(quad, np.float32))
    rg, val = [], []
    for f_ in range(9):
        p = np.array([25 + f_*50.0, 25 + r*50.0, 1.0])
        xy = H @ p
        xy = (xy[:2] / xy[2]).astype(int)
        patch = img[max(0,xy[1]-14):xy[1]+14, max(0,xy[0]-14):xy[0]+14].astype(np.float32)
        b, g, rr = patch[..., 0], patch[..., 1], patch[..., 2]
        gray = rr * 0.299 + g * 0.587 + b * 0.114
        k = max(1, int(patch[..., 0].size * 0.25))
        idx = np.argsort(gray, axis=None)[:k]
        rg.append(float((rr - g).flatten()[idx].mean()))
        val.append(float(gray.mean()))
    return rg, val

# f195 用诊断输出的 EMA 四角；f10 用正常四角
quad_f195 = [[160, 361], [902, 390], [929, 1197], [60, 1175]]
quad_f10 = [[159, 360], [900, 392], [920, 1171], [52, 1131]]
cap = cv2.VideoCapture(V2)
for idx0, tag, quad in [(9, "f10(正常段)", quad_f10), (194, "f195(全黑段)", quad_f195)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, im = cap.read()
    rg_r, v_r = ink_redness(im, quad, 9)   # 红方底线
    rg_b, v_b = ink_redness(im, quad, 0)   # 黑方底线
    print(f"\nvideo_02 {tag}: 全图亮度 {im.mean():.0f}")
    print(f"  红底线(r9) 墨色R-G: {[f'{x:+.1f}' for x in rg_r]}  均值{np.mean(rg_r):+.1f}")
    print(f"  黑底线(r0) 墨色R-G: {[f'{x:+.1f}' for x in rg_b]}  均值{np.mean(rg_b):+.1f}")
    print(f"  红底线亮度 {[f'{x:.0f}' for x in v_r]} / 黑底线 {[f'{x:.0f}' for x in v_b]}")
cap.release()
