"""旋转增强：训练集图像按步进角度旋转（默认 20°，共 17 个副本），仅增强 train.txt。

- 旋转后扩画布到外接矩形（对角线尺寸），114 填充（与 letterbox 一致）；
- 标签框四角经仿射变换后取轴对齐外接矩形（旋转后的圆形棋子外接框即紧框）；
- val.txt 不动（跨版本指标可比）；
- 幂等：旋转副本已存在则跳过重编码；train.txt 每次从原始条目重建
  （原始条目 = 文件名不匹配 _r\\d{3}\\.jpg 的条目）。

用法：python augment_rotations.py <dataset_dir> [step_deg]
"""
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))


def to_win(p):
    p = p.replace("\\", "/")
    if p.startswith("/mnt/l/"):
        return "L:/" + p[len("/mnt/l/"):]
    return p


def to_wsl(p):
    p = p.replace("\\", "/")
    if p[:3].lower() == "l:/":
        p = "/mnt/l/" + p[3:]
    return p


def make_variant(img_path, lbl_path, dst_img, dst_lbl, deg):
    img = cv2.imread(img_path)
    if img is None:
        return False
    H, W = img.shape[:2]
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    NW = int(round(abs(W * c) + abs(H * s)))
    NH = int(round(abs(W * s) + abs(H * c)))
    tx = NW / 2 - (W / 2 * c - H / 2 * s)
    ty = NH / 2 - (W / 2 * s + H / 2 * c)
    M = np.float32([[c, -s, tx], [s, c, ty]])
    out = cv2.warpAffine(img, M, (NW, NH), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    lines = []
    if os.path.exists(lbl_path):
        for ln in open(lbl_path):
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            pts = np.array([[(cx - w / 2) * W, (cy - h / 2) * H],
                            [(cx + w / 2) * W, (cy - h / 2) * H],
                            [(cx + w / 2) * W, (cy + h / 2) * H],
                            [(cx - w / 2) * W, (cy + h / 2) * H]], np.float32)
            dst = cv2.transform(pts[None], M)[0]
            x1, y1 = max(0.0, dst[:, 0].min()), max(0.0, dst[:, 1].min())
            x2, y2 = min(float(NW), dst[:, 0].max()), min(float(NH), dst[:, 1].max())
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{cls} {(x1 + x2) / 2 / NW:.6f} {(y1 + y2) / 2 / NH:.6f} "
                         f"{(x2 - x1) / NW:.6f} {(y2 - y1) / NH:.6f}")
    if not os.path.exists(dst_img):
        cv2.imwrite(dst_img, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(dst_lbl, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return True


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DBG, "train_v8_yolo")
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    angles = list(range(step, 360, step))
    img_dir = os.path.join(ds, "images")
    lbl_dir = os.path.join(ds, "labels")
    train_txt = os.path.join(ds, "train.txt")

    orig = [ln.strip() for ln in open(train_txt)
            if ln.strip() and not re.search(r"_r\d{3}\.jpg$", os.path.basename(ln.strip()))]
    print(f"原始 train 条目 {len(orig)}，旋转角度 {angles}")
    entries = list(orig)
    made = skipped = 0
    for i, e in enumerate(orig):
        wp = to_win(e)
        base = os.path.splitext(os.path.basename(wp))[0]
        lbl = os.path.join(lbl_dir, base + ".txt")
        for deg in angles:
            name = f"{base}_r{deg:03d}"
            dst_img = os.path.join(img_dir, name + ".jpg")
            dst_lbl = os.path.join(lbl_dir, name + ".txt")
            if os.path.exists(dst_img) and os.path.exists(dst_lbl):
                skipped += 1
            else:
                if not make_variant(wp, lbl, dst_img, dst_lbl, deg):
                    continue
                made += 1
            entries.append(to_wsl(dst_img))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(orig)}（新建 {made}，跳过 {skipped}）", flush=True)
    with open(train_txt, "w") as fh:
        fh.write("\n".join(entries) + "\n")
    print(f"[done] 新建 {made} 副本（已存在跳过 {skipped}），train.txt 条目 {len(orig)} → {len(entries)}")


if __name__ == "__main__":
    main()
