#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一瞳孔检测原型 v3：阈值候选 + 双眼配对 + 瞳孔验证裁决。

用法：
    python tools/eye_proto/pupil_detect.py debug_frames/video_*.mp4 --step 30 --out debug_frames/pupil_out

针对自拍特写构图（双眼恒在画面上半部）的三级策略：
1. 候选：左右半幅各自适应阈值找暗团块（形状/亮环过滤后取 top-K）——虹膜、眉毛、
   眼睑阴影条都可能进来，不做单打独斗的裁决；
2. 配对：左右候选两两组合，约束 |Δy|<0.07H、尺寸比<2.4（眉毛对虽同高，但见下条）；
3. 瞳孔验证裁决：对每个候选拟合瞳孔（成分内最暗分位→fitEllipse+圆度+绝对尺寸+
   瞳径/虹膜径校验），配对排序键 = (通过验证的瞳孔数, 候选基础分之和)。真眼对会
   双双拟合出圆瞳孔，眉毛/阴影条对双双失败——即使粘连成横条（半眯眼视频），瞳孔
   仍是最暗点可从条内取出。
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

# —— 参数（1080p 自拍特写标定）——
ADAPT_BS, ADAPT_C = 91, 10                   # 自适应阈值：bs=51 会把软对焦虹膜打碎
TOP_K = 4                                    # 每半幅保留候选数
IRIS_AREA_MIN, IRIS_AREA_MAX = 3500, 120000  # 团块面积（px²）；上限放宽容纳粘连条
IRIS_ASPECT = (0.5, 3.2)                     # 上限放宽：半眯眼时虹膜+阴影粘连成横条
IRIS_EXTENT_MIN = 0.22
RING_CONTRAST_MIN = 12.0                     # 亮环-团块灰度差
RING_EXPAND = 1.45
PAIR_DY_MAX = 0.07                           # 双眼 y 差上限（画幅高占比）
PAIR_SIZE_MAX = 2.4                          # 双眼 bbox 对角线比上限
SEED_QUANTILE = 0.05                         # 条内最暗分位 → 种子暗核
SEED_ERODE = 9                               # 种子暗核腐蚀半径（压掉薄阴影带）
RAY_N, RAY_SKIP, RAY_MAX = 32, 8, 90         # Starburst 射线数/起扫半径/最大半径
RAY_GRAD_MIN = 8.0                           # 认作瞳孔边界的上升梯度阈值
RAY_PTS_MIN = 16                             # 拟合椭圆所需最少边界点
PUPIL_IRIS_RATIO = (0.12, 0.95)              # 瞳径/虹膜径合法区间
PUPIL_D_MIN, PUPIL_D_MAX = 25, 160           # 瞳孔绝对直径（px，1080p 标定）
PUPIL_ROUND_MAX = 0.45                       # |a-b|/max(a,b) 上限
REGION_Y = (0.28, 0.72)                      # 眼睛搜索纵向窗口（顶部是眉毛/发际）
REGION_X = ((0.06, 0.52), (0.48, 0.94))      # 左/右眼横向窗口（画面边缘是头发）


def find_candidates(gray, x0, x1, y0, y1, k=TOP_K):
    """在区域 [x0,x1)×[y0,y1) 内找暗团块候选，返回 [(bbox, mask, score)]（降序，≤k 个）。"""
    roi = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (9, 9), 0)
    binv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, ADAPT_BS, ADAPT_C)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, kern, iterations=1)
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kern, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    rh, rw = roi.shape
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (IRIS_AREA_MIN <= area <= IRIS_AREA_MAX):
            continue
        aspect = w / max(h, 1)
        if not (IRIS_ASPECT[0] <= aspect <= IRIS_ASPECT[1]):
            continue
        extent = area / float(w * h)
        if extent < IRIS_EXTENT_MIN:
            continue
        comp_mask = labels == i
        blob_mean = float(np.mean(roi[comp_mask]))
        ex0 = max(0, int(x - w * (RING_EXPAND - 1) / 2)); ey0 = max(0, int(y - h * (RING_EXPAND - 1) / 2))
        ex1 = min(rw, int(x + w * (1 + (RING_EXPAND - 1) / 2))); ey1 = min(rh, int(y + h * (1 + (RING_EXPAND - 1) / 2)))
        ring = roi[ey0:ey1, ex0:ex1]
        ring_mask = comp_mask[ey0:ey1, ex0:ex1]
        if ring.size == 0 or (~ring_mask).sum() < 50:
            continue
        contrast = float(np.mean(ring[~ring_mask])) - blob_mean
        if contrast < RING_CONTRAST_MIN:
            continue
        y_norm = (y + h / 2.0) / rh
        height_pen = 1.0 - max(0.0, 0.5 - y_norm) * 1.2
        score = contrast * extent * height_pen
        out.append(((int(x), int(y), int(w), int(h)), comp_mask, score))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:k]


def starburst_edges(gray, seed, r_max=RAY_MAX, n_rays=RAY_N,
                    r_skip=RAY_SKIP, grad_min=RAY_GRAD_MIN):
    """从 seed 放射采样，取每射线上最远的强上升梯度点（暗→亮 = 瞳孔外沿）。"""
    pts = []
    h, w = gray.shape
    for k in range(n_rays):
        th = 2 * np.pi * k / n_rays
        cos, sin = np.cos(th), np.sin(th)
        rs = np.arange(r_skip, r_max, 2)
        xs = (seed[0] + rs * cos).astype(int)
        ys = (seed[1] + rs * sin).astype(int)
        ok = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
        if not ok.any():
            continue
        vals = gray[ys[ok], xs[ok]].astype(float)
        g = vals[1:] - vals[:-1]
        idx = np.where(g > grad_min)[0]
        if len(idx) == 0:
            continue
        r_hit = rs[ok][idx[-1] + 1]
        pts.append((seed[0] + r_hit * cos, seed[1] + r_hit * sin))
    return pts


def fit_pupil(roi, cand):
    """候选 bbox 内找瞳孔：bbox 全域暗分位种子（含亮巩膜，分位才有区分度）
    → Starburst 边界点 → fitEllipse。

    强度阈值直接取瞳孔轮廓在软对焦/阴影下不可靠（实测圆度 0.3-0.8 乱跳），
    Starburst 按梯度找边界，实测圆度稳定 0.11-0.29。
    注意：分位不能只在成分像素上算——成分全是暗像素，阈值会过低，
    9px 腐蚀后薄核直接消失（实测 47/55 帧死种子）。
    """
    (ix, iy, iw, ih), comp_mask, _ = cand
    if comp_mask.sum() < 200:
        return None
    crop = roi[iy:iy + ih, ix:ix + iw]
    thr = np.quantile(crop, SEED_QUANTILE)
    seed_u8 = np.where(crop <= thr, 255, 0).astype(np.uint8)
    seed_u8 = cv2.erode(seed_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEED_ERODE, SEED_ERODE)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed_u8, 8)
    comps = [(st[i][4], i) for i in range(1, n) if st[i][4] > 30]
    if not comps:
        return None
    _, bi = max(comps)
    ys_, xs_ = np.where(lab == bi)
    seed = (ix + float(xs_.mean()), iy + float(ys_.mean()))
    pts = starburst_edges(roi, seed)
    if len(pts) < RAY_PTS_MIN:
        return None
    (cx, cy), (a, b), ang = cv2.fitEllipse(np.float32(pts).reshape(-1, 1, 2))
    iris_d = (iw + ih) / 2.0
    pupil_d = (a + b) / 2.0
    if not (PUPIL_D_MIN <= pupil_d <= PUPIL_D_MAX):
        return None
    if not (PUPIL_IRIS_RATIO[0] * iris_d <= pupil_d <= PUPIL_IRIS_RATIO[1] * iris_d):
        return None
    if abs(a - b) / max(a, b) > PUPIL_ROUND_MAX:
        return None
    if not (0.05 * iw <= cx - ix <= 0.95 * iw and 0.05 * ih <= cy - iy <= 0.95 * ih):
        return None
    return (cx, cy), (a, b), ang


def detect_eyes(gray):
    """整帧双眼检测：候选→配对→瞳孔验证。返回 {'L': (iris, pupil)|None, 'R': ...}。"""
    H, W = gray.shape
    y0, y1 = int(H * REGION_Y[0]), int(H * REGION_Y[1])
    rois = {}
    cands = {}
    for eye, (xa, xb) in zip('LR', REGION_X):
        ex0, ex1 = int(W * xa), int(W * xb)
        rois[eye] = gray[y0:y1, ex0:ex1]
        cands[eye] = find_candidates(gray, ex0, ex1, y0, y1)
    for eye in 'LR':
        cands[eye] = [(c, fit_pupil(rois[eye], c)) for c in cands[eye]]

    def cy_of(cand):
        (x, y, w, h) = cand[0]
        return y + h / 2.0

    def diag_of(cand):
        (x, y, w, h) = cand[0]
        return (w * w + h * h) ** 0.5

    best_pair, best_key = None, (-1, -1.0)
    for cl, pl in cands['L']:
        for cr, pr in cands['R']:
            if abs(cy_of(cl) - cy_of(cr)) > PAIR_DY_MAX * (y1 - y0):
                continue
            dl, dr = diag_of(cl), diag_of(cr)
            if max(dl, dr) > PAIR_SIZE_MAX * min(dl, dr):
                continue
            npup = int(pl is not None) + int(pr is not None)
            key = (npup, cl[2] + cr[2])
            if key > best_key:
                best_key, best_pair = key, (('L', cl, pl), ('R', cr, pr))
    if best_pair is None:
        # 无合法配对：各自退回最佳候选（可能一眼被遮挡/出画）
        return {eye: (cands[eye][0] if cands[eye] else None) for eye in 'LR'}
    result = {}
    for eye, cand, pup in best_pair:
        result[eye] = (cand, pup)
    return result


def process_video(path, out_dir, step, preview_n=4, save_all=False):
    cap = cv2.VideoCapture(path)
    name = os.path.splitext(os.path.basename(path))[0]
    rows = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sub = os.path.join(out_dir, name) if save_all else out_dir
    os.makedirs(out_sub, exist_ok=True)
    preview_idx = set()
    if save_all:
        preview_idx = None  # None = 全部处理帧都保存
    elif preview_n:
        preview_idx = {min(total - 1, int(round(t / step)) * step)
                       for t in np.linspace(0, total - 1, preview_n)}
    H = W = None
    fi = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            want_img = preview_idx is None or fi in preview_idx
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            y0, y1 = int(H * REGION_Y[0]), int(H * REGION_Y[1])
            eyes = detect_eyes(gray)
            for eye in 'LR':
                ex0 = int(W * REGION_X[0 if eye == 'L' else 1][0])
                row = dict(video=name, frame=fi, eye=eye, ok=0,
                           iris_x='', iris_y='', iris_w='', iris_h='',
                           pupil_x='', pupil_y='', pupil_a='', pupil_b='', pupil_deg='')
                hit = eyes.get(eye)
                if hit:
                    cand, pup = hit
                    (ix, iy, iw, ih), _, _ = cand
                    row.update(iris_x=ex0 + ix, iris_y=y0 + iy, iris_w=iw, iris_h=ih)
                    if pup:
                        (px, py), (a, b), ang = pup
                        row.update(ok=1, pupil_x=round(px + ex0, 1), pupil_y=round(py + y0, 1),
                                   pupil_a=round(a, 1), pupil_b=round(b, 1),
                                   pupil_deg=round(ang, 1))
                        if want_img:
                            cv2.ellipse(frame, (int(px + ex0), int(py + y0)),
                                        (int(a / 2), int(b / 2)), ang, 0, 360, (0, 0, 255), 3)
                    if want_img:
                        cv2.rectangle(frame, (ex0 + ix, y0 + iy),
                                      (ex0 + ix + iw, y0 + iy + ih), (0, 255, 0), 2)
                rows.append(row)
            if want_img:
                cv2.imwrite(os.path.join(out_sub, f'f{fi:05d}.jpg'), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                saved += 1
        fi += 1
    cap.release()
    print(f'  {name}: {fi} frames, {saved} annotated saved -> {out_sub}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='视频路径/glob')
    ap.add_argument('--step', type=int, default=30, help='抽帧步长（帧）')
    ap.add_argument('--save-all', action='store_true',
                    help='每个处理帧都存标注图（按视频分子目录），配合 --step 1 即逐帧')
    ap.add_argument('--out', default='debug_frames/pupil_out')
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.exit('没有匹配到视频')
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for p in paths:
        rows = process_video(p, args.out, args.step, save_all=args.save_all)
        all_rows.extend(rows)
        for eye in ('L', 'R'):
            sub = [r for r in rows if r['eye'] == eye]
            n_ok = sum(r['ok'] for r in sub)
            n_iris = sum(1 for r in sub if r['iris_w'] != '')
            if sub:
                print(f'{os.path.basename(p)} 眼{eye}: 瞳孔 {n_ok}/{len(sub)} '
                      f'({100.0 * n_ok / len(sub):.0f}%), 虹膜 {n_iris}/{len(sub)} '
                      f'({100.0 * n_iris / len(sub):.0f}%)')

    if all_rows:
        keys = list(all_rows[0].keys())
        with open(os.path.join(args.out, 'pupil.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f'CSV: {os.path.join(args.out, "pupil.csv")} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
