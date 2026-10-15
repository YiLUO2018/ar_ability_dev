#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM 瞳孔检测 spike：验证「裁剪+放大 → 点提示 → 瞳孔（而非虹膜）分割」可行性。

设计（与象棋项目 SAM 经验对齐）：
- 经典管线（pupil_detect.py）出的虹膜 bbox 当 ROI：裁 1.8×bbox 边长方形 → resize 1024
  （瞳孔在 SAM 视野里 ~120-165px 直径，语义/分辨率双达标）；
- 种子 = bbox 内距离变换峰值（最厚暗核，压睫毛阴影）；
- 3 种提示变体共用一次编码：P1 单正点 / P2 正点+4 角负点 / P3 框提示；
- 掩码选择按面积带（不是分数——象棋血泪：碎块分数可虚高 1.0，溢出块面积超带），
  期望瞳孔面积 = π×(0.45×虹膜径)²/4，带 [0.4, 2.2]×；
- 输出：放大拼图（掩码轮廓蓝 + 椭圆红）+ 全掩码行为日志（看 SAM 到底返回什么）。

用法：
    python tools/eye_proto/sam_pupil_spike.py --per-video 2 --out debug_frames/pupil_sam
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor  # noqa: E402

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
CSV_PATH = 'debug_frames/pupil_all/pupil.csv'
CROP_GAIN = 1.8          # 裁剪边长 = 1.8 × max(iw, ih)
PUPIL_IRIS_D = 0.45      # 期望瞳径/虹膜径（室内光瞳孔偏大）
AREA_LO, AREA_HI = 0.4, 2.2


def dt_seed(gray, ix, iy, iw, ih):
    """bbox 内距离变换峰值 = 最厚暗核（瞳孔核），返回 (x,y) 原图坐标。"""
    crop = gray[iy:iy + ih, ix:ix + iw]
    thr = np.quantile(crop, 0.25)
    dark = np.where(crop <= thr, 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dt = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    _, mx, _, loc = cv2.minMaxLoc(dt)
    if mx < 6:
        return None
    return (ix + loc[0], iy + loc[1])


def mask_stats(mask_u8, scale):
    """最大连通域轮廓 → (面积1024, 面积原图, 等效直径原图, 圆度, 椭圆)。"""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50 or len(c) < 5:
        return None
    circ = 4 * np.pi * area / max(cv2.arcLength(c, True) ** 2, 1.0)
    (cx, cy), (a, b), ang = cv2.fitEllipse(c)
    return dict(area1024=area, area=area / scale ** 2,
                d=(a + b) / 2 / scale, circ=circ,
                ellipse=(cx / scale, cy / scale, a / scale, b / scale, ang),
                contour=c)


def daugman_refine(crop_bgr, mask_ellipse):
    """掩码椭圆邻域 ×3 放大 → CLAHE → Daugman 积分微分圆搜索。

    搜 (cx,cy,r)：中心在掩码中心 ±0.25d 的 9×9 网格，半径 [0.5,1.25]×d/2，
    得分 = 内外环灰度差(外亮内暗) × √沿圆梯度幅值均值。返回 (cx,cy,d)（原图px）或 None。
    """
    ex, ey, a, b, ang = mask_ellipse
    d = (a + b) / 2
    side = int(d * 2.0)
    H, W = crop_bgr.shape[:2]
    x0 = int(np.clip(ex - side / 2, 0, max(0, W - side)))
    y0 = int(np.clip(ey - side / 2, 0, max(0, H - side)))
    sub = crop_bgr[y0:y0 + side, x0:x0 + side]
    if sub.shape[0] < 60:
        return None
    Z = 3
    big = cv2.resize(sub, (sub.shape[1] * Z, sub.shape[0] * Z), interpolation=cv2.INTER_CUBIC)
    gray8 = cv2.createCLAHE(2.0, (16, 16)).apply(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray8.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    gmag = np.hypot(gx, gy)
    sm = cv2.GaussianBlur(gray, (7, 7), 0)
    cen0 = ((ex - x0) * Z, (ey - y0) * Z)
    dZ = d * Z
    best = (-1.0, None)
    thetas = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)
    h, w = gmag.shape
    for dxc in np.linspace(-0.25 * dZ, 0.25 * dZ, 9):
        for dyc in np.linspace(-0.25 * dZ, 0.25 * dZ, 9):
            ccx, ccy = cen0[0] + dxc, cen0[1] + dyc
            for rr in np.arange(0.5 * dZ / 2, 1.25 * dZ / 2, 2.0):
                xs = (ccx + rr * ct).astype(int)
                ys = (ccy + rr * st).astype(int)
                if xs.min() < 1 or ys.min() < 1 or xs.max() >= w - 1 or ys.max() >= h - 1:
                    continue
                xi = (ccx + (rr - 2) * ct).astype(int)
                yi = (ccy + (rr - 2) * st).astype(int)
                xo = (ccx + (rr + 2) * ct).astype(int)
                yo = (ccy + (rr + 2) * st).astype(int)
                contrast = sm[yo, xo].mean() - sm[yi, xi].mean()
                strength = gmag[ys, xs].mean()
                score = contrast * np.sqrt(max(strength, 1e-3))
                if score > best[0]:
                    best = (score, (ccx / Z + x0, ccy / Z + y0, 2 * rr / Z))
    return best[1] if best[1] is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-video', type=int, default=2)
    ap.add_argument('--out', default='debug_frames/pupil_sam')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print('加载 SAM ViT-B ...', flush=True)
    t0 = time.time()
    sam = sam_model_registry['vit_b'](checkpoint=CHECKPOINT)
    sam.to('cpu')
    predictor = SamPredictor(sam)
    print(f'  {time.time() - t0:.0f}s', flush=True)

    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    caps = {}

    def get_frame(vid, fi):
        if vid not in caps:
            caps[vid] = cv2.VideoCapture(f'debug_frames/{vid}.mp4')
        c = caps[vid]
        c.set(cv2.CAP_PROP_POS_FRAMES, fi)
        return c.read()[1]

    videos = sorted(set(r['video'] for r in rows))
    for vid in videos:
        tiles = {v: [] for v in ('P1', 'P2', 'P3')}
        # 每视频抽 per_video 帧（虹膜检出的），跳过开头 60 帧避开起手不稳
        frames = sorted({int(r['frame']) for r in rows
                         if r['video'] == vid and r['iris_w'] and int(r['frame']) > 60})
        picks = frames[::max(1, len(frames) // max(1, args.per_video))][:args.per_video]
        for fi in picks:
            frame = get_frame(vid, fi)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            for eye in 'LR':
                r = next((x for x in rows if x['video'] == vid and x['frame'] == str(fi)
                          and x['eye'] == eye and x['iris_w']), None)
                if r is None:
                    continue
                gx, gy = int(r['iris_x']), int(r['iris_y'])
                iw, ih = int(r['iris_w']), int(r['iris_h'])
                seed = dt_seed(gray, gx, gy, iw, ih)
                if seed is None:
                    print(f'{vid[-6:]} f{fi} {eye}: 无种子，跳过')
                    continue
                # 裁剪方形（bbox 中心，边 1.8×max(iw,ih)，钳制画幅）
                side = int(CROP_GAIN * max(iw, ih))
                cx0 = gx + iw // 2
                cy0 = gy + ih // 2
                x0 = max(0, min(W - side, cx0 - side // 2))
                y0 = max(0, min(H - side, cy0 - side // 2))
                crop = frame[y0:y0 + side, x0:x0 + side]
                scale = 1024.0 / side
                img1024 = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                sx, sy = (seed[0] - x0) * scale, (seed[1] - y0) * scale
                predictor.set_image(cv2.cvtColor(img1024, cv2.COLOR_BGR2RGB))
                iris_d1024 = (iw + ih) / 2.0 * scale
                want_area = np.pi * (PUPIL_IRIS_D * iris_d1024 / 2) ** 2

                variants = {
                    'P1': (np.float32([[sx, sy]]), np.int32([1]), None),
                    'P2': (np.float32([[sx, sy], [70, 70], [954, 70], [70, 954], [954, 954]]),
                           np.int32([1, 0, 0, 0, 0]), None),
                    'P3': (np.float32([[sx, sy]]), np.int32([1]),
                           np.float32([sx - 0.55 * ih * scale, sy - 0.55 * ih * scale,
                                       sx + 0.55 * ih * scale, sy + 0.55 * ih * scale])),
                }
                for vname, (pc, pl, box) in variants.items():
                    kw = dict(point_coords=pc, point_labels=pl, multimask_output=True)
                    if box is not None:
                        kw['box'] = box
                    masks, scores, _ = predictor.predict(**kw)
                    picked, picked_stat = None, None
                    for mi, m in enumerate(masks):
                        st = mask_stats(m.astype(np.uint8) * 255, scale)
                        if st is None:
                            continue
                        in_band = AREA_LO * want_area <= st['area1024'] <= AREA_HI * want_area
                        print(f"  {vid[-6:]} f{fi} {eye} {vname} m{mi} "
                              f"score={scores[mi]:.2f} d={st['d']:.0f}px "
                              f"circ={st['circ']:.2f} band={'Y' if in_band else 'N'}")
                        if in_band and (picked is None or scores[mi] > picked[1]):
                            picked, picked_stat = (mi, scores[mi]), st
                    tile = img1024.copy()
                    if picked_stat is not None:
                        c = picked_stat['contour']
                        cv2.drawContours(tile, [c], -1, (255, 200, 0), 3)
                        ex, ey, a, b, ang = picked_stat['ellipse']
                        cv2.ellipse(tile, (int(ex * scale), int(ey * scale)),
                                    (int(a * scale / 2), int(b * scale / 2)), ang,
                                    0, 360, (0, 0, 255), 3)
                    tile = cv2.resize(tile, (640, 640))
                    cv2.putText(tile, f'{vid[-6:]} f{fi} {eye} {vname}'
                                + ('' if picked else ' NO-BAND'),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    tiles[vname].append(tile)
        for vname, tl in tiles.items():
            if tl:
                while len(tl) % 4:
                    tl.append(np.zeros_like(tl[0]))
                sheet = np.vstack([np.hstack(tl[i:i + 4]) for i in range(0, len(tl), 4)])
                out = os.path.join(args.out, f'{vid[-6:]}_{vname}.jpg')
                cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
                print(f'sheet: {out}')
    for c in caps.values():
        c.release()


if __name__ == '__main__':
    main()
