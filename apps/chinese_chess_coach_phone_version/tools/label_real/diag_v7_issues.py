"""诊断 v7 回放中用户报告的 5 个问题帧（权威数据，不靠肉眼看图）。

video_01: f5（漏检+配比不符）、f95（红3炮+2卒假阳）、f185（右上角点错）
video_02: f183（棋盘漏检）、f212（渲染态疑似全黑卒 —— 源头是 f195/f210 的检出）

用法：python diag_v7_issues.py
"""
import io
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from replay_pipeline import (
    EdgeSamDetector, YoloDetectorPy, QuadRefinerPy, snap_to_grid, vote_grid,
    h_transform, CLASS_NAMES, PT, MARGIN, CELL,
)

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
V1 = os.path.join(DBG, "debug_video_01.mp4")
V2 = os.path.join(DBG, "debug_video_02.mp4")


def read_frame(video, idx0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


# ---------------------------------------------------------- EdgeSAM 带诊断的 detect

def detect_dbg(edge, img, tag):
    """复刻 EdgeSamDetector.detect，逐关口打印失败原因。"""
    h, w = img.shape[:2]
    emb, nw, nh, scale = edge._encode(img)
    px, py = edge.prompt_x * nw, edge.prompt_y * nh
    scores, masks = edge._decode(emb, px, py)
    mask_size = masks.shape[1] * masks.shape[2]
    lo, hi = int(mask_size * 0.02), int(mask_size * 0.35)
    cands = []
    for i in range(masks.shape[0]):
        a = edge._largest_component_area(masks[i] > 0, masks.shape[2], masks.shape[1])
        cands.append((i, a, float(scores[i])))
    best, best_area = -1, 0
    for i, a, sc in cands:
        if lo <= a <= hi and a > best_area:
            best_area, best = a, i
    print(f"  [{tag}] prompt=({edge.prompt_x:.2f},{edge.prompt_y:.2f}) "
          f"候选(面积占比%,分数): "
          + " ".join(f"#{i}:{a*100/mask_size:.1f}%/{sc:.2f}" for i, a, sc in cands)
          + f"  合法带[{lo*100/mask_size:.0f}%,{hi*100/mask_size:.0f}%]")
    if best < 0:
        print(f"  [{tag}] FAIL: 无面积合法候选掩码")
        return None
    quad_mask = edge._extract_corners(masks[best] > 0)
    if quad_mask is None:
        print(f"  [{tag}] FAIL: 掩码提不出四角")
        return None
    corners = (quad_mask * (1024.0 / masks.shape[1]) * np.array([w / nw, h / nh])).astype(np.float32)
    print(f"  [{tag}] raw_corners={np.round(corners,0).tolist()}")
    if not edge._is_valid_quad(corners, w, h):
        print(f"  [{tag}] FAIL: isValidQuad 不过"); return None
    if edge._is_cut_off(corners, w, h):
        print(f"  [{tag}] FAIL: 贴边裁切拒（角点出画面边缘 8px）"); return None
    if edge.prev_corners is not None and edge._is_jump(corners, edge.prev_corners):
        consistent = (edge.jump_candidate is not None
                      and not edge._is_jump(corners, edge.jump_candidate))
        edge.jump_streak = edge.jump_streak + 1 if consistent else 1
        edge.jump_candidate = corners.copy()
        if edge.jump_streak < edge.JUMP_ACCEPT_STREAK:
            print(f"  [{tag}] FAIL: 时间一致性闸拦截 streak={edge.jump_streak}")
            return None
        print(f"  [{tag}] 时间闸：连续第3帧一致，接受新位置")
        edge.jump_streak, edge.jump_candidate = 0, None
    else:
        edge.jump_streak, edge.jump_candidate = 0, None
    cen = corners.mean(0)
    edge.prompt_x = float(np.clip(cen[0] / w, 0.02, 0.98))
    edge.prompt_y = float(np.clip(cen[1] / h, 0.02, 0.98))
    smoothed = edge._smooth(corners, w, h).astype(np.float32)
    print(f"  [{tag}] EMA后={np.round(smoothed,0).tolist()}")
    return edge._build_info(smoothed)


def fmt_dets(dets, info):
    out = []
    for d in dets:
        cx, cy, x1, y1, x2, y2, c, s = d
        p = h_transform(info["imageToGrid"], [[cx, cy]])[0]
        f = int(round((p[0] - MARGIN) / CELL))
        r = int(round((p[1] - MARGIN) / CELL))
        out.append(f"{CLASS_NAMES[c]}{'红' if c<=6 else '黑'}{s:.2f}@({f},{r})")
    return out


def hist(dets):
    from collections import Counter
    return dict(Counter(CLASS_NAMES[d[6]] + ("红" if d[6] <= 6 else "黑") for d in dets))


# ---------------------------------------------------------- video_01：带状态复跑到 f185

def video1():
    print("=" * 30, "video_01 f5 / f95 / f185", "=" * 30)
    edge, yolo, refiner = EdgeSamDetector(), YoloDetectorPy(), QuadRefinerPy()
    cap = cv2.VideoCapture(V1)
    n_det = 0
    recent, last_centroids = [], []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= 185:
            break
        f += 1
        if f % 5:
            continue
        tag = f"f{f}"
        detected = detect_dbg(edge, img, tag)
        if detected is None:
            continue
        info = detected
        refined_tag = ""
        if len(last_centroids) >= 12:
            refined = refiner.refine(info["corners"], last_centroids)
            if refined is not None:
                new_info = edge.refine_quad(refined, img.shape[1], img.shape[0])
                if new_info is not None:
                    refined_tag = f" 精修→{np.round(new_info['corners'],0).tolist()}"
                    info = new_info
                else:
                    refined_tag = " 精修被拒"
        dets, _ = yolo.detect(img, info["corners"])
        last_centroids = [[d[0], d[1]] for d in dets]
        grid = snap_to_grid(dets, info)
        recent.append(grid)
        while len(recent) > 5:
            recent.pop(0)
        voted = vote_grid(recent)
        n_pc = sum(1 for r in range(10) for ff in range(9) if voted[r][ff])
        if f in (5, 95, 185):
            print(f"  >>> f{f} 检出{len(dets)} 投票后{n_pc}子{refined_tag}")
            print(f"      dets: {' '.join(fmt_dets(dets, info))}")
            print(f"      直方图: {hist(dets)}")
    cap.release()


def occupancy_gt(frame_idx0):
    """video_01 m3 已验证占位真值（grayMin 分类器）。"""
    from m3_occupancy import read_frame, board_homography, occupancy_features, SAM_PHYS
    img = read_frame(V1, frame_idx0)
    H, _ = board_homography(img, SAM_PHYS)
    dark, std, red, sat = occupancy_features(img, H)
    occ = []
    for r in range(10):
        row = ""
        for f_ in range(9):
            i = r * 9 + f_
            is_red = red[i] > 25.0 and sat[i] > 50.0
            row += "." if not (dark[i] < 50.0 or std[i] > 50.0 or is_red) else "R" if is_red else "B"
        occ.append(row + f"  r{r}")
    return occ


def video1_gt():
    print("\n" + "=" * 30, "video_01 f5 占位真值（m3 grayMin 网格，B=墨色 R=红）", "=" * 30)
    for row in occupancy_gt(4):
        print("   ", row)
    print("    标准开局应有：r0 黑9子 r2 黑2炮 r3 黑5卒 r6 红5兵 r7 红2炮 r9 红9子")


# ---------------------------------------------------------- video_02：f183 漏检 + f212 渲染源

def video2():
    print("\n" + "=" * 30, "video_02 f165~f215 棋盘检测关口诊断", "=" * 30)
    edge, yolo = EdgeSamDetector(), YoloDetectorPy()
    cap = cv2.VideoCapture(V2)
    f = 0
    last_ok = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= 215:
            break
        f += 1
        if f % 5:
            continue
        info = detect_dbg(edge, img, f"f{f}")
        if info is None:
            print(f"  f{f} 漏检（距上次成功 {f-last_ok} 帧）")
            continue
        last_ok = f
        dets, _ = yolo.detect(img, info["corners"])
        if f in (195, 210):
            print(f"  >>> f{f} 检出{len(dets)}:")
            print(f"      dets: {' '.join(fmt_dets(dets, info))}")
            print(f"      直方图: {hist(dets)}")
    cap.release()


if __name__ == "__main__":
    video1()
    video1_gt()
    video2()
