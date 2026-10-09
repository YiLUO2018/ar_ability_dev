"""P3-a 探针 v2：完全镜像新版 detect()（时间稳定性仲裁）+ 逐候选打印。"""
import io
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_pipeline import EdgeSamDetector

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))


def detect_dbg(edge, img, tag):
    h, w = img.shape[:2]
    emb, nw, nh, scale = edge._encode(img)
    px, py = edge.prompt_x * nw, edge.prompt_y * nh
    scores, masks = edge._decode(emb, px, py)
    mask_size = masks.shape[1] * masks.shape[2]
    lo, hi = int(mask_size * 0.02), int(mask_size * 0.35)
    cands = []
    for i in range(masks.shape[0]):
        a = edge._largest_component_area(masks[i] > 0, masks.shape[2], masks.shape[1])
        if lo <= a <= hi:
            cands.append((a, i, float(scores[i])))
    cands.sort(reverse=True)
    print(f"[{tag}] 带内候选: " + " ".join(f"#{i}:{a*100/mask_size:.1f}%/{s:.2f}" for a, i, s in cands)
          + (f"  (prev={'有' if edge.prev_corners is not None else '无'})" ))
    chosen = None
    for pass_recon in (False, True):
        acceptables = []
        for a, i, sc in cands:
            quad_mask, recon, comp = edge._extract_corners_ex(masks[i] > 0, nw / 4.0, nh / 4.0)
            if quad_mask is None:
                print(f"    #{i}: 提不出四角")
                continue
            corners = (quad_mask * (1024.0 / masks.shape[1]) * np.array([w / nw, h / nh])).astype(np.float32)
            if not edge._is_valid_quad(corners, w, h):
                print(f"    #{i}(recon={recon}): isValidQuad不过 {np.round(corners,0).tolist()}")
                continue
            if pass_recon != recon:
                continue
            if not recon and edge._is_cut_off(corners, w, h):
                print(f"    #{i}(recon={recon}): 贴边拒 {np.round(corners,0).tolist()}")
                continue
            if recon and (corners[:, 0].min() < -w * 0.5 or corners[:, 0].max() > w * 1.5 or
                          corners[:, 1].min() < -h * 0.5 or corners[:, 1].max() > h * 1.5):
                print(f"    #{i}(recon): 复原过远 {np.round(corners,0).tolist()}")
                continue
            d = (float(np.hypot(*(corners - edge.prev_corners).T).mean())
                 if edge.prev_corners is not None else -1.0)
            acceptables.append((corners, recon, comp, i, d))
        if acceptables:
            if edge.prev_corners is None:
                chosen = acceptables[0]
            else:
                chosen = min(acceptables, key=lambda c: c[4])
            for c in acceptables:
                mark = " ←选" if c is chosen else ""
                print(f"    #{c[3]}(recon={c[1]}): 可接受 平均位移={c[4]:.0f}px{mark}")
            break
    if chosen is None:
        print(f"[{tag}] FAIL: 无可接受候选")
        return None
    corners, recon, comp, ci, cd = chosen
    if edge.prev_corners is not None and edge._is_jump(corners, edge.prev_corners):
        consistent = (edge.jump_candidate is not None
                      and not edge._is_jump(corners, edge.jump_candidate))
        edge.jump_streak = edge.jump_streak + 1 if consistent else 1
        edge.jump_candidate = corners.copy()
        if edge.jump_streak < edge.JUMP_ACCEPT_STREAK:
            print(f"[{tag}] 时间闸拦截 streak={edge.jump_streak} corners={np.round(corners,0).tolist()}")
            return None
        print(f"[{tag}] 时间闸3连接受")
        edge.jump_streak, edge.jump_candidate = 0, None
    else:
        edge.jump_streak, edge.jump_candidate = 0, None
    if recon:
        ys, xs = np.nonzero(comp)
        cx = xs.mean() * (1024.0 / masks.shape[1]) * (w / nw)
        cy = ys.mean() * (1024.0 / masks.shape[1]) * (h / nh)
        edge.prompt_x = float(np.clip(cx / w, 0.02, 0.98))
        edge.prompt_y = float(np.clip(cy / h, 0.02, 0.98))
    else:
        cen = corners.mean(0)
        edge.prompt_x = float(np.clip(cen[0] / w, 0.02, 0.98))
        edge.prompt_y = float(np.clip(cen[1] / h, 0.02, 0.98))
    smoothed = edge._smooth(corners, w, h).astype(np.float32)
    print(f"[{tag}] OK #{ci} recon={recon} raw={np.round(corners,0).tolist()} EMA={np.round(smoothed,0).tolist()}")
    return edge._build_info(smoothed)


if __name__ == "__main__":
    vid = sys.argv[1]
    lo, hi = sys.argv[2].split("-")
    edge = EdgeSamDetector()
    cap = cv2.VideoCapture(os.path.join(DBG, f"debug_video_{vid}.mp4"))
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= int(hi):
            break
        f += 1
        if f % 5 == 0 and int(lo) <= f:
            detect_dbg(edge, img, f"f{f}")
    cap.release()
