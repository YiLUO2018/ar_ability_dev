"""P3-a 新检测逻辑逐关口探针（与 EdgeSamDetector.detect 同构 + 打印）。"""
import io
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_pipeline import EdgeSamDetector, h_transform

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))


def read_frame(video, idx0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def probe(edge, img, tag):
    h, w = img.shape[:2]
    emb, nw, nh, scale = edge._encode(img)
    px, py = edge.prompt_x * nw, edge.prompt_y * nh
    scores, masks = edge._decode(emb, px, py)
    mask_size = masks.shape[1] * masks.shape[2]
    lo, hi = int(mask_size * 0.02), int(mask_size * 0.35)
    cands = []
    for i in range(masks.shape[0]):
        a = edge._largest_component_area(masks[i] > 0, masks.shape[2], masks.shape[1])
        cands.append((a, i, float(scores[i])))
        if lo <= a <= hi:
            cands[-1] = (a, i, float(scores[i]))
    print(f"[{tag}] 候选: " + " ".join(f"#{i}:{a*100/mask_size:.1f}%/{s:.2f}"
                                       for a, i, s in cands))
    cands = sorted([(a, i) for a, i, s in cands if lo <= a <= hi], reverse=True)
    if not cands:
        print(f"[{tag}] FAIL: 无面积合法候选")
        return None
    for a, i in cands:
        quad_mask, recon, comp = edge._extract_corners_ex(masks[i] > 0)
        if quad_mask is None:
            print(f"  #{i}(面积{a*100/mask_size:.1f}%): 提不出四角")
            continue
        corners = (quad_mask * (1024.0 / masks.shape[1]) * np.array([w / nw, h / nh])).astype(np.float32)
        if not edge._is_valid_quad(corners, w, h):
            print(f"  #{i}(面积{a*100/mask_size:.1f}%,recon={recon}): isValidQuad 不过 "
                  f"corners={np.round(corners,0).tolist()}")
            continue
        if not recon and edge._is_cut_off(corners, w, h):
            print(f"  #{i}(面积{a*100/mask_size:.1f}%,recon={recon}): 贴边裁切拒 "
                  f"corners={np.round(corners,0).tolist()}")
            continue
        if recon and (corners[:, 0].min() < -w * 0.5 or corners[:, 0].max() > w * 1.5 or
                      corners[:, 1].min() < -h * 0.5 or corners[:, 1].max() > h * 1.5):
            print(f"  #{i}(面积{a*100/mask_size:.1f}%,recon={recon}): 复原角点过远 "
                  f"corners={np.round(corners,0).tolist()}")
            continue
        # 到此为可接受候选（先全可见后复原的优先级由调用方保证，这里打印全部可接受者）
        jump = edge.prev_corners is not None and edge._is_jump(corners, edge.prev_corners)
        print(f"  #{i}(面积{a*100/mask_size:.1f}%,recon={recon}): 可接受 corners="
              f"{np.round(corners,0).tolist()} jump={jump}")
        return corners, recon, comp
    print(f"[{tag}] FAIL: 所有候选被拒")
    return None


def run(video, frames):
    edge = EdgeSamDetector()
    cap = cv2.VideoCapture(video)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= max(frames):
            break
        f += 1
        if f not in frames:
            continue
        r = probe(edge, img, f"f{f}")
        if r is not None:
            corners, recon, comp = r
            if edge.prev_corners is not None and edge._is_jump(corners, edge.prev_corners):
                consistent = (edge.jump_candidate is not None
                              and not edge._is_jump(corners, edge.jump_candidate))
                edge.jump_streak = edge.jump_streak + 1 if consistent else 1
                edge.jump_candidate = corners.copy()
                if edge.jump_streak < edge.JUMP_ACCEPT_STREAK:
                    print(f"    → 时间闸拦截 streak={edge.jump_streak}")
                    continue
                edge.jump_streak, edge.jump_candidate = 0, None
            else:
                edge.jump_streak, edge.jump_candidate = 0, None
            if recon:
                ys, xs = np.nonzero(comp)
                cx = xs.mean() * (1024.0 / 256) * (img.shape[1] / nw_cache[0]) if False else 0
            cen = corners.mean(0)
            edge.prompt_x = float(np.clip(cen[0] / img.shape[1], 0.02, 0.98))
            edge.prompt_y = float(np.clip(cen[1] / img.shape[0], 0.02, 0.98))
            smoothed = edge._smooth(corners, img.shape[1], img.shape[0]).astype(np.float32)
            edge.prev_corners = smoothed.copy()
            print(f"    → 接受 EMA后={np.round(smoothed,0).tolist()}")
    cap.release()


if __name__ == "__main__":
    vid = sys.argv[1]
    frames = [int(x) for x in sys.argv[2].split(",")]
    run(os.path.join(DBG, f"debug_video_{vid}.mp4"), frames)
