"""诊断 video_01 的 TR/BR 角误差根因：掩码边界 vs 真值边的偏差剖面。

对最差 N 帧（按 legacy trace 的角误差排序）：跑检测，取被选中候选的掩码，
沿真值四条边采样，量掩码外沿相对真值边的偏移（正值=掩码越界/凸出，负=内缩）。
输出每帧：最差角、顶边/右边偏移剖面（左/中/右三段）、掩码是否触边、提取路径。

用法：python diag_tr.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_pipeline import EdgeSamDetector

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
VIDEO = os.path.join(DBG, "debug_video_01.mp4")
TRACE = os.path.join(DBG, "trace_01_legacy.csv")


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def edge_profile(comp, p0, p1, n=9, band=40):
    """沿 p0→p1 边采样掩码边界相对真值边的位置。

    法向约定：nrm=左法向（顶边 TL→TR 时指向下方=掩码内侧）。对每个采样点在
    [-band,+band] 扫描，边界偏移 = 最小 True 的 s：0=边界恰在真值线上；
    负=掩码越过真值线外凸；正=掩码未达真值线（内缩）。
    """
    d = p1 - p0
    L = np.hypot(*d)
    u = d / L
    nrm = np.array([-u[1], u[0]])          # 顶边时=(0,1)=向下=内侧
    H, W = comp.shape
    offs = []
    for t in np.linspace(0.08, 0.92, n):
        pt = p0 + d * t
        b = np.nan
        for s in np.linspace(-band, band, 161):
            q = pt + nrm * s
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < W and 0 <= y < H and comp[y, x]:
                b = s
                break
        offs.append(b)
    return np.array(offs)


def main():
    d = np.genfromtxt(TRACE, delimiter=",", names=True)
    ok = d[d["ok"] == 1]
    C = np.stack([ok[f"c{i}"] for i in range(8)], 1).reshape(-1, 4, 2)
    gt = np.median(C, 0)
    err = np.linalg.norm(C - gt, axis=2).mean(1)
    order = np.argsort(err)[::-1]
    worst = [int(ok["frame"][i]) for i in order[:12]]
    good = [int(ok["frame"][i]) for i in order[len(order)//2: len(order)//2+3]]
    print(f"GT corners: {np.round(gt).astype(int).tolist()}")
    print(f"worst frames: {worst}\ngood frames:  {good}\n")

    det = EdgeSamDetector()
    det.relock_fixes = False
    det.shape_gate = False

    for fr in worst + good:
        img = read_frame(VIDEO, fr)
        if img is None:
            continue
        h, w = img.shape[:2]
        emb, nw, nh, scale = det._encode(img)
        scores, masks = det._decode(emb, det.prompt_x * nw, det.prompt_y * nh)
        cb_w, cb_h = nw / 4.0, nh / 4.0
        # 与 _select 相同的候选遍历，但记录掩码
        mask_size = masks.shape[1] * masks.shape[2]
        lo, hi = int(mask_size * 0.02), int(mask_size * 0.35)
        pick = None
        for i in range(masks.shape[0]):
            m = masks[i] > 0
            a = det._largest_component_area(m, masks.shape[2], masks.shape[1])
            if not (lo <= a <= hi):
                continue
            quad, recon, comp = det._extract_corners_ex(m, cb_w, cb_h)
            if quad is None:
                continue
            q = quad * (1024.0 / masks.shape[1])
            q = (q * np.array([w / nw, h / nh])).astype(np.float32)
            if not det._is_valid_quad(q, w, h) or det._is_cut_off(q, w, h):
                continue
            pick = (q, recon, comp)
            break
        tag = "good" if fr in good else "WORST"
        if pick is None:
            print(f"f{fr} [{tag}] 无候选")
            continue
        q, recon, comp = pick
        e = np.linalg.norm(q - gt, axis=1)
        # 掩码是 256 画布分辨率，comp 放大到原图
        mask_up = cv2.resize(comp.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        top = edge_profile(mask_up, gt[0], gt[1])       # TL→TR
        right = edge_profile(mask_up, gt[1], gt[2])     # TR→BR
        print(f"f{fr} [{tag}] recon={int(recon)} err per corner: "
              f"{[f'{x:.0f}' for x in e]}  (TL TR BR BL)")
        print(f"   顶边掩码外沿偏移(左→右, +外凸): {np.round(top).astype(int)}")
        print(f"   右边掩码外沿偏移(上→下, +外凸): {np.round(right).astype(int)}")


if __name__ == "__main__":
    main()
