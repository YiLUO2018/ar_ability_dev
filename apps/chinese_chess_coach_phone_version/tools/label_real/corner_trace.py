"""棋盘框检测性能分析（corner_trace）：对静止视频逐检测帧输出角点轨迹 CSV + 指标。

背景（2026-10-11）：vis2 回放显示 video_01（手机/棋盘静止、偶发手遮挡角点）存在
① 角点识别错误/外框形状错 ② 角点持续跳变。静止视频的真值角点恒定 → 用全体成功帧的
逐坐标中位数当 GT，量化「错误形状帧」（err > 阈值）与「跳变事件」。

只跑 EdgeSAM 检测层（不含 YOLO/Refiner/引擎），每 5 帧一次 + 失败加速（fixed 模式），
输出 CSV：frame,ok,rescue,recon,area,prev_null,c0..c7。

用法：python corner_trace.py <video> <csv_out> [--mode fixed|legacy] [--every 5]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_pipeline import (EdgeSamDetector, PHYSICAL_CORNERS, MARGIN, CELL,
                             h_compute, h_transform)

CANON_NODES = np.array([[MARGIN + f * CELL, MARGIN + r * CELL] for r in range(10) for f in range(9)],
                        np.float64)


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    video = sys.argv[1]
    csv_out = sys.argv[2]
    mode = opt("--mode", "fixed")
    every = int(opt("--every", "5"))

    det = EdgeSamDetector()
    det.relock_fixes = (mode == "fixed")
    det.shape_gate = (mode == "fixed")
    quad_mode = opt("--quad-mode", "hull")
    EdgeSamDetector.QUAD_MODE = quad_mode
    # D1/D2 PC 模拟：--imu static 启用静止锚闸；--inject-move F 在 F 帧把锚平移 1 格
    # （模拟静止中棋盘被挪），验证超时仲裁在 ANCHOR_REJ_TIMEOUT 次连续拒绝后重锚。
    imu = opt("--imu", "off")
    det.set_imu_mode(imu)               # static=锚闸 | moving=救援二级 | off=UNKNOWN(S7)
    cell_thr = float(opt("--anchor-cell", "0.2") or 0.2)
    det.ANCHOR_CORNER = cell_thr
    inject_move = int(opt("--inject-move", "0") or 0)
    injected = [False]

    # C7① 参考 = GT 网格节点（**物理参考**，端上为 YOLO 棋子质心）。不可用「上帧 quad
    # 反投影节点」——那是自指涉：一次外扩型伪成功会把节点集毒化到真板之外，此后所有
    # 好候选包含率 <0.85 全被拒 → 永久死亡螺旋（f173-f∞ 实测复现）。静止视频的 GT
    # 节点=理想棋子质心超集，与端上语义等价且不受检测误差污染。
    # GT 取 legacy trace 成功帧的中位数（video_01 静止）。
    gt_csv = os.path.join(os.path.dirname(csv_out), "trace_01_legacy.csv")
    ref_pts = None
    if os.path.exists(gt_csv):
        g = np.genfromtxt(gt_csv, delimiter=",", names=True)
        gok = g[g["ok"] == 1]
        G = np.stack([gok[f"c{i}"] for i in range(8)], 1).reshape(-1, 4, 2)
        gt_corners = np.median(G, 0)
        ref_pts = h_transform(h_compute(PHYSICAL_CORNERS, gt_corners), CANON_NODES)
    print(f"[trace] mode={mode} quad={quad_mode} every={every} ref={'GT节点' if ref_pts is not None else '无'}")

    cap = cv2.VideoCapture(video)
    rows = []
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frame += 1
        # 失败/未锁定期间加速采样（fixed 模式，与端上 B 同）；legacy 固定 every。
        unlocked = (len(rows) > 0 and rows[-1][1] == 0)
        step = 1 if (mode == "fixed" and (unlocked or not rows)) else every
        if frame % step != 0:
            continue
        prev_null = det.prev_corners is None
        info = det.detect(img, ref_pts)
        if (inject_move and not injected[0] and frame >= inject_move
                and det.anchor is not None):
            cell = det._cell_size(det.anchor)
            det.anchor = det.anchor.copy()
            det.anchor[:, 0] += cell      # 模拟棋盘被挪 1 格（手机仍静止）
            injected[0] = True
            print(f"[inject] f{frame}: 锚平移 +1 格（模拟挪盘），等仲裁重锚", flush=True)
        if info is None:
            rows.append((frame, 0, 0, 0, 0, int(prev_null)) + (0.0,) * 8)
        else:
            c = np.asarray(info["corners"], np.float64).ravel()
            rescue = 1 if det._last_rescue else 0
            rows.append((frame, 1, rescue, int(det._last_recon), int(det._last_area),
                         int(prev_null)) + tuple(float(x) for x in c))
    cap.release()

    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("frame,ok,rescue,recon,area,prev_null,c0,c1,c2,c3,c4,c5,c6,c7\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # ---- 指标 ----
    oks = [r for r in rows if r[1] == 1]
    fails = [r for r in rows if r[1] == 0]
    print(f"mode={mode}: 检测尝试={len(rows)} 成功={len(oks)} 失败={len(fails)} "
          f"救援={sum(r[2] for r in rows)} 失败时prev已清={sum(r[5] for r in fails)} "
          f"锚拒绝={det.n_anchor_rej}")
    if not oks:
        return
    C = np.array([r[6:14] for r in oks])
    gt = np.median(C, axis=0)
    err = np.linalg.norm((C - gt).reshape(-1, 4, 2), axis=2).mean(1)
    frames = np.array([r[0] for r in oks])
    for thr in (8.0, 15.0, 40.0):
        n = int((err > thr).sum())
        print(f"  err>{thr:4.0f}px: {n:4d} 帧 ({n/len(oks)*100:5.1f}%)")
    # 跳变事件：相邻成功检测间（间隔 ≤ 4 个检测周期）平均角位移 > 8px（静止场景）。
    jumps = []
    for i in range(1, len(oks)):
        gap = (frames[i] - frames[i-1]) / every
        d = np.linalg.norm((C[i] - C[i-1]).reshape(-1, 4, 2), axis=2).mean()
        if d > 8.0 and gap <= 4:
            jumps.append((int(frames[i]), float(d)))
    print(f"  跳变事件(相邻成功位移>8px): {len(jumps)}")
    if jumps[:20]:
        print("   首批:", [(f, f"{d:.0f}px") for f, d in jumps[:12]])
    # 角点坐标滑动标准差（10 成功帧窗口的均值）
    if len(C) >= 10:
        stds = [C[i:i+10].std(0).mean() for i in range(len(C) - 9)]
        print(f"  滑动10帧角点坐标std: 均值={np.mean(stds):.2f}px 中位={np.median(stds):.2f}px max={np.max(stds):.1f}px")


if __name__ == "__main__":
    main()
