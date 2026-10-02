"""合成数据量化验证：扰动粗角点 → 几何精修 → 对比 GT 角点误差。

在合成场景图（角点 GT 已知）上，把 GT 角点加高斯噪声模拟「粗模型输出」，
再跑 ``refine_corners`` 精修，统计精修前后相对 GT 的角点误差（像素）。

用法（在 WSL 的 le_wm 环境）：
    python validate_synthetic.py --n 100 --noise 10,20,30 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GEN = os.path.normpath(os.path.join(HERE, "..", "data_gen"))
sys.path.insert(0, DATA_GEN)

import render as rd  # noqa: E402
from refine import refine_corners  # noqa: E402


def corner_error_px(a_norm: np.ndarray, b_norm: np.ndarray, W: int, H: int) -> np.ndarray:
    """两套归一化角点（TL,TR,BR,BL）的逐角点像素误差。"""
    scale = np.array([W, H], dtype=np.float32)
    a = a_norm.astype(np.float32) * scale
    b = b_norm.astype(np.float32) * scale
    return np.linalg.norm(a - b, axis=1)


def perturb(corners_norm: np.ndarray, W: int, H: int, noise_px: float, rng) -> np.ndarray:
    """对角点（归一化）加高斯扰动，模拟粗模型估计。"""
    scale = np.array([W, H], dtype=np.float32)
    px = corners_norm.astype(np.float32) * scale
    px = px + rng.normal(0.0, noise_px, size=px.shape).astype(np.float32)
    return np.clip(px / scale, 0.001, 0.999).astype(np.float32)


def summarize(errors: list[np.ndarray]) -> dict:
    """errors: list of 每样本逐角点误差 (4,)。返回整体中位/均值/p90/失败数。"""
    if not errors:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "n": 0}
    flat = np.concatenate(errors)
    return {
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "p90": float(np.percentile(flat, 90)),
        "n": len(errors),
    }


def run(noise_px: float, n: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    W = H = size
    init_errs, ref_errs = [], []
    n_fail = 0

    for _ in range(n):
        scene, gt = rd.generate_corner_sample(rng, W, H)
        init = perturb(gt, W, H, noise_px, rng)
        res = refine_corners(scene, init)
        ref = res["corners_norm"]

        if not np.isfinite(ref).all():
            n_fail += 1
            continue

        init_errs.append(corner_error_px(init, gt, W, H))
        ref_errs.append(corner_error_px(ref, gt, W, H))

    s_init = summarize(init_errs)
    s_ref = summarize(ref_errs)
    print(f"\nnoise={noise_px:>4.0f}px  n={s_init['n']:>4d}  fail={n_fail:>3d}")
    print(f"  初始误差:  median={s_init['median']:6.2f}  mean={s_init['mean']:6.2f}  p90={s_init['p90']:6.2f} px")
    print(f"  精修误差:  median={s_ref['median']:6.2f}  mean={s_ref['mean']:6.2f}  p90={s_ref['p90']:6.2f} px")
    if s_init["median"] > 0:
        print(f"  误差中位降低 {100 * (1 - s_ref['median'] / s_init['median']):.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="每种噪声水平的样本数")
    ap.add_argument("--noise", type=str, default="10,20,30", help="扰动噪声(px)，逗号分隔")
    ap.add_argument("--size", type=int, default=640, help="场景边长")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    noises = [float(x) for x in args.noise.split(",")]
    for noise in noises:
        run(noise, args.n, args.size, args.seed + int(noise))


if __name__ == "__main__":
    main()
