"""P3-a 外推复原合成测试：全可见 / 1角出画 / 2对角出画 / 整边出画（应拒绝）。"""
import io
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_pipeline import EdgeSamDetector

CASES = [
    # (名称, 真值四角[掩码坐标], 期望触发复原, 期望角点精确)
    ("全可见", [(30, 40), (220, 30), (230, 220), (20, 210)], False, True),
    ("TR出画(右+上)", [(30, 60), (280, -10), (200, 230), (20, 200)], True, True),
    ("TL+BR对角出画", [(-30, -20), (230, 40), (290, 270), (30, 230)], True, True),
    ("TL+TR整边出画(应拒)", [(-20, -30), (270, -20), (230, 200), (30, 230)], False, False),
    ("仅TR角微出画", [(40, 40), (258, 45), (210, 220), (25, 205)], True, True),
]

ok_all = True
for name, quad, expect_recon, expect_exact in CASES:
    canvas = np.zeros((256, 256), np.uint8)
    cv2.fillPoly(canvas, [np.array(quad, np.int32)], 1)
    corners, recon, _ = EdgeSamDetector._extract_corners_ex(canvas > 0)
    if corners is None:
        print(f"[{name}] FAIL: 提不出四角")
        ok_all = False
        continue
    # 与真值最近角配对比较
    q = np.array(quad, np.float64)
    errs = []
    for t in q:
        d = np.hypot(*(corners - t).T)
        errs.append(d.min())
    max_err = max(errs)
    if expect_exact:
        status = "PASS" if (recon == expect_recon and max_err < 4.0) else "FAIL"
    else:  # 应拒用例：复原不触发即可（兜底角点会被上游贴边校验拒绝）
        status = "PASS" if recon == expect_recon else "FAIL"
    if status == "FAIL":
        ok_all = False
    print(f"[{name}] {status} recon={recon}（期望 {expect_recon}）max角误差={max_err:.1f}px "
          f"(掩码坐标) 复原角={np.round(corners, 0).tolist()}")

print("\n全部通过" if ok_all else "\n存在失败")
