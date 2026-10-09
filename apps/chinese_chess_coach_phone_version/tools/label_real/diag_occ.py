"""诊断 flywheel 帧 (2,0) 占位不一致：是真子(SAM漏) 还是 YOLO 假子。

把 frame_00000 按 metadata.corners warp 到 canonical，看 (2,0) 及其邻域 patch 的实际像素，
并列出吸附到 (2,0) 的 YOLO 原始框（cx/cy/label/conf）。不跑 SAM。
"""
import os
import sys
import json

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sam_piece_spike import PHYSICAL_CORNERS, canonical_nodes

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
FLY = os.path.join(DBG, "flywheel_real")


def main():
    meta = None
    with open(os.path.join(FLY, "metadata.jsonl"), encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    img = cv2.imread(os.path.join(FLY, meta["file"] + ".jpg"))
    corners = np.array(meta["corners"], dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(corners, PHYSICAL_CORNERS)
    warped = cv2.warpPerspective(img, M, (450, 500))
    nodes = canonical_nodes()  # rank-major, i = r*9+f

    # (2,0) 及其邻域 patch 的像素统计
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    print("== canonical patch 统计（节点 -> grayMin, grayMean, std）==")
    for (f, r) in [(2, 0), (3, 0), (4, 0), (1, 0), (2, 1), (2, 2)]:
        i = r * 9 + f
        x, y = int(round(nodes[i][0])), int(round(nodes[i][1]))
        p = gray[y - 15:y + 16, x - 15:x + 16]
        print(f"  ({f},{r}) xy=({x},{y}) grayMin={p.min():.0f} mean={p.mean():.0f} std={p.std():.1f}")

    # 吸附到 (2,0) 的 YOLO 框（原图坐标）
    i20 = 0 * 9 + 2
    nx, ny = nodes[i20]
    M_inv = np.linalg.inv(M)
    def proj(x, y):
        p = M_inv @ np.array([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]
    tx, ty = proj(nx, ny)
    print(f"\n== (2,0) 原图坐标 ≈ ({tx:.0f},{ty:.0f})，附近 YOLO 框 ==")
    for d in meta["detections"]:
        cx, cy, lab, conf = d["cx"], d["cy"], d["label"], d["conf"]
        dist = np.hypot(cx - tx, cy - ty)
        if dist < 60:
            print(f"  label={lab} conf={conf:.3f} cx={cx:.0f} cy={cy:.0f} dist={dist:.0f}")

    # 保存 (2,0) 邻域 3x3 patch 拼图（放大看）
    cell = 80
    mont = np.full((3 * cell, 3 * cell, 3), 255, dtype=np.uint8)
    for rr in range(3):
        for cc in range(3):
            f = 2 + cc - 1
            r = 0 + rr
            if f < 0 or f > 8 or r < 0 or r > 9:
                continue
            i = r * 9 + f
            x, y = int(round(nodes[i][0])), int(round(nodes[i][1]))
            patch = warped[max(0, y - 20):y + 20, max(0, x - 20):x + 20]
            patch = cv2.resize(patch, (cell, cell), interpolation=cv2.INTER_CUBIC)
            mont[rr * cell:(rr + 1) * cell, cc * cell:(cc + 1) * cell] = patch
            cv2.putText(mont, f"({f},{r})", (cc * cell + 4, rr * cell + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    out = os.path.join(DBG, "diag_20_patch.png")
    cv2.imwrite(out, mont)
    print(f"\n[vis] {out}")


if __name__ == "__main__":
    main()
