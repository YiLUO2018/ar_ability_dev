"""M3：占位检测（便宜、稳定）+ 棋盘精修。

视频棋子是**浅色圆盘**（红/黑只是盘面字色，盘体与木板同色）。关键结论：
  - **黑子靠「墨色字符」**：盘面深色字符使 25×25 patch 的灰度最小值 grayMin 显著
    低于空位。黑子 grayMin 27–55，空位（含右下反光区）grayMin 57–117。
  - **反光免疫**：反光只会让像素更亮、不会更暗，故 grayMin 对右下角反光区稳定；
    而 std 会因反光抖动（反光区空位 std 可达 ~36，与扁平棋子重叠）。
  - **红子靠颜色**：R-G>25 且 S>50 单独兜底。

因此「墨色 + 强起伏 + 红」三信号联合，取代单纯 std，优先保证**帧间稳定**
（from→to 需要的是稳定，不是高召回；系统性漏检/误检可被时间投票 + 规则传播消掉）。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import CANON_CORNERS, BOARD_W, BOARD_H, canonical_nodes, warp_points
from refine import refine_corners

VIDEO = os.environ.get(
    "CHESS_VIDEO",
    r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\debug_video_01.mp4",
)
PHYSICAL_CORNERS = np.array(
    [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float32
)
SAM_PHYS = np.array([[58, 145], [956, 123], [1029, 1071], [22, 1090]], dtype=np.float32)

DARK_MIN = 50.0    # 黑子墨色：grayMin < 50（空位 >=51，反光免疫）
STD_HI = 50.0      # 强起伏兜底：std > 50（凸起圆盘，远高于反光区 ~36）
RED_DG = 25.0      # 红子 R-G 下限
RED_S = 50.0       # 红子饱和度下限

_NODES = canonical_nodes()


def read_frame(video, idx=0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def board_homography(frame, phys_corners):
    H, W = frame.shape[:2]
    Hp = cv2.getPerspectiveTransform(PHYSICAL_CORNERS, phys_corners.astype(np.float32))
    grid = warp_points(CANON_CORNERS, Hp)
    res = refine_corners(frame, grid / np.array([W, H], dtype=np.float32))
    return res["H"], res


def occupancy_features(frame, H):
    """返回 (dark 90, std 90, red 90, sat 90)：每交点的灰度最小值/标准差/红度/饱和度。"""
    warped = cv2.warpPerspective(frame, np.linalg.inv(H), (BOARD_W, BOARD_H))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bgr = warped
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    dark = np.zeros(90, dtype=np.float32)
    std = np.zeros(90, dtype=np.float32)
    red = np.zeros(90, dtype=np.float32)
    sat = np.zeros(90, dtype=np.float32)
    for i in range(90):
        x, y = int(round(_NODES[i][0])), int(round(_NODES[i][1]))
        cell = gray[y - 12:y + 13, x - 12:x + 13]
        dark[i] = float(cell.min())
        std[i] = float(cell.std())
        b = float(bgr[y - 12:y + 13, x - 12:x + 13, 0].mean())
        g = float(bgr[y - 12:y + 13, x - 12:x + 13, 1].mean())
        r = float(bgr[y - 12:y + 13, x - 12:x + 13, 2].mean())
        red[i] = r - g
        sat[i] = float(hsv[y - 12:y + 13, x - 12:x + 13, 1].mean())
    return dark, std, red, sat


def classify_occupancy(frame, H, dark_min=DARK_MIN, std_hi=STD_HI):
    dark, std, red, sat = occupancy_features(frame, H)
    occ = ((dark < dark_min) | (std > std_hi)
           | ((red > RED_DG) & (sat > RED_S))).astype(np.uint8)
    tags = ["X" if o else "." for o in occ]
    return occ, tags, std


def print_grid(tags):
    for r in range(10):
        print(f"rank{r}: " + "".join(tags[r * 9 + f] for f in range(9)))


def main():
    frame = read_frame(VIDEO, 0)
    H, res = board_homography(frame, SAM_PHYS)
    print(f"refine: grid_rms={res['grid_rms']:.2f}")
    occ, tags, std = classify_occupancy(frame, H)
    print(f"\nstd+红 占位检出 {int(occ.sum())} 子：")
    print_grid(tags)


if __name__ == "__main__":
    main()
