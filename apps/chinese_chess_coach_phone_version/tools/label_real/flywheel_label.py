"""数据飞轮离线标注：把 FlywheelCollector 采集的难帧 → YOLO 检测训练集。

输入 = 设备 pull 下来的 `flywheel/`（`frame_*.jpg` + `metadata.jsonl`）。
`metadata.jsonl` 每行含 `file`（jpeg 名）、`corners`（8 float，物理棋盘四角 TL,TR,BR,BL）、
`fen`、`detections`（端上 YOLO 原始框：cx/cy/label/conf，全帧像素坐标）。

两种标注模式（`--mode`）：
- `standard`（旧）：只用 `m2_opening.match_standard` 标「确为标准开局」的帧（缺子≤2 补齐身份）。
  适合「开局标准、但想采多角度」的帧；中局帧全跳过。
- `yolo`（默认，YOLO-in-the-loop）：SAM 定位置 + 端上 YOLO 定身份 + 规则/判色/置信度三重校验，
  可标「任意规则可验证的中局」。保守策略：**三关全过才落标签，任何冲突跳过并记入 review.jsonl**。

`yolo` 模式三道闸门（宁缺毋滥）：
  1. **占位一致**：SAM 与 YOLO 在 90 个交点的「有无棋子」必须完全一致（防 SAM 漏检 / YOLO 漏检）。
  2. **硬规则**：每方恰 1 帅/将且在九宫、每种子数上限、仕/相/兵位置区、总数 ≤ 32。
  3. **置信度 + 外观判色**：每子 conf ≥ CONF_MIN；外观颜色（HSV 红度/灰度墨色）与 YOLO 颜色一致。

类别顺序（与端上 `YoloPieceDetector` / `train_yolo.py` 一致）：
  0–6 红 帅/仕/相/车/马/炮/兵，7–13 黑 将/士/象/车/马/炮/卒。

用法（Windows 原生 CPU，SAM ViT-B 单帧 ~50s；**禁止 GPU**）：
  python flywheel_label.py --mode yolo --dir debug_frames/flywheel --out debug_frames/flywheel_yolo
设备 pull（debuggable app）：
  adb exec-out run-as com.arability.chinese_chess_coach_phone_version tar cf - files/flywheel > flywheel.tar

已知局限：车/马/炮（无位置约束）与同侧象/马的单帧身份无法用规则裁决，仍继承 YOLO 的域差；
这类「YOLO 判错但规则分不开」的帧会因冲突被跳过，需 M3/M4 视频规则传播（运动定身份）或人工 review。
"""
import os
import sys
import json
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

from sam_piece_spike import (
    CELL, MARGIN, BOARD_W, BOARD_H, PHYSICAL_CORNERS, canonical_nodes, project,
)
from sam_detect_pieces import detect_pieces
from m2_opening import match_standard

CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']

# FEN 棋种字符 ↔ 类别基序（红 0–6；黑 +7）。
TYPE2BASE = {'k': 0, 'a': 1, 'b': 2, 'r': 3, 'n': 4, 'c': 5, 'p': 6}
BASE2CHAR = {v: k for k, v in TYPE2BASE.items()}

# 每种子数上限（红黑同）：帅/将 1、仕/士 2、相/象 2、车 2、马 2、炮 2、兵/卒 5。
MAX_COUNTS = {'k': 1, 'a': 2, 'b': 2, 'r': 2, 'n': 2, 'c': 2, 'p': 5}

PAD = 40        # 棋盘外 padding（像素）
CELL_BOX = 0.85 # standard 模式：SAM 漏检标准棋子的合成 bbox 边长（格）

# yolo 模式参数
SNAP_DIST = 0.6  # YOLO 框中心吸附到最近交点的最大距离（格）
CONF_MIN = 0.5   # 落标签的最低置信度
RED_DG = 25.0    # 外观判色：红子 R-G 下限
RED_S = 50.0     # 外观判色：红子饱和度下限
DARK_MIN = 50.0  # 外观判色：黑子墨色 grayMin 上限（< 此值判黑）


def class_id(color, t):
    """(color, type) → 14 类类别序号。color ∈ {'r','b'}，t 为小写棋种字符。"""
    return TYPE2BASE[t] + (0 if color == 'r' else 7)


def yolo_identity(label):
    """YOLO 类别序号 → (color, type)。label ∈ 0..13。"""
    color = 'r' if label <= 6 else 'b'
    return color, BASE2CHAR[label % 7]


def mask_bbox(mask):
    """bool mask → (x, y, w, h) 原图像素 bbox。空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


def validate_grid(grid):
    """校验 10×9 身份网格（每格 (color, type) 或 None）是否满足硬规则。

    与端上 `FlywheelCollector.isIllegalPosition` 同语义：双帅/缺帅、帅仕象卒越界、
    每种子数超上限、总数 > 32 均判非法。
    """
    red_count, black_count = {}, {}
    red_general = black_general = False
    total = 0
    for r in range(10):
        for f in range(9):
            g = grid[r][f]
            if g is None:
                continue
            color, t = g
            total += 1
            d = red_count if color == 'r' else black_count
            d[t] = d.get(t, 0) + 1
            if color == 'r':
                if t == 'k':
                    if red_general:
                        return False
                    red_general = True
                    if not (3 <= f <= 5 and 7 <= r <= 9):
                        return False
                elif t == 'a' and not (3 <= f <= 5 and 7 <= r <= 9):
                    return False
                elif t == 'b' and not (5 <= r <= 9):
                    return False
                elif t == 'p' and not (0 <= r <= 6):
                    return False
            else:
                if t == 'k':
                    if black_general:
                        return False
                    black_general = True
                    if not (3 <= f <= 5 and 0 <= r <= 2):
                        return False
                elif t == 'a' and not (3 <= f <= 5 and 0 <= r <= 2):
                    return False
                elif t == 'b' and not (0 <= r <= 4):
                    return False
                elif t == 'p' and not (3 <= r <= 9):
                    return False
    if total > 32:
        return False
    if not red_general or not black_general:
        return False
    for d in (red_count, black_count):
        for t, c in d.items():
            if c > MAX_COUNTS[t]:
                return False
    return True


def appearance_colors(warped):
    """外观判色：canonical 棋盘（450×500 BGR）→ 90 交点各 'r'/'b'/'?'（rank-major）。

    红靠 R-G 与饱和度，黑靠 25×25 patch 的灰度最小值 grayMin（墨色字符拉低最小值）。
    '?' 表示两者都不满足（外观不明确），不参与否决。
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    bgr = warped
    nodes = canonical_nodes()
    out = []
    for i in range(90):
        x, y = int(round(nodes[i][0])), int(round(nodes[i][1]))
        cell_g = gray[y - 12:y + 13, x - 12:x + 13]
        dark = float(cell_g.min())
        b = float(bgr[y - 12:y + 13, x - 12:x + 13, 0].mean())
        g = float(bgr[y - 12:y + 13, x - 12:x + 13, 1].mean())
        r_ = float(bgr[y - 12:y + 13, x - 12:x + 13, 2].mean())
        sat = float(hsv[y - 12:y + 13, x - 12:x + 13, 1].mean())
        if (r_ - g > RED_DG) and (sat > RED_S):
            out.append('r')
        elif dark < DARK_MIN:
            out.append('b')
        else:
            out.append('?')
    return out


def parse_args():
    def opt(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    flywheel_dir = os.path.abspath(opt("--dir", os.path.join(DBG, "flywheel")))
    out_dir = os.path.abspath(opt("--out", os.path.join(DBG, "flywheel_yolo")))
    checkpoint = opt("--sam", CHECKPOINT)
    max_frames = int(opt("--max", "0"))
    mode = opt("--mode", "yolo")
    do_color = "--no-color-check" not in sys.argv
    return flywheel_dir, out_dir, checkpoint, max_frames, mode, do_color


def setup_frame(img, meta):
    """每帧几何准备：corners、单应 M（image→canonical）、90 交点原图坐标、格尺寸。"""
    corners = np.array(meta["corners"], dtype=np.float32).reshape(4, 2)  # TL,TR,BR,BL
    M = cv2.getPerspectiveTransform(corners, PHYSICAL_CORNERS)
    nodes_img = project(canonical_nodes(), np.linalg.inv(M))

    cell_w, cell_h, cell_areas = [], [], []
    for r in range(10):
        for f in range(9):
            i = r * 9 + f
            p = nodes_img[i]
            fr = f + 1 if f < 8 else f - 1
            rd = r + 1 if r < 9 else r - 1
            cw = float(np.linalg.norm(p - nodes_img[r * 9 + fr]))
            ch = float(np.linalg.norm(p - nodes_img[rd * 9 + f]))
            cell_w.append(cw); cell_h.append(ch); cell_areas.append(cw * ch)
    return corners, M, nodes_img, cell_w, cell_h, cell_areas


def label_standard(detected, nodes_img, cell_w, cell_h):
    """standard 模式：标准开局身份。返回 (boxes 或 None, skip 原因或 None)。"""
    occ = np.zeros((10, 9), dtype=int)
    for (f, r) in detected:
        occ[r][f] = 1
    is_std, grid = match_standard(occ)
    if not is_std:
        return None, f"非标准开局（检出 {int(occ.sum())} 子）"

    boxes = []
    for r in range(10):
        for f in range(9):
            ident = grid[r][f]
            if ident is None:
                continue
            cls = class_id(*ident)
            i = r * 9 + f
            b = mask_bbox(detected[(f, r)]["mask"]) if (f, r) in detected else None
            if b is None:
                px, py = nodes_img[i]
                bw, bh = CELL_BOX * cell_w[i], CELL_BOX * cell_h[i]
                b = (px - bw / 2.0, py - bh / 2.0, bw, bh)
            boxes.append((cls,) + tuple(b))
    return boxes, None


def label_yolo(detected, meta, nodes_img, cell_areas, warped, do_color):
    """yolo 模式：SAM 位置 + YOLO 身份 + 规则/判色/置信度校验。返回 (boxes 或 None, 原因或 None)。"""
    # 1) YOLO 框中心吸附到最近交点（同一交点保留最高置信度；远处孤儿忽略）。
    yolo_grid = [None] * 90  # (label, conf)
    for d in meta.get("detections", []):
        cx, cy, label, conf = d["cx"], d["cy"], int(d["label"]), float(d["conf"])
        best_i = int(np.argmin([np.hypot(cx - nodes_img[i][0], cy - nodes_img[i][1])
                                for i in range(90)]))
        best_d = float(np.hypot(cx - nodes_img[best_i][0], cy - nodes_img[best_i][1]))
        if best_d > SNAP_DIST * np.sqrt(cell_areas[best_i]):
            continue
        cur = yolo_grid[best_i]
        if cur is None or conf > cur[1]:
            yolo_grid[best_i] = (label, conf)

    # 2) 占位：SAM 为准。YOLO 假子（SAM 空但 YOLO 有）直接丢弃，不否决整帧；
    #    YOLO 漏检（SAM 有但 YOLO 无）则无法定身份，跳过整帧。
    for i in range(90):
        f, r = i % 9, i // 9
        sam_occ = (f, r) in detected
        if sam_occ and yolo_grid[i] is None:
            return None, f"YOLO 漏检 @({f},{r})"
        if not sam_occ:
            yolo_grid[i] = None  # 丢弃 YOLO 假子

    # 3) 置信度 + 身份网格。
    grid = [[None] * 9 for _ in range(10)]
    for i in range(90):
        if yolo_grid[i] is None:
            continue
        label, conf = yolo_grid[i]
        if conf < CONF_MIN:
            f, r = i % 9, i // 9
            return None, f"低置信 @({f},{r}) conf={conf:.2f}"
        f, r = i % 9, i // 9
        grid[r][f] = yolo_identity(label)

    # 4) 硬规则。
    if not validate_grid(grid):
        return None, "规则校验失败"

    # 5) 外观判色交叉校验。
    if do_color:
        ap = appearance_colors(warped)
        for i in range(90):
            if yolo_grid[i] is None:
                continue
            yc = 'r' if yolo_grid[i][0] <= 6 else 'b'
            if ap[i] != '?' and ap[i] != yc:
                f, r = i % 9, i // 9
                return None, f"颜色冲突 @({f},{r}) 外观={ap[i]} yolo={yc}"

    # 6) 出框：一律用 SAM mask bbox（比 YOLO 中心点准）。
    boxes = []
    for i in range(90):
        if yolo_grid[i] is None:
            continue
        cls = yolo_grid[i][0]
        f, r = i % 9, i // 9
        b = mask_bbox(detected[(f, r)]["mask"])
        if b is None:
            return None, f"缺 mask @({f},{r})"
        boxes.append((cls,) + tuple(b))
    return boxes, None


def write_frame(img, corners, boxes, img_dir, lbl_dir, out_name):
    """裁剪棋盘区域（四角外扩 PAD），写 images/labels。boxes = [(cls, x, y, w, h), ...]。"""
    H, W = img.shape[:2]
    x0 = max(0, int(corners[:, 0].min()) - PAD)
    x1 = min(W, int(corners[:, 0].max()) + PAD)
    y0 = max(0, int(corners[:, 1].min()) - PAD)
    y1 = min(H, int(corners[:, 1].max()) + PAD)
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return False

    lines = []
    for cls, bx, by_, bw, bh in boxes:
        cx = min(max((bx - x0 + bw / 2.0) / cw, 0.0), 1.0)
        cy = min(max((by_ - y0 + bh / 2.0) / ch, 0.0), 1.0)
        wn = min(max(bw / cw, 0.0), 1.0)
        hn = min(max(bh / ch, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

    crop = img[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(img_dir, out_name + ".jpg"), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(lbl_dir, out_name + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return True


def main():
    flywheel_dir, out_dir, checkpoint, max_frames, mode, do_color = parse_args()
    meta_path = os.path.join(flywheel_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        print(f"[err] 找不到 {meta_path}，先 adb pull 采集目录。")
        return

    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    metas = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metas.append(json.loads(line))

    t0 = time.time()
    print(f"加载 SAM ViT-B ...（mode={mode} color_check={do_color}）", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    print(f"模型就绪 {time.time() - t0:.1f}s", flush=True)

    written = 0
    skipped = 0
    reviews = []
    for meta in metas:
        name = meta.get("file")
        if max_frames and written >= max_frames:
            print(f"[stop] 已达 --max {max_frames} 张，停止。")
            break
        jpeg = os.path.join(flywheel_dir, name + ".jpg")
        if not os.path.exists(jpeg):
            print(f"  [warn] 缺图 {jpeg}，跳过", flush=True)
            skipped += 1
            continue
        img = cv2.imread(jpeg)
        if img is None:
            print(f"  [warn] 读图失败 {jpeg}，跳过", flush=True)
            skipped += 1
            continue

        corners, M, nodes_img, cell_w, cell_h, cell_areas = setup_frame(img, meta)

        # SAM 逐子检测（每帧重建 image embedding）
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detected = detect_pieces(predictor, nodes_img, cell_w, cell_h, cell_areas)

        if mode == "standard":
            boxes, reason = label_standard(detected, nodes_img, cell_w, cell_h)
        else:
            warped = cv2.warpPerspective(img, M, (BOARD_W, BOARD_H))
            boxes, reason = label_yolo(detected, meta, nodes_img, cell_areas, warped, do_color)

        if reason is not None:
            print(f"  [skip] {name}: {reason}", flush=True)
            skipped += 1
            if mode == "yolo":
                reviews.append({"file": name, "reason": reason})
            continue

        out_name = "flywheel_" + name
        if write_frame(img, corners, boxes, img_dir, lbl_dir, out_name):
            written += 1
            print(f"  {out_name}: {len(boxes)} boxes", flush=True)
        else:
            print(f"  [warn] {name}: crop 非法，跳过", flush=True)
            skipped += 1

    # data.yaml
    yaml_lines = [
        f"# 象棋棋子检测数据集（飞轮标注 mode={mode}，{written} 张图）",
        f"path: {out_dir.replace(os.sep, '/')}",
        "train: images",
        "val: images",
        "",
        "nc: 14",
        "names: [" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]",
    ]
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(yaml_lines) + "\n")

    # review.jsonl（yolo 模式冲突帧）
    if reviews:
        with open(os.path.join(out_dir, "review.jsonl"), "w", encoding="utf-8") as fh:
            for r in reviews:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[review] {len(reviews)} 个冲突帧 -> {out_dir}/review.jsonl（供人工/视频管线）", flush=True)

    print(f"\n[done] 标注 {written} 张，跳过 {skipped} 张 -> {out_dir}", flush=True)
    print(f"总耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
