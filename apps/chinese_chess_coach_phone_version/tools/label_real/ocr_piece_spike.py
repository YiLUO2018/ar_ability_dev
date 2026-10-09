"""OCR 单字读字验证：已知框 → warp → crop → RapidOCR rec 读 11 个汉字。

目的：验证「用开源 OCR 读棋子单字」能否辅助/替代 YOLO 定身份（尤其车/马/炮、象/马/将混淆）。
输入：board_21.png（斜拍真实棋盘）+ m5_bboxes.csv（SAM 已检出的 38 个棋子 (file,rank)）。
做法：用 SAM 已确认四角 warp 到 canonical 正视角 → 按交点 crop 单字 patch → 灰度+CLAHE+放大
      → RapidOCR 只跑 rec（use_det=False）读字 → 约束到 11 字词表。
输出：board_21_ocr.png（10×9 拼图，每格单字 patch + 识别结果）+ 控制台逐子结果。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sam_piece_spike import (
    BOARD_CORNERS, PHYSICAL_CORNERS, CELL, MARGIN, BOARD_W, BOARD_H, canonical_nodes,
)

from rapidocr import RapidOCR

DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))
IMG = os.path.join(DBG, "board_21.png")
CSV = os.path.join(DBG, "m5_bboxes.csv")
OUT = os.path.join(DBG, "board_21_ocr.png")

# 11 个不同汉字（红黑合并去重；车/马/炮红黑同字）。
CHARS = ['帅', '仕', '相', '车', '马', '炮', '兵', '将', '士', '象', '卒']

PATCH = 46      # crop 边长（px，canonical 内，略大于棋子直径 ~45px）
SIZE = 96       # 放大后喂 rec 的边长


def preprocess(patch_bgr):
    """单字 patch → 灰度 + CLAHE 对比度增强 + 放大（白底深字更利于 rec）。"""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def nearest_char(text):
    """OCR 输出文本 → 11 字词表里最接近的字。空/不匹配返回 '?'。"""
    if not text:
        return '?'
    for ch in text:
        if ch in CHARS:
            return ch
    return '?'


def read_char(engine, patch_bgr):
    """读一个单字 patch，返回 (识别字, 置信度)。"""
    img = preprocess(patch_bgr)
    result, _ = engine(img, use_det=False, use_cls=False, use_rec=True)
    # RapidOCR rec-only 输出：可能是 [{text, score}] 或 [[text, score], ...]，防御性解析。
    text, score = '', 0.0
    items = result if isinstance(result, (list, tuple)) else [result]
    for it in items:
        if isinstance(it, dict):
            t = it.get('text', '')
            s = float(it.get('score', 0.0))
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            t, s = str(it[0]), float(it[1])
        else:
            continue
        if s > score:
            text, score = t, s
    return nearest_char(text), score


def main():
    img = cv2.imread(IMG)
    M = cv2.getPerspectiveTransform(BOARD_CORNERS, PHYSICAL_CORNERS)
    warped = cv2.warpPerspective(img, M, (BOARD_W, BOARD_H))

    # 读 SAM 检出的棋子 (file, rank)
    occ = np.zeros((10, 9), dtype=bool)
    with open(CSV, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            f, r = int(line.split(',')[0]), int(line.split(',')[1])
            occ[r][f] = True
    n_pieces = int(occ.sum())
    print(f"棋子数 {n_pieces}（来自 {os.path.basename(CSV)}）", flush=True)

    print("加载 RapidOCR ...", flush=True)
    engine = RapidOCR()
    print("引擎就绪", flush=True)

    nodes = canonical_nodes()  # rank-major, i = r*9+f

    # 拼图：每格 patch（96×96）+ 底部标注条
    cell = SIZE
    label_h = 26
    board = np.full((10 * (cell + label_h), 9 * cell, 3), 255, dtype=np.uint8)

    correct_hint = {'帅': 0, '仕': 0, '相': 0, '车': 0, '马': 0, '炮': 0, '兵': 0,
                    '将': 0, '士': 0, '象': 0, '卒': 0, '?': 0}
    results = {}
    for r in range(10):
        for f in range(9):
            if not occ[r][f]:
                continue
            x, y = nodes[r * 9 + f]
            x0, y0 = int(round(x - PATCH / 2)), int(round(y - PATCH / 2))
            patch = warped[max(0, y0):y0 + PATCH, max(0, x0):x0 + PATCH]
            ch, score = read_char(engine, patch)
            results[(f, r)] = (ch, score)
            correct_hint[ch] = correct_hint.get(ch, 0) + 1

            # 拼图：patch 放大到 96×96 放到对应格
            disp = cv2.resize(patch, (cell, cell), interpolation=cv2.INTER_CUBIC)
            board[r * (cell + label_h):r * (cell + label_h) + cell, f * cell:(f + 1) * cell] = disp
            # 标注条
            ly = r * (cell + label_h) + cell
            board[ly:ly + label_h, f * cell:(f + 1) * cell] = (235, 235, 235)
            txt = ch if ch != '?' else f"?({score:.2f})"
            cv2.putText(board, txt, (f * cell + 6, ly + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(OUT, board)
    print(f"\n[vis] 已保存 {OUT}\n", flush=True)

    print("逐子识别结果（file,rank -> 字 置信度）:")
    for (f, r), (ch, score) in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  ({f},{r}) -> {ch}  {score:.3f}")

    print("\n字频统计:")
    for ch in ['帅', '仕', '相', '车', '马', '炮', '兵', '将', '士', '象', '卒', '?']:
        if correct_hint.get(ch, 0):
            print(f"  {ch}: {correct_hint[ch]}")


if __name__ == "__main__":
    main()
