"""类别契约 + 布局生成 —— 单一事实来源，必须与 Kotlin 端保持一致。

类别映射（与 ``TFLitePieceRecognizer`` 的 labelToPiece 对齐）：
  0      = 空位
  1..7   = 红方 帅/仕/相/马/车/炮/兵（对应 ``PieceType.entries``: GENERAL..PAWN）
  8..14  = 黑方 将/士/象/马/车/炮/卒

颜色/棋种换算：
  label = 1 + type_index + (7 if black else 0)
  type_index = (label - 1) % 7
  color = BLACK if label >= 8 else RED
"""
from __future__ import annotations

import numpy as np

EMPTY = 0
NUM_CLASSES = 15

# PieceType.entries 顺序: GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN
GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN = range(7)
TYPE_NAMES = ["GENERAL", "ADVISOR", "ELEPHANT", "HORSE", "CHARIOT", "CANNON", "PAWN"]

# 每种棋子的 (红字, 黑字)
GLYPHS = [
    ("帅", "将"),
    ("仕", "士"),
    ("相", "象"),
    ("马", "马"),
    ("车", "车"),
    ("炮", "炮"),
    ("兵", "卒"),
]

RED, BLACK = "red", "black"


def label_of(color: str, type_index: int) -> int:
    """由颜色与棋种索引得到类别标签。"""
    assert 0 <= type_index < 7, type_index
    return 1 + type_index + (7 if color == BLACK else 0)


def type_index_of(label: int) -> int:
    return (label - 1) % 7


def color_of(label: int) -> str:
    return BLACK if label >= 8 else RED


def standard_initial_layout() -> dict:
    """标准开局：黑方在上（rank 0..4），红方在下（rank 6..9）。

    返回 ``{(file, rank): (color, type_index)}``。
    """
    back = [CHARIOT, HORSE, ELEPHANT, ADVISOR, GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT]
    layout: dict = {}

    for f, t in enumerate(back):
        layout[(f, 0)] = (BLACK, t)
        layout[(f, 9)] = (RED, t)

    layout[(1, 2)] = (BLACK, CANNON)
    layout[(7, 2)] = (BLACK, CANNON)
    layout[(1, 7)] = (RED, CANNON)
    layout[(7, 7)] = (RED, CANNON)

    for f in (0, 2, 4, 6, 8):
        layout[(f, 3)] = (BLACK, PAWN)
        layout[(f, 6)] = (RED, PAWN)

    return layout


def random_midgame_layout(rng: np.random.Generator) -> dict:
    """随机中局布局：双方将/帅各一枚在各自九宫内，其余按颜色偏置随机摆放，
    并按各棋种的合法上限（将1 仕/相/马/车/炮各2 兵5）控制数量，保证局面基本合法。

    返回 ``{(file, rank): (color, type_index)}``。
    """
    # 每种棋子的上限（GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN）
    MAX_COUNT = (1, 2, 2, 2, 2, 2, 5)

    layout: dict = {}
    counts = {RED: [0] * 7, BLACK: [0] * 7}

    def place(color, t, f, r):
        layout[(f, r)] = (color, t)
        counts[color][t] += 1

    # 红帅在红方九宫（file 3..5, rank 7..9）；黑将在黑方九宫（file 3..5, rank 0..2）
    place(RED, GENERAL, int(rng.integers(3, 6)), int(rng.integers(7, 10)))
    place(BLACK, GENERAL, int(rng.integers(3, 6)), int(rng.integers(0, 3)))

    n_target = int(rng.integers(10, 32))
    placed = 2
    attempts = 0
    while placed < n_target and attempts < 1000:
        attempts += 1
        f = int(rng.integers(0, 9))
        color = RED if rng.random() < 0.5 else BLACK
        # 按颜色偏置行号（红多在下半，黑多在上半），并允许过河交叉
        if color == RED:
            r = int(np.clip(rng.normal(7.0, 2.0), 0, 9))
        else:
            r = int(np.clip(rng.normal(2.0, 2.0), 0, 9))
        if (f, r) in layout:
            continue
        t = int(rng.integers(1, 7))  # 不再额外放将/帅
        if counts[color][t] >= MAX_COUNT[t]:
            continue
        place(color, t, f, r)
        placed += 1

    return layout


def layout_to_grid(layout: dict) -> np.ndarray:
    """把布局字典转成 10×9 的标签矩阵（rank-major，grid[rank][file]），空位为 0。"""
    grid = np.zeros((10, 9), dtype=np.int32)
    for (f, r), (color, t) in layout.items():
        grid[r, f] = label_of(color, t)
    return grid
