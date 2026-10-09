"""M2：标准开局识别（占位掩码匹配 → 判定是否标准开局，是则 32 子身份免费到手）。

输入 = 占位网格（9×10 rank-major 0/1，来自 M1/`m0_metrics.csv`），输出 = 是否标准开局 +
（若是）32 子的 (color, type) 身份。

匹配规则（容忍 M0 的偏移漏检，但不容忍「多出来的棋子」）：
- 多子（occ=1 但标准=0）→ 说明有子移动过，非标准开局 → 拒绝；
- 缺子（occ=0 但标准=1）→ 可能是 M0 漏检，容忍（≤ 阈值）。
"""
import os
import sys

import numpy as np

# 标准开局 FEN（rank0 黑方顶 → rank9 红方底）
STD_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def parse_fen(fen):
    """FEN 局面串 → grid[rank][file] = (color, type) 或 None。color ∈ {'r','b'}，type ∈ 小写棋种。"""
    grid = [[None] * 9 for _ in range(10)]
    ranks = fen.split()[0].split("/")
    for r, rank_str in enumerate(ranks):
        f = 0
        for ch in rank_str:
            if ch.isdigit():
                f += int(ch)
            else:
                color = "r" if ch.isupper() else "b"
                grid[r][f] = (color, ch.lower())
                f += 1
    return grid


STD_GRID = parse_fen(STD_FEN)
STD_OCC = np.array([[1 if STD_GRID[r][f] else 0 for f in range(9)] for r in range(10)])


def match_standard(occ, max_missing=2):
    """匹配标准开局。

    occ: 9×10 rank-major 0/1 占位。返回 (is_std, grid)：
    - is_std=True 时 grid 为 32 子身份（含用标准位置补齐的漏检子）；
    - is_std=False 时 grid 为 None。
    """
    occ = np.asarray(occ, dtype=int)
    if occ.shape != (10, 9):
        return False, None
    extra = (occ == 1) & (STD_OCC == 0)          # 多出来的子 → 非标准
    missing = (occ == 0) & (STD_OCC == 1)         # 漏检的子 → 可容忍
    if extra.any() or missing.sum() > max_missing:
        return False, None
    # 标准开局：把漏检位置用标准身份补齐，直接得到 32 子身份
    grid = [[STD_GRID[r][f] for f in range(9)] for r in range(10)]
    return True, grid


def occupancy_from_csv(csv_path):
    """从 m0_metrics.csv 读 90 交点占位，返回 9×10 rank-major 0/1。"""
    occ = np.zeros((10, 9), dtype=int)
    with open(csv_path, encoding="utf-8") as fh:
        fh.readline()  # header
        for line in fh:
            file_, rank, is_piece = line.split(",")[:3]
            occ[int(rank)][int(file_)] = int(is_piece)
    return occ


def main():
    # 测试 1：标准开局自身 → 应匹配
    ok, grid = match_standard(STD_OCC)
    print(f"[标准开局自匹配] is_std={ok}  (期望 True)")

    # 测试 2：标准开局漏检 2 子 → 应仍匹配（容忍）
    occ2 = STD_OCC.copy()
    occ2[3][4] = 0   # 黑卒 file4/rank3 漏检
    occ2[6][4] = 0   # 红兵 file4/rank6 漏检
    ok2, _ = match_standard(occ2)
    print(f"[标准开局漏检 2 子] is_std={ok2}  (期望 True)")

    # 测试 3：board_21 占位（非标准，38 子/检出 35）→ 应拒绝
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug_frames", "m0_metrics.csv"
    )
    board21 = occupancy_from_csv(csv_path)
    n = int(board21.sum())
    ok3, _ = match_standard(board21)
    print(f"[board_21] 检出 {n} 子，is_std={ok3}  (期望 False，非标准开局)")
    # 打印 board_21 与标准的差异（多子/缺子）
    extra = (board21 == 1) & (STD_OCC == 0)
    missing = (board21 == 0) & (STD_OCC == 1)
    print(f"  多出的子位置: {sorted(zip(*np.nonzero(extra)))}")
    print(f"  相对标准缺的子: {sorted(zip(*np.nonzero(missing)))}")


if __name__ == "__main__":
    main()
