"""M4 身份传播：初始身份 + 象棋规则 + 多帧投票占用 → 全视频逐子身份。

输入：m4_detect_multi.json（每段多帧 detect_pieces 投票后的 (file,rank) 集合）
输出：走子序列 + 每段棋盘身份（JSON）。

核心：初始 32 子身份已知。每段之间至多一步合法走子。枚举当前方所有合法走法，
用「走子专用打分」（源格清空 + 目标格被占）找最匹配的一步；低于阈值判为无走子（手噪声）。
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xiangqi import parse_identity_table, legal_moves, apply_move, NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DET_JSON = os.path.normpath(os.path.join(
    HERE, "..", "..", "debug_frames", "m3_debug", "m4_detect_multi.json"))
OUT_JSON = os.path.normpath(os.path.join(
    HERE, "..", "..", "debug_frames", "m3_debug", "m4_propagate_result.json"))

MIN_PIECES = 28      # 检测子数少于此视为遮挡段，跳过走子判定
ACCEPT_SCORE = 1.0   # 走子分 >= 此值才接受（+2=源清空且目标被占，+1=仅其一，0=抵消，负=矛盾）


def move_score(frm, to, board, detected):
    s = 0.0
    s += 1.0 if frm not in detected else -1.0   # 源格应清空
    s += 1.0 if to in detected else -1.0        # 目标格应被占（含吃子后仍占）
    return s


def board_to_str(board):
    rows = []
    for r in range(10):
        row = []
        for f in range(9):
            p = board.get((f, r))
            row.append(NAMES[p] if p else ".")
        rows.append(f"rank{r}: " + "".join(row))
    return "\n".join(rows)


def main():
    det = json.load(open(DET_JSON))
    segs = det["segs"]
    cells_list = [set(tuple(c) for c in cells) for cells in det["cells"]]

    board = parse_identity_table()
    side = 'B'   # 红炮 (7,7)->(4,7) 后轮到黑方

    moves = []
    states = [dict(board)]

    print(f"初始（{len(board)} 子）：")
    print(board_to_str(board))
    print()

    for k in range(1, len(cells_list)):
        detected = cells_list[k]
        if len(detected) < MIN_PIECES:
            print(f"seg{k} [{segs[k][0]},{segs[k][1]}]: 遮挡段({len(detected)}子) 跳过")
            states.append(dict(board))
            continue

        legal = legal_moves(board, side)
        best = None
        best_score = -1e9
        for (frm, to) in legal:
            sc = move_score(frm, to, board, detected)
            if sc > best_score:
                best_score = sc
                best = (frm, to)

        if best is None or best_score < ACCEPT_SCORE:
            print(f"seg{k} [{segs[k][0]},{segs[k][1]}]: 无走子（best={best_score:.0f}）")
            states.append(dict(board))
            continue

        frm, to = best
        piece = board[frm]
        captured = board.get(to)
        board = apply_move(board, frm, to)
        side = 'R' if side == 'B' else 'B'
        moves.append((k, frm, to, piece, captured))
        states.append(dict(board))
        cap = f" 吃{NAMES[captured]}" if captured else ""
        print(f"seg{k} [{segs[k][0]},{segs[k][1]}]: {NAMES[piece]}({frm[0]},{frm[1]})->"
              f"({to[0]},{to[1]}){cap}  [score={best_score:.0f}]")

    print(f"\n===== 共 {len(moves)} 步走子 =====")
    for k, frm, to, piece, captured in moves:
        cap = f" 吃{NAMES[captured]}" if captured else ""
        print(f"  seg{k}: {NAMES[piece]}{frm}->{to}{cap}")

    out = {
        "segs": segs,
        "moves": [{"seg": k, "from": list(frm), "to": list(to),
                   "piece": list(piece), "captured": list(captured) if captured else None}
                  for (k, frm, to, piece, captured) in moves],
        "states": [{"seg": i, "board": {f"{f},{r}": list(p) for (f, r), p in st.items()}}
                   for i, st in enumerate(states)],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    print(f"\n[done] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
