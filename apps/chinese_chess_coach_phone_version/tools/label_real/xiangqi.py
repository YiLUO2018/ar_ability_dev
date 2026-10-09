"""中国象棋走法规则引擎（轻量、无 UI，供标注/身份传播/记谱复用）。

棋盘坐标：(file, rank)，file 0-8（图左→右），rank 0-9（图顶→底）。
黑方在 rank 0-4（顶），红方在 rank 5-9（底）。黑向下（rank+）走，红向上（rank-）走。

棋子编码：color ∈ {'R','B'}（红/黑），type ∈ {'K','A','E','R','H','C','P'}
  K=帅/将  A=仕/士  E=相/象  R=车  H=马  C=炮  P=兵/卒
"""
from __future__ import annotations

# (color, type) 的合法中文名，用于记谱/输出
NAMES = {
    ('R', 'K'): '帅', ('R', 'A'): '仕', ('R', 'E'): '相', ('R', 'R'): '车',
    ('R', 'H'): '马', ('R', 'C'): '炮', ('R', 'P'): '兵',
    ('B', 'K'): '将', ('B', 'A'): '士', ('B', 'E'): '象', ('B', 'R'): '车',
    ('B', 'H'): '马', ('B', 'C'): '炮', ('B', 'P'): '卒',
}


def in_board(f, r):
    return 0 <= f <= 8 and 0 <= r <= 9


def in_palace(f, r, color):
    if color == 'B':
        return 3 <= f <= 5 and 0 <= r <= 2
    return 3 <= f <= 5 and 7 <= r <= 9


def own_side(f, r, color):
    """棋子是否在本方半场（用于相/象不能过河）。"""
    return (0 <= r <= 4) if color == 'B' else (5 <= r <= 9)


def crossed_river(r, color):
    """兵/卒是否已过河。黑卒过河 = rank>=5；红兵过河 = rank<=4。"""
    return r >= 5 if color == 'B' else r <= 4


def _slide_moves(board, f, r, color, type_, dirs):
    """车/炮沿正交方向滑动。炮 capture 需隔子。返回 [(from, to)]。"""
    moves = []
    for df, dr in dirs:
        cf, cr = f + df, r + dr
        jumped = False
        while in_board(cf, cr):
            occ = board.get((cf, cr))
            if occ is None:
                if type_ != 'C' or not jumped:
                    moves.append(((f, r), (cf, cr)))
            else:
                if type_ == 'C':
                    if not jumped:
                        jumped = True          # 炮：第一个子作炮架
                    else:
                        if occ[0] != color:    # 隔一子吃
                            moves.append(((f, r), (cf, cr)))
                        break
                else:
                    if occ[0] != color:
                        moves.append(((f, r), (cf, cr)))
                    break
            cf += df; cr += dr
    return moves


def legal_moves(board, color):
    """返回 color 方所有合法落点 [(from, to)]。"""
    moves = []
    for (f, r), (c, t) in board.items():
        if c != color:
            continue
        if t == 'K':
            for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nf, nr = f + df, r + dr
                if not (in_palace(nf, nr, color) and in_board(nf, nr)):
                    continue
                occ = board.get((nf, nr))
                if occ is None or occ[0] != color:
                    moves.append(((f, r), (nf, nr)))
        elif t == 'A':
            for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                nf, nr = f + df, r + dr
                if not in_palace(nf, nr, color):
                    continue
                occ = board.get((nf, nr))
                if occ is None or occ[0] != color:
                    moves.append(((f, r), (nf, nr)))
        elif t == 'E':
            for df, dr in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                nf, nr = f + df, r + dr
                if not (in_board(nf, nr) and own_side(nf, nr, color)):
                    continue
                ef, er = f + df // 2, r + dr // 2   # 象眼
                if board.get((ef, er)) is not None:
                    continue
                occ = board.get((nf, nr))
                if occ is None or occ[0] != color:
                    moves.append(((f, r), (nf, nr)))
        elif t == 'R':
            moves += _slide_moves(board, f, r, color, 'R', ((1, 0), (-1, 0), (0, 1), (0, -1)))
        elif t == 'C':
            moves += _slide_moves(board, f, r, color, 'C', ((1, 0), (-1, 0), (0, 1), (0, -1)))
        elif t == 'H':
            for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
                nf, nr = f + df, r + dr
                if not in_board(nf, nr):
                    continue
                # 马腿：先直走一步
                lf, lr = f + (df // 2 if abs(df) == 2 else 0), r + (dr // 2 if abs(dr) == 2 else 0)
                if board.get((lf, lr)) is not None:
                    continue
                occ = board.get((nf, nr))
                if occ is None or occ[0] != color:
                    moves.append(((f, r), (nf, nr)))
        elif t == 'P':
            fwd = 1 if color == 'B' else -1
            # 前进
            nf, nr = f, r + fwd
            if in_board(nf, nr):
                occ = board.get((nf, nr))
                if occ is None or occ[0] != color:
                    moves.append(((f, r), (nf, nr)))
            # 过河后可横走
            if crossed_river(r, color):
                for df in (-1, 1):
                    nf, nr = f + df, r
                    if in_board(nf, nr):
                        occ = board.get((nf, nr))
                        if occ is None or occ[0] != color:
                            moves.append(((f, r), (nf, nr)))
    return moves


def apply_move(board, frm, to):
    """返回新 board（含吃子）。不校验合法性。"""
    nb = dict(board)
    piece = nb.pop(frm)
    nb[to] = piece
    return nb


def parse_identity_table():
    """从 annotation_identity.md 已知结论硬编码初始身份（32 子）。

    返回 board dict {(f,r): (color,type)}。
    """
    B, R = 'B', 'R'
    board = {
        (0, 0): (B, 'R'), (1, 0): (B, 'H'), (2, 0): (B, 'E'), (3, 0): (B, 'A'),
        (4, 0): (B, 'K'), (5, 0): (B, 'A'), (6, 0): (B, 'E'), (7, 0): (B, 'H'),
        (8, 0): (B, 'R'),
        (1, 2): (B, 'C'), (7, 2): (B, 'C'),
        (0, 3): (B, 'P'), (2, 3): (B, 'P'), (4, 3): (B, 'P'), (6, 3): (B, 'P'),
        (8, 3): (B, 'P'),
        (0, 6): (R, 'P'), (2, 6): (R, 'P'), (4, 6): (R, 'P'), (6, 6): (R, 'P'),
        (8, 6): (R, 'P'),
        (1, 7): (R, 'C'), (4, 7): (R, 'C'),
        (0, 9): (R, 'R'), (1, 9): (R, 'H'), (2, 9): (R, 'E'), (3, 9): (R, 'A'),
        (4, 9): (R, 'K'), (5, 9): (R, 'A'), (6, 9): (R, 'E'), (7, 9): (R, 'H'),
        (8, 9): (R, 'R'),
    }
    return board
