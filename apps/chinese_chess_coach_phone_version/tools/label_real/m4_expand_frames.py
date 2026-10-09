"""M4 第 4 步：把段级身份扩散到每一帧（M4 最终交付）。

读 m4_propagate_result.json（每段 board 身份 + 段边界），把段内每帧映射到该段身份，
输出 per-frame 身份 CSV：frame, 然后 90 列（每个交点的棋子，空为 '.'）。

burst（段间运动）帧标记为 '?'（过渡，无干净身份），供 M5 排除。
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
RES_JSON = os.path.normpath(os.path.join(
    HERE, "..", "..", "debug_frames", "m3_debug", "m4_propagate_result.json"))
OUT_CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "debug_frames", "m4_identity_per_frame.csv"))

# 编码：红帅仕相车马炮兵 / 黑将士象车马炮卒 → 单字符（便于 CSV 肉眼核对）
CODE = {('R', 'K'): '帅', ('R', 'A'): '仕', ('R', 'E'): '相', ('R', 'R'): '车',
        ('R', 'H'): '马', ('R', 'C'): '炮', ('R', 'P'): '兵',
        ('B', 'K'): '将', ('B', 'A'): '士', ('B', 'E'): '象', ('B', 'R'): '车',
        ('B', 'H'): '马', ('B', 'C'): '炮', ('B', 'P'): '卒'}


def main():
    res = json.load(open(RES_JSON))
    segs = res["segs"]
    states = res["states"]

    # 每个交点 → 当前身份（按段推进）
    # states[i]["board"] 是 { "f,r": [color,type] }
    N = 3384  # 总帧数
    with open(OUT_CSV, "w", encoding="utf-8") as fh:
        fh.write("frame," + ",".join(str(i) for i in range(90)) + "\n")
        seg_idx = 0
        for frame in range(N):
            # 找到 frame 所属段
            while seg_idx < len(segs) - 1 and frame > segs[seg_idx][1]:
                seg_idx += 1
            s, e = segs[seg_idx]
            if s <= frame <= e:
                board = states[seg_idx]["board"]
                row = ["."] * 90
                for key, piece in board.items():
                    f, r = key.split(",")
                    i = int(r) * 9 + int(f)
                    row[i] = CODE[tuple(piece)]
                fh.write(f"{frame}," + ",".join(row) + "\n")
            else:
                # 段间过渡
                fh.write(f"{frame}," + ",".join(["?"] * 90) + "\n")
    print(f"[done] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
