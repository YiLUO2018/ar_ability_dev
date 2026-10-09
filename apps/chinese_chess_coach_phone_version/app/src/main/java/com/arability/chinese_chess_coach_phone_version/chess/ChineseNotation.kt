package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.abs

/**
 * 把一步走法转成标准中国象棋记谱（「炮二平五」「马2进3」「前炮平五」）。
 *
 * 规则（见《项目全景梳理与实施方案.md》§6.5.2）：
 * - 红列号 一~九 从右到左（file8=一 … file0=九）；黑列号 1~9 从左到右（file0=1 … file8=9）。
 * - 动词：平（同行）/ 进（向对方，红 rank 减小、黑 rank 增大）/ 退（反向）。
 * - 数字：车/炮/帅/兵 = 步数（进/退）或目标列号（平）；马/象/仕 = 目标列号。
 * - 同列同色同名双子时用「前/后」代替源列号（前=更靠近对方）。
 */
object ChineseNotation {

    /** 红方列号（按 file 0..8 索引）。 */
    private val RED_FILE_NUM = arrayOf("九", "八", "七", "六", "五", "四", "三", "二", "一")

    /** 步数中文数字（进/退的步数，红方用中文、黑方用阿拉伯）。 */
    private val CN_NUM = arrayOf("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")

    fun of(board: Board, move: Move): String {
        val color = move.piece.color
        val name = move.piece.name

        // 动词 + 数字
        val verb: Char
        val num: String
        if (move.fromRank == move.toRank) {
            verb = '平'
            num = columnOf(color, move.toFile)
        } else {
            val forward = if (color == PieceColor.RED) -1 else +1
            verb = if ((move.toRank - move.fromRank) * forward > 0) '进' else '退'
            num = when (move.piece.type) {
                PieceType.HORSE, PieceType.ELEPHANT, PieceType.ADVISOR -> columnOf(color, move.toFile)
                else -> stepNum(color, abs(move.toRank - move.fromRank))
            }
        }

        // 源列（或前/后消歧）；前/后置于棋子名前，列号置于其后。
        val src = sourceOf(board, move, color)
        return if (src == "前" || src == "后") "$src$name$verb$num" else "$name$src$verb$num"
    }

    private fun stepNum(color: PieceColor, step: Int): String =
        if (color == PieceColor.RED) CN_NUM[step] else step.toString()

    /** 源列号；若同列还有另一枚同色同名子，改用前/后。 */
    private fun sourceOf(board: Board, move: Move, color: PieceColor): String {
        var otherRank = -1
        for (r in 0..9) {
            if (r == move.fromRank) continue
            val p = board.pieceAt(move.fromFile, r) ?: continue
            if (p.color == color && p.type == move.piece.type) {
                otherRank = r
                break
            }
        }
        if (otherRank < 0) return columnOf(color, move.fromFile)

        val isFront = if (color == PieceColor.RED) move.fromRank < otherRank
        else move.fromRank > otherRank
        return if (isFront) "前" else "后"
    }

    private fun columnOf(color: PieceColor, file: Int): String =
        if (color == PieceColor.RED) RED_FILE_NUM[file] else (file + 1).toString()
}
