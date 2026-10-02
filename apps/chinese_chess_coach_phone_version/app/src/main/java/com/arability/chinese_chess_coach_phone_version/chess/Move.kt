package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 一步走法。
 *
 * 坐标约定：file ∈ [0,8]（列，左→右），rank ∈ [0,9]（行，上→下）；
 * rank 0 为黑方底线，rank 9 为红方底线。
 *
 * @param fromFile / [fromRank] 起点坐标。
 * @param toFile / [toRank] 终点坐标。
 * @param piece 移动的棋子。
 * @param captured 被吃掉的棋子（无则 null）。
 */
data class Move(
    val fromFile: Int,
    val fromRank: Int,
    val toFile: Int,
    val toRank: Int,
    val piece: Piece,
    val captured: Piece? = null
) {
    /** 中国象棋记谱（如「炮二平五」的近似坐标表示）。 */
    fun toNotation(): String {
        val from = "${'a' + fromFile}${fromRank + 1}"
        val to = "${'a' + toFile}${toRank + 1}"
        return "${piece.name}$from$to"
    }

    override fun toString(): String = toNotation()
}
