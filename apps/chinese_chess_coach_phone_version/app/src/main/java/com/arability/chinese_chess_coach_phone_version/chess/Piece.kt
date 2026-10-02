package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 一枚棋子：所属方 + 类别。
 */
data class Piece(val color: PieceColor, val type: PieceType) {

    /** FEN 字符（黑=小写，红=大写）。 */
    val fenChar: Char
        get() = if (color == PieceColor.RED) type.fenChar.uppercaseChar() else type.fenChar

    /** 中文名（按方显示）。 */
    val name: String
        get() = type.name(color)
}
