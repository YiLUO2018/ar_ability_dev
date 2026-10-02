package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 棋子类别（共 7 种）。
 *
 * @property fenChar FEN 记谱字符（小写=黑方，大写=红方由 [PieceColor] 决定）。
 * @property redName 红方棋子名。
 * @property blackName 黑方棋子名。
 */
enum class PieceType(val fenChar: Char, val redName: String, val blackName: String) {
    GENERAL('k', "帅", "将"),
    ADVISOR('a', "仕", "士"),
    ELEPHANT('b', "相", "象"),
    HORSE('n', "马", "马"),
    CHARIOT('r', "车", "车"),
    CANNON('c', "炮", "炮"),
    PAWN('p', "兵", "卒");

    fun name(color: PieceColor): String = if (color == PieceColor.RED) redName else blackName

    companion object {
        /** 按 FEN 字符解析（小写=黑，大写=红），未识别返回 null。 */
        fun fromFenChar(c: Char): PieceType? = entries.firstOrNull { it.fenChar == c.lowercaseChar() }
    }
}
