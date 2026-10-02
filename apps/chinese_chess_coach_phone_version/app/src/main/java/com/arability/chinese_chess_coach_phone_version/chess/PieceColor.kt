package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 棋子所属方。红方在棋盘下方（rank 大），黑方在上方（rank 小）。
 */
enum class PieceColor {
    RED, BLACK;

    fun opponent(): PieceColor = if (this == RED) BLACK else RED
}
