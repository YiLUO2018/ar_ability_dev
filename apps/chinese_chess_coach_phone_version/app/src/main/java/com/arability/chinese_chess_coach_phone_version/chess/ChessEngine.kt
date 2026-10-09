package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 象棋引擎接口。上层只依赖本接口，不关心具体实现。
 * 当前实现：XQWLight（Java 移植版），备选 KotlinChessEngine（纯 Kotlin 降级）。
 */
interface ChessEngine {
    /**
     * 计算给定局面下 [Board.sideToMove] 一方的最佳走法。
     *
     * @param board 当前局面（引擎会内部拷贝，不修改入参）。
     * @param depth 搜索深度（半回合数）。
     * @return 最佳走法；无合法走法（被将死/困毙）返回 null。
     */
    fun bestMove(board: Board, depth: Int = DEFAULT_DEPTH): Move?

    fun close() {}

    companion object {
        const val DEFAULT_DEPTH = 3
    }
}
