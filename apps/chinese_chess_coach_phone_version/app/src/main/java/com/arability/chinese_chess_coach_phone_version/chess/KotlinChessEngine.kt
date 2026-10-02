package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 纯 Kotlin 实现的中国象棋引擎（走法生成 + negamax/alpha-beta 搜索 + 子力评估）。
 *
 * 作为「可替换引擎」的默认实现：棋力有限但无需 NDK/外部二进制，
 * 用于先跑通「识别 → 建议」闭环；后续可用 XQWLight（JNI）替换本类。
 */
class KotlinChessEngine : ChessEngine {

    override fun bestMove(board: Board, depth: Int): Move? {
        val b = board.copy()
        val color = b.sideToMove
        val moves = orderMoves(b.generateMoves(color))
        if (moves.isEmpty()) return null

        var best: Move? = null
        var alpha = NEG_INF
        for (m in moves) {
            val undo = b.makeMove(m)
            val score = -negamax(b, color.opponent(), depth - 1, NEG_INF, -alpha)
            b.unmakeMove(undo)
            if (score > alpha) {
                alpha = score
                best = m
            }
        }
        return best
    }

    private fun negamax(board: Board, color: PieceColor, depth: Int, alpha: Int, beta: Int): Int {
        if (depth == 0) return evaluate(board, color)

        val moves = orderMoves(board.generateMoves(color))
        if (moves.isEmpty()) {
            // 无棋可走：被将死或困毙，视为负；深度越大（离现在越远）惩罚越小
            return -MATE_VALUE + depth
        }

        var a = alpha
        for (m in moves) {
            val undo = board.makeMove(m)
            val score = -negamax(board, color.opponent(), depth - 1, -beta, -a)
            board.unmakeMove(undo)
            if (score >= beta) return beta
            if (score > a) a = score
        }
        return a
    }

    /** 从 [color] 视角评估局面（正 = 有利）。 */
    private fun evaluate(board: Board, color: PieceColor): Int {
        var score = 0
        for (rank in 0..9) {
            for (file in 0..8) {
                val p = board.pieceAt(file, rank) ?: continue
                val v = pieceValue(p, rank)
                score += if (p.color == color) v else -v
            }
        }
        return score
    }

    private fun pieceValue(p: Piece, rank: Int): Int = when (p.type) {
        PieceType.GENERAL -> 100000
        PieceType.CHARIOT -> 1000
        PieceType.CANNON -> 500
        PieceType.HORSE -> 450
        PieceType.ELEPHANT -> 200
        PieceType.ADVISOR -> 200
        PieceType.PAWN -> {
            val crossed = if (p.color == PieceColor.RED) rank <= 4 else rank >= 5
            if (crossed) 200 else 100
        }
    }

    /** 简单走法排序：吃子价值高者优先，提升 alpha-beta 剪枝效率。 */
    private fun orderMoves(moves: List<Move>): List<Move> =
        moves.sortedByDescending { it.captured?.let { c -> pieceValue(c, it.toRank) } ?: 0 }

    companion object {
        private const val MATE_VALUE = 1_000_000
        private const val NEG_INF = -Int.MAX_VALUE / 2
    }
}
