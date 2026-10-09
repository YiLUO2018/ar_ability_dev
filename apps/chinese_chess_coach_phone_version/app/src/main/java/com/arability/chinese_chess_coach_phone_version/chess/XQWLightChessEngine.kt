package com.arability.chinese_chess_coach_phone_version.chess

import xqwlight.Position
import xqwlight.Search

/**
 * 基于 XQWLight（象棋小巫师，Java 移植版，GPL-2.0）的象棋引擎。
 *
 * XQWLight 是成熟的 UCCI 引擎，带迭代加深 + 置换表 + 空着裁剪 + 静态搜索 + 位置评估，
 * 棋力远强于 [KotlinChessEngine]（后者只算子力 / 无位置评估 / 深度 3）。
 *
 * 桥接方式：本 App 的 [Board.toFen] 与 XQWLight 的 `Position.fromFen` 使用同一套 FEN 约定
 * （k/a/b/n/r/c/p，黑=小写/红=大写，rank 0=黑方底线在顶），故直接把 FEN 喂给引擎即可；
 * 引擎返回的走法编码 `src + (dst << 8)` 再解码回本 App 的 (file, rank) 坐标。
 *
 * 由于 XQWLight 本质是「限时迭代加深」而非「固定深度」，[bestMove] 忽略 [depth] 参数、
 * 改用固定时间预算 [TIME_LIMIT_MS] 控制棋力与响应速度。
 */
class XQWLightChessEngine : ChessEngine {

    // 单例可复用：搜索开始前 searchMain 会重置置换表 / 历史表 / 杀手着法，
    // 且本 App 通过单线程 Executor 串行调用，无并发问题。
    private val position = Position()
    private val search = Search(position, HASH_LEVEL)

    @Synchronized
    override fun bestMove(board: Board, depth: Int): Move? {
        // XQWLight 在根节点无合法走法（被将死/困毙）时会调用 setBestMove(0, …)，
        // 其 historyIndex(0) 算出负下标导致越界崩溃。先用本 App 走法生成器判断，
        // 无合法走法直接返回 null（符合 ChessEngine 接口契约），绕开该崩溃。
        if (board.generateMoves(board.sideToMove).isEmpty()) return null

        position.fromFen(board.toFen())
        val mv = search.searchMain(MAX_DEPTH, TIME_LIMIT_MS)
        if (mv <= 0) return null
        return decodeMove(board, mv)
    }

    override fun close() {
        // XQWLight 无外部资源需释放；Position/Search 均为纯内存对象。
    }

    /** 把 XQWLight 走法编码 `src + (dst << 8)` 解码为本 App 的 [Move]。 */
    private fun decodeMove(board: Board, mv: Int): Move? {
        val src = mv and 0xff
        val dst = mv shr 8
        val fromFile = (src and 0x0f) - Position.FILE_LEFT
        val fromRank = (src shr 4) - Position.RANK_TOP
        val toFile = (dst and 0x0f) - Position.FILE_LEFT
        val toRank = (dst shr 4) - Position.RANK_TOP

        val piece = board.pieceAt(fromFile, fromRank) ?: return null
        val captured = board.pieceAt(toFile, toRank)
        return Move(fromFile, fromRank, toFile, toRank, piece, captured)
    }

    companion object {
        /** 置换表等级：2^16 = 65536 项，约 2MB，一次性分配。 */
        private const val HASH_LEVEL = 16

        /** 迭代加深上限（XQWLight 内部 LIMIT_DEPTH=64）。 */
        private const val MAX_DEPTH = 64

        /** 单次搜索时间预算（毫秒）。1s 约可达深度 8–12，明显强于原 Kotlin 引擎。 */
        private const val TIME_LIMIT_MS = 1000
    }
}
