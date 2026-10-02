package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.max
import kotlin.math.min

/**
 * 中国象棋棋盘（9 列 × 10 行，共 90 个交叉点）。
 *
 * 坐标约定：`grid[rank][file]`，file ∈ [0,8]，rank ∈ [0,9]；
 * rank 0 为黑方底线（顶部），rank 9 为红方底线（底部）。
 *
 * 本类负责局面存储、FEN 解析/导出、走法生成与合法性校验，
 * 是引擎与识别结果的中间表示。
 */
class Board {

    private val grid: Array<Array<Piece?>> = Array(10) { arrayOfNulls(9) }

    /** 轮到哪一方走棋（默认红方）。 */
    var sideToMove: PieceColor = PieceColor.RED

    fun pieceAt(file: Int, rank: Int): Piece? =
        if (file in 0..8 && rank in 0..9) grid[rank][file] else null

    fun set(file: Int, rank: Int, piece: Piece?) {
        if (file in 0..8 && rank in 0..9) grid[rank][file] = piece
    }

    fun clear() {
        for (r in 0..9) for (f in 0..8) grid[r][f] = null
    }

    /** 深拷贝一份局面（引擎搜索前使用，避免污染调用方）。 */
    fun copy(): Board {
        val b = Board()
        for (r in 0..9) for (f in 0..8) b.grid[r][f] = grid[r][f]
        b.sideToMove = sideToMove
        return b
    }

    // ---------------------------------------------------------------- 走法

    /** 走法撤销信息（用于 make/unmake）。 */
    class Undo(
        val fromFile: Int,
        val fromRank: Int,
        val toFile: Int,
        val toRank: Int,
        val moved: Piece,
        val captured: Piece?
    )

    fun makeMove(m: Move): Undo {
        val captured = grid[m.toRank][m.toFile]
        grid[m.toRank][m.toFile] = m.piece
        grid[m.fromRank][m.fromFile] = null
        return Undo(m.fromFile, m.fromRank, m.toFile, m.toRank, m.piece, captured)
    }

    fun unmakeMove(u: Undo) {
        grid[u.fromRank][u.fromFile] = u.moved
        grid[u.toRank][u.toFile] = u.captured
    }

    /** 生成某方全部合法走法（已过滤被将军 / 将帅照面）。 */
    fun generateMoves(color: PieceColor): List<Move> {
        val result = ArrayList<Move>()
        for (m in generatePseudoLegalMoves(color)) {
            val undo = makeMove(m)
            val legal = !isInCheck(color) && !generalsFacing()
            unmakeMove(undo)
            if (legal) result.add(m)
        }
        return result
    }

    /**
     * 生成某方「伪合法」走法：满足各棋子自身移动规则，但不校验己方是否被将军。
     * 用于攻击检测与合法走法的中间步骤。
     */
    fun generatePseudoLegalMoves(color: PieceColor): List<Move> {
        val moves = ArrayList<Move>()
        for (rank in 0..9) {
            for (file in 0..8) {
                val p = grid[rank][file] ?: continue
                if (p.color != color) continue
                addPieceMoves(moves, file, rank, p)
            }
        }
        return moves
    }

    private fun addPieceMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        when (p.type) {
            PieceType.GENERAL -> addGeneralMoves(moves, file, rank, p)
            PieceType.ADVISOR -> addAdvisorMoves(moves, file, rank, p)
            PieceType.ELEPHANT -> addElephantMoves(moves, file, rank, p)
            PieceType.HORSE -> addHorseMoves(moves, file, rank, p)
            PieceType.CHARIOT -> addChariotMoves(moves, file, rank, p)
            PieceType.CANNON -> addCannonMoves(moves, file, rank, p)
            PieceType.PAWN -> addPawnMoves(moves, file, rank, p)
        }
    }

    /** 尝试落子：目标为空或敌子则加入走法。 */
    private fun tryAdd(
        moves: MutableList<Move>,
        fromFile: Int, fromRank: Int,
        toFile: Int, toRank: Int,
        p: Piece
    ) {
        if (toFile !in 0..8 || toRank !in 0..9) return
        val target = grid[toRank][toFile]
        if (target != null && target.color == p.color) return
        moves.add(Move(fromFile, fromRank, toFile, toRank, p, target))
    }

    private fun palaceRanks(color: PieceColor): IntRange = if (color == PieceColor.RED) 7..9 else 0..2

    private fun inPalace(file: Int, rank: Int, color: PieceColor): Boolean =
        file in 3..5 && rank in palaceRanks(color)

    private fun addGeneralMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        for ((df, dr) in ORTHO) {
            val nf = file + df; val nr = rank + dr
            if (inPalace(nf, nr, p.color)) tryAdd(moves, file, rank, nf, nr, p)
        }
    }

    private fun addAdvisorMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        for ((df, dr) in DIAG) {
            val nf = file + df; val nr = rank + dr
            if (inPalace(nf, nr, p.color)) tryAdd(moves, file, rank, nf, nr, p)
        }
    }

    private fun addElephantMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        // 相/象走田字，不能过河；象眼（中点）需为空。
        val ownSide = if (p.color == PieceColor.RED) 5..9 else 0..4
        for ((df, dr) in DIAG) {
            val eyeFile = file + df; val eyeRank = rank + dr
            val nf = file + 2 * df; val nr = rank + 2 * dr
            if (nf !in 0..8 || nr !in ownSide) continue
            if (grid[eyeRank][eyeFile] != null) continue
            tryAdd(moves, file, rank, nf, nr, p)
        }
    }

    private fun addHorseMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        // 马走日字，蹩马腿（长方向上的相邻正交点需为空）。
        for ((df, dr, lf, lr) in HORSE_LEGS) {
            val legFile = file + lf
            val legRank = rank + lr
            if (legFile !in 0..8 || legRank !in 0..9) continue
            if (grid[legRank][legFile] != null) continue
            tryAdd(moves, file, rank, file + df, rank + dr, p)
        }
    }

    private fun addChariotMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        for ((df, dr) in ORTHO) {
            var nf = file + df; var nr = rank + dr
            while (nf in 0..8 && nr in 0..9) {
                val target = grid[nr][nf]
                if (target == null) {
                    moves.add(Move(file, rank, nf, nr, p, null))
                } else {
                    if (target.color != p.color) moves.add(Move(file, rank, nf, nr, p, target))
                    break
                }
                nf += df; nr += dr
            }
        }
    }

    private fun addCannonMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        for ((df, dr) in ORTHO) {
            // 不吃子：沿直线走到第一个棋子前
            var nf = file + df; var nr = rank + dr
            while (nf in 0..8 && nr in 0..9 && grid[nr][nf] == null) {
                moves.add(Move(file, rank, nf, nr, p, null))
                nf += df; nr += dr
            }
            // 吃子：越过第一个棋子（炮架）后，再找下一个棋子，若为敌子则可吃
            var sf = nf + df; var sr = nr + dr
            while (sf in 0..8 && sr in 0..9 && grid[sr][sf] == null) {
                sf += df; sr += dr
            }
            if (sf in 0..8 && sr in 0..9) {
                val target = grid[sr][sf]
                if (target != null && target.color != p.color) {
                    moves.add(Move(file, rank, sf, sr, p, target))
                }
            }
        }
    }

    private fun addPawnMoves(moves: MutableList<Move>, file: Int, rank: Int, p: Piece) {
        val forward = if (p.color == PieceColor.RED) -1 else +1
        val crossedRiver = if (p.color == PieceColor.RED) rank <= 4 else rank >= 5

        tryAdd(moves, file, rank, file, rank + forward, p)
        if (crossedRiver) {
            tryAdd(moves, file, rank, file - 1, rank, p)
            tryAdd(moves, file, rank, file + 1, rank, p)
        }
    }

    // ---------------------------------------------------------------- 将军

    /** 某方是否被将军（对方存在伪合法走法可吃己方将/帅）。 */
    fun isInCheck(color: PieceColor): Boolean {
        val king = findKing(color) ?: return false
        for (m in generatePseudoLegalMoves(color.opponent())) {
            if (m.toFile == king.first && m.toRank == king.second) return true
        }
        return false
    }

    /** 将帅是否照面（同列且中间无子）。 */
    fun generalsFacing(): Boolean {
        val rk = findKing(PieceColor.RED) ?: return false
        val bk = findKing(PieceColor.BLACK) ?: return false
        if (rk.first != bk.first) return false
        val f = rk.first
        val lo = min(rk.second, bk.second)
        val hi = max(rk.second, bk.second)
        for (r in lo + 1 until hi) {
            if (grid[r][f] != null) return false
        }
        return true
    }

    /** 找某方将/帅坐标 (file, rank)，不存在返回 null。 */
    fun findKing(color: PieceColor): Pair<Int, Int>? {
        for (rank in 0..9) {
            for (file in 0..8) {
                val p = grid[rank][file] ?: continue
                if (p.color == color && p.type == PieceType.GENERAL) return file to rank
            }
        }
        return null
    }

    // ---------------------------------------------------------------- FEN

    /** 导出中国象棋 FEN 的棋子排布 + 行棋方。 */
    fun toFen(): String {
        val sb = StringBuilder()
        for (rank in 0..9) {
            var empty = 0
            for (file in 0..8) {
                val p = grid[rank][file]
                if (p == null) {
                    empty++
                } else {
                    if (empty > 0) { sb.append(empty); empty = 0 }
                    sb.append(p.fenChar)
                }
            }
            if (empty > 0) sb.append(empty)
            if (rank < 9) sb.append('/')
        }
        sb.append(' ')
        sb.append(if (sideToMove == PieceColor.RED) 'w' else 'b')
        return sb.toString()
    }

    companion object {
        /** 标准开局 FEN。 */
        const val INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

        private val ORTHO = arrayOf(intArrayOf(1, 0), intArrayOf(-1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        private val DIAG = arrayOf(intArrayOf(1, 1), intArrayOf(1, -1), intArrayOf(-1, 1), intArrayOf(-1, -1))

        // 马腿：target 偏移 (df,dr) 与对应的腿偏移 (lf,lr)
        private val HORSE_LEGS = arrayOf(
            intArrayOf(2, 1, 1, 0),
            intArrayOf(2, -1, 1, 0),
            intArrayOf(-2, 1, -1, 0),
            intArrayOf(-2, -1, -1, 0),
            intArrayOf(1, 2, 0, 1),
            intArrayOf(1, -2, 0, -1),
            intArrayOf(-1, 2, 0, 1),
            intArrayOf(-1, -2, 0, -1)
        )

        /** 从 FEN 构建局面。解析失败返回 null。 */
        fun fromFen(fen: String): Board? {
            val parts = fen.trim().split(Regex("\\s+"))
            val placement = parts.getOrNull(0) ?: return null
            val board = Board()
            val ranks = placement.split('/')
            if (ranks.size != 10) return null
            for (rank in 0..9) {
                var file = 0
                for (ch in ranks[rank]) {
                    if (ch.isDigit()) {
                        file += ch - '0'
                    } else {
                        if (file > 8) return null
                        val type = PieceType.fromFenChar(ch) ?: return null
                        val color = if (ch.isUpperCase()) PieceColor.RED else PieceColor.BLACK
                        board.set(file, rank, Piece(color, type))
                        file++
                    }
                }
                if (file != 9) return null
            }
            board.sideToMove = if (parts.getOrNull(1) == "b") PieceColor.BLACK else PieceColor.RED
            return board
        }
    }
}
