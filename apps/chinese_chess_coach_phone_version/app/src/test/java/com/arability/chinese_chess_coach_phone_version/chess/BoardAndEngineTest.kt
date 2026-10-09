package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BoardAndEngineTest {

    @Test
    fun initialFenRoundTrip() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        assertEquals(Board.INITIAL_FEN, board.toFen())
    }

    @Test
    fun initialPositionHasExpectedLegalMoves() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        val moves = board.generateMoves(PieceColor.RED)
        fun has(fromF: Int, fromR: Int, toF: Int, toR: Int) =
            moves.any { it.fromFile == fromF && it.fromRank == fromR && it.toFile == toF && it.toRank == toR }

        assertTrue("马(1,9) 应可跳到 (2,7)", has(1, 9, 2, 7))
        assertTrue("马(1,9) 应可跳到 (0,7)", has(1, 9, 0, 7))
        assertTrue("兵(0,6) 应可前进到 (0,5)", has(0, 6, 0, 5))
        assertTrue("帅(4,9) 应可在宫内前进到 (4,8)", has(4, 9, 4, 8))
        // 马(1,9) 往 (3,8) 被相(2,9) 蹩马腿
        assertFalse("马(1,9) 不应能跳到 (3,8)", has(1, 9, 3, 8))
    }

    @Test
    fun makeUnmakeRoundTrip() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        val before = board.toFen()
        val move = board.generateMoves(PieceColor.RED).first { it.fromFile == 1 && it.fromRank == 9 }
        val undo = board.makeMove(move)
        board.unmakeMove(undo)
        assertEquals(before, board.toFen())
    }

    @Test
    fun chariotCheckIsDetected() {
        val board = Board()
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(4, 4, Piece(PieceColor.RED, PieceType.CHARIOT))
        assertTrue(board.isInCheck(PieceColor.BLACK))

        val blackMoves = board.generateMoves(PieceColor.BLACK)
        fun has(toF: Int, toR: Int) = blackMoves.any { it.toFile == toF && it.toRank == toR }
        assertTrue(has(3, 0))
        assertTrue(has(5, 0))
        assertFalse("将不应移动到仍被车攻击的 (4,1)", has(4, 1))
    }

    @Test
    fun generalsFacingIsDetected() {
        val board = Board()
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(4, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        assertTrue(board.generalsFacing())
        // 中间有子则不算照面
        board.set(4, 5, Piece(PieceColor.RED, PieceType.PAWN))
        assertFalse(board.generalsFacing())
    }

    @Test
    fun engineReturnsBestMove() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        val engine = KotlinChessEngine()
        val move = engine.bestMove(board, depth = 2)
        assertNotNull(move)
        // 引擎不应修改入参局面
        assertEquals(Board.INITIAL_FEN, board.toFen())
    }

    @Test
    fun xqwLightEngineReturnsLegalMove() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        val engine = XQWLightChessEngine()
        val move = engine.bestMove(board)
        assertNotNull("XQWLight 应在开局局面给出走法", move)
        val legal = board.generateMoves(board.sideToMove)
        assertTrue(
            "XQWLight 返回的走法应属于合法走法集：$move",
            legal.any {
                it.fromFile == move!!.fromFile && it.fromRank == move.fromRank &&
                    it.toFile == move.toFile && it.toRank == move.toRank
            }
        )
        // 引擎不应修改入参局面
        assertEquals(Board.INITIAL_FEN, board.toFen())
    }

    @Test
    fun xqwLightEngineNoMovesWhenMated() {
        val board = Board()
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(3, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.set(4, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.set(5, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.sideToMove = PieceColor.BLACK
        assertNull(XQWLightChessEngine().bestMove(board))
    }

    @Test
    fun chineseNotationBasicMoves() {
        // 炮二平五：红炮 (7,7) → (4,7)
        val board = Board()
        val cannon = Piece(PieceColor.RED, PieceType.CANNON)
        board.set(7, 7, cannon)
        assertEquals("炮二平五", ChineseNotation.of(board, Move(7, 7, 4, 7, cannon)))

        // 马八进七：红马 (1,9) → (2,7)
        val horse = Piece(PieceColor.RED, PieceType.HORSE)
        assertEquals("马八进七", ChineseNotation.of(board, Move(1, 9, 2, 7, horse)))

        // 兵七进一：红兵 (2,6) → (2,5)
        val pawn = Piece(PieceColor.RED, PieceType.PAWN)
        assertEquals("兵七进一", ChineseNotation.of(board, Move(2, 6, 2, 5, pawn)))

        // 马2进3：黑马 (1,0) → (2,2)
        val bHorse = Piece(PieceColor.BLACK, PieceType.HORSE)
        assertEquals("马2进3", ChineseNotation.of(board, Move(1, 0, 2, 2, bHorse)))

        // 车一平四：红车 (8,9) → (5,9)
        val chariot = Piece(PieceColor.RED, PieceType.CHARIOT)
        assertEquals("车一平四", ChineseNotation.of(board, Move(8, 9, 5, 9, chariot)))
    }

    @Test
    fun chineseNotationDisambiguation() {
        // 同列两枚红炮 (7,7) 与 (7,5)，前炮（更靠近黑方，rank 小者）平五 → 前炮平五
        val board = Board()
        val cannon = Piece(PieceColor.RED, PieceType.CANNON)
        board.set(7, 7, cannon)
        board.set(7, 5, cannon)
        assertEquals("前炮平五", ChineseNotation.of(board, Move(7, 5, 4, 5, cannon)))
        assertEquals("后炮平五", ChineseNotation.of(board, Move(7, 7, 4, 7, cannon)))
    }

    @Test
    fun engineNoMovesWhenMated() {
        // 三车困毙：黑将 (4,0)，红车控制 3/4/5 三列，黑无合法走法
        val board = Board()
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(3, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.set(4, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.set(5, 9, Piece(PieceColor.RED, PieceType.CHARIOT))
        board.sideToMove = PieceColor.BLACK

        assertTrue(board.isInCheck(PieceColor.BLACK))
        val moves = board.generateMoves(PieceColor.BLACK)
        assertTrue("黑方应无合法走法", moves.isEmpty())
        val engine = KotlinChessEngine()
        assertNull(engine.bestMove(board, depth = 1))
    }
}
