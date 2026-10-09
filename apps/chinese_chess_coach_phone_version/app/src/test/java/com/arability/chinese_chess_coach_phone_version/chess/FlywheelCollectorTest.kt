package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlywheelCollectorTest {

    @Test
    fun initialPositionIsLegal() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        assertFalse(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun doubleGeneralIsIllegal() {
        val board = Board()
        board.set(4, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(3, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun missingGeneralIsIllegal() {
        val board = Board()
        board.set(0, 0, Piece(PieceColor.BLACK, PieceType.CHARIOT))
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun generalOutsidePalaceIsIllegal() {
        val board = Board()
        board.set(0, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun elephantWrongSideIsIllegal() {
        val board = Board()
        board.set(4, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(2, 4, Piece(PieceColor.RED, PieceType.ELEPHANT)) // 红相过河（rank4）
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun pawnOnBackRankIsIllegal() {
        val board = Board()
        board.set(4, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(0, 8, Piece(PieceColor.RED, PieceType.PAWN)) // 红兵在自身底线后（rank8）
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun threeAdvisorsIsIllegal() {
        val board = Board()
        board.set(4, 9, Piece(PieceColor.RED, PieceType.GENERAL))
        board.set(4, 0, Piece(PieceColor.BLACK, PieceType.GENERAL))
        board.set(3, 9, Piece(PieceColor.RED, PieceType.ADVISOR))
        board.set(5, 9, Piece(PieceColor.RED, PieceType.ADVISOR))
        board.set(3, 7, Piece(PieceColor.RED, PieceType.ADVISOR)) // 第 3 个仕
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }

    @Test
    fun tooManyPiecesIsIllegal() {
        val board = Board.fromFen(Board.INITIAL_FEN)!!
        board.set(1, 5, Piece(PieceColor.RED, PieceType.PAWN))
        board.set(3, 5, Piece(PieceColor.RED, PieceType.PAWN))
        board.set(5, 5, Piece(PieceColor.RED, PieceType.PAWN)) // 32 + 3 = 35 > 32
        assertTrue(FlywheelCollector.isIllegalPosition(board))
    }
}
