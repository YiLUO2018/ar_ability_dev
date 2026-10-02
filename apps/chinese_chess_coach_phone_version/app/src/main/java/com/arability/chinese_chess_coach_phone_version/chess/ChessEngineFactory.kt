package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 引擎工厂。切换引擎实现（如 XQWLight JNI 版）时只改这里，调用方无需改动。
 */
object ChessEngineFactory {
    fun create(): ChessEngine = KotlinChessEngine()
}
