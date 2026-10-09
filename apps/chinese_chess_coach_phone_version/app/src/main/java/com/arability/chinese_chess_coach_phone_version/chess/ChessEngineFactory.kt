package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 引擎工厂。切换引擎实现时只改这里，调用方无需改动。
 * 当前使用 XQWLight（Java 移植版）；KotlinChessEngine 保留作为纯 Kotlin 降级实现。
 */
object ChessEngineFactory {
    fun create(): ChessEngine = XQWLightChessEngine()
}
