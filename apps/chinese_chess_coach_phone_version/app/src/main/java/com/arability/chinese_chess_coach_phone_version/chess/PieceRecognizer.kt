package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 棋子识别器接口。上层只依赖本接口，不关心具体实现。
 * 后续可替换为 TFLite 分类模型实现，改工厂即可。
 */
interface PieceRecognizer {
    /**
     * 识别规整棋盘图中每个交叉点的棋子。
     *
     * @return 9 列 × 10 行的棋子网格（下标 [rank][file]），空位为 null。
     */
    fun recognize(boardInfo: BoardInfo): Array<Array<Piece?>>

    /**
     * 释放识别器占用的资源。默认空实现，供需要清理的实现（如 TFLite 解释器）覆盖。
     */
    fun close() {}
}
