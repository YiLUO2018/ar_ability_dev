package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap

/**
 * 单个棋子检测框（逐子检测器如 YOLO 的原始输出，供数据飞轮采集/诊断使用）。
 *
 * @param cx / [cy] 检测框中心在源图像中的像素坐标。
 * @param label 类别序号（0–13，顺序见 YoloPieceDetector 契约）。
 * @param confidence 置信度 [0,1]。
 */
data class PieceDetection(
    val cx: Float,
    val cy: Float,
    val label: Int,
    val confidence: Float
)

/**
 * 棋子识别器接口。上层只依赖本接口，不关心具体实现。
 * 后续可替换为 TFLite 分类模型实现，改工厂即可。
 */
interface PieceRecognizer {
    /**
     * 识别棋盘图中每个交叉点的棋子。
     *
     * @param bitmap 原始帧（逐子检测器如 YOLO 需在原帧上检出 bbox；
     *               基于 warped 的整盘分类器可忽略此参数）。
     * @param boardInfo 棋盘定位结果（四角 + 规整图 + 单应映射）。
     * @return 9 列 × 10 行的棋子网格（下标 [rank][file]），空位为 null。
     */
    fun recognize(bitmap: Bitmap, boardInfo: BoardInfo): Array<Array<Piece?>>

    /**
     * 最近一次 [recognize] 的原始逐子检测框。非逐子检测器（整盘分类/模板）返回空列表。
     * 供数据飞轮判断「难帧」与离线标注作先验。
     */
    fun lastDetections(): List<PieceDetection> = emptyList()

    /**
     * 释放识别器占用的资源。默认空实现，供需要清理的实现（如 TFLite 解释器）覆盖。
     */
    fun close() {}
}
