package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap

/**
 * 棋盘定位结果。
 *
 * @param corners 源图像中棋盘四角像素坐标，顺序 [TL, TR, BR, BL]，共 8 个 float（x0,y0,x1,y1,...）。
 * @param warped 透视校正后的规整棋盘图（canonical），交叉点 (file, rank) 位于
 *               (originX + file*cellSize, originY + rank*cellSize)。
 * @param cellSize 规整图中相邻交叉点间距（像素）。
 * @param originX / [originY] 规整图原点偏移（左上角交叉点 (0,0) 的像素坐标）。
 */
class BoardInfo(
    val corners: FloatArray,
    val warped: Bitmap,
    val cellSize: Int,
    val originX: Int = 0,
    val originY: Int = 0
) {
    val cols: Int get() = 9
    val rows: Int get() = 10

    fun intersectionX(file: Int): Int = originX + file * cellSize
    fun intersectionY(rank: Int): Int = originY + rank * cellSize
}
