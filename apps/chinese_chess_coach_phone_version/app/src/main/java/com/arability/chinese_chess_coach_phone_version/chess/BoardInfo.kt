package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap
import kotlin.math.roundToInt

/**
 * 棋盘定位结果。
 *
 * @param corners 源图像中棋盘四角像素坐标，顺序 [TL, TR, BR, BL]，共 8 个 float（x0,y0,x1,y1,...）。
 * @param warped 透视校正后的规整棋盘图（canonical），交叉点 (file, rank) 位于
 *               (originX + file*cellSize, originY + rank*cellSize)。
 * @param cellSize 规整图中相邻交叉点间距（像素）。
 * @param originX / [originY] 规整图原点偏移（左上角交叉点 (0,0) 的像素坐标）。
 * @param imageToGrid 源图像像素 → canonical 坐标的 3×3 单应矩阵（行优先，h33=1）。
 *                    供逐子检测器（YOLO）把检测框中心映射回 (file, rank)。
 *                    语义因检测器而异（物理四角 or 网格交点），由各检测器按自身 corners 语义计算。
 */
class BoardInfo(
    val corners: FloatArray,
    val warped: Bitmap,
    val cellSize: Int,
    val originX: Int = 0,
    val originY: Int = 0,
    val imageToGrid: FloatArray? = null
) {
    val cols: Int get() = 9
    val rows: Int get() = 10

    fun intersectionX(file: Int): Int = originX + file * cellSize
    fun intersectionY(rank: Int): Int = originY + rank * cellSize

    /** 源图像像素坐标 → (file, rank)；无单应或越界时返回 null。 */
    fun imagePointToGrid(x: Float, y: Float): Pair<Int, Int>? {
        val h = imageToGrid ?: return null
        val p = Homography.transform(h, x, y)
        val file = ((p[0] - originX) / cellSize).roundToInt()
        val rank = ((p[1] - originY) / cellSize).roundToInt()
        if (file !in 0..8 || rank !in 0..9) return null
        return file to rank
    }
}
