package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 棋盘 canonical 几何常量（单一事实来源）。
 *
 * 与 ``tools/data_gen`` / ``tools/label_real/geometry.py`` 保持同一套常量：
 * CELL=50、MARGIN=25，网格 9 列（file 0..8）× 10 行（rank 0..9）。
 * 网格最外层交点（即棋盘四角）在 canonical 坐标中位于 MARGIN 处。
 *
 * canonical 坐标系经 [BoardInfo.imageToGrid] 单应建立（历史 warped 位图已移除，C1/C8），
 * 交叉点 (file, rank) 位于 (MARGIN + file*CELL, MARGIN + rank*CELL)。
 */
object BoardGeometry {
    const val CELL = 50
    const val MARGIN = 25

    /** canonical 棋盘宽（含两侧 MARGIN）。 */
    const val BOARD_W = MARGIN * 2 + CELL * 8   // 450

    /** canonical 棋盘高（含上下 MARGIN）。 */
    const val BOARD_H = MARGIN * 2 + CELL * 9   // 500

    /** canonical 四角（网格最外层交点），顺序 TL, TR, BR, BL（8 个 float）。 */
    val corners: FloatArray = floatArrayOf(
        MARGIN.toFloat(), MARGIN.toFloat(),
        (MARGIN + 8 * CELL).toFloat(), MARGIN.toFloat(),
        (MARGIN + 8 * CELL).toFloat(), (MARGIN + 9 * CELL).toFloat(),
        MARGIN.toFloat(), (MARGIN + 9 * CELL).toFloat(),
    )

    /**
     * canonical 物理棋盘边界四角（含边框外沿，即整张 canonical 图的四角），顺序 TL, TR, BR, BL。
     * 分割掩码给出的是「物理棋盘边」，故应 warp 到本组边界，而非 [corners]（网格交点，内缩 MARGIN）。
     */
    val physicalCorners: FloatArray = floatArrayOf(
        0f, 0f,
        BOARD_W.toFloat(), 0f,
        BOARD_W.toFloat(), BOARD_H.toFloat(),
        0f, BOARD_H.toFloat(),
    )
}
