package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 标准开局约定重置（replay_pipeline.py 同款，2026-10-14 移植）：
 * 占位匹配标准开局时 refBoard := 32 交点真值——把「约定」钉到真实物理位置
 * （不依赖任何检测四角），纠正种子级偏差（track7 教训：f41 种子 BR 外扩
 * 25-40px 被棋锚纠偏忠实继承）。任意一子走动即产生越位 → [matches] 自动失效，
 * 中局回退质心投影约定。
 *
 * 几何性判据（on_std ≥20 且 extra ≤4）：越位子只是 RANSAC 外点、身份与位置
 * 拟合无关；biased 四角下个别子 snap 偏 1 格 / 幻影框 / 走子都不影响。
 */
object StdOpening {

    /** 标准开局 32 交点（棋盘系 xy 交错；底线=车马象士将士象马车 9 列全占）。 */
    val REF_BOARD: FloatArray = buildList {
        val pawnFiles = intArrayOf(0, 2, 4, 6, 8)
        for (f in 0..8) for (r in intArrayOf(0, 9)) {
            add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
        for (f in intArrayOf(1, 7)) for (r in intArrayOf(2, 7)) {
            add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
        for (f in pawnFiles) for (r in intArrayOf(3, 6)) {
            add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
    }.toFloatArray()

    /** 标准开局占位表（rank*9+file）。 */
    private val OCC = BooleanArray(90).also { occ ->
        val pawnFiles = intArrayOf(0, 2, 4, 6, 8)
        for (f in 0..8) for (r in intArrayOf(0, 9)) occ[r * 9 + f] = true
        for (f in intArrayOf(1, 7)) for (r in intArrayOf(2, 7)) occ[r * 9 + f] = true
        for (f in pawnFiles) for (r in intArrayOf(3, 6)) occ[r * 9 + f] = true
    }

    fun matches(grid: Array<Array<Piece?>>): Boolean {
        var onStd = 0
        var extra = 0
        for (r in 0..9) for (f in 0..8) {
            if (grid[r][f] != null) {
                if (OCC[r * 9 + f]) onStd++ else extra++
            }
        }
        return onStd >= 20 && extra <= 4
    }
}
