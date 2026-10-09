package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * 棋子质心反馈的四角精修（P0-b）：用 YOLO 检出的棋子质心校正棋盘四角的系统性偏差。
 *
 * 动机：分割掩码在近端会「溢」到桌面/阴影条带（哪怕白桌面也有阴影条），导致底部节点
 * 整体低于真实棋子 → 棋子被就近吸附到上一行（真机实测红底线 8/9 颗跑到 rank8）。
 * 棋子质心与节点分配无关、稳定可靠，可当几何锚点。
 *
 * 做法（离线 fit_std_homography.py 思想的端上移植，无标准开局先验版）：
 *  1. 由当前四角算 90 交点（canonical → 图像）；
 *  2. 每个质心就近吸附到交点 → 候选对应（node canonical 坐标 ↔ 质心）；
 *  3. RANSAC 拟合校正单应，评分用「唯一性内点」（每个质心至多贡献一个内点，
 *     拒退化解——离线版实测教训：普通内点计分会被秩 2 退化单应骗到满分）；
 *  4. 精修四角 = 校正单应投影物理四角；校正量被钳制在 [REFINE_MAX_CORNER_MOVE] 内
 *     （这是「精修」不是「重检测」，防错误大跳）。
 *
 * 与离线版的关键区别：候选对应来自 app 自己的定向网格，**无 180°/镜像翻转歧义**。
 */
object QuadRefiner {

    /** 尝试精修的最少质心数。 */
    private const val MIN_CENTROIDS = 12

    /** 质心吸附到交点的最大距离（格）。 */
    private const val SNAP_CELL = 0.8f

    /** 内点判据（格）：canonical 交点投影到质心的最大距离。 */
    private const val INLIER_CELL = 0.35f

    /** RANSAC 迭代次数（候选对 ~32 个，4 点采样足够）。 */
    private const val RANSAC_ITERS = 500

    /** 单角最大校正量（格）：超过视为异常，拒绝精修。 */
    private const val REFINE_MAX_CORNER_MOVE = 1.5f

    /** 精修结果的最低唯一内点数。 */
    private const val MIN_INLIERS = 16

    /** 内点结构扩散校验：至少覆盖这么多列（file 0..8）与行（rank 0..9）。 */
    private const val MIN_SPREAD_FILES = 6
    private const val MIN_SPREAD_RANKS = 6

    /**
     * 精修棋盘四角。
     *
     * @param corners 当前物理四角（TL,TR,BR,BL，图像像素，8 float）。
     * @param centroids 检出棋子质心列表（图像像素 [x,y]）。
     * @return 精修后的四角（8 float）；条件不满足 / 拟合失败返回 null。
     */
    fun refine(corners: FloatArray, centroids: List<FloatArray>): FloatArray? {
        if (corners.size != 8 || centroids.size < MIN_CENTROIDS) return null

        // 1. canonical → 图像 的初始单应与 90 交点。
        val h0 = Homography.compute(BoardGeometry.physicalCorners, corners)
        val nodes = Array(90) { i ->
            val f = i % 9
            val r = i / 9
            Homography.transform(
                h0,
                BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat(),
                BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat(),
            )
        }

        // 格边长（图像像素）：取所有交点行内相邻距的中位数。
        val spacings = ArrayList<Float>(80)
        for (r in 0 until 10) {
            for (f in 0 until 8) {
                val a = nodes[r * 9 + f]
                val b = nodes[r * 9 + f + 1]
                spacings.add(hypot(b[0] - a[0], b[1] - a[1]))
            }
        }
        spacings.sort()
        val cell = spacings[spacings.size / 2]
        if (cell <= 0f) return null

        // 2. 质心 → 最近交点，构造候选对应（canonical 节点 ↔ 质心）。
        val pairNode = ArrayList<FloatArray>(centroids.size)   // canonical (x,y)
        val pairCent = ArrayList<FloatArray>(centroids.size)   // 图像质心 (x,y)
        for (c in centroids) {
            var bestI = -1
            var bestD = Float.MAX_VALUE
            for (i in 0 until 90) {
                val d = hypot(c[0] - nodes[i][0], c[1] - nodes[i][1])
                if (d < bestD) { bestD = d; bestI = i }
            }
            if (bestI < 0 || bestD > SNAP_CELL * cell) continue
            val f = bestI % 9
            val r = bestI / 9
            pairNode.add(
                floatArrayOf(
                    BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat(),
                    BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat(),
                )
            )
            pairCent.add(c)
        }
        if (pairNode.size < MIN_CENTROIDS) return null

        // 3. RANSAC + 唯一性内点计分。
        val rng = java.util.Random(42)
        val n = pairNode.size
        val thr = INLIER_CELL * cell
        val inputArea = quadArea(corners)
        val inputDiag = max(
            hypot(corners[2] - corners[0], corners[3] - corners[1]),
            hypot(corners[4] - corners[6], corners[5] - corners[7]),
        )

        var bestH: FloatArray? = null
        var bestScore = 0
        for (it in 0 until RANSAC_ITERS) {
            val i0 = rng.nextInt(n); val i1 = rng.nextInt(n)
            val i2 = rng.nextInt(n); val i3 = rng.nextInt(n)
            if (i0 == i1 || i0 == i2 || i0 == i3 || i1 == i2 || i1 == i3 || i2 == i3) continue
            val idx = intArrayOf(i0, i1, i2, i3)
            val src = FloatArray(8)
            val dst = FloatArray(8)
            for (k in 0 until 4) {
                src[k * 2] = pairNode[idx[k]][0]; src[k * 2 + 1] = pairNode[idx[k]][1]
                dst[k * 2] = pairCent[idx[k]][0]; dst[k * 2 + 1] = pairCent[idx[k]][1]
            }
            val h = try { Homography.compute(src, dst) } catch (t: Throwable) { continue }
            if (!isValidRefine(h, corners, inputArea, inputDiag, cell)) continue
            val result = uniqueInliers(h, pairNode, pairCent, thr)
            if (result.score > bestScore) { bestScore = result.score; bestH = h }
        }
        val h = bestH ?: return null
        // 结构双闸：内点数 + 行列覆盖（垃圾/压缩解能凑内点数，但覆盖不了整个棋盘）。
        if (bestScore < MIN_INLIERS) return null
        val final = uniqueInliers(h, pairNode, pairCent, thr)
        if (final.score < MIN_INLIERS ||
            final.files < MIN_SPREAD_FILES || final.ranks < MIN_SPREAD_RANKS
        ) return null

        // 4. 精修四角 = 校正单应投影物理四角。
        val out = FloatArray(8)
        for (k in 0 until 4) {
            val p = Homography.transform(
                h, BoardGeometry.physicalCorners[k * 2], BoardGeometry.physicalCorners[k * 2 + 1]
            )
            out[k * 2] = p[0]; out[k * 2 + 1] = p[1]
        }
        return out
    }

    /** 校正单应有效性：投影物理四角面积不塌缩 / 不爆炸，且单角位移受钳制。 */
    private fun isValidRefine(
        h: FloatArray, corners: FloatArray, inputArea: Float, inputDiag: Float, cell: Float
    ): Boolean {
        val maxMove = REFINE_MAX_CORNER_MOVE * cell
        var moved = 0f
        for (k in 0 until 4) {
            val p = Homography.transform(
                h, BoardGeometry.physicalCorners[k * 2], BoardGeometry.physicalCorners[k * 2 + 1]
            )
            val d = hypot(p[0] - corners[k * 2], p[1] - corners[k * 2 + 1])
            if (d > maxMove) return false
            moved = max(moved, d)
        }
        val projected = FloatArray(8)
        for (k in 0 until 4) {
            val p = Homography.transform(
                h, BoardGeometry.physicalCorners[k * 2], BoardGeometry.physicalCorners[k * 2 + 1]
            )
            projected[k * 2] = p[0]; projected[k * 2 + 1] = p[1]
        }
        val area = quadArea(projected)
        if (inputArea <= 0f) return false
        val ratio = area / inputArea
        return ratio in 0.7f..1.4f && inputDiag > 0f
    }

    /** 唯一性内点评估结果：一对一内点数 + 内点覆盖的列数/行数。 */
    private class InlierStats(val score: Int, val files: Int, val ranks: Int)

    /** 唯一性内点：每个质心至多被一个节点认领（拒退化线解），并统计行列覆盖。 */
    private fun uniqueInliers(
        h: FloatArray, nodes: List<FloatArray>, cents: List<FloatArray>, thr: Float
    ): InlierStats {
        val claimed = BooleanArray(cents.size)
        val nodeUsed = BooleanArray(nodes.size)
        // 按距离升序贪心配对，保证一对一。
        val pairs = ArrayList<Triple<Float, Int, Int>>(nodes.size)
        for (i in nodes.indices) {
            val p = Homography.transform(h, nodes[i][0], nodes[i][1])
            for (j in cents.indices) {
                val d = hypot(p[0] - cents[j][0], p[1] - cents[j][1])
                if (d < thr) pairs.add(Triple(d, i, j))
            }
        }
        pairs.sortBy { it.first }
        val filesSeen = HashSet<Int>()
        val ranksSeen = HashSet<Int>()
        var score = 0
        for ((d, i, j) in pairs) {
            if (nodeUsed[i] || claimed[j]) continue
            nodeUsed[i] = true; claimed[j] = true; score++
            val f = ((nodes[i][0] - BoardGeometry.MARGIN) / BoardGeometry.CELL).toInt()
            val r = ((nodes[i][1] - BoardGeometry.MARGIN) / BoardGeometry.CELL).toInt()
            filesSeen.add(f)
            ranksSeen.add(r)
        }
        return InlierStats(score, filesSeen.size, ranksSeen.size)
    }

    private fun quadArea(c: FloatArray): Float {
        var s = 0f
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            s += c[i * 2] * c[j * 2 + 1] - c[j * 2] * c[i * 2 + 1]
        }
        return abs(s) / 2f
    }
}
