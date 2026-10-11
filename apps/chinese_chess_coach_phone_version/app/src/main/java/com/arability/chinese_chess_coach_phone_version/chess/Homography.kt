package com.arability.chinese_chess_coach_phone_version.chess

/**
 * 单应性（透视变换）工具：从 4 组点对应求解 3×3 单应矩阵，并做透视校正采样。
 */
object Homography {

    /**
     * 由 4 组点对应求解单应矩阵，使 src 映射到 dst。
     *
     * @param src 4 个源点，顺序 x0,y0,x1,y1,...（8 个 float）。
     * @param dst 4 个目标点，同上。
     * @return 3×3 矩阵（h33 = 1），按行优先排列（9 个 float）。
     */
    fun compute(src: FloatArray, dst: FloatArray): FloatArray {
        // 构建 8x9 增广矩阵，未知量 h0..h7（h8=1）
        val a = Array(8) { FloatArray(9) }
        for (i in 0 until 4) {
            val x = src[i * 2]
            val y = src[i * 2 + 1]
            val u = dst[i * 2]
            val v = dst[i * 2 + 1]
            a[i * 2] = floatArrayOf(x, y, 1f, 0f, 0f, 0f, -x * u, -y * u, u)
            a[i * 2 + 1] = floatArrayOf(0f, 0f, 0f, x, y, 1f, -x * v, -y * v, v)
        }
        val h = FloatArray(8)
        gaussSolve(a, h)
        return floatArrayOf(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1f)
    }

    /**
     * 应用单应矩阵，把源点 (x, y) 映射到目标坐标。
     *
     * @param h 由 [compute] 求得的 3×3 矩阵（h33=1）。
     */
    fun transform(h: FloatArray, x: Float, y: Float): FloatArray {
        val w = h[6] * x + h[7] * y + h[8]
        val tx = (h[0] * x + h[1] * y + h[2]) / w
        val ty = (h[3] * x + h[4] * y + h[5]) / w
        return floatArrayOf(tx, ty)
    }

    // 历史 warp 位图采样（warp/sampleBilinear）已随 BoardInfo.warped 移除（2026-10-11，C1/C8）。

    /** 高斯消元求解 8×9 增广矩阵，结果写入 [out]（长度 8）。 */
    private fun gaussSolve(a: Array<FloatArray>, out: FloatArray) {
        val n = 8
        for (col in 0 until n) {
            var pivot = col
            for (row in col + 1 until n) {
                if (kotlin.math.abs(a[row][col]) > kotlin.math.abs(a[pivot][col])) pivot = row
            }
            val tmp = a[col]; a[col] = a[pivot]; a[pivot] = tmp
            val div = a[col][col]
            if (kotlin.math.abs(div) < 1e-9f) continue
            for (j in col until n + 1) a[col][j] /= div
            for (row in 0 until n) {
                if (row == col) continue
                val factor = a[row][col]
                for (j in col until n + 1) a[row][j] -= factor * a[col][j]
            }
        }
        for (i in 0 until n) out[i] = a[i][n]
    }
}
