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
     * 透视校正采样：用「目标→源」的单应矩阵 [hInv] 把 src 图像映射到 dstW×dstH。
     *
     * @param src 源像素（ARGB）。
     * @param hInv 使 dst 坐标映射回 src 坐标的单应矩阵。
     */
    fun warp(src: IntArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int, hInv: FloatArray): IntArray {
        val dst = IntArray(dstW * dstH)
        for (dy in 0 until dstH) {
            for (dx in 0 until dstW) {
                val w = hInv[6] * dx + hInv[7] * dy + hInv[8]
                val sx = (hInv[0] * dx + hInv[1] * dy + hInv[2]) / w
                val sy = (hInv[3] * dx + hInv[4] * dy + hInv[5]) / w
                dst[dy * dstW + dx] = sampleBilinear(src, srcW, srcH, sx, sy)
            }
        }
        return dst
    }

    private fun sampleBilinear(src: IntArray, w: Int, h: Int, x: Float, y: Float): Int {
        val x0 = x.toInt()
        val y0 = y.toInt()
        if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) {
            val cx = x.toInt().coerceIn(0, w - 1)
            val cy = y.toInt().coerceIn(0, h - 1)
            return src[cy * w + cx]
        }
        val fx = x - x0
        val fy = y - y0
        val p00 = src[y0 * w + x0]
        val p10 = src[y0 * w + x0 + 1]
        val p01 = src[(y0 + 1) * w + x0]
        val p11 = src[(y0 + 1) * w + x0 + 1]

        var r = 0; var g = 0; var b = 0
        r += ((1 - fx) * (1 - fy) * ((p00 shr 16) and 0xFF)).toInt()
        g += ((1 - fx) * (1 - fy) * ((p00 shr 8) and 0xFF)).toInt()
        b += ((1 - fx) * (1 - fy) * (p00 and 0xFF)).toInt()
        r += (fx * (1 - fy) * ((p10 shr 16) and 0xFF)).toInt()
        g += (fx * (1 - fy) * ((p10 shr 8) and 0xFF)).toInt()
        b += (fx * (1 - fy) * (p10 and 0xFF)).toInt()
        r += ((1 - fx) * fy * ((p01 shr 16) and 0xFF)).toInt()
        g += ((1 - fx) * fy * ((p01 shr 8) and 0xFF)).toInt()
        b += ((1 - fx) * fy * (p01 and 0xFF)).toInt()
        r += (fx * fy * ((p11 shr 16) and 0xFF)).toInt()
        g += (fx * fy * ((p11 shr 8) and 0xFF)).toInt()
        b += (fx * fy * (p11 and 0xFF)).toInt()
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

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
