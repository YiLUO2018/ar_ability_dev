package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * P3-a 角点外推复原合成测试（与 PC 复刻 test_geom_reconstruct.py 同 5 例）：
 * 全可见 / 1 角出画 / 2 对角出画 / 整边出画（应拒，交由贴边校验兜底）/ 角微出画。
 */
class ClippedQuadReconstructTest {

    private fun fillQuad(w: Int, h: Int, quad: Array<Pair<Float, Float>>): ByteArray {
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val px = x + 0.5f
                val py = y + 0.5f
                var inside = true
                for (i in 0 until 4) {
                    val a = quad[i]
                    val b = quad[(i + 1) % 4]
                    val cr = (b.first - a.first) * (py - a.second) -
                        (b.second - a.second) * (px - a.first)
                    if (cr < 0f) inside = false
                }
                if (inside) out[y * w + x] = 1
            }
        }
        return out
    }

    private fun maxCornerError(corners: FloatArray, quad: Array<Pair<Float, Float>>): Float {
        var worst = 0f
        for (t in quad) {
            var best = Float.MAX_VALUE
            for (i in 0 until 4) {
                val d = kotlin.math.hypot(
                    (corners[i * 2] - t.first).toDouble(),
                    (corners[i * 2 + 1] - t.second).toDouble()
                ).toFloat()
                if (d < best) best = d
            }
            if (best > worst) worst = best
        }
        return worst
    }

    private fun runCase(quad: Array<Pair<Float, Float>>): Pair<FloatArray, Boolean>? {
        val binary = fillQuad(256, 256, quad)
        val mq = SegmentationBoardDetector.extractCornersEx(binary, 256, 256, 256f, 256f)
            ?: return null
        return mq.corners to mq.reconstructed
    }

    @Test
    fun fullVisible_noReconstruction() {
        val quad = arrayOf(
            30f to 40f, 220f to 30f, 230f to 220f, 20f to 210f
        )
        val r = runCase(quad)
        assertNotNull(r)
        assertFalse(r!!.second)
        assertTrue(maxCornerError(r.first, quad) < 4f)
    }

    @Test
    fun oneCornerOutOfFrame_reconstructed() {
        val quad = arrayOf(
            30f to 60f, 280f to -10f, 200f to 230f, 20f to 200f
        )
        val r = runCase(quad)
        assertNotNull(r)
        assertTrue(r!!.second)
        assertTrue("maxErr=${maxCornerError(r.first, quad)}", maxCornerError(r.first, quad) < 4f)
    }

    @Test
    fun twoOppositeCornersOutOfFrame_reconstructed() {
        val quad = arrayOf(
            -30f to -20f, 230f to 40f, 290f to 270f, 30f to 230f
        )
        val r = runCase(quad)
        assertNotNull(r)
        assertTrue(r!!.second)
        assertTrue(maxCornerError(r.first, quad) < 4f)
    }

    @Test
    fun wholeEdgeOutOfFrame_notReconstructed() {
        // 顶边整条出画：无该边信息，复原应不触发（返回贪心兜底角点，由贴边校验拒绝）。
        val quad = arrayOf(
            -20f to -30f, 270f to -20f, 230f to 200f, 30f to 230f
        )
        val r = runCase(quad)
        assertNotNull(r)
        assertFalse(r!!.second)
    }

    @Test
    fun tinyClip_reconstructed() {
        val quad = arrayOf(
            40f to 40f, 258f to 45f, 210f to 220f, 25f to 205f
        )
        val r = runCase(quad)
        assertNotNull(r)
        assertTrue(r!!.second)
        assertTrue(maxCornerError(r.first, quad) < 4f)
    }
}
