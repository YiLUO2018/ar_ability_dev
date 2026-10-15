package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** BoardTracker 纯 Kotlin 数学层（HomographyMath）单测——不依赖 OpenCV，JVM 直跑。 */
class HomographyMathTest {

    private fun approx(a: FloatArray, b: FloatArray, tol: Float = 0.5f): Boolean {
        for (i in a.indices) if (abs(a[i] - b[i]) > tol) return false
        return true
    }

    @Test
    fun `mat mul and invert round trip to identity`() {
        val h = floatArrayOf(
            1.2f, 0.1f, 30f,
            -0.05f, 0.9f, -12f,
            0.0001f, -0.0002f, 1f,
        )
        val inv = HomographyMath.invert(h)
        assertNotNull(inv)
        val identity = HomographyMath.mul(h, inv!!)
        val expect = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        assertTrue(approx(identity, expect, 1e-3f))
    }

    @Test
    fun `apply maps the 4 defining points back`() {
        val src = floatArrayOf(0f, 0f, 100f, 0f, 100f, 80f, 0f, 80f)
        val dst = floatArrayOf(12f, 34f, 210f, 30f, 205f, 190f, 8f, 180f)
        val h = Homography.compute(src, dst)
        assertTrue(approx(HomographyMath.apply(h, src), dst, 0.05f))
    }

    @Test
    fun `ransac recovers homography with outliers present`() {
        // 真单应：canonical 棋盘 → 图像（平移+缩放+轻微透视）
        val hTrue = Homography.compute(
            floatArrayOf(0f, 0f, 450f, 0f, 450f, 500f, 0f, 500f),
            floatArrayOf(40f, 60f, 890f, 45f, 905f, 950f, 25f, 930f),
        )
        val n = 20
        val src = FloatArray(n * 2); val dst = FloatArray(n * 2)
        for (i in 0 until n) {
            src[i * 2] = 25f + i * 19f; src[i * 2 + 1] = 25f + (i % 7) * 61f
            val p = HomographyMath.apply(hTrue, floatArrayOf(src[i * 2], src[i * 2 + 1]))
            dst[i * 2] = p[0]; dst[i * 2 + 1] = p[1]
        }
        // 5 个外点（错配 80px）
        for (i in 15 until n) { dst[i * 2] += 80f; dst[i * 2 + 1] += 30f }
        val res = HomographyMath.ransacHomography(src, dst, n, 5f, 12)
        assertNotNull(res)
        assertEquals(15, res!!.second)
        // 恢复的 H 应把内点映准
        val mapped = HomographyMath.apply(res.first, src)
        for (i in 0 until 15) {
            assertTrue(abs(mapped[i * 2] - dst[i * 2]) < 3f)
            assertTrue(abs(mapped[i * 2 + 1] - dst[i * 2 + 1]) < 3f)
        }
    }

    @Test
    fun `ransac returns null when too few inliers`() {
        val src = FloatArray(16); val dst = FloatArray(16)
        for (i in 0 until 8) {
            src[i * 2] = i * 30f; src[i * 2 + 1] = i * 10f
            dst[i * 2] = i * 30f + 3f; dst[i * 2 + 1] = i * 10f - 2f
        }
        val res = HomographyMath.ransacHomography(src, dst, 8, 2f, 12)
        assertNull(res)
    }

    @Test
    fun `uniqueNearest matches one-to-one within radius`() {
        val proj = floatArrayOf(10f, 10f, 100f, 10f, 200f, 10f, 305f, 10f)
        val det = floatArrayOf(12f, 11f, 98f, 12f, 100.5f, 9f, 500f, 500f)
        val pairs = HomographyMath.uniqueNearest(proj, det, 30f)
        // (200,10)/(305,10) 的最近质心都 >30 → 不匹配；剩下两对一对一
        assertEquals(2, pairs.size / 2)
        val matched = HashSet<Pair<Int, Int>>()
        for (k in 0 until pairs.size / 2) matched.add(pairs[k * 2] to pairs[k * 2 + 1])
        assertEquals(setOf(0 to 0, 1 to 2), matched)
    }

    @Test
    fun `pcaPairSets recovers correspondence under rotation scale translation`() {
        // 棋盘系参考云（标准开局 32 交点，去 3 个点打破 180° 旋转对称——对称云下
        // PCA 180° 组合同样 count 满分，需旋转一致闸区分，见下个测试）
        val ref = ArrayList<Float>()
        for (f in intArrayOf(0, 2, 4, 6, 8)) {
            for (r in intArrayOf(0, 9)) {
                if (f == 8 && r == 0) continue   // 打破对称
                ref.add(25f + f * 50f); ref.add(25f + r * 50f)
            }
        }
        for (r in intArrayOf(2, 7)) {
            for (f in intArrayOf(1, 7)) { ref.add(25f + f * 50f); ref.add(25f + r * 50f) }
        }
        for (f in intArrayOf(0, 2, 4, 6, 8)) {
            for (r in intArrayOf(3, 6)) {
                if (f == 2 && r == 6) continue   // 再打破对称
                ref.add(25f + f * 50f); ref.add(25f + r * 50f)
            }
        }
        val refArr = ref.toFloatArray()
        // 相似变换：旋转 25° + 尺度 1.6 + 平移
        val ang = Math.toRadians(25.0); val ca = cos(ang).toFloat(); val sa = sin(ang).toFloat()
        val scale = 1.6f; val tx = 300f; val ty = -50f
        val n = refArr.size / 2
        val detArr = FloatArray(n * 2)
        for (i in 0 until n) {
            val x = refArr[i * 2]; val y = refArr[i * 2 + 1]
            detArr[i * 2] = (ca * x - sa * y) * scale + tx
            detArr[i * 2 + 1] = (sa * x + ca * y) * scale + ty
        }
        val sets = HomographyMath.pcaPairSets(refArr, detArr, 1.2f * 50f * scale)
        assertTrue(sets.isNotEmpty())
        val best = sets.first()
        // 正确组合应匹配几乎全部点（≥ n-2）；允许 ≤2 对边界交叉（相邻点吸附越界，
        // 实际由 RANSAC 内点闸兜住为外点）
        assertTrue("best=${best.count} n=$n", best.count >= n - 2)
        val sameIdx = (0 until best.count).count { best.pairs[it * 2] == best.pairs[it * 2 + 1] }
        assertTrue("sameIdx=$sameIdx", sameIdx >= n - 2)
        // 拟合出的 H det>0（非反射）且能把参考映到质心
        val fit = HomographyMath.ransacHomography(best.src, best.dst, best.count, 10f, 10)
        assertNotNull(fit)
        assertTrue(HomographyMath.det2(fit!!.first) > 0f)
        val mapped = HomographyMath.apply(fit.first, refArr)
        assertTrue(approx(mapped, detArr, 5f))
    }

    @Test
    fun `axisAngleBetween separates flipped homography from truth`() {
        // 真单应 vs 180° 翻转单应（标准开局云 180° 对称 → 两者都把参考映到棋子上）
        val hTrue = Homography.compute(
            floatArrayOf(0f, 0f, 450f, 0f, 450f, 500f, 0f, 500f),
            floatArrayOf(100f, 80f, 980f, 60f, 1000f, 990f, 80f, 960f),
        )
        val c = BoardGeometry.BOARD_W / 2f
        val cy = BoardGeometry.BOARD_H / 2f
        // 自身 → ~0°
        assertTrue(abs(HomographyMath.axisAngleBetween(hTrue, hTrue, c, cy)) < 0.01f)
        // 180° 旋转的单应（把棋盘系绕中心转 180° 再映像）→ 轴向夹角 ~180°
        val rot180 = floatArrayOf(-1f, 0f, 450f, 0f, -1f, 500f, 0f, 0f, 1f)
        val hFlip = HomographyMath.mul(hTrue, rot180)
        val angFlip = abs(HomographyMath.axisAngleBetween(hFlip, hTrue, c, cy))
        assertTrue("angFlip=$angFlip", angFlip > Math.toRadians(170.0))
        // 小旋转（10°）→ 通过 40° 闸
        val t = Math.toRadians(10.0)
        val rot10 = floatArrayOf(
            cos(t).toFloat(), -sin(t).toFloat(), 0f,
            sin(t).toFloat(), cos(t).toFloat(), 0f,
            0f, 0f, 1f,
        )
        val hRot = HomographyMath.mul(hTrue, rot10)
        assertTrue(abs(HomographyMath.axisAngleBetween(hRot, hTrue, c, cy)) < Math.toRadians(40.0))
    }
}
