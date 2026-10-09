package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.hypot
import kotlin.random.Random

/**
 * QuadRefiner 单测：合成已知单应下的 32 个标准开局棋子质心（含噪声与离群点），
 * 初始四角带系统性偏差（近端底边下移 25px，模拟掩码溢出/阴影条带），
 * 断言精修能收敛回真实四角。
 */
class QuadRefinerTest {

    /** 真实物理四角（图像像素，TL,TR,BR,BL）。 */
    private val trueCorners = floatArrayOf(
        100f, 120f,
        420f, 100f,
        450f, 540f,
        80f, 560f,
    )

    /** 标准开局 32 子的 (file, rank)。 */
    private val stdNodes: List<Pair<Int, Int>> = buildList {
        for (f in 0..8) add(f to 0)          // 黑底线
        add(1 to 2); add(7 to 2)             // 黑炮
        for (f in 0..8 step 2) add(f to 3)   // 黑卒
        for (f in 0..8 step 2) add(f to 6)   // 红兵
        add(1 to 7); add(7 to 7)             // 红炮
        for (f in 0..8) add(f to 9)          // 红底线
    }

    private fun canonicalNode(f: Int, r: Int): FloatArray = floatArrayOf(
        BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat(),
        BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat(),
    )

    private fun maxCornerErr(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in 0 until 4) {
            m = maxOf(m, hypot(a[i * 2] - b[i * 2], a[i * 2 + 1] - b[i * 2 + 1]))
        }
        return m
    }

    @Test
    fun `refine recovers true corners from biased quad`() {
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        val rng = Random(7)
        val centroids = stdNodes.map { (f, r) ->
            val p = Homography.transform(hTrue, canonicalNode(f, r)[0], canonicalNode(f, r)[1])
            floatArrayOf(p[0] + rng.nextFloat() * 4 - 2, p[1] + rng.nextFloat() * 4 - 2)
        }.toMutableList()
        // 两个离群质心（假阳/噪声），RANSAC 应忽略。
        centroids.add(floatArrayOf(30f, 40f))
        centroids.add(floatArrayOf(500f, 20f))

        // 偏差四角：底边（BR,BL）下移 25px——模拟近端掩码溢出。
        val biased = trueCorners.copyOf()
        biased[5] += 25f   // BR.y
        biased[7] += 25f   // BL.y
        assertTrue("初始偏差应显著", maxCornerErr(biased, trueCorners) >= 24f)

        val refined = QuadRefiner.refine(biased, centroids)
        assertTrue("精修应成功", refined != null)
        val err = maxCornerErr(refined!!, trueCorners)
        assertTrue("精修后角点误差应 < 8px（噪声量级），实际 $err", err < 8f)
    }

    @Test
    fun `refine returns null when centroids too few`() {
        val few = stdNodes.take(8).map { (f, r) ->
            val p = Homography.transform(
                Homography.compute(BoardGeometry.physicalCorners, trueCorners),
                canonicalNode(f, r)[0], canonicalNode(f, r)[1],
            )
            floatArrayOf(p[0], p[1])
        }
        assertNull(QuadRefiner.refine(trueCorners, few))
    }

    @Test
    fun `refine returns null on garbage centroids`() {
        val rng = Random(1)
        val garbage = (0 until 32).map {
            floatArrayOf(rng.nextFloat() * 480, rng.nextFloat() * 640)
        }
        assertNull(QuadRefiner.refine(trueCorners, garbage))
    }

    @Test
    fun `refine keeps already correct quad stable`() {
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        val rng = Random(3)
        val centroids = stdNodes.map { (f, r) ->
            val p = Homography.transform(hTrue, canonicalNode(f, r)[0], canonicalNode(f, r)[1])
            floatArrayOf(p[0] + rng.nextFloat() * 4 - 2, p[1] + rng.nextFloat() * 4 - 2)
        }
        val refined = QuadRefiner.refine(trueCorners, centroids)
        assertTrue(refined != null)
        val err = maxCornerErr(refined!!, trueCorners)
        // 4 点精确解会过拟合 ±2px 噪声（实测 ~9.5px ≈ 0.24 格，远小于吸附容差），EMA 会再平滑。
        assertTrue("已正确的四角精修后应保持稳定（< 12px），实际 $err", err < 12f)
    }
}
