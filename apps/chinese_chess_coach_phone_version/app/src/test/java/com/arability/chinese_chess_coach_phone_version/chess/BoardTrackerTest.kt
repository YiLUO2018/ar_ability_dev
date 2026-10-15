package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.abs
import kotlin.math.hypot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** BoardTracker 纯逻辑单测（LK/OpenCV 层不在此覆盖，真机矩阵验证）。 */
class BoardTrackerTest {

    private val trueCorners = floatArrayOf(
        100f, 120f, 980f, 140f, 1010f, 880f, 70f, 860f)

    /** 标准开局 32 交点（棋盘系），180° 旋转对称。 */
    private fun stdRefBoard(): FloatArray {
        val out = ArrayList<Float>(64)
        val files = intArrayOf(0, 2, 4, 6, 8)
        for (f in files) for (r in intArrayOf(0, 9)) {
            out.add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            out.add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
        for (f in intArrayOf(1, 7)) for (r in intArrayOf(2, 7)) {
            out.add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            out.add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
        for (f in files) for (r in intArrayOf(3, 6)) {
            out.add(BoardGeometry.MARGIN + f * BoardGeometry.CELL.toFloat())
            out.add(BoardGeometry.MARGIN + r * BoardGeometry.CELL.toFloat())
        }
        return out.toFloatArray()
    }

    private fun approx(a: FloatArray, b: FloatArray, tol: Float): Boolean {
        for (i in a.indices) if (abs(a[i] - b[i]) > tol) return false
        return true
    }

    @Test
    fun `seed then corners round trip`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        assertNotNull(t.H)
        assertTrue(approx(t.corners()!!, trueCorners, 0.5f))
        assertEquals(0, t.age)
        assertTrue(t.featuresStale)
    }

    @Test
    fun `accumulate applies lk homography and increments age`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        // 合成 Hlk：平移 (20,5) + 尺度 1.02
        val s = 1.02f
        val hlk = floatArrayOf(s, 0f, 20f, 0f, s, 5f, 0f, 0f, 1f)
        val n = 50
        val src = FloatArray(n * 2); val dst = FloatArray(n * 2)
        val rnd = kotlin.random.Random(7)
        for (i in 0 until n) {
            src[i * 2] = 100f + rnd.nextFloat() * 800f
            src[i * 2 + 1] = 120f + rnd.nextFloat() * 700f
            val p = HomographyMath.apply(hlk, floatArrayOf(src[i * 2], src[i * 2 + 1]))
            dst[i * 2] = p[0]; dst[i * 2 + 1] = p[1]
        }
        assertTrue(t.accumulate(src, dst, n, 1920, 1080))
        assertEquals(1, t.age)
        val expected = HomographyMath.apply(
            HomographyMath.mul(hlk, Homography.compute(BoardGeometry.physicalCorners, trueCorners)),
            BoardGeometry.physicalCorners)
        assertTrue(approx(t.corners()!!, expected, 2f))
    }

    @Test
    fun `accumulate rejects too few pairs`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        val src = FloatArray(20); val dst = FloatArray(20)
        for (i in 0 until 10) { src[i * 2] = i * 30f; src[i * 2 + 1] = i * 20f; dst[i * 2] = i * 30f + 2f; dst[i * 2 + 1] = i * 20f }
        assertFalse(t.accumulate(src, dst, 10, 1920, 1080))
        assertEquals(1, t.nFail)
    }

    @Test
    fun `pieceCorrect pulls drifted H back to piece truth`() {
        val t = BoardTracker()
        val refBoard = stdRefBoard()
        // 注入漂移：图像系平移 (30,20)px —— seed 到漂移四角
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        val drift = floatArrayOf(1f, 0f, 30f, 0f, 1f, 20f, 0f, 0f, 1f)
        val driftedCorners = HomographyMath.apply(
            HomographyMath.mul(drift, hTrue), BoardGeometry.physicalCorners)
        t.seed(driftedCorners)
        // YOLO 质心 = 真值位置（±1.5px 噪声）
        val detPts = HomographyMath.apply(hTrue, refBoard)
        for (i in detPts.indices) detPts[i] += if (i % 2 == 0) 1.5f else -1f
        val shift = t.pieceCorrect(refBoard, detPts)
        assertNotNull(shift)
        assertTrue("shift=$shift", shift!! > 15f)
        assertTrue(approx(t.corners()!!, trueCorners, 4f))
        assertTrue(t.isBacked)
    }

    @Test
    fun `pieceCorrect recovers large drift via pca stage`() {
        val t = BoardTracker()
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        // 漂移 400px（超出一级 0.75 格捕获范围 → 二级 PCA）
        val drift = floatArrayOf(1f, 0f, 400f, 0f, 1f, -260f, 0f, 0f, 1f)
        val driftedCorners = HomographyMath.apply(HomographyMath.mul(drift, hTrue), BoardGeometry.physicalCorners)
        t.seed(driftedCorners)
        val refBoard = stdRefBoard()
        val detPts = HomographyMath.apply(hTrue, refBoard)
        val shift = t.pieceCorrect(refBoard, detPts)
        assertNotNull(shift)
        assertTrue(approx(t.corners()!!, trueCorners, 6f))
    }

    @Test
    fun `pieceCorrect rejects flipped correspondence on symmetric cloud`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        val refBoard = stdRefBoard()
        // detPts = 180° 翻转后的位置（对称云上与原点集重合，但对应关系翻转）
        val rot180 = floatArrayOf(-1f, 0f, BoardGeometry.BOARD_W.toFloat(),
                                  0f, -1f, BoardGeometry.BOARD_H.toFloat(), 0f, 0f, 1f)
        val detPts = HomographyMath.apply(
            HomographyMath.mul(hTrue, rot180), refBoard)
        val before = t.corners()!!.copyOf()
        val shift = t.pieceCorrect(refBoard, detPts)
        // 翻转拟合必须被旋转一致闸拒绝：返回 null 且 H 不变
        if (shift != null) {
            // 唯一合法例外：拟合恰好收敛回真值（对称点集上不可能——180° 组合的 H 就是翻转）
            assertTrue(approx(t.corners()!!, before, 3f))
        } else {
            assertTrue(approx(t.corners()!!, before, 0.01f))
        }
    }

    @Test
    fun `backing expires after CORRECT_BACKING frames`() {
        val t = BoardTracker()
        val refBoard = stdRefBoard()
        val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)
        val detPts = HomographyMath.apply(hTrue, refBoard)
        val driftCorners = HomographyMath.apply(
            HomographyMath.mul(floatArrayOf(1f, 0f, 25f, 0f, 1f, 15f, 0f, 0f, 1f), hTrue),
            BoardGeometry.physicalCorners)
        t.seed(driftCorners)
        assertNotNull(t.pieceCorrect(refBoard, detPts))
        assertTrue(t.isBacked)
        // 推进 age：合成零运动帧（src==dst 精确一致 → Hlk=identity）
        val n = 50
        val rnd = kotlin.random.Random(3)
        val src = FloatArray(n * 2); val dst = FloatArray(n * 2)
        for (i in 0 until n) {
            val x = 100f + rnd.nextFloat() * 800f
            val y = 120f + rnd.nextFloat() * 700f
            src[i * 2] = x; src[i * 2 + 1] = y
            dst[i * 2] = x; dst[i * 2 + 1] = y
        }
        repeat(BoardTracker.CORRECT_BACKING + 1) {
            assertTrue(t.accumulate(src, dst, n, 1920, 1080))
        }
        assertFalse(t.isBacked)
    }

    @Test
    fun `cellSizeOf scales with quad diagonal`() {
        val cell = BoardTracker.cellSizeOf(trueCorners)
        val diag = hypot(1010f - 100f, 880f - 120f)
        val expected = diag * BoardGeometry.CELL / hypot(
            BoardGeometry.BOARD_W.toFloat(), BoardGeometry.BOARD_H.toFloat())
        assertTrue(abs(cell - expected) < 0.5f)
        // 期望格宽 ~85px 量级（910px 对角 / 672.7 × 50）
        assertTrue("cell=$cell", cell in 60f..95f)
    }
}
