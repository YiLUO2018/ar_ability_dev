package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 准入闸 v2+v3 与标准开局约定重置单测（PC replay 定稿行为）。 */
class AdmissionGateTest {

    private val trueCorners = floatArrayOf(
        100f, 120f, 980f, 140f, 1010f, 880f, 70f, 860f)
    private val hTrue = Homography.compute(BoardGeometry.physicalCorners, trueCorners)

    private fun stdGrid(): Array<Array<Piece?>> {
        val g = Array(10) { arrayOfNulls<Piece>(9) }
        val p = Piece(PieceColor.RED, PieceType.PAWN)
        val pawnFiles = intArrayOf(0, 2, 4, 6, 8)
        for (f in 0..8) for (r in intArrayOf(0, 9)) g[r][f] = p
        for (f in intArrayOf(1, 7)) for (r in intArrayOf(2, 7)) g[r][f] = p
        for (f in pawnFiles) for (r in intArrayOf(3, 6)) g[r][f] = p
        return g
    }

    @Test
    fun `stdOpening matches full and tolerant opening but not midgame`() {
        val full = stdGrid()
        assertEquals(32, StdOpening.REF_BOARD.size / 2)
        assertTrue(StdOpening.matches(full))
        // 缺 8 子（斜视角远端漏检）仍匹配（on_std = 24 ≥ 20 且 extra = 0）
        for (f in 0..8 step 2) full[0][f] = null
        for (f in 0..2) full[9][f] = null
        assertTrue("on_std=${count(full)}", StdOpening.matches(full))
        // 中局：6 子走离标准位 → on_std 跌破 20 → 不匹配（约定回退质心投影）
        val mid = stdGrid()
        for (f in intArrayOf(0, 2, 4, 6, 8)) {
            mid[3][f]?.let { mid[3][f] = null; mid[4][f] = it }
        }
        mid[0][1]?.let { mid[0][1] = null; mid[1][1] = it }
        assertFalse(StdOpening.matches(mid))
    }

    private fun count(g: Array<Array<Piece?>>): Int {
        var n = 0
        for (r in 0..9) for (f in 0..8) if (g[r][f] != null) n++
        return n
    }

    @Test
    fun `containment and minEdgeDist basics`() {
        val quad = floatArrayOf(0f, 0f, 100f, 0f, 100f, 100f, 0f, 100f)
        val inside = floatArrayOf(10f, 10f, 90f, 10f, 50f, 50f, 200f, 50f)  // 3/4 in
        assertEquals(0.75f, BoardTracker.containmentRatio(quad, inside), 1e-4f)
        assertTrue(BoardTracker.pointInQuad(quad, 50f, 50f))
        assertFalse(BoardTracker.pointInQuad(quad, 150f, 50f))
        // 内部点距边 ~10；贴边点 ~1
        val pts = floatArrayOf(10f, 10f, 1f, 1f)
        assertTrue(BoardTracker.minEdgeDist(quad, pts) in 0.5f..1.5f)
    }

    @Test
    fun `admitReanchor accepts close quad and holds deviants when backed`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        // 制造背书：先纠偏一次
        val ref = StdOpening.REF_BOARD
        val detPts = HomographyMath.apply(hTrue, ref)
        // 先注入小漂移再纠偏（同 BoardTrackerTest 手法）
        val drifted = HomographyMath.apply(
            HomographyMath.mul(floatArrayOf(1f, 0f, 25f, 0f, 1f, 15f, 0f, 0f, 1f), hTrue),
            BoardGeometry.physicalCorners)
        t.seed(drifted)
        assertTrue(t.pieceCorrect(ref, detPts) != null)
        assertTrue(t.isBacked)
        val refImg = t.refToImage(ref)!!
        // ① 贴合传播的好候选 → ACCEPT
        val good = t.corners()!!
        assertEquals(ReanchorDecision.ACCEPT,
            BoardTracker.admitReanchor(good, t, refImg, true))
        // ② 内收候选（包含率 <0.9）→ HOLD
        val inset = HomographyMath.apply(
            HomographyMath.mul(floatArrayOf(0.80f, 0f, 60f, 0f, 0.80f, 40f, 0f, 0f, 1f), hTrue),
            BoardGeometry.physicalCorners)
        assertEquals(ReanchorDecision.HOLD,
            BoardTracker.admitReanchor(inset, t, refImg, true))
        // ③ 形变/膨胀候选（过包含闸但逐角差 >1.5 格）→ HOLD（v3）
        val far = trueCorners.map { it }.toFloatArray()
        far[4] += 300f; far[5] += 300f   // BR 外拉 300px ≈ 4 格
        assertEquals(ReanchorDecision.HOLD,
            BoardTracker.admitReanchor(far, t, refImg, true))
        // ④ 无参考（首次种子）→ ACCEPT
        assertEquals(ReanchorDecision.ACCEPT,
            BoardTracker.admitReanchor(trueCorners, t, null, true))
    }

    @Test
    fun `dead propagation accepts reanchor even when backed`() {
        // LK 死亡死锁回归（2026-10-15 真机教训）：背书未过期（isBacked=true）但传播
        // 已死（age 冻结）→ v3 若照常生效会永远拦 SAM → 无法重播种。必须 ACCEPT。
        val t = BoardTracker()
        val ref = StdOpening.REF_BOARD
        val detPts = HomographyMath.apply(hTrue, ref)
        val drifted = HomographyMath.apply(
            HomographyMath.mul(floatArrayOf(1f, 0f, 25f, 0f, 1f, 15f, 0f, 0f, 1f), hTrue),
            BoardGeometry.physicalCorners)
        t.seed(drifted)
        assertTrue(t.pieceCorrect(ref, detPts) != null)
        assertTrue(t.isBacked)
        val far = trueCorners.copyOf()
        far[4] += 300f; far[5] += 300f
        assertEquals(ReanchorDecision.ACCEPT,
            BoardTracker.admitReanchor(far, t, t.refToImage(ref), false))
    }

    @Test
    fun `admitReanchor stale escapes when old and unbacked`() {
        val t = BoardTracker()
        t.seed(trueCorners)
        // age 推到 STALE_AGE+（无背书）：合成零运动帧
        val n = 50
        val rnd = kotlin.random.Random(3)
        val src = FloatArray(n * 2); val dst = FloatArray(n * 2)
        for (i in 0 until n) {
            val x = 100f + rnd.nextFloat() * 800f
            val y = 120f + rnd.nextFloat() * 700f
            src[i * 2] = x; src[i * 2 + 1] = y
            dst[i * 2] = x; dst[i * 2 + 1] = y
        }
        repeat(BoardTracker.STALE_AGE + 1) { t.accumulate(src, dst, n, 1920, 1080) }
        assertFalse(t.isBacked)
        val ref = StdOpening.REF_BOARD
        val refImg = t.refToImage(ref)!!
        val inset = HomographyMath.apply(
            HomographyMath.mul(floatArrayOf(0.80f, 0f, 60f, 0f, 0.80f, 40f, 0f, 0f, 1f), hTrue),
            BoardGeometry.physicalCorners)
        // 陈旧且无背书：包含率不达标 → 陈旧逃生（信 SAM 重建）
        assertEquals(ReanchorDecision.STALE_RESET,
            BoardTracker.admitReanchor(inset, t, refImg, true))
    }
}
