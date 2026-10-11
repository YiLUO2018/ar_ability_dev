package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.hypot

/**
 * C4 形状闸 / C7① 质心先验的纯几何单测（companion 函数，无 Android 依赖）。
 */
class CandidateSelectionTest {

    // ------------------------------------------------------------- C4 hasBoardShape

    @Test
    fun shapeGate_acceptsNormalBoardQuad() {
        // 典型斜拍四角（来自真机 board_21 实测）：TL,TR,BR,BL。
        val q = floatArrayOf(
            116f, 76f, 350f, 76f, 375f, 299f, 102f, 306f
        )
        assertTrue(SegmentationBoardDetector.hasBoardShape(q))
    }

    @Test
    fun shapeGate_acceptsStrongPerspectiveQuad() {
        // 强透视：远边（上）只有近边（下）一半长，内角偏离 90°——仍应在宽松界内。
        val q = floatArrayOf(
            160f, 60f, 320f, 60f, 400f, 340f, 60f, 340f
        )
        assertTrue(SegmentationBoardDetector.hasBoardShape(q))
    }

    @Test
    fun shapeGate_rejectsSharpParallelogram() {
        // 强剪切平行四边形（内角 ~27°/153°）——溢出/噪声掩码常见形态，应被拒。
        val sharp = floatArrayOf(
            0f, 0f, 300f, 0f, 460f, 80f, 160f, 80f
        )
        assertFalse(SegmentationBoardDetector.hasBoardShape(sharp))
        // 矩形细条（四角皆 90°、对边比 1）——形状闸不拒，归 isValidQuad 面积下界管。
        val thin = floatArrayOf(
            0f, 0f, 300f, 0f, 300f, 10f, 0f, 10f
        )
        assertTrue(SegmentationBoardDetector.hasBoardShape(thin))
    }

    @Test
    fun shapeGate_rejectsExtremeEdgeRatio() {
        // 上边 100、下边 400——对边比 0.25 超下界 0.35。
        val q = floatArrayOf(
            150f, 0f, 250f, 0f, 400f, 200f, 0f, 200f
        )
        assertFalse(SegmentationBoardDetector.hasBoardShape(q))
    }

    // ------------------------------------------------------------- C7① pointInQuad / containment

    private val boardQuad = floatArrayOf(
        100f, 100f, 400f, 100f, 400f, 400f, 100f, 400f
    )

    @Test
    fun pointInQuad_insideAndOutside() {
        assertTrue(SegmentationBoardDetector.pointInQuad(boardQuad, 250f, 250f))
        assertTrue(SegmentationBoardDetector.pointInQuad(boardQuad, 101f, 101f))
        assertFalse(SegmentationBoardDetector.pointInQuad(boardQuad, 50f, 250f))
        assertFalse(SegmentationBoardDetector.pointInQuad(boardQuad, 450f, 450f))
    }

    @Test
    fun pointInQuad_worksForClockwiseAndCounterClockwise() {
        // 逆时针序（BL,BR,TR,TL）：同样的几何区域。
        val ccw = floatArrayOf(
            100f, 400f, 400f, 400f, 400f, 100f, 100f, 100f
        )
        assertTrue(SegmentationBoardDetector.pointInQuad(ccw, 250f, 250f))
        assertFalse(SegmentationBoardDetector.pointInQuad(ccw, 0f, 0f))
    }

    @Test
    fun containmentRatio_countsCorrectly() {
        val cents = listOf(
            floatArrayOf(150f, 150f),   // in
            floatArrayOf(350f, 350f),   // in
            floatArrayOf(0f, 0f),       // out
            floatArrayOf(250f, 250f),   // in
        )
        assertEquals(0.75f, SegmentationBoardDetector.containmentRatio(boardQuad, cents), 1e-4f)
        assertEquals(0f, SegmentationBoardDetector.containmentRatio(boardQuad, emptyList()), 1e-4f)
    }

    /**
     * C7① 核心语义：真棋盘 = 包住全部质心的**最小**四角。
     * 溢出候选（棋盘+桌面，面积大）同样包住全部质心但更大；碎块候选包不住。
     * 期望挑选棋盘四角而非溢出四角。
     */
    @Test
    fun minimalContainingQuad_beatsOverflowCandidate() {
        // 棋盘内部散布 12 个质心（模拟 YOLO 检出）。
        val cents = ArrayList<FloatArray>()
        for (i in 0 until 3) {
            for (j in 0 until 4) {
                cents.add(floatArrayOf(130f + j * 80f, 130f + i * 100f))
            }
        }
        // 候选1 = 真棋盘（包住全部）。
        val board = floatArrayOf(100f, 100f, 400f, 100f, 400f, 400f, 100f, 400f)
        // 候选2 = 溢出到桌面（包住棋盘 + 一大片桌面，面积远大）。
        val overflow = floatArrayOf(40f, 40f, 480f, 30f, 500f, 470f, 20f, 480f)
        // 候选3 = 碎块（只包住左上角一颗子区域）。
        val fragment = floatArrayOf(120f, 120f, 220f, 120f, 220f, 240f, 120f, 240f)

        assertTrue(SegmentationBoardDetector.containmentRatio(board, cents) >= 0.85f)
        assertTrue(SegmentationBoardDetector.containmentRatio(overflow, cents) >= 0.85f)
        assertTrue(SegmentationBoardDetector.containmentRatio(fragment, cents) < 0.85f)

        // 在包含率达标（≥0.85）的候选中取面积最小 → board（而非 overflow）。
        val qualified = listOf(board, overflow, fragment)
            .filter { SegmentationBoardDetector.containmentRatio(it, cents) >= 0.85f }
        val winner = qualified.minByOrNull { SegmentationBoardDetector.quadAreaOf(it) }!!
        assertEquals(300f * 300f, SegmentationBoardDetector.quadAreaOf(winner), 1f)
        // 胜者与真棋盘四角一致。
        assertTrue(winner.contentEquals(board))
    }

    /** 边界情形：质心已被打乱（棋盘大幅移动后的陈旧质心）→ 无候选达标 → 回退现行为。 */
    @Test
    fun staleCentroids_noCandidateQualified() {
        val staleCents = List(12) { floatArrayOf(1000f + it * 10f, 1000f) }
        val candidates = listOf(
            floatArrayOf(100f, 100f, 400f, 100f, 400f, 400f, 100f, 400f),
            floatArrayOf(40f, 40f, 480f, 30f, 500f, 470f, 20f, 480f),
        )
        assertTrue(candidates.none {
            SegmentationBoardDetector.containmentRatio(it, staleCents) >= 0.85f
        })
    }

    /** hasBoardShape 与视频极端斜角的兼容冒烟：远/近边比 0.5 应通过（界 0.35）。 */
    @Test
    fun shapeGate_boundaryRatio() {
        val q = floatArrayOf(
            125f, 0f, 275f, 0f, 300f, 200f, 0f, 200f
        )
        val top = hypot(q[2] - q[0], q[3] - q[1])
        val bottom = hypot(q[6] - q[4], q[7] - q[5])
        assertEquals(0.5f, top / bottom, 0.01f)
        assertTrue(SegmentationBoardDetector.hasBoardShape(q))
    }
}
