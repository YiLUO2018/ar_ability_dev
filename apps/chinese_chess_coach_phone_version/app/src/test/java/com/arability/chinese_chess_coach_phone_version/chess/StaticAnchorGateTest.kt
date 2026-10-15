package com.arability.chinese_chess_coach_phone_version.chess

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** D1 静止锚闸纯逻辑单测（IMU 静止分支设计方案.md §4-§5）。 */
class StaticAnchorGateTest {

    /** 450×500×1.28 的投影四角（TL,TR,BR,BL）：对角 860 → 格宽 ≈63.9px。 */
    private fun baseQuad() = floatArrayOf(
        0f, 0f, 576f, 0f, 576f, 640f, 0f, 640f,
    )

    private fun shifted(q: FloatArray, cornerIdx: Int, dx: Float, dy: Float): FloatArray =
        q.copyOf().also {
            it[cornerIdx * 2] += dx
            it[cornerIdx * 2 + 1] += dy
        }

    private fun settledGate(): StaticAnchorGate =
        StaticAnchorGate().apply {
            repeat(SETTLE_FEEDS) { feed(baseQuad()) }
        }

    @Test
    fun `建锚需要连续一致成功检测`() {
        val g = StaticAnchorGate()
        assertNull(g.anchor)
        repeat(4) { g.feed(baseQuad()) }
        assertNull("第 4 次还未建锚（需 5 次）", g.anchor)
        g.feed(baseQuad())
        assertNotNull("第 5 次建锚", g.anchor)
        assertEquals(baseQuad().toList(), g.anchor!!.toList())
    }

    @Test
    fun `建锚中途不一致即清零重数`() {
        val g = StaticAnchorGate()
        g.feed(baseQuad())
        g.feed(baseQuad())
        // 第 3 次跳到远处（>5px 抖动阈）：窗口清零，从它重新数
        g.feed(shifted(baseQuad(), 0, 50f, 0f))
        g.feed(baseQuad())
        g.feed(baseQuad())
        assertNull("窗口被打断后 3 次不足以建锚", g.anchor)
        g.feed(baseQuad())
        g.feed(baseQuad())
        g.feed(baseQuad())   // 新窗口第 5 次（第 4 次重启后 q,q,q,q,q）→ 建锚
        assertNotNull(g.anchor)
        // 锚取组中位数：最后一组全为 baseQuad → 锚 = baseQuad
        assertEquals(baseQuad().toList(), g.anchor!!.toList())
    }

    @Test
    fun `小偏移接受 大漂移拒绝`() {
        val g = settledGate()
        assertTrue("8px（<0.2 格≈12.8px）应接受", g.check(shifted(baseQuad(), 1, 8f, 0f)))
        assertFalse("20px（>0.2 格）应拒绝", g.check(shifted(baseQuad(), 1, 20f, 0f)))
        assertEquals(1, g.rejectCount)
        // 接受后拒绝计数归零
        assertTrue(g.check(shifted(baseQuad(), 1, 5f, 0f)))
        assertFalse(g.check(shifted(baseQuad(), 2, 0f, 20f)))
        assertEquals(2, g.rejectCount)
    }

    @Test
    fun `静止态持续拒绝不自动重锚_恢复依赖clear`() {
        val g = settledGate()
        val moved = shifted(baseQuad(), 0, 100f, 0f)
        repeat(200) { assertFalse(g.check(moved)) }   // 单角漂移无论拒绝多少次：锚不失效（vis4 教训）
        assertEquals(200, g.rejectCount)
        assertNotNull("锚仍在，拒绝由上层保持显示", g.anchor)
        assertEquals(baseQuad().toList(), g.anchor!!.toList())
        // IMU 离开静止 → clear → 从新位置重新积累
        g.clear()
        assertNull(g.anchor)
        repeat(5) { g.feed(moved) }
        assertEquals(moved.toList(), g.anchor!!.toList())
    }

    @Test
    fun `双确认超时候选要求四角整体平移`() {
        val g = settledGate()
        // 单角漂移（TR +40px，其余角不动）：攒满阈值也不给候选（几何确认否决）
        val drift = shifted(baseQuad(), 1, 40f, 0f)
        repeat(60) { assertFalse(g.check(drift)) }
        assertNull("单角漂移不应给出重锚候选", g.pendingTimeout(60))
        // 四角整体平移 60px（>0.5 格≈32px）：给出候选
        val moved = FloatArray(8) { baseQuad()[it] + 60f }
        assertFalse(g.check(moved))
        val cand = g.pendingTimeout(60)
        assertNotNull("整体平移应给出候选", cand)
        assertEquals(moved.toList(), cand!!.toList())
        // 内容验证否决 → 计数重置 → 不再给候选，需重新攒
        g.resetRejects()
        assertNull(g.pendingTimeout(60))
        assertFalse("重置后再拒一次不达阈值", g.pendingTimeout(59) != null)
    }

    @Test
    fun `clear 清空全部状态`() {
        val g = settledGate()
        g.clear()
        assertNull(g.anchor)
        assertTrue("清锚后放行", g.check(shifted(baseQuad(), 0, 500f, 0f)))
    }

    @Test
    fun `cellSizeOf 按canonical比例换算`() {
        // 450×500 四角 → 对角 672.68 → 格宽恰 50
        val q = floatArrayOf(0f, 0f, 450f, 0f, 450f, 500f, 0f, 500f)
        assertEquals(50f, StaticAnchorGate.cellSizeOf(q), 0.01f)
    }

    companion object {
        private const val SETTLE_FEEDS = 5
    }
}
