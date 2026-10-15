package com.arability.chinese_chess_coach_phone_version.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** IMU 静止判据三态机单测（IMU 静止分支设计方案.md §3：进慢出快迟滞）。 */
class ImuStaticDetectorTest {

    private val t0 = 1_000_000_000L     // 任意起点（ns）
    private val tick = 20_000_000L      // 50Hz

    /** 共享时钟：多次 pushSeconds 连续推进（时间戳回退会污染窗口保持计时）。 */
    private var clockNs = t0

    private fun push(d: ImuStaticDetector, t: Long, gyro: Float, acc: Float) {
        d.pushGyro(t, gyro)
        d.pushAcc(t, acc)
    }

    private fun pushSeconds(d: ImuStaticDetector, seconds: Double, gyro: Float, acc: Float) {
        val n = (seconds * 1e9 / tick).toInt()
        repeat(n) {
            push(d, clockNs, gyro, acc)
            clockNs += tick
        }
    }

    @Test
    fun `静止序列保持1秒后进入STATIC`() {
        val d = ImuStaticDetector()
        assertEquals(ImuState.UNKNOWN, d.state)
        pushSeconds(d, 0.5, 0.01f, 9.8f)
        assertTrue("0.5s（<T_HOLD_IN=1s）不应进静止：${d.state}", d.state != ImuState.STATIC)
        // 窗口判据从首个「数据足够」样本（~0.2s）起算，需再保持 ≥1s → 总时长 ≥1.5s
        pushSeconds(d, 1.2, 0.01f, 9.8f)
        assertEquals(ImuState.STATIC, d.state)
    }

    @Test
    fun `出快_静止中单帧破坏立即MOVING`() {
        val d = ImuStaticDetector()
        pushSeconds(d, 1.5, 0.01f, 9.8f)
        assertEquals(ImuState.STATIC, d.state)
        push(d, t0 + 1_500_000_000L, 0.5f, 9.8f)   // 一个大陀螺样本
        assertEquals(ImuState.MOVING, d.state)
    }

    @Test
    fun `数据不足保持UNKNOWN`() {
        val d = ImuStaticDetector()
        repeat(5) { push(d, t0 + it * tick, 0.01f, 9.8f) }
        assertEquals(ImuState.UNKNOWN, d.state)
    }

    @Test
    fun `加速度抖动判移动`() {
        val d = ImuStaticDetector()
        var t = t0
        repeat(100) {
            val acc = if (it % 2 == 0) 9.5f else 10.5f   // std≈0.5 > 0.15
            push(d, t, 0.01f, acc)
            t += tick
        }
        assertEquals(ImuState.MOVING, d.state)
    }

    @Test
    fun `模长偏移重力带判移动`() {
        val d = ImuStaticDetector()
        pushSeconds(d, 1.5, 0.01f, 8.0f)   // 自由落体段：‖a‖≈0，均值出带
        assertEquals(ImuState.MOVING, d.state)
    }

    @Test
    fun `移动后恢复静止需重新保持1秒`() {
        val d = ImuStaticDetector()
        pushSeconds(d, 1.5, 0.01f, 9.8f)
        assertEquals(ImuState.STATIC, d.state)
        // 移动 0.5s
        pushSeconds(d, 0.5, 0.3f, 9.8f)
        assertEquals(ImuState.MOVING, d.state)
        // 恢复静止：0.5s 不够（窗口仍含移动样本，判据尚未开始保持）
        pushSeconds(d, 0.5, 0.01f, 9.8f)
        assertTrue(d.state != ImuState.STATIC)
        // 判据自窗口干净起保持 ≥1s → 再推 1.2s 应回到静止
        pushSeconds(d, 1.2, 0.01f, 9.8f)
        assertEquals(ImuState.STATIC, d.state)
    }
}
