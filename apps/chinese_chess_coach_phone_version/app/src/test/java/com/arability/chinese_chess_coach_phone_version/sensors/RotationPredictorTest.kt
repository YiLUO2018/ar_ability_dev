package com.arability.chinese_chess_coach_phone_version.sensors

import kotlin.math.abs
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** T2 旋转预测器单测（纯逻辑）。 */
class RotationPredictorTest {

    private val nsPerSec = 1_000_000_000L

    @Test
    fun `rodrigues round trips with rotation vector`() {
        val rv = floatArrayOf(0.02f, -0.05f, 0.01f)
        val r = RotationPredictor.rodrigues(rv[0], rv[1], rv[2])
        val back = RotationPredictor.rotationVector(r)
        assertArrayEquals(rv, back, 1e-4f)
    }

    @Test
    fun `deltaRotation integrates constant angular rate`() {
        val p = RotationPredictor.fromFov(1920, 1080, 66f)
        // 绕 y 轴 0.2 rad/s，50Hz，2 秒
        var t = 0L
        for (i in 0 until 100) {
            t = i * nsPerSec / 50
            p.pushGyro(t, 0f, 0.2f, 0f)
        }
        val t1 = 2L * nsPerSec
        val dr = p.deltaRotation(0, t1)!!
        // 绕 y 转 0.4 rad：x 轴单位向量映为 (cosθ, 0, −sinθ)（右手系 Ry(+θ)）
        val v = floatArrayOf(dr[0], dr[3], dr[6])
        assertTrue("v0=${v[0]}", abs(v[0] - kotlin.math.cos(0.4f)) < 1e-3f)
        assertTrue("v2=${v[2]}", abs(v[2] + kotlin.math.sin(0.4f)) < 1e-3f)
    }

    @Test
    fun `deltaRotation returns null when window not covered`() {
        val p = RotationPredictor.fromFov(1920, 1080, 66f)
        p.pushGyro(0, 0f, 0f, 0f)
        assertNull(p.deltaRotation(0, nsPerSec))
    }

    @Test
    fun `rotationHomography maps image points like 3d rotation`() {
        val p = RotationPredictor.fromFov(1920, 1080, 66f)
        // 绕 z（光轴）0.1 rad/s × 1s
        for (i in 0 until 50) p.pushGyro(i * nsPerSec / 50, 0f, 0f, 0.1f)
        val h = p.rotationHomography(0, nsPerSec)!!
        assertTrue(abs(h[8] - 1f) < 1e-5f)
        // 中心点不动（绕光轴转，主点即旋转中心）
        val c = com.arability.chinese_chess_coach_phone_version.chess.HomographyMath.apply(
            h, floatArrayOf(p.cx, p.cy))
        assertTrue(abs(c[0] - p.cx) < 0.5f && abs(c[1] - p.cy) < 0.5f)
        // 中心右侧 100px 的点绕中心转 0.1 rad
        val q = com.arability.chinese_chess_coach_phone_version.chess.HomographyMath.apply(
            h, floatArrayOf(p.cx + 100f, p.cy))
        val ex = p.cx + 100f * kotlin.math.cos(0.1f)
        val ey = p.cy + 100f * kotlin.math.sin(0.1f)
        assertTrue("q=(${q[0]},${q[1]})", abs(q[0] - ex) < 1f && abs(q[1] - ey) < 1f)
    }

    @Test
    fun `updateBias converges toward true gyro bias`() {
        val p = RotationPredictor.fromFov(1920, 1080, 66f)
        val trueBias = floatArrayOf(0.01f, -0.006f, 0.004f)
        // 真实角速度 = 0（静止），陀螺输出 = bias
        val t0 = 0L
        for (i in 0 until 50) p.pushGyro(i * nsPerSec / 50, trueBias[0], trueBias[1], trueBias[2])
        val t1 = nsPerSec
        // 视觉 H_lk = 恒等（画面没动）
        val identity = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        repeat(200) { p.updateBias(t0, t1, identity) }
        // EMA 200 步 × α=0.05 → 残差 e^-10 ≈ 4.5e-5
        assertTrue("bias=${p.bias.toList()}", abs(p.bias[0] - trueBias[0]) < 1e-3f)
        assertTrue(abs(p.bias[1] - trueBias[1]) < 1e-3f)
        assertTrue(abs(p.bias[2] - trueBias[2]) < 1e-3f)
        // 零偏校正后 ΔR 应回到恒等附近
        val dr = p.deltaRotation(t0, t1)!!
        assertTrue(abs(dr[0] - 1f) < 1e-2f)
    }

    @Test
    fun `updateBias translation contamination stays small`() {
        // 纯平移单应经 K⁻¹HK 的反对称提取产生 O(t/2f) 假旋转信号（每帧 ~t/(2f)
        // ≈ 8/3000 ≈ 0.003 rad/s 量级）——已知近似性质：有界且方向随平移变化，
        // 稳态污染 ≪ GYRO_STATIC_MAX(0.08)。断言有界而非为零。
        val p = RotationPredictor.fromFov(1920, 1080, 66f)
        p.pushGyro(0, 0f, 0f, 0f)
        p.pushGyro(nsPerSec / 2, 0f, 0f, 0f)
        val hTrans = floatArrayOf(1f, 0f, 8f, 0f, 1f, -3f, 0f, 0f, 1f)
        repeat(200) { p.updateBias(0, nsPerSec, hTrans) }
        assertTrue("bias=${p.bias.toList()}", abs(p.bias[0]) < 0.005f)
        assertTrue(abs(p.bias[1]) < 0.005f)
        assertTrue(abs(p.bias[2]) < 0.005f)
    }
}
