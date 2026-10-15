package com.arability.chinese_chess_coach_phone_version.sensors

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * C9-T2 旋转预测器（《BoardTracker帧间跟踪方案.md》§3 移植）：LK 失效（运动模糊/
 * 快甩）时用陀螺积分的旋转单应 H_rot = K·ΔR·K⁻¹ 补位传播，直至 LK 恢复或
 * EdgeSAM 重锚。LK 成功帧用「ΔR_gyro 与 H_lk 旋转分量差」在线估计陀螺零偏。
 *
 * 分工：LK 为主、IMU 旋转预测补位——两者失效模式互补（LK 怕模糊/快甩/弱纹理，
 * IMU 怕漂移但旋转短期精确）。纯逻辑，JVM 可单测。
 *
 * 必需工程件（真机接线，见方案 §3）：
 *  - 内参 K：fx=fy=W/(2·tan(fovH/2))，主点取画面中心（畸变忽略——主摄棋盘区域
 *    边缘畸变 <1-2px）；fovH 由 Camera2Interop 读或标定一次。
 *  - 帧-IMU 时间戳对齐：相机时基（imageProxy.timestamp）vs 传感器时基
 *    （SensorEvent.timestamp）的常数偏移由 [setTimeOffset] 注入；「静止时 ΔR 应为
 *    零」可在线校准（真机调试项）。
 */
class RotationPredictor(
    /** 相机内参（像素）。 */
    val fx: Float, val fy: Float, val cx: Float, val cy: Float,
) {
    companion object {
        /** 零偏 EMA 平滑系数（每成功帧）。 */
        const val BIAS_ALPHA = 0.05f
        /** 内积缓冲样本数上限（~50Hz×4s，防泄漏）。 */
        private const val MAX_SAMPLES = 256

        /** 由图像宽高 + 水平 FOV（度）构造。 */
        fun fromFov(imgW: Int, imgH: Int, fovHDeg: Float): RotationPredictor {
            val f = imgW.toFloat() / (2f * kotlin.math.tan(
                Math.toRadians(fovHDeg.toDouble()) / 2.0).toFloat())
            return RotationPredictor(f, f, imgW / 2f, imgH / 2f)
        }

        /** 旋转向量 → 3×3 旋转矩阵（Rodrigues，行优先 9 float）。 */
        fun rodrigues(rx: Float, ry: Float, rz: Float): FloatArray {
            val theta = sqrt(rx * rx + ry * ry + rz * rz)
            if (theta < 1e-8f) {
                val x = rx / 2f; val y = ry / 2f; val z = rz / 2f   // 一阶近似
                return floatArrayOf(
                    1f, -z, y,
                    z, 1f, -x,
                    -y, x, 1f)
            }
            val kx = rx / theta; val ky = ry / theta; val kz = rz / theta
            val c = cos(theta); val s = sin(theta); val t = 1f - c
            return floatArrayOf(
                t * kx * kx + c, t * kx * ky - s * kz, t * kx * kz + s * ky,
                t * kx * ky + s * kz, t * ky * ky + c, t * ky * kz - s * kx,
                t * kx * kz - s * ky, t * ky * kz + s * kx, t * kz * kz + c)
        }

        /** 3×3 旋转矩阵 → 旋转向量（轴角；行优先）。 */
        fun rotationVector(r: FloatArray): FloatArray {
            val cosA = ((r[0] + r[4] + r[8]) - 1f) / 2f
            val a = kotlin.math.acos(cosA.coerceIn(-1f, 1f))
            if (a < 1e-6f) {
                return floatArrayOf((r[7] - r[5]) / 2f, (r[2] - r[6]) / 2f, (r[3] - r[1]) / 2f)
            }
            var rx = r[7] - r[5]; var ry = r[2] - r[6]; var rz = r[3] - r[1]
            val norm = sqrt(rx * rx + ry * ry + rz * rz).coerceAtLeast(1e-12f)
            rx = rx / norm; ry = ry / norm; rz = rz / norm
            return floatArrayOf(rx * a, ry * a, rz * a)
        }

        private fun mul3(a: FloatArray, b: FloatArray): FloatArray {
            val out = FloatArray(9)
            for (i in 0 until 3) for (j in 0 until 3) {
                var s = 0f
                for (k in 0 until 3) s += a[i * 3 + k] * b[k * 3 + j]
                out[i * 3 + j] = s
            }
            return out
        }

        private fun apply3(m: FloatArray, x: Float, y: Float, z: Float): FloatArray =
            floatArrayOf(
                m[0] * x + m[1] * y + m[2] * z,
                m[3] * x + m[4] * y + m[5] * z,
                m[6] * x + m[7] * y + m[8] * z)
    }

    /** 相机时基 − 传感器时基 的常数偏移（ns）：相机帧 t_cam 对应传感器时刻 t_cam − offset。 */
    var timeOffsetNs: Long = 0L

    /** 陀螺零偏（rad/s），在线估计。 */
    var bias = FloatArray(3)
        private set

    private val ts = LongArray(MAX_SAMPLES)
    private val wx = FloatArray(MAX_SAMPLES)
    private val wy = FloatArray(MAX_SAMPLES)
    private val wz = FloatArray(MAX_SAMPLES)
    private var head = 0    // 环形缓冲：[head, head+count)
    private var count = 0

    /** 传感器时基时间戳（SensorEvent.timestamp）。 */
    fun pushGyro(tNs: Long, x: Float, y: Float, z: Float) {
        val i = (head + count) % MAX_SAMPLES
        ts[i] = tNs; wx[i] = x; wy[i] = y; wz[i] = z
        if (count < MAX_SAMPLES) count++ else head = (head + 1) % MAX_SAMPLES
    }

    /**
     * [t0Ns, t1Ns)（传感器时基）区间的 ΔR，含零偏校正。
     * 样本不足/区间无覆盖返回 null。
     */
    fun deltaRotation(t0Ns: Long, t1Ns: Long): FloatArray? {
        if (t1Ns <= t0Ns || count < 2) return null
        var r = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        var prevT = t0Ns
        var any = false
        var lastX = 0f; var lastY = 0f; var lastZ = 0f
        forEachSampleInRange(t0Ns, t1Ns) { t, x, y, z ->
            val dt = (t - prevT) / 1e9f
            prevT = t
            if (dt > 0f) {
                r = mul3(rodrigues(
                    (x - bias[0]) * dt, (y - bias[1]) * dt, (z - bias[2]) * dt), r)
                any = true
            }
            lastX = x; lastY = y; lastZ = z
        }
        // 尾段零阶保持：最后样本的 ω 覆盖 [prevT, t1]
        if (any && prevT < t1Ns) {
            val dt = (t1Ns - prevT) / 1e9f
            r = mul3(rodrigues(
                (lastX - bias[0]) * dt, (lastY - bias[1]) * dt, (lastZ - bias[2]) * dt), r)
        }
        return if (any) r else null
    }

    /** H_rot = K·ΔR·K⁻¹（2D 单应，行优先 9 float，h33≈1）。 */
    fun rotationHomography(t0Ns: Long, t1Ns: Long): FloatArray? {
        val dr = deltaRotation(t0Ns, t1Ns) ?: return null
        val k = floatArrayOf(fx, 0f, cx, 0f, fy, cy, 0f, 0f, 1f)
        val kInv = floatArrayOf(1f / fx, 0f, -cx / fx, 0f, 1f / fy, -cy / fy, 0f, 0f, 1f)
        val h = mul3(mul3(k, dr), kInv)
        // 归一化 h33=1（Homography.compute 约定）
        val s = h[8]
        if (abs(s) < 1e-12f) return null
        return FloatArray(9) { h[it] / s }
    }

    /**
     * 零偏在线估计（LK 成功帧调用）：H_lk 的旋转分量（K⁻¹·H_lk·K 的反对称部分）
     * vs 原始陀螺积分（不带零偏），差值/dt 为零偏样本，EMA 平滑。
     */
    fun updateBias(t0Ns: Long, t1Ns: Long, hLk: FloatArray) {
        if (t1Ns <= t0Ns || count < 2) return
        val k = floatArrayOf(fx, 0f, cx, 0f, fy, cy, 0f, 0f, 1f)
        val kInv = floatArrayOf(1f / fx, 0f, -cx / fx, 0f, 1f / fy, -cy / fy, 0f, 0f, 1f)
        val m = mul3(mul3(kInv, hLk), k)
        // 归一化尺度（平移项 t·nᵀ/d 会混入；一阶取均值尺度）
        val sc = (sqrt(abs(m[0] * m[4] * m[8])) + sqrt(abs(m[8]))) / 2f
        if (sc < 1e-9f) return
        val vision = floatArrayOf(
            (m[7] - m[5]) / 2f / sc, (m[2] - m[6]) / 2f / sc, (m[3] - m[1]) / 2f / sc)
        // 原始陀螺积分（不扣零偏）
        val savedBias = bias.copyOf()
        bias = FloatArray(3)
        val raw = deltaRotation(t0Ns, t1Ns)
        bias = savedBias
        val rvRaw = raw?.let { rotationVector(it) } ?: return
        val dt = (t1Ns - t0Ns) / 1e9f
        if (dt <= 0f) return
        for (i in 0 until 3) {
            val sample = (rvRaw[i] - vision[i]) / dt
            bias[i] += BIAS_ALPHA * (sample - bias[i])
        }
    }

    fun reset() {
        head = 0; count = 0
        bias = FloatArray(3)
    }

    /** 遍历 (t0, t1] 内样本（升序）；末尾补一个 t1 的“保持”样本以覆盖区间尾部。 */
    private inline fun forEachSampleInRange(
        t0Ns: Long, t1Ns: Long,
        crossinline block: (Long, Float, Float, Float) -> Unit,
    ) {
        val n = count
        // 收集区间内样本（保持时间顺序：环形缓冲从旧到新）
        for (idx in 0 until n) {
            val i = (head + idx) % MAX_SAMPLES
            val t = ts[i]
            if (t <= t0Ns) continue
            if (t > t1Ns) break
            block(t, wx[i], wy[i], wz[i])
        }
    }
}
