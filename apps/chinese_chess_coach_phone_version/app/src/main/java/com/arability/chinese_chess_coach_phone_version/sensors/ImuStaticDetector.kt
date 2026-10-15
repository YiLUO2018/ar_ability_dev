package com.arability.chinese_chess_coach_phone_version.sensors

/**
 * IMU 静止判定三态：UNKNOWN（数据不足/传感器缺失，静止特权不启用，行为=现状）、
 * MOVING、STATIC（静止特权：锚闸 D1 + 不清屏 D2）。
 */
enum class ImuState { STATIC, MOVING, UNKNOWN }

/**
 * 静止判据纯逻辑（IMU 静止分支，`docs/01-现行设计/IMU静止分支设计方案.md` §3）。
 *
 * 双滑窗判据（窗宽 [T_WIN_NS]）：
 *  - 陀螺静：窗内所有样本各轴幅值 < [GYRO_STATIC_MAX]（盖过 residual bias，典型 <0.02 rad/s；
 *    手持微抖 0.1-0.5 rad/s 会被正确判移动）；
 *  - 加速度静：窗内 ‖a‖ 标准差 < [ACC_STD_MAX] 且均值 ∈ [ACC_NORM_LO, ACC_NORM_HI]。
 *
 * 状态机迟滞「进慢出快」：判据需**连续保持 [T_HOLD_IN_NS]** 才进 STATIC（防移动中的
 * 瞬时安静期误入）；STATIC 中任一窗口判据破坏**立即**出为 MOVING（静止特权只该在
 * 真静止时给）。已知盲区（匀速纯平移）与兜底见设计方案 §3.4。
 *
 * 纯逻辑无 Android 依赖，JVM 可单测（`ImuStaticDetectorTest`）。
 */
class ImuStaticDetector(
    private val gyroMaxThr: Float = GYRO_STATIC_MAX,
    private val accStdThr: Float = ACC_STD_MAX,
    private val accNormLo: Float = ACC_NORM_LO,
    private val accNormHi: Float = ACC_NORM_HI,
    private val tWinNs: Long = T_WIN_NS,
    private val tHoldInNs: Long = T_HOLD_IN_NS,
) {
    /** 传感器线程写、相机线程读（经 [StaticMotionMonitor.state] 快照）→ volatile 可见性。 */
    @Volatile
    var state = ImuState.UNKNOWN
        private set

    /** 诊断快照（标定/落盘用）。 */
    var lastGyroWinMax = 0f
        private set
    var lastAccStd = 0f
        private set
    var lastAccMean = 0f
        private set

    private val gyroT = LongArray(RING)
    private val gyroV = FloatArray(RING)
    private val accT = LongArray(RING)
    private val accV = FloatArray(RING)
    private var gN = 0
    private var aN = 0
    private var criteriaSinceNs = 0L

    fun pushGyro(tNs: Long, maxAxisRadS: Float) {
        val i = gN % RING
        gyroT[i] = tNs
        gyroV[i] = maxAxisRadS
        gN++
        evaluate(tNs)
    }

    fun pushAcc(tNs: Long, normMs2: Float) {
        val i = aN % RING
        accT[i] = tNs
        accV[i] = normMs2
        aN++
        evaluate(tNs)
    }

    private fun evaluate(nowNs: Long) {
        // 窗内样本不足/跨度不够 → 数据不足，保持现状（UNKNOWN 起步）。
        val gOk = windowStats(gyroT, gyroV, gN, nowNs)
        val aOk = windowStats(accT, accV, aN, nowNs)
        val gyroEnough = gOk != null && gOk.count >= MIN_SAMPLES && gOk.spanNs >= tWinNs * 2 / 5
        val accEnough = aOk != null && aOk.count >= MIN_SAMPLES && aOk.spanNs >= tWinNs * 2 / 5
        if (!gyroEnough || !accEnough) return

        lastGyroWinMax = gOk!!.max
        val mean = aOk!!.sum / aOk.count
        val varSum = run {
            var s = 0.0
            for (k in 0 until aOk.count) {
                val d = (aOk.values[k] - mean).toDouble()
                s += d * d
            }
            s
        }
        val std = kotlin.math.sqrt(varSum / aOk.count).toFloat()
        lastAccStd = std
        lastAccMean = mean

        val criteria = gOk.max < gyroMaxThr && std < accStdThr &&
            mean in accNormLo..accNormHi
        if (criteria) {
            if (state != ImuState.STATIC) {
                if (criteriaSinceNs == 0L) criteriaSinceNs = nowNs
                if (nowNs - criteriaSinceNs >= tHoldInNs) state = ImuState.STATIC
            }
        } else {
            criteriaSinceNs = 0L
            state = ImuState.MOVING      // 出快：任一判据破坏立即离开静止
        }
    }

    private class WinStats(val count: Int, val spanNs: Long, val max: Float, val sum: Float, val values: FloatArray)

    private fun windowStats(t: LongArray, v: FloatArray, n: Int, nowNs: Long): WinStats? {
        if (n == 0) return null
        val lo = nowNs - tWinNs
        var count = 0
        var max = 0f
        var sum = 0f
        val values = FloatArray(minOf(n, RING))
        var firstTs = Long.MAX_VALUE
        var lastTs = Long.MIN_VALUE
        val start = if (n > RING) n - RING else 0
        for (k in start until n) {
            val i = k % RING
            if (t[i] < lo) continue
            count++
            max = maxOf(max, v[i])
            sum += v[i]
            values[count - 1] = v[i]
            firstTs = minOf(firstTs, t[i])
            lastTs = maxOf(lastTs, t[i])
        }
        if (count == 0) return null
        return WinStats(count, lastTs - firstTs, max, sum, values.copyOf(count))
    }

    companion object {
        const val GYRO_STATIC_MAX = 0.08f     // rad/s
        const val ACC_STD_MAX = 0.15f         // m/s²
        const val ACC_NORM_LO = 9.3f          // m/s²（重力 ±0.5）
        const val ACC_NORM_HI = 10.3f
        const val T_WIN_NS = 500_000_000L     // 0.5s 滑窗
        const val T_HOLD_IN_NS = 1_000_000_000L  // 连续保持 1.0s 才进 STATIC
        private const val MIN_SAMPLES = 10
        private const val RING = 128          // 50Hz×0.5s=25/传感器，128 足够余量
    }
}
