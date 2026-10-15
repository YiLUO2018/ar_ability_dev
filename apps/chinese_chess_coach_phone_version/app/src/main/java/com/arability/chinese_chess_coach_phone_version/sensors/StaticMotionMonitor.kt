package com.arability.chinese_chess_coach_phone_version.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * IMU 静止监视器（IMU 静止分支）：陀螺仪 + 加速度计 → [ImuStaticDetector] 三态。
 *
 * - 传感器回调跑在专用 [HandlerThread]（不占主线程，回调只做 O(1) 环形缓冲累加）；
 * - [state] 为 @Volatile 快照，processFrame（相机线程）直接读，无需加锁——
 *   静止分支用滑窗统计而非逐帧积分，**天然不需要 IMU-相机时间戳对齐**（那是
 *   §9 T2 移动预测分支的命门）；
 * - 传感器缺失/注册失败 → state 恒 UNKNOWN → 静止特权不启用，行为退回现状；
 * - 标定落盘：启动后前 [CALIB_SECONDS] 每 5s 向 debugDir/imu_calib.log 追加一行
 *   窗口统计（gyroWinMax/accStd/accMean/state），供真机阈值标定（设备 logcat 被过滤，
 *   一律走落盘证据）。
 */
class StaticMotionMonitor(
    context: Context,
    private val debugDir: File? = null,
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val detector = ImuStaticDetector()

    /** 三态快照（相机线程读；detector.state 已 @Volatile）。 */
    val state: ImuState get() = detector.state

    /** 陀螺原始样本钩子（C9-T2 RotationPredictor 消费；传感器线程回调）。 */
    @Volatile
    var onGyroSample: ((timestampNs: Long, wx: Float, wy: Float, wz: Float) -> Unit)? = null

    private var thread: HandlerThread? = null
    private var startedAtMs = 0L
    private var lastCalibMs = 0L

    fun start() {
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (gyro == null && acc == null) {
            Log.w(TAG, "无陀螺仪/加速度计，IMU 静止分支停用（UNKNOWN）")
            return
        }
        val ht = HandlerThread("imu-monitor").also { it.start() }
        thread = ht
        val handler = Handler(ht.looper)
        if (gyro != null) {
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME, handler)
        }
        if (acc != null) {
            sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_GAME, handler)
        }
        startedAtMs = System.currentTimeMillis()
        Log.i(TAG, "IMU 监视启动 gyro=${gyro != null} acc=${acc != null}")
    }

    fun stop() {
        try {
            sensorManager.unregisterListener(this)
        } catch (t: Throwable) {
            Log.w(TAG, "注销传感器失败", t)
        }
        thread?.quitSafely()
        thread = null
    }

    override fun onSensorChanged(event: SensorEvent) {
        val v = event.values
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val mag = maxOf(abs(v[0]), abs(v[1]), abs(v[2]))
                detector.pushGyro(event.timestamp, mag)
                onGyroSample?.invoke(event.timestamp, v[0], v[1], v[2])
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                detector.pushAcc(event.timestamp, norm)
            }
        }
        maybeCalibLog()
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit

    /** 前 60s 每 5s 落一行窗口统计，用真机实测分布校准 §6 阈值。 */
    private fun maybeCalibLog() {
        val dir = debugDir ?: return
        val now = System.currentTimeMillis()
        if (now - startedAtMs > CALIB_SECONDS * 1000L) return
        if (now - lastCalibMs < 5_000L) return
        lastCalibMs = now
        try {
            if (!dir.exists()) dir.mkdirs()
            File(dir, "imu_calib.log").appendText(
                "t=${(now - startedAtMs) / 1000}s state=${detector.state} " +
                    "gyroWinMax=${"%.4f".format(detector.lastGyroWinMax)} " +
                    "accStd=${"%.3f".format(detector.lastAccStd)} " +
                    "accMean=${"%.3f".format(detector.lastAccMean)}\n"
            )
        } catch (t: Throwable) {
            Log.w(TAG, "imu 标定日志写失败", t)
        }
    }

    companion object {
        private const val TAG = "StaticMotionMonitor"
        private const val CALIB_SECONDS = 60
    }
}
