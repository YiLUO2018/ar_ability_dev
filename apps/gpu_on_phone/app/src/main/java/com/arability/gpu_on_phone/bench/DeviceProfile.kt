package com.arability.gpu_on_phone.bench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import org.json.JSONObject

/** 设备静态信息（G0-1 的 App 侧部分；SoC/温度的系统级摸底由 PC 侧 adb 完成） */
object DeviceProfile {

    fun summaryText(context: Context): String = buildString {
        appendLine("device=${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})")
        if (Build.VERSION.SDK_INT >= 31) {
            appendLine("soc=${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
        } else {
            appendLine("soc=n/a(api<31) hardware=${Build.HARDWARE}")
        }
        appendLine("android=${Build.VERSION.RELEASE} (api ${Build.VERSION.SDK_INT})")
        append("battery=${batteryPct(context)}% temp=${batteryTempC(context)}°C")
    }

    fun metaJson(context: Context): JSONObject = JSONObject().apply {
        put("device_model", Build.MODEL)
        put("device_device", Build.DEVICE)
        put("manufacturer", Build.MANUFACTURER)
        if (Build.VERSION.SDK_INT >= 31) {
            put("soc_manufacturer", Build.SOC_MANUFACTURER)
            put("soc_model", Build.SOC_MODEL)
        }
        put("hardware", Build.HARDWARE)
        put("android_release", Build.VERSION.RELEASE)
        put("api_level", Build.VERSION.SDK_INT)
    }

    fun batteryPct(context: Context): Int = try {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
    } catch (_: Exception) { -1 }

    /** 电池温度（°C，读不到返回 NaN）——作为热状态的低成本代理指标 */
    fun batteryTempC(context: Context): Double = try {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val t = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        if (t == Int.MIN_VALUE) Double.NaN else t / 10.0
    } catch (_: Exception) { Double.NaN }
}

/** 跑批前后的环境采样（方法学：记录电量/温度，热节流批次可事后作废） */
interface EnvSampler {
    fun sample(): JSONObject?

    object Noop : EnvSampler {
        override fun sample(): JSONObject? = null
    }
}

class DeviceEnvSampler(private val context: Context) : EnvSampler {
    override fun sample(): JSONObject? = try {
        JSONObject()
            .put("battery_pct", DeviceProfile.batteryPct(context))
            .put("battery_temp_c", if (DeviceProfile.batteryTempC(context).isNaN()) JSONObject.NULL else DeviceProfile.batteryTempC(context))
    } catch (_: Exception) { null }
}
