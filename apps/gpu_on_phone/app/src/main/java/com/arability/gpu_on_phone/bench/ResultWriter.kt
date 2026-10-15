package com.arability.gpu_on_phone.bench

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 结果落盘（方法学：每次 run 一个 JSON + index.jsonl 一行摘要，adb 拉回 PC 汇总）。
 * 目录：<filesDir>/bench_results/（.gitignore 已排除，PC 侧工具从 adb 拉取）。
 */
class ResultWriter(private val context: Context) {

    fun write(result: BenchResult): File {
        val dir = File(context.filesDir, "bench_results").apply { mkdirs() }

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "${result.spec.key}_${result.backendId}_$stamp.json"
        val json = buildJson(result)
        File(dir, fileName).writeText(json.toString(2), Charsets.UTF_8)

        File(dir, "index.jsonl").appendText(
            JSONObject()
                .put("ts", result.timestampIso)
                .put("model", result.spec.key)
                .put("backend", result.backendId)
                .put("median_ms", round2(result.stats.median))
                .put("p99_ms", round2(result.stats.p99))
                .put("first_run_ms", round2(result.firstRunMs.toDouble()))
                .put("file", fileName)
                .toString() + "\n",
            Charsets.UTF_8,
        )
        return File(dir, fileName)
    }

    private fun buildJson(result: BenchResult): JSONObject = JSONObject().apply {
        put("schema", 1)
        put("meta", JSONObject().apply {
            put("timestamp", result.timestampIso)
            put("runtime", result.runtimeName)
            put("runtime_version", result.runtimeVersion)
            put("backend_id", result.backendId)
            put("backend", result.backendDisplayName)
            put("model_key", result.spec.key)
            put("model_file", result.spec.fileName)
            put("precision", "fp32")
            put("warmup", result.warmup)
            put("iterations", result.iterations)
            put("input_source", "synthetic_seed42")
            put("device", DeviceProfile.metaJson(context))
        })
        put("setup_ms", round2(result.setupMs.toDouble()))
        put("first_run_ms", round2(result.firstRunMs.toDouble()))
        put("env_start", result.envStart ?: JSONObject.NULL)
        put("env_end", result.envEnd ?: JSONObject.NULL)
        put("stats", JSONObject().apply {
            val s = result.stats
            put("n", s.n)
            put("median_ms", round2(s.median))
            put("mean_ms", round2(s.mean))
            put("min_ms", round2(s.min))
            put("max_ms", round2(s.max))
            put("p90_ms", round2(s.p90))
            put("p99_ms", round2(s.p99))
            put("variance", round2(s.variance))
            put("cv", round4(s.cv))
        })
        put("latencies_ms", JSONArray().apply {
            result.latenciesMs.forEach { put(round2(it.toDouble())) }
        })
    }

    private fun round2(v: Double): Double = Math.round(v * 100.0) / 100.0
    private fun round4(v: Double): Double = Math.round(v * 10000.0) / 10000.0
}
