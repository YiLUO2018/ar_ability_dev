package com.arability.gpu_on_phone.bench

import com.arability.gpu_on_phone.backends.Backend
import com.arability.gpu_on_phone.models.ModelSpec
import org.json.JSONObject
import java.io.File

/** 一次跑批的完整结果（交给 ResultWriter 落盘） */
data class BenchResult(
    val spec: ModelSpec,
    val backendId: String,
    val backendDisplayName: String,
    val runtimeName: String,
    val runtimeVersion: String,
    val warmup: Int,
    val iterations: Int,
    /** setup = 模型加载 + session 创建 + 固定输入构造（不含推理） */
    val setupMs: Long,
    /** 首帧：session 建立后的第一次推理（含编译/懒加载，单列不入统计——NNAPI/QNN 首编可达数秒） */
    val firstRunMs: Long,
    /** 正式测量的每次延迟（warmup 之后） */
    val latenciesMs: LongArray,
    val stats: Stats,
    val envStart: JSONObject?,
    val envEnd: JSONObject?,
    val timestampIso: String,
)

/**
 * 计时路径唯一（设计文档 §7）：所有后端共用本 runner，
 * 首帧单列 → warmup → 正式测量 → 统计。
 */
class BenchRunner(
    private val sampler: EnvSampler = EnvSampler.Noop,
    private val clock: () -> Long = System::nanoTime,
) {
    fun run(
        modelsDir: File,
        spec: ModelSpec,
        backend: Backend,
        warmup: Int,
        iterations: Int,
        runtimeName: String = "onnxruntime",
        runtimeVersion: String = "unknown",
        progress: (String) -> Unit = {},
    ): BenchResult {
        require(warmup >= 0) { "warmup >= 0 required" }
        require(iterations >= 1) { "iterations >= 1 required" }

        val envStart = sampler.sample()
        try {
            progress("setup：加载 ${spec.fileName}（${backend.displayName}）…")
            var t = clock()
            backend.setup(modelsDir, spec)
            val setupMs = elapsed(t)

            progress("首帧（含编译/加载，单列）…")
            t = clock()
            backend.runOnce()
            val firstRunMs = elapsed(t)

            if (warmup > 0) progress("warmup ×$warmup…")
            repeat(warmup) { backend.runOnce() }

            progress("正式测量 ×$iterations…")
            val latencies = LongArray(iterations) {
                t = clock()
                backend.runOnce()
                elapsed(t)
            }
            val envEnd = sampler.sample()

            return BenchResult(
                spec = spec,
                backendId = backend.id,
                backendDisplayName = backend.displayName,
                runtimeName = runtimeName,
                runtimeVersion = runtimeVersion,
                warmup = warmup,
                iterations = iterations,
                setupMs = setupMs,
                firstRunMs = firstRunMs,
                latenciesMs = latencies,
                stats = Stats.of(latencies),
                envStart = envStart,
                envEnd = envEnd,
                timestampIso = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", java.util.Locale.US).format(java.util.Date()),
            )
        } finally {
            try {
                backend.close()
            } catch (_: Exception) {
                // close 失败不影响结果保留
            }
        }
    }

    private fun elapsed(t0: Long): Long = (clock() - t0) / 1_000_000
}
