package com.arability.gpu_on_phone

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.arability.gpu_on_phone.backends.OrtBackend
import com.arability.gpu_on_phone.backends.OrtEp
import com.arability.gpu_on_phone.bench.BenchResult
import com.arability.gpu_on_phone.bench.BenchRunner
import com.arability.gpu_on_phone.bench.DeviceEnvSampler
import com.arability.gpu_on_phone.bench.DeviceProfile
import com.arability.gpu_on_phone.bench.ResultWriter
import com.arability.gpu_on_phone.databinding.ActivityMainBinding
import com.arability.gpu_on_phone.models.ModelSpecs
import ai.onnxruntime.OrtEnvironment
import java.io.File
import java.util.concurrent.Executors

/**
 * benchmark 载体（单 Activity，仓库约定形态）：配置选择 → 跑批 → 结果展示/分享。
 * 计时逻辑全部在 bench/ 内，本类只做 UI 编排。
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val TAG = "GpuBench"
    }

    private lateinit var binding: ActivityMainBinding
    private val executor = Executors.newSingleThreadExecutor()
    private var lastResultFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.spinnerTarget.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_item, ModelSpecs.all.map { it.title }).apply {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
        binding.spinnerBackend.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_item, OrtEp.entries.map { it.displayName }).apply {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }

        refreshStatus()
        binding.btnRun.setOnClickListener { startRun() }
        binding.btnShare.setOnClickListener { shareLast() }
    }

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }

    private fun refreshStatus() {
        binding.deviceInfo.text = DeviceProfile.summaryText(this) + "\nORT " + ortVersion()
        val dir = resolveModelsDir(this)
        val lines = ModelSpecs.all.map { spec ->
            (if (File(dir, spec.fileName).exists()) "✓" else "✗") + " " + spec.fileName
        }
        binding.modelsStatus.text = "模型目录（adb push 到此）：\n${dir.absolutePath}\n" + lines.joinToString("\n")
    }

    /**
     * 模型目录解析：优先应用外部专有目录（adb push 免权限），
     * 其下无 onnx 时回退 filesDir/models。
     */
    private fun resolveModelsDir(context: Context): File {
        val ext = context.getExternalFilesDir(null)
        val candidates = buildList {
            if (ext != null) add(File(ext, "models"))
            add(File(context.filesDir, "models"))
        }
        return candidates.firstOrNull { d -> d.isDirectory && d.listFiles()?.any { it.extension == "onnx" } == true }
            ?: candidates.first()
    }

    private fun ortVersion(): String = try {
        OrtEnvironment.getEnvironment().version
    } catch (e: Throwable) {
        Log.i(TAG, "ort version unavailable: ${e.message}")
        "unknown"
    }

    private fun startRun() {
        val spec = ModelSpecs.all[binding.spinnerTarget.selectedItemPosition.coerceIn(0, ModelSpecs.all.size - 1)]
        val ep = OrtEp.entries[binding.spinnerBackend.selectedItemPosition.coerceIn(0, OrtEp.entries.size - 1)]
        val warmup = binding.editWarmup.text.toString().toIntOrNull()?.coerceIn(0, 100) ?: 5
        val iterations = binding.editIterations.text.toString().toIntOrNull()?.coerceIn(1, 1000) ?: 30
        val dir = resolveModelsDir(this)

        if (!File(dir, spec.fileName).exists()) {
            binding.textResult.text = "模型缺失：${spec.fileName}\n先 adb push 到 ${dir.absolutePath}"
            return
        }

        binding.btnRun.isEnabled = false
        binding.textResult.text = "运行中…"
        val sampler = DeviceEnvSampler(this)
        executor.execute {
            try {
                val result = BenchRunner(sampler).run(
                    modelsDir = dir,
                    spec = spec,
                    backend = OrtBackend(ep),
                    warmup = warmup,
                    iterations = iterations,
                    runtimeVersion = ortVersionSafe(),
                    progress = { msg -> runOnUiThread { binding.textResult.text = msg } },
                )
                val file = ResultWriter(this@MainActivity).write(result)
                lastResultFile = file
                runOnUiThread {
                    binding.textResult.text = formatSummary(result, file)
                    refreshStatus()
                }
            } catch (e: Throwable) {
                Log.i(TAG, "bench failed", e)
                runOnUiThread { binding.textResult.text = "失败：${e.message}" }
            } finally {
                runOnUiThread { binding.btnRun.isEnabled = true }
            }
        }
    }

    private fun ortVersionSafe(): String = try {
        OrtEnvironment.getEnvironment().version
    } catch (_: Throwable) { "unknown" }

    private fun formatSummary(r: BenchResult, file: File): String = buildString {
        appendLine("${r.spec.title} × ${r.backendDisplayName}（${r.runtimeName} ${r.runtimeVersion}）")
        appendLine("setup      ${r.setupMs} ms")
        appendLine("first_run  ${r.firstRunMs} ms（含编译/加载，单列）")
        appendLine("median     ${fmt(r.stats.median)} ms")
        appendLine("p90 / p99  ${fmt(r.stats.p90)} / ${fmt(r.stats.p99)} ms")
        appendLine("min / max  ${fmt(r.stats.min)} / ${fmt(r.stats.max)} ms")
        appendLine("cv         ${"%.3f".format(r.stats.cv)}")
        append("已保存：${file.name}")
    }

    private fun fmt(v: Double): String = "%.1f".format(v)

    private fun shareLast() {
        val file = lastResultFile ?: run {
            Toast.makeText(this, "还没有结果", Toast.LENGTH_SHORT).show()
            return
        }
        val text = file.readText(Charsets.UTF_8)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, "${file.name}\n\n$text")
        }
        startActivity(Intent.createChooser(intent, file.name))
    }
}
