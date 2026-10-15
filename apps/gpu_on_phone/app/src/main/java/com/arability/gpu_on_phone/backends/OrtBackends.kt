package com.arability.gpu_on_phone.backends

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.arability.gpu_on_phone.models.ModelSpec
import com.arability.gpu_on_phone.models.ModelSpecs
import java.io.File
import java.util.EnumSet

/**
 * ONNX Runtime 的 EP 变体（实验计划 A3/B1）：
 * - CPU：默认 CPU EP（A1/A2 基线）
 * - XNNPACK：addXnnpack（A3）
 * - NNAPI / NNAPI_FP16：addNnapi（B1 关闭性实测；USE_FP16 flag 对照）
 *
 * 注：ORT Android 官方包无 GPU EP（调研结论 §1.3）；GPU 路线由后续 LiteRT/ncnn/MNN 后端接入。
 */
enum class OrtEp(
    /** 后端稳定 id，写入结果 JSON */
    val id: String,
    val displayName: String,
) {
    CPU("ort-cpu", "ORT CPU") {
        override fun configure(options: OrtSession.SessionOptions) = Unit
    },
    XNNPACK("ort-xnnpack", "ORT XNNPACK") {
        override fun configure(options: OrtSession.SessionOptions) {
            options.addXnnpack(emptyMap())
        }
    },
    NNAPI("ort-nnapi", "ORT NNAPI") {
        override fun configure(options: OrtSession.SessionOptions) {
            options.addNnapi()
        }
    },
    NNAPI_FP16("ort-nnapi-fp16", "ORT NNAPI fp16") {
        override fun configure(options: OrtSession.SessionOptions) {
            options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
        }
    };

    /** EP 装配逻辑（子类实现） */
    abstract fun configure(options: OrtSession.SessionOptions)
}

/** 有状态的 ORT 后端实例（一次 setup → runOnce×N → close 生命周期对应一次跑批） */
class OrtBackend(private val ep: OrtEp) : Backend {

    override val id: String = ep.id
    override val displayName: String = ep.displayName

    private var session: OrtSession? = null
    private var inputs: Map<String, OnnxTensor> = emptyMap()

    override fun setup(modelsDir: File, spec: ModelSpec) {
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        ep.configure(options)
        val s = env.createSession(File(modelsDir, spec.fileName).absolutePath, options)
        session = s
        inputs = ModelSpecs.bindInputs(s, spec, ModelSpecs.buildInputs(env, modelsDir, spec))
    }

    override fun runOnce() {
        val s = session ?: error("backend not setup")
        s.run(inputs).use { /* 计时区间只推进计算；输出仅 decoder 预跑时读取 */ }
    }

    override fun close() {
        inputs.values.forEach { it.close() }
        inputs = emptyMap()
        session?.close()
        session = null
    }
}
