package com.arability.gpu_on_phone.models

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Random

/** 单个输入张量的名字与形状（形状用于合成输入构造与 sanity check） */
data class TensorSpec(val name: String, val shape: LongArray)

/**
 * 被测模型规格。inputs 的顺序必须与 ONNX 图输入顺序一致（名字不匹配时按位置兜底）。
 */
data class ModelSpec(
    val key: String,
    val title: String,
    val fileName: String,
    val inputs: List<TensorSpec>,
)

/**
 * 基准模型注册表（设计文档 §5：只用真实业务模型）。
 *
 * 输入契约来自象棋 App 实测（chinese_chess_coach_phone_version）：
 * - EdgeSAM encoder：image [1,3,1024,1024] fp32 NCHW → image_embeddings [1,256,64,64]
 * - EdgeSAM decoder：image_embeddings + point_coords [1,1,2]（1024 帧坐标）+ point_labels [1,1]
 * - YOLOv8n chess：images [1,3,480,640] → [1,18,N]（已 sigmoid，车在马前）
 */
object ModelSpecs {
    private const val TAG = "ModelSpecs"

    /** 合成输入固定种子：跨后端/跨次运行逐位一致 */
    const val SEED = 42L

    val ENCODER = ModelSpec(
        key = "edge_sam_encoder",
        title = "EdgeSAM encoder",
        fileName = "edge_sam_3x_encoder.onnx",
        inputs = listOf(TensorSpec("image", longArrayOf(1, 3, 1024, 1024))),
    )

    val DECODER = ModelSpec(
        key = "edge_sam_decoder",
        title = "EdgeSAM decoder",
        fileName = "edge_sam_3x_decoder.onnx",
        inputs = listOf(
            TensorSpec("image_embeddings", longArrayOf(1, 256, 64, 64)),
            TensorSpec("point_coords", longArrayOf(1, 1, 2)),
            TensorSpec("point_labels", longArrayOf(1, 1)),
        ),
    )

    val YOLO = ModelSpec(
        key = "yolov8n_chess",
        title = "YOLOv8n chess",
        fileName = "yolov8n_chess.onnx",
        inputs = listOf(TensorSpec("images", longArrayOf(1, 3, 480, 640))),
    )

    val all = listOf(ENCODER, DECODER, YOLO)

    fun byKey(key: String): ModelSpec = all.first { it.key == key }

    /**
     * 为 spec 构造固定输入张量（顺序与 spec.inputs 一致）。
     * decoder 的 embeddings 依赖 encoder 先跑一次（CPU、不计入 decoder 计时）。
     */
    fun buildInputs(env: OrtEnvironment, modelsDir: File, spec: ModelSpec): List<OnnxTensor> =
        when (spec.key) {
            ENCODER.key -> listOf(randomTensor(env, ENCODER.inputs[0].shape))
            YOLO.key -> listOf(randomTensor(env, YOLO.inputs[0].shape))
            DECODER.key -> listOf(
                runEncoderOnceForEmbeddings(env, modelsDir),
                fixedTensor(env, longArrayOf(1, 1, 2), floatArrayOf(512f, 512f)),
                fixedTensor(env, longArrayOf(1, 1), floatArrayOf(1f)),
            )
            else -> error("unknown spec ${spec.key}")
        }

    /**
     * 把按 spec 顺序构造的张量绑定到 session 实际输入名上：
     * 名字全匹配则按名绑定；否则按位置绑定（并打日志——名字漂移是要留意的信号）。
     */
    fun bindInputs(session: OrtSession, spec: ModelSpec, tensors: List<OnnxTensor>): Map<String, OnnxTensor> {
        val actual = session.inputNames.toList()
        require(actual.size == tensors.size) {
            "input count mismatch: graph=${actual} spec=${spec.inputs.map { it.name }}"
        }
        return if (spec.inputs.all { it.name in actual }) {
            spec.inputs.mapIndexed { i, s -> s.name to tensors[i] }.toMap()
        } else {
            Log.i(TAG, "input names mismatch, binding positionally: graph=$actual spec=${spec.inputs.map { it.name }}")
            actual.zip(tensors).toMap()
        }
    }

    /** 固定种子随机填充（模拟归一化后的图像张量幅度 [-1,1)） */
    private fun randomTensor(env: OrtEnvironment, shape: LongArray): OnnxTensor {
        val count = shape.fold(1L) { acc, d -> acc * d }.toInt()
        val buf = directFloatBuffer(count)
        val rnd = Random(SEED)
        for (i in 0 until count) buf.put(2f * rnd.nextFloat() - 1f)
        buf.flip()
        return OnnxTensor.createTensor(env, buf, shape)
    }

    private fun fixedTensor(env: OrtEnvironment, shape: LongArray, values: FloatArray): OnnxTensor {
        val buf = directFloatBuffer(values.size)
        buf.put(values)
        buf.flip()
        return OnnxTensor.createTensor(env, buf, shape)
    }

    /**
     * 用 CPU 跑一次 encoder 提取 image_embeddings（decoder 的输入依赖）。
     * 只执行一次、不计时；内存拷贝为 direct buffer 以复用 createTensor。
     */
    private fun runEncoderOnceForEmbeddings(env: OrtEnvironment, modelsDir: File): OnnxTensor {
        val session = env.createSession(
            File(modelsDir, ENCODER.fileName).absolutePath,
            OrtSession.SessionOptions(),
        )
        session.use { s ->
            val input = randomTensor(env, ENCODER.inputs[0].shape)
            input.use { t ->
                val result = s.run(mapOf(ENCODER.inputs[0].name to t))
                result.use { r ->
                    val out = r[0] as? OnnxTensor ?: error("encoder output is not a tensor")
                    val src = out.floatBuffer
                    val dst = directFloatBuffer(src.remaining())
                    dst.put(src)
                    dst.flip()
                    return OnnxTensor.createTensor(env, dst, longArrayOf(1, 256, 64, 64))
                }
            }
        }
    }

    private fun directFloatBuffer(capacity: Int): FloatBuffer =
        ByteBuffer.allocateDirect(capacity * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
}
