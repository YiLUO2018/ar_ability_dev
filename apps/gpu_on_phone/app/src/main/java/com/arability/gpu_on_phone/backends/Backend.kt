package com.arability.gpu_on_phone.backends

import com.arability.gpu_on_phone.models.ModelSpec
import java.io.File

/**
 * 推理后端统一接口。横评只换实现不换计时路径（设计文档 §7）。
 *
 * 生命周期：setup（加载模型 + 准备固定输入）→ runOnce*（首帧/warmup/正式计时）→ close。
 * 输入张量在 setup 内创建并复用——计时区间只含纯推理，不含输入构造（方法学约定）。
 */
interface Backend {
    /** 稳定 id，写入结果 JSON（如 "ort-cpu" / "litert-gpu"） */
    val id: String

    /** UI 显示名 */
    val displayName: String

    fun setup(modelsDir: File, spec: ModelSpec)

    fun runOnce()

    fun close()
}
