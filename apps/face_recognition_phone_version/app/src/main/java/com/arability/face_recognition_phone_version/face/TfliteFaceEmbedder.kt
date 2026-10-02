package com.arability.face_recognition_phone_version.face

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * 基于 TFLite + MobileFaceNet 的人脸特征提取实现（纯离线，无 GMS）。
 *
 * 模型：mobile_facenet.tflite（112x112 输入，128 维输出）。
 * 该模型为双输入（img1/img2）人脸比对结构，这里同一张脸喂给两个输入，
 * 取第一个输出作为特征向量。
 */
class TfliteFaceEmbedder(context: Context) : FaceEmbedder {

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val nchw: Boolean
    private val outputRows: Int
    private val embeddingDim: Int

    init {
        val model = loadModelFile(context, MODEL_PATH)
        interpreter = Interpreter(model)

        val inShape = interpreter.getInputTensor(0).shape()
        if (inShape.size == 4 && inShape[1] == 3) {
            // NCHW：[1,3,H,W]
            nchw = true
            inputHeight = inShape[2]
            inputWidth = inShape[3]
        } else {
            // NHWC：[1,H,W,3]
            nchw = false
            inputHeight = inShape[1]
            inputWidth = inShape[2]
        }

        val outShape = interpreter.getOutputTensor(0).shape()
        outputRows = outShape[0]
        embeddingDim = outShape.last()
    }

    override fun embed(faceBitmap: Bitmap): FloatArray {
        val buffer = preprocess(faceBitmap)
        // 双输入模型，喂同一张脸；取第一个输出向量
        val inputs = arrayOf<Any>(buffer, buffer)
        val output = Array(outputRows) { FloatArray(embeddingDim) }
        interpreter.runForMultipleInputsOutputs(inputs, mapOf(0 to output))
        return l2Normalize(output[0])
    }

    override fun close() {
        interpreter.close()
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val buffer = ByteBuffer.allocateDirect(CHANNELS * inputWidth * inputHeight * FLOAT_BYTES)
        buffer.order(ByteOrder.nativeOrder())

        if (nchw) {
            for (c in 0 until CHANNELS) {
                for (p in pixels) {
                    buffer.putFloat(channelValue(p, c) / 255f)
                }
            }
        } else {
            for (p in pixels) {
                buffer.putFloat(channelValue(p, 0) / 255f)
                buffer.putFloat(channelValue(p, 1) / 255f)
                buffer.putFloat(channelValue(p, 2) / 255f)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun channelValue(pixel: Int, channel: Int): Float = when (channel) {
        0 -> ((pixel shr 16) and 0xFF).toFloat() // R
        1 -> ((pixel shr 8) and 0xFF).toFloat()  // G
        else -> (pixel and 0xFF).toFloat()        // B
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sum = 0f
        for (v in vector) sum += v * v
        val norm = sqrt(sum)
        if (norm < 1e-6f) return vector
        return FloatArray(vector.size) { i -> vector[i] / norm }
    }

    companion object {
        private const val MODEL_PATH = "models/mobile_facenet.tflite"
        private const val CHANNELS = 3
        private const val FLOAT_BYTES = 4

        private fun loadModelFile(context: Context, path: String): MappedByteBuffer {
            val fd = context.assets.openFd(path)
            val stream = FileInputStream(fd.fileDescriptor)
            return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }
}
