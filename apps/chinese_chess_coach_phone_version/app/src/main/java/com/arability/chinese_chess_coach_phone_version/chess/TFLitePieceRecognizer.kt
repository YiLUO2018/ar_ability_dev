package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 基于 TFLite 的棋子识别实现：对每个交叉点裁剪 ROI，缩放到模型输入尺寸，argmax 15 类。
 *
 * 模型契约（[DEFAULT_MODEL_PATH]）：
 *  - 输入：float32 [1, H, W, 3]，RGB，取值 [0,1]（H、W 从模型输入张量形状读取，约定为正方形）。
 *  - 输出：float32 [1, 15]，取 argmax 作为类别。
 *  - 类别映射：0 = 空位；1–7 = 红方 帅/仕/相/马/车/炮/兵；8–14 = 黑方 将/士/象/马/车/炮/卒。
 *
 * 模型缺失或加载失败时，由 [PieceRecognizerFactory] 降级为 [TemplatePieceRecognizer]。
 */
class TFLitePieceRecognizer(
    private val interpreter: Interpreter,
    private val inputSize: Int,
    private val confidenceThreshold: Float
) : PieceRecognizer {

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(NUM_CLASSES) }
    private val roi = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(inputSize * inputSize)

    override fun recognize(boardInfo: BoardInfo): Array<Array<Piece?>> {
        val warped = boardInfo.warped
        val size = boardInfo.cellSize
        val half = size / 2
        val result = Array(10) { arrayOfNulls<Piece>(9) }

        for (rank in 0..9) {
            for (file in 0..8) {
                val cx = boardInfo.intersectionX(file)
                val cy = boardInfo.intersectionY(rank)
                val left = cx - half
                val top = cy - half
                if (left < 0 || top < 0 || left + size > warped.width || top + size > warped.height) continue

                val src = Rect(left, top, left + size, top + size)
                val dst = Rect(0, 0, inputSize, inputSize)
                Canvas(roi).drawBitmap(warped, src, dst, null)

                fillInput()
                interpreter.run(inputBuffer, output)

                val probs = output[0]
                var best = 0
                var bestScore = probs[0]
                for (i in 1 until NUM_CLASSES) {
                    if (probs[i] > bestScore) {
                        bestScore = probs[i]
                        best = i
                    }
                }
                result[rank][file] = labelToPiece(best, bestScore)
            }
        }
        return result
    }

    private fun fillInput() {
        roi.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        inputBuffer.rewind()
        for (p in inputPixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()
    }

    private fun labelToPiece(label: Int, score: Float): Piece? {
        if (label == 0 || score < confidenceThreshold) return null
        val color = if (label <= 7) PieceColor.RED else PieceColor.BLACK
        val type = PieceType.entries[(label - 1) % 7]
        return Piece(color, type)
    }

    override fun close() {
        try {
            interpreter.close()
        } catch (t: Throwable) {
            Log.w(TAG, "关闭解释器失败", t)
        }
    }

    companion object {
        private const val TAG = "TFLitePieceRecognizer"
        const val NUM_CLASSES = 15
        const val DEFAULT_MODEL_PATH = "models/piece_classifier.tflite"
        const val DEFAULT_INPUT_SIZE = 64
        const val DEFAULT_CONFIDENCE = 0.5f

        /** 从 assets 加载模型并创建识别器；模型缺失/损坏时抛出异常，由工厂降级处理。 */
        fun load(context: Context): TFLitePieceRecognizer {
            val fd = context.assets.openFd(DEFAULT_MODEL_PATH)
            val buffer: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
            val interpreter = Interpreter(buffer)
            val shape = interpreter.getInputTensor(0).shape()
            val inputSize = if (shape.size >= 3) shape[1] else DEFAULT_INPUT_SIZE
            return TFLitePieceRecognizer(interpreter, inputSize, DEFAULT_CONFIDENCE)
        }
    }
}
