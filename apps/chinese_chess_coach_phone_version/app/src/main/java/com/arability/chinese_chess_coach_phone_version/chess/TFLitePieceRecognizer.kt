package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 基于 TFLite 的整盘棋子识别实现：把规整棋盘一次送入分类器，输出 90 个交叉点的类别。
 *
 * 模型契约（[DEFAULT_MODEL_PATH]，与 tools/train 的 PieceClassifierExport 对齐）：
 *  - 输入：float32 [1, H, W, 3]，RGB，取值 [0,1]。H、W 从模型输入张量形状读取，
 *    约定为 canonical 棋盘（450×500）的 0.8×，即 H=400、W=360。
 *  - 输出：float32 [1, 15, 10, 9] logits，维度依次为 (batch, class, rank, file)。
 *    对每个交叉点取 argmax，类别 0 = 空位。
 *  - 类别映射：0 = 空位；1–7 = 红方 帅/仕/相/马/车/炮/兵；8–14 = 黑方 将/士/象/马/车/炮/卒。
 *
 * 模型缺失或加载失败时，由 [PieceRecognizerFactory] 降级为 [TemplatePieceRecognizer]。
 */
class TFLitePieceRecognizer(
    private val interpreter: Interpreter,
    private val inputW: Int,
    private val inputH: Int
) : PieceRecognizer {

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputW * inputH * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { Array(NUM_CLASSES) { Array(ROWS) { FloatArray(COLS) } } }
    private val inputPixels = IntArray(inputW * inputH)
    private var logCounter = 0

    override fun recognize(bitmap: Bitmap, boardInfo: BoardInfo): Array<Array<Piece?>> {
        // 规整棋盘（450×500，canonical 带 MARGIN）→ 分类器输入尺寸（0.8×）。
        val scaled = if (boardInfo.warped.width == inputW && boardInfo.warped.height == inputH) {
            boardInfo.warped
        } else {
            Bitmap.createScaledBitmap(boardInfo.warped, inputW, inputH, true)
        }

        fillInput(scaled)
        interpreter.run(inputBuffer, output)

        val logits = output[0]  // [class][rank][file]
        val result = Array(ROWS) { arrayOfNulls<Piece>(COLS) }
        for (rank in 0 until ROWS) {
            for (file in 0 until COLS) {
                var best = 0
                var bestScore = logits[0][rank][file]
                for (c in 1 until NUM_CLASSES) {
                    if (logits[c][rank][file] > bestScore) {
                        bestScore = logits[c][rank][file]
                        best = c
                    }
                }
                result[rank][file] = labelToPiece(best)
            }
        }
        if (logCounter++ % 20 == 0) {
            val sb = StringBuilder(ROWS * (COLS + 1))
            for (rank in 0 until ROWS) {
                if (rank > 0) sb.append('/')
                for (file in 0 until COLS) {
                    sb.append(result[rank][file]?.fenChar ?: '.')
                }
            }
            Log.i(TAG, "grid=$sb")
        }
        return result
    }

    private fun fillInput(bitmap: Bitmap) {
        bitmap.getPixels(inputPixels, 0, inputW, 0, 0, inputW, inputH)
        inputBuffer.rewind()
        for (p in inputPixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()
    }

    private fun labelToPiece(label: Int): Piece? {
        if (label == 0) return null
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
        const val ROWS = 10
        const val COLS = 9
        const val DEFAULT_MODEL_PATH = "models/piece_classifier.tflite"

        /** 从 assets 加载模型并创建识别器；模型缺失/损坏时抛出异常，由工厂降级处理。 */
        fun load(context: Context): TFLitePieceRecognizer {
            val fd = context.assets.openFd(DEFAULT_MODEL_PATH)
            val buffer: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
            val interpreter = Interpreter(buffer)
            val shape = interpreter.getInputTensor(0).shape()  // [1, H, W, 3]
            val inputH = shape[1]
            val inputW = shape[2]
            return TFLitePieceRecognizer(interpreter, inputW, inputH)
        }
    }
}
