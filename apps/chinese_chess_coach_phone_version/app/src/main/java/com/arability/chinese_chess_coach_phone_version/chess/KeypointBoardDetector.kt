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
 * 基于 TFLite 棋盘角点模型的定位器：整图回归四角 + 置信度，再透视校正为 canonical 棋盘。
 *
 * 模型契约（[DEFAULT_MODEL_PATH]，与 tools/train 的 CornerModelExport 对齐）：
 *  - 输入：float32 [1, S, S, 3]，RGB，取值 [0,1]。S 从模型输入张量形状读取（训练默认 320）。
 *  - 输出：float32 [1, 12] = x0,y0,x1,y1,x2,y2,x3,y3,c0,c1,c2,c3。
 *    前 8 个为四角 TL,TR,BR,BL 的归一化坐标 [0,1]（相对输入图），后 4 个为各角置信度。
 *
 * 输入图按训练一致的方式「拉伸」缩放为正方形；归一化坐标再按原图宽/高映射回像素坐标。
 * 置信度低于 [confidenceThreshold] 视为「无棋盘」，返回 null。
 */
class KeypointBoardDetector(
    private val interpreter: Interpreter,
    private val inputSize: Int,
    private val confidenceThreshold: Float,
    private val debugDir: java.io.File? = null
) : BoardDetector {

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(12) }
    private val inputPixels = IntArray(inputSize * inputSize)
    private var logCounter = 0

    override fun detect(bitmap: Bitmap): BoardInfo? {
        val w = bitmap.width
        val h = bitmap.height
        if (w == 0 || h == 0) return null

        // 拉伸缩放为正方形（与训练 cv2.resize 到 (S,S) 一致）。
        val scaled = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }
        fillInput(scaled)
        interpreter.run(inputBuffer, output)

        val o = output[0]
        val c0 = o[8]; val c1 = o[9]; val c2 = o[10]; val c3 = o[11]
        val avgConf = (c0 + c1 + c2 + c3) / 4f
        if (avgConf < confidenceThreshold) {
            return null
        }

        // 归一化坐标 → 原图像素坐标（x 按宽、y 按高）。
        val corners = FloatArray(8)
        for (i in 0 until 4) {
            corners[i * 2] = o[i * 2] * w
            corners[i * 2 + 1] = o[i * 2 + 1] * h
        }

        if (logCounter++ % 20 == 0) {
            Log.i(TAG, "corners=[${corners[0].toInt()},${corners[1].toInt()}] [${corners[2].toInt()},${corners[3].toInt()}] [${corners[4].toInt()},${corners[5].toInt()}] [${corners[6].toInt()},${corners[7].toInt()}] conf=$avgConf")
            saveDebugFrame(bitmap, logCounter)
        }

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // canonical（带 MARGIN）→ 源图像，采样得到规整棋盘。
        val hInv = Homography.compute(BoardGeometry.corners, corners)
        val warpedPixels = Homography.warp(
            pixels, w, h, BoardGeometry.BOARD_W, BoardGeometry.BOARD_H, hInv
        )
        val warped = Bitmap.createBitmap(
            BoardGeometry.BOARD_W, BoardGeometry.BOARD_H, Bitmap.Config.ARGB_8888
        )
        warped.setPixels(warpedPixels, 0, BoardGeometry.BOARD_W, 0, 0, BoardGeometry.BOARD_W, BoardGeometry.BOARD_H)

        return BoardInfo(
            corners = corners,
            warped = warped,
            cellSize = BoardGeometry.CELL,
            originX = BoardGeometry.MARGIN,
            originY = BoardGeometry.MARGIN,
            imageToGrid = Homography.compute(corners, BoardGeometry.corners)
        )
    }

    private fun saveDebugFrame(bitmap: Bitmap, idx: Int) {
        val dir = debugDir ?: return
        try {
            if (!dir.exists()) dir.mkdirs()
            val file = java.io.File(dir, "board_$idx.png")
            java.io.FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.i(TAG, "已保存调试帧: ${file.absolutePath}")
        } catch (t: Throwable) {
            Log.w(TAG, "保存调试帧失败", t)
        }
    }

    private fun fillInput(bitmap: Bitmap) {
        bitmap.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        inputBuffer.rewind()
        for (p in inputPixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()
    }

    override fun close() {
        try {
            interpreter.close()
        } catch (t: Throwable) {
            Log.w(TAG, "关闭解释器失败", t)
        }
    }

    companion object {
        private const val TAG = "KeypointBoardDetector"
        const val DEFAULT_MODEL_PATH = "models/board_corner.tflite"
        const val DEFAULT_CONFIDENCE = 0.5f

        /** 从 assets 加载模型并创建定位器；模型缺失/损坏时抛出异常，由工厂降级处理。 */
        fun load(context: Context, debugDir: java.io.File? = null): KeypointBoardDetector {
            val fd = context.assets.openFd(DEFAULT_MODEL_PATH)
            val buffer: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
            val interpreter = Interpreter(buffer)
            val shape = interpreter.getInputTensor(0).shape()  // [1, S, S, 3]
            val inputSize = shape[1]
            return KeypointBoardDetector(interpreter, inputSize, DEFAULT_CONFIDENCE, debugDir)
        }
    }
}
