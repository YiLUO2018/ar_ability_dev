package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 基于自训 yolov8n（ONNX Runtime）的逐子棋子检测器。
 *
 * 在**原始帧**上检出棋子 bbox + 类别，再把每框中心经 [BoardInfo.imageToGrid] 单应映射到最近交叉点，
 * 填成 9×10 网格。相较整盘分类器，不假设棋子压在交点正上方，对偏移子更鲁棒。
 *
 * 模型契约（`models/yolov8n_chess.onnx`，与 tools/label_real/export_onnx.py 对齐）：
 *  - 输入：`images` [1,3,640,640] float32 NCHW，RGB 归一化 [0,1]（letterbox 到 640×640）。
 *  - 输出：`output0` [1,18,8400] float32：前 4 通道 = box（xywh，letterbox 像素坐标），
 *    后 14 通道 = 14 类分数（导出时已 sigmoid，取值 [0,1]）。8400 = 80²+40²+20² 三尺度 anchor。
 *  - 类别顺序：0–6 红 帅/仕/相/车/马/炮/兵，7–13 黑 将/士/象/车/马/炮/卒。
 */
class YoloPieceDetector(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : PieceRecognizer {

    private val inputBuffer: FloatBuffer =
        ByteBuffer.allocateDirect(3 * IMG_SIZE * IMG_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    private var logCounter = 0
    private var lastDets: List<PieceDetection> = emptyList()

    override fun recognize(bitmap: Bitmap, boardInfo: BoardInfo): Array<Array<Piece?>> {
        val result = Array(ROWS) { arrayOfNulls<Piece>(COLS) }
        val h = boardInfo.imageToGrid ?: run {
            Log.w(TAG, "缺少 imageToGrid 单应，跳过检测")
            return result
        }

        // 1. 裁剪棋盘区域（外扩 padding，与训练时 crop 一致）。
        val crop = cropBoard(bitmap, boardInfo.corners) ?: return result

        // 2. letterbox 到 640×640（居中、114 填充）。
        val lb = letterbox(crop.bitmap)
        val (canvas, r, padX, padY) = lb

        // 3. RGB 归一化 + NCHW。
        fillInput(canvas)

        // 4. 推理。
        val out = runInference() ?: return result

        // 5. 解码 + NMS。
        val dets = decode(out, r, padX, padY, crop.left, crop.top)
        lastDets = dets
        if (dets.isEmpty()) {
            if (logCounter++ % 20 == 0) Log.i(TAG, "检出 0 子")
            return result
        }

        // 6. 按置信度降序填格子（先到先得，避免相邻框重复占格）。
        for (d in dets) {
            val gr = boardInfo.imagePointToGrid(d.cx, d.cy) ?: continue
            val (file, rank) = gr
            if (result[rank][file] == null) {
                result[rank][file] = labelToPiece(d.label)
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
            Log.i(TAG, "检出 ${dets.size} 子 grid=$sb")
        }

        crop.bitmap.recycle()
        return result
    }

    // ------------------------------------------------------------- 裁剪 / letterbox

    /** 按四角外扩 CROP_PAD_RATIO 裁剪棋盘区域。 */
    private fun cropBoard(bitmap: Bitmap, corners: FloatArray): Cropped? {
        val xs = floatArrayOf(corners[0], corners[2], corners[4], corners[6])
        val ys = floatArrayOf(corners[1], corners[3], corners[5], corners[7])
        val minX = xs.minOrNull()!!; val maxX = xs.maxOrNull()!!
        val minY = ys.minOrNull()!!; val maxY = ys.maxOrNull()!!
        val bw = maxX - minX
        val bh = maxY - minY
        if (bw <= 0 || bh <= 0) return null
        val padX = bw * CROP_PAD_RATIO
        val padY = bh * CROP_PAD_RATIO
        val left = (minX - padX).toInt().coerceIn(0, bitmap.width - 1)
        val top = (minY - padY).toInt().coerceIn(0, bitmap.height - 1)
        val right = (maxX + padX).toInt().coerceIn(0, bitmap.width)
        val bottom = (maxY + padY).toInt().coerceIn(0, bitmap.height)
        val w = right - left
        val h = bottom - top
        if (w <= 0 || h <= 0) return null
        val bmp = Bitmap.createBitmap(bitmap, left, top, w, h)
        return Cropped(bmp, left, top)
    }

    private data class Cropped(val bitmap: Bitmap, val left: Int, val top: Int)

    /** letterbox：长边缩到 IMG_SIZE（保比例），居中放入 IMG_SIZE² 画布（114 灰填充）。 */
    private fun letterbox(bitmap: Bitmap): Letterboxed {
        val w = bitmap.width
        val h = bitmap.height
        val r = IMG_SIZE.toFloat() / max(w, h)
        val newW = max(1, (w * r).roundToInt())
        val newH = max(1, (h * r).roundToInt())
        val resized = if (newW == w && newH == h) bitmap
            else Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val padX = (IMG_SIZE - newW) / 2f
        val padY = (IMG_SIZE - newH) / 2f

        val canvas = IntArray(IMG_SIZE * IMG_SIZE)
        val gray = (0xFF shl 24) or (114 shl 16) or (114 shl 8) or 114
        for (i in canvas.indices) canvas[i] = gray

        val rp = IntArray(newW * newH)
        resized.getPixels(rp, 0, newW, 0, 0, newW, newH)
        val ox = padX.toInt()
        val oy = padY.toInt()
        for (y in 0 until newH) {
            System.arraycopy(rp, y * newW, canvas, (y + oy) * IMG_SIZE + ox, newW)
        }
        if (resized !== bitmap) resized.recycle()
        return Letterboxed(canvas, r, padX, padY)
    }

    private data class Letterboxed(val canvas: IntArray, val r: Float, val padX: Float, val padY: Float)

    // ------------------------------------------------------------- 推理

    private fun fillInput(canvas: IntArray) {
        inputBuffer.rewind()
        val plane = IMG_SIZE * IMG_SIZE
        // NCHW，逐通道平面写。
        for (c in 0 until 3) {
            for (i in 0 until plane) {
                val p = canvas[i]
                val v = when (c) {
                    0 -> (p shr 16) and 0xFF
                    1 -> (p shr 8) and 0xFF
                    else -> p and 0xFF
                }
                inputBuffer.put(c * plane + i, v / 255f)
            }
        }
        inputBuffer.rewind()
    }

    private fun runInference(): FloatArray? {
        val inputTensor = OnnxTensor.createTensor(
            env, inputBuffer, longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        )
        try {
            val result = session.run(mapOf("images" to inputTensor))
            try {
                val outTensor = result.get("output0").get() as OnnxTensor
                val fb = outTensor.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return arr
            } finally {
                result.close()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "YOLO 推理失败", t)
            return null
        } finally {
            inputTensor.close()
        }
    }

    // ------------------------------------------------------------- 解码

    private fun decode(
        out: FloatArray,
        r: Float,
        padX: Float,
        padY: Float,
        cropLeft: Int,
        cropTop: Int
    ): List<PieceDetection> {
        val numAnchors = out.size / (4 + NUM_CLASSES)
        val stride = numAnchors

        // 收集阈值以上的候选。
        val boxes = ArrayList<FloatArray>(64)   // xyxy（全帧坐标）
        val classes = ArrayList<Int>(64)
        val scores = ArrayList<Float>(64)

        for (a in 0 until numAnchors) {
            var bestCls = -1
            var bestScore = CONF_THRESHOLD
            for (c in 0 until NUM_CLASSES) {
                val s = out[(4 + c) * stride + a]
                if (s > bestScore) {
                    bestScore = s
                    bestCls = c
                }
            }
            if (bestCls < 0) continue

            val cx = out[a]
            val cy = out[stride + a]
            val w = out[2 * stride + a]
            val h = out[3 * stride + a]

            // xywh（letterbox）→ xyxy（letterbox）→ 去 letterbox → 全帧坐标。
            var x1 = (cx - w / 2f - padX) / r + cropLeft
            var y1 = (cy - h / 2f - padY) / r + cropTop
            var x2 = (cx + w / 2f - padX) / r + cropLeft
            var y2 = (cy + h / 2f - padY) / r + cropTop

            boxes.add(floatArrayOf(x1, y1, x2, y2))
            classes.add(bestCls)
            scores.add(bestScore)
        }

        // 按类别分组做 NMS。
        val keep = nmsPerClass(boxes, classes, scores, IOU_THRESHOLD)
        val dets = ArrayList<PieceDetection>(keep.size)
        for (i in boxes.indices) {
            if (!keep[i]) continue
            val b = boxes[i]
            dets.add(PieceDetection((b[0] + b[2]) / 2f, (b[1] + b[3]) / 2f, classes[i], scores[i]))
        }
        dets.sortByDescending { it.confidence }
        return dets
    }

    /** 每类独立 NMS，返回保留标记。 */
    private fun nmsPerClass(
        boxes: List<FloatArray>,
        classes: List<Int>,
        scores: List<Float>,
        iouThr: Float
    ): BooleanArray {
        val n = boxes.size
        val keep = BooleanArray(n) { true }
        for (i in 0 until n) {
            if (!keep[i]) continue
            for (j in i + 1 until n) {
                if (!keep[j] || classes[i] != classes[j]) continue
                if (iou(boxes[i], boxes[j]) > iouThr) {
                    keep[j] = false
                }
            }
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = max(a[0], b[0])
        val y1 = max(a[1], b[1])
        val x2 = minOf(a[2], b[2])
        val y2 = minOf(a[3], b[3])
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun labelToPiece(label: Int): Piece {
        val color = if (label <= 6) PieceColor.RED else PieceColor.BLACK
        val type = PieceType.entries[TYPE_ORDINALS[label % 7]]
        return Piece(color, type)
    }

    override fun lastDetections(): List<PieceDetection> = lastDets

    override fun close() {
        try { session.close() } catch (t: Throwable) { Log.w(TAG, "关闭 session 失败", t) }
        try { env.close() } catch (t: Throwable) { Log.w(TAG, "关闭 env 失败", t) }
    }

    companion object {
        private const val TAG = "YoloPieceDetector"
        const val MODEL_PATH = "models/yolov8n_chess.onnx"
        const val IMG_SIZE = 640
        const val NUM_CLASSES = 14
        const val ROWS = 10
        const val COLS = 9

        /** 置信度阈值。 */
        private const val CONF_THRESHOLD = 0.25f

        /** NMS IoU 阈值（与 ultralytics 默认一致）。 */
        private const val IOU_THRESHOLD = 0.45f

        /** 棋盘裁剪外扩比例（相对棋盘宽高）。 */
        private const val CROP_PAD_RATIO = 0.06f

        /** 类别序号 → PieceType ordinal（类别顺序：帅/仕/相/车/马/炮/兵）。 */
        private val TYPE_ORDINALS = intArrayOf(
            PieceType.GENERAL.ordinal,  // 0 帅/将
            PieceType.ADVISOR.ordinal,  // 1 仕/士
            PieceType.ELEPHANT.ordinal, // 2 相/象
            PieceType.CHARIOT.ordinal,  // 3 车
            PieceType.HORSE.ordinal,    // 4 马
            PieceType.CANNON.ordinal,   // 5 炮
            PieceType.PAWN.ordinal,     // 6 兵/卒
        )

        /** 从 assets 加载模型并创建检测器；模型缺失/损坏时抛出异常，由工厂降级。 */
        fun load(context: Context): YoloPieceDetector {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                val cores = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
                setIntraOpNumThreads(cores)
            }
            val session = env.createSession(readAsset(context, MODEL_PATH), opts)
            return YoloPieceDetector(env, session)
        }

        private fun readAsset(context: Context, path: String): ByteArray =
            context.assets.open(path).use { it.readBytes() }
    }
}
