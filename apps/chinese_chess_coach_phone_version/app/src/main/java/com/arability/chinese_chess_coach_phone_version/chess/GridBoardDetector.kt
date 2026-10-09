package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 纯 Kotlin 棋盘定位实现（灰度 → Sobel 边缘 → Hough 直线 → 极值线求角 → 透视校正）。
 *
 * 作为「可替换定位器」的默认实现，无 OpenCV/NDK 依赖；后续可替换为 OpenCV 版。
 * 阈值与聚类参数为经验值，需在真机上按实际光照/角度微调。
 */
class GridBoardDetector(
    private val maxDimension: Int = 480,
    private val debugDir: java.io.File? = null
) : BoardDetector {

    private var debugCount = 0
    private var debugSaved = false

    override fun detect(bitmap: Bitmap): BoardInfo? {
        // 1. 缩放到 maxDimension 内以加速
        val scale = 1f.coerceAtMost(maxDimension.toFloat() / max(bitmap.width, bitmap.height))
        val sw = max(1, (bitmap.width * scale).toInt())
        val sh = max(1, (bitmap.height * scale).toInt())
        val scaled = if (sw == bitmap.width && sh == bitmap.height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, sw, sh, true)
        }
        val pixels = IntArray(sw * sh)
        scaled.getPixels(pixels, 0, sw, 0, 0, sw, sh)

        // 2. 灰度 + 边缘 + 直线
        val gray = grayscale(pixels, sw, sh)
        val edges = sobel(gray, sw, sh)
        if (!debugSaved) {
            debugSaved = true
            saveDebug(scaled, edges, sw, sh)
        }
        val lines = houghLines(edges, sw, sh)
        if (lines == null) {
            debugLog("detect fail: no Hough lines (no edge votes) sw=$sw sh=$sh edgePx=${countEdges(edges)}")
            return null
        }

        // 3. 分类：水平线（y 坐标）与竖直线（x 坐标）。
        //    只保留接近水平/竖直且票数足够高的直线，过滤棋子圆边等斜向噪声。
        val maxVotes = lines.maxOf { it.votes }
        val minVote = (maxVotes * MIN_VOTE_RATIO).toInt()
        val marginX = sw * BORDER_MARGIN
        val marginY = sh * BORDER_MARGIN

        val horiz = ArrayList<Float>()
        val vert = ArrayList<Float>()
        for (l in lines) {
            if (l.votes < minVote) continue
            val rad = l.theta * PI / 180.0
            val s = sin(rad)
            val c = cos(rad)
            if (l.theta in H_BAND) {
                // 水平线：取其在图像中心列处的 y 坐标
                val y = ((l.rho - c * sw / 2) / s).toFloat()
                if (y in marginY..(sh - marginY)) horiz.add(y)
            } else if (l.theta in V_BAND_LOW || l.theta in V_BAND_HIGH) {
                // 竖直线：取其在图像中心行处的 x 坐标
                val x = ((l.rho - s * sh / 2) / c).toFloat()
                if (x in marginX..(sw - marginX)) vert.add(x)
            }
        }
        if (horiz.size < 2 || vert.size < 2) {
            debugLog("detect fail: peaks=${lines.size} horiz=${horiz.size} vert=${vert.size} edgePx=${countEdges(edges)}")
            return null
        }

        val top = horiz.minOrNull()!!
        val bottom = horiz.maxOrNull()!!
        val left = vert.minOrNull()!!
        val right = vert.maxOrNull()!!

        // 4. 四角（缩放坐标）
        val tl = floatArrayOf(left, top)
        val tr = floatArrayOf(right, top)
        val br = floatArrayOf(right, bottom)
        val bl = floatArrayOf(left, bottom)

        // 5. 缩放坐标映射回原始图像坐标
        val inv = 1f / scale
        val src = floatArrayOf(
            tl[0] * inv, tl[1] * inv,
            tr[0] * inv, tr[1] * inv,
            br[0] * inv, br[1] * inv,
            bl[0] * inv, bl[1] * inv
        )

        // 6. 目标：canonical 棋盘（带 MARGIN），网格最外层交点即四角。
        val cw = BoardGeometry.BOARD_W
        val ch = BoardGeometry.BOARD_H
        val dst = BoardGeometry.corners

        val hInv = Homography.compute(dst, src) // canonical → src
        val warpedPixels = Homography.warp(pixels, sw, sh, cw, ch, hInv)
        val warped = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
        warped.setPixels(warpedPixels, 0, cw, 0, 0, cw, ch)

        debugLog("detect ok: horiz=${horiz.size} vert=${vert.size} top=$top bottom=$bottom left=$left right=$right")
        return BoardInfo(
            corners = src,
            warped = warped,
            cellSize = BoardGeometry.CELL,
            originX = BoardGeometry.MARGIN,
            originY = BoardGeometry.MARGIN,
            imageToGrid = Homography.compute(src, BoardGeometry.corners)
        )
    }

    // ------------------------------------------------------------- 基础算子

    private fun grayscale(pixels: IntArray, w: Int, h: Int): IntArray {
        val gray = IntArray(w * h)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = (0.299f * r + 0.587f * g + 0.114f * b).toInt()
        }
        return gray
    }

    private fun sobel(gray: IntArray, w: Int, h: Int): ByteArray {
        val edges = ByteArray(w * h)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val gx = -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] +
                    gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1]
                val gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] +
                    gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1]
                val mag = sqrt((gx * gx + gy * gy).toDouble())
                if (mag > EDGE_THRESHOLD) edges[i] = 1
            }
        }
        return edges
    }

    /** Hough 直线检测，返回非极大值抑制后的峰值直线（含票数）。 */
    private fun houghLines(edges: ByteArray, w: Int, h: Int): List<Peak>? {
        val diag = sqrt((w * w + h * h).toDouble()).toInt()
        val thetaCount = 180
        val rhoCount = 2 * diag + 1
        val acc = IntArray(thetaCount * rhoCount)

        var maxVotes = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (edges[y * w + x] == 0.toByte()) continue
                for (t in 0 until thetaCount) {
                    val rad = t * PI / 180.0
                    val rho = x * cos(rad) + y * sin(rad)
                    val ri = (round(rho) + diag).toInt()
                    if (ri in 0 until rhoCount) {
                        val idx = t * rhoCount + ri
                        acc[idx]++
                        if (acc[idx] > maxVotes) maxVotes = acc[idx]
                    }
                }
            }
        }
        if (maxVotes == 0) return null

        val threshold = (maxVotes * PEAK_RATIO).toInt()
        val candidates = ArrayList<IntArray>()
        for (t in 0 until thetaCount) {
            for (r in 0 until rhoCount) {
                val v = acc[t * rhoCount + r]
                if (v >= threshold) candidates.add(intArrayOf(t, r, v))
            }
        }

        // 贪心非极大值抑制：按票数降序，跳过与已选峰值过近的候选。
        // 注意：比较必须使用「原始 rho 索引」这一同一坐标系。
        candidates.sortByDescending { it[2] }
        val peaks = ArrayList<Peak>()
        for ((t, r, v) in candidates) {
            val tooClose = peaks.any { abs(t - it.theta) <= MERGE_DEG && abs(r - it.rawRho) <= MERGE_RHO }
            if (tooClose) continue
            peaks.add(Peak(theta = t, rho = (r - diag).toFloat(), rawRho = r, votes = v))
        }
        return peaks
    }

    private class Peak(val theta: Int, val rho: Float, val rawRho: Int, val votes: Int)

    private fun countEdges(edges: ByteArray): Int {
        var n = 0
        for (e in edges) if (e == 1.toByte()) n++
        return n
    }

    private fun saveDebug(scaled: Bitmap, edges: ByteArray, w: Int, h: Int) {
        try {
            val dir = debugDir ?: return
            dir.mkdirs()
            java.io.FileOutputStream(java.io.File(dir, "frame.png")).use {
                scaled.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            val edgeBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val px = IntArray(w * h)
            for (i in px.indices) px[i] = if (edges[i] == 1.toByte()) -0x1 else -0x1000000
            edgeBmp.setPixels(px, 0, w, 0, 0, w, h)
            java.io.FileOutputStream(java.io.File(dir, "edges.png")).use {
                edgeBmp.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            Log.d(TAG, "debug images saved to ${dir.absolutePath}")
        } catch (t: Throwable) {
            Log.w(TAG, "saveDebug failed", t)
        }
    }

    private fun debugLog(msg: String) {
        // 每 5 次检测输出一次，避免刷屏
        if (debugCount++ % 5 == 0) Log.d(TAG, msg)
    }

    private companion object {
        const val TAG = "GridBoardDetector"
        const val EDGE_THRESHOLD = 100.0
        const val PEAK_RATIO = 0.25
        const val MERGE_DEG = 8
        const val MERGE_RHO = 12

        // 只保留接近水平/竖直的直线（过滤棋子圆边等斜向噪声）
        val H_BAND = 84..96          // 水平线 theta 范围（°）
        val V_BAND_LOW = 0..6        // 竖直线 theta 范围（°）
        val V_BAND_HIGH = 174..180   // 竖直线 theta 范围（°，接近 180°）

        const val MIN_VOTE_RATIO = 0.45f  // 保留票数不低于最高票 45% 的直线
        const val BORDER_MARGIN = 0.02f   // 剔除贴图像边界的伪直线（比例）
    }
}
