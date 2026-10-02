package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 纯 Kotlin 棋子识别实现：交叉点占用检测 + 红黑颜色分类 + 字形模板匹配。
 *
 * 作为「可替换识别器」的默认（降级）实现，无模型依赖，先跑通闭环；
 * 字形识别依赖程序渲染的模板，鲁棒性有限，后续应替换为 TFLite 分类模型。
 */
class TemplatePieceRecognizer : PieceRecognizer {

    private val templates: MutableMap<Piece, IntArray> = HashMap()

    override fun recognize(boardInfo: BoardInfo): Array<Array<Piece?>> {
        val warped = boardInfo.warped
        val w = warped.width
        val h = warped.height
        val pixels = IntArray(w * h)
        warped.getPixels(pixels, 0, w, 0, 0, w, h)
        val background = estimateBackground(pixels)
        val size = boardInfo.cellSize

        val result = Array(10) { arrayOfNulls<Piece>(9) }
        for (rank in 0..9) {
            for (file in 0..8) {
                val cx = boardInfo.intersectionX(file)
                val cy = boardInfo.intersectionY(rank)
                val roi = cropGray(pixels, w, h, cx, cy, size)
                if (roi == null) continue
                if (occupancy(roi, background) < OCCUPY_THRESHOLD) continue
                val color = classifyColor(roi)
                val type = matchType(roi, color, size)
                result[rank][file] = Piece(color, type)
            }
        }
        return result
    }

    // ------------------------------------------------------------- 占用/颜色

    /** 估计棋盘木色背景（全图平均，木色占绝对多数）。 */
    private fun estimateBackground(pixels: IntArray): Int {
        var r = 0L; var g = 0L; var b = 0L
        for (p in pixels) {
            r += (p shr 16) and 0xFF
            g += (p shr 8) and 0xFF
            b += p and 0xFF
        }
        val n = pixels.size
        return Color.rgb((r / n).toInt(), (g / n).toInt(), (b / n).toInt())
    }

    /** 与背景差异明显的像素占比（棋子圆饼覆盖 ROI 大部分面积）。 */
    private fun occupancy(roi: IntArray, background: Int): Float {
        val br = (background shr 16) and 0xFF
        val bg = (background shr 8) and 0xFF
        val bb = background and 0xFF
        var count = 0
        for (p in roi) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (abs(r - br) + abs(g - bg) + abs(b - bb) > BG_DIFF_THRESHOLD) count++
        }
        return count.toFloat() / roi.size
    }

    /** 红黑分类：红色棋子的红通道显著高于绿/蓝。 */
    private fun classifyColor(roi: IntArray): PieceColor {
        var r = 0L; var g = 0L; var b = 0L
        for (p in roi) {
            r += (p shr 16) and 0xFF
            g += (p shr 8) and 0xFF
            b += p and 0xFF
        }
        val n = roi.size
        val mr = (r / n).toInt()
        val mg = (g / n).toInt()
        val mb = (b / n).toInt()
        return if (mr - max(mg, mb) > RED_DOMINANCE) PieceColor.RED else PieceColor.BLACK
    }

    // ------------------------------------------------------------- 字形匹配

    private fun matchType(roi: IntArray, color: PieceColor, size: Int): PieceType {
        val candidates = PieceType.entries.map { Piece(color, it) }
        var bestType = PieceType.PAWN
        var bestScore = -Float.MAX_VALUE
        for (piece in candidates) {
            val tmpl = templates.getOrPut(piece) { renderTemplate(piece, size) }
            val score = ncc(roi, tmpl)
            if (score > bestScore) {
                bestScore = score
                bestType = piece.type
            }
        }
        return bestType
    }

    /** 归一化互相关系数（NCC），衡量 ROI 与模板的相似度。 */
    private fun ncc(a: IntArray, b: IntArray): Float {
        val n = a.size
        var ma = 0.0; var mb = 0.0
        for (i in 0 until n) { ma += a[i]; mb += b[i] }
        ma /= n; mb /= n
        var num = 0.0; var da = 0.0; var db = 0.0
        for (i in 0 until n) {
            val xa = a[i] - ma
            val xb = b[i] - mb
            num += xa * xb
            da += xa * xa
            db += xb * xb
        }
        val den = sqrt(da * db)
        if (den < 1e-6) return 0f
        return (num / den).toFloat()
    }

    /** 渲染一枚棋子的灰度模板（圆饼 + 汉字）。 */
    private fun renderTemplate(piece: Piece, size: Int): IntArray {
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val textColor = if (piece.color == PieceColor.RED) Color.rgb(200, 40, 40) else Color.BLACK

        val disc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(230, 200, 140)
        }
        canvas.drawCircle(size / 2f, size / 2f, size * 0.48f, disc)

        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = textColor
            textSize = size * 0.56f
            typeface = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }
        val baseline = size / 2f - (text.descent() + text.ascent()) / 2f
        canvas.drawText(piece.name, size / 2f, baseline, text)

        val gray = IntArray(size * size)
        val px = IntArray(size * size)
        bitmap.getPixels(px, 0, size, 0, 0, size, size)
        for (i in px.indices) {
            val p = px[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = (0.299f * r + 0.587f * g + 0.114f * b).toInt()
        }
        bitmap.recycle()
        return gray
    }

    // ------------------------------------------------------------- ROI

    /** 以 (cx, cy) 为中心裁剪 size×size 灰度 ROI；越界返回 null。 */
    private fun cropGray(pixels: IntArray, w: Int, h: Int, cx: Int, cy: Int, size: Int): IntArray? {
        val half = size / 2
        val left = cx - half
        val top = cy - half
        if (left < 0 || top < 0 || left + size > w || top + size > h) return null
        val gray = IntArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val p = pixels[(top + y) * w + (left + x)]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                gray[y * size + x] = (0.299f * r + 0.587f * g + 0.114f * b).toInt()
            }
        }
        return gray
    }

    private fun abs(v: Int): Int = if (v < 0) -v else v

    private companion object {
        const val OCCUPY_THRESHOLD = 0.3f
        const val BG_DIFF_THRESHOLD = 60
        const val RED_DOMINANCE = 30
    }
}
