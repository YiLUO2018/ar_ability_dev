package com.arability.chinese_chess_coach_phone_version.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.arability.chinese_chess_coach_phone_version.chess.BoardInfo
import com.arability.chinese_chess_coach_phone_version.chess.Move
import kotlin.math.max

/**
 * 叠加在预览层上的绘制层：绘制检测到的棋盘四角（反馈）与建议走法（高亮起点/终点 + 箭头）。
 */
class BoardOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boardPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val movePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val arrowPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    private val pointPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    private var boardInfo: BoardInfo? = null
    private var move: Move? = null
    private var imageWidth = 0
    private var imageHeight = 0

    fun setImageSize(width: Int, height: Int) {
        if (imageWidth != width || imageHeight != height) {
            imageWidth = width
            imageHeight = height
            invalidate()
        }
    }

    fun setSuggestion(boardInfo: BoardInfo?, move: Move?) {
        this.boardInfo = boardInfo
        this.move = move
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val info = boardInfo ?: return

        if (info.corners.size >= 8) {
            val p0 = toView(info.corners[0], info.corners[1])
            val p1 = toView(info.corners[2], info.corners[3])
            val p2 = toView(info.corners[4], info.corners[5])
            val p3 = toView(info.corners[6], info.corners[7])
            canvas.drawLine(p0[0], p0[1], p1[0], p1[1], boardPaint)
            canvas.drawLine(p1[0], p1[1], p2[0], p2[1], boardPaint)
            canvas.drawLine(p2[0], p2[1], p3[0], p3[1], boardPaint)
            canvas.drawLine(p3[0], p3[1], p0[0], p0[1], boardPaint)
        }

        val m = move ?: return
        val from = toView(intersectionX(info, m.fromFile), intersectionY(info, m.fromRank))
        val to = toView(intersectionX(info, m.toFile), intersectionY(info, m.toRank))
        drawArrow(canvas, from[0], from[1], to[0], to[1])
        val r = pointPaint.textSize.coerceAtLeast(14f)
        canvas.drawCircle(from[0], from[1], r, pointPaint)
        canvas.drawCircle(to[0], to[1], r, pointPaint)
    }

    /** 将棋盘交叉点映射到源图像坐标（四角双线性插值）。 */
    private fun intersectionX(info: BoardInfo, file: Int): Float {
        val fx = file / 8f
        val fy = 0f
        val topX = lerp(info.corners[0], info.corners[2], fx)
        val bottomX = lerp(info.corners[6], info.corners[4], fx)
        return lerp(topX, bottomX, fy)
    }

    private fun intersectionY(info: BoardInfo, rank: Int): Float {
        val fx = 0f
        val fy = rank / 9f
        val topY = lerp(info.corners[1], info.corners[3], fx)
        val bottomY = lerp(info.corners[7], info.corners[5], fx)
        return lerp(topY, bottomY, fy)
    }

    /** 源图像坐标 → 视图坐标（PreviewView FILL_CENTER 缩放 + 居中裁剪）。 */
    private fun toView(srcX: Float, srcY: Float): FloatArray {
        val scale: Float
        val offsetX: Float
        val offsetY: Float
        if (imageWidth > 0 && imageHeight > 0) {
            scale = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
            val displayW = imageWidth * scale
            val displayH = imageHeight * scale
            offsetX = (width - displayW) / 2f
            offsetY = (height - displayH) / 2f
        } else {
            scale = 1f
            offsetX = 0f
            offsetY = 0f
        }
        return floatArrayOf(srcX * scale + offsetX, srcY * scale + offsetY)
    }

    private fun drawArrow(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float) {
        val dx = x1 - x0
        val dy = y1 - y0
        val len = kotlin.math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
        if (len < 1f) return
        val ux = dx / len
        val uy = dy / len
        val headLen = 26f
        val headWidth = 14f
        // 箭头线段（终点回缩，避免盖住终点圆）
        canvas.drawLine(x0, y0, x1 - ux * headLen, y1 - uy * headLen, movePaint)
        // 箭头头部三角形
        val bx = x1 - ux * headLen
        val by = y1 - uy * headLen
        val px = -uy * headWidth
        val py = ux * headWidth
        val path = android.graphics.Path().apply {
            moveTo(x1, y1)
            lineTo(bx + px, by + py)
            lineTo(bx - px, by - py)
            close()
        }
        canvas.drawPath(path, arrowPaint)
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t
}
