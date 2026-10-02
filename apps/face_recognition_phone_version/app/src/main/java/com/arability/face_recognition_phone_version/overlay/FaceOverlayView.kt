package com.arability.face_recognition_phone_version.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import com.arability.face_recognition_phone_version.face.FaceBox
import kotlin.math.max

/**
 * 叠加在预览层上的绘制层，用于绘制人脸 BBox 与识别标签。
 */
class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val labelPaint = Paint().apply {
        color = Color.GREEN
        textSize = 42f * resources.displayMetrics.density
        isAntiAlias = true
    }

    private val labelBackground = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val powerPaint = Paint().apply {
        color = Color.GREEN
        textSize = 34f * resources.displayMetrics.density
        isAntiAlias = true
        typeface = Typeface.MONOSPACE
    }

    private var faces: List<FaceBox> = emptyList()
    private var labels: List<String?> = emptyList()

    // 检测源图像的尺寸（upright），用于按预览 FILL_CENTER 缩放映射归一化坐标
    private var imageWidth = 0
    private var imageHeight = 0

    fun setFaces(faces: List<FaceBox>, labels: List<String?>) {
        this.faces = faces
        this.labels = labels
        invalidate()
    }

    fun setImageSize(width: Int, height: Int) {
        if (imageWidth != width || imageHeight != height) {
            imageWidth = width
            imageHeight = height
            invalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (faces.isEmpty()) return

        // PreviewView 默认 FILL_CENTER：放大图像填满 view，超出部分居中裁剪。
        val scale: Float
        val displayW: Float
        val displayH: Float
        val offsetX: Float
        val offsetY: Float
        if (imageWidth > 0 && imageHeight > 0) {
            scale = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
            displayW = imageWidth * scale
            displayH = imageHeight * scale
            offsetX = (width - displayW) / 2f
            offsetY = (height - displayH) / 2f
        } else {
            scale = 1f
            displayW = width.toFloat()
            displayH = height.toFloat()
            offsetX = 0f
            offsetY = 0f
        }

        for (i in faces.indices) {
            val face = faces[i]
            val left = face.left * displayW + offsetX
            val top = face.top * displayH + offsetY
            val right = face.right * displayW + offsetX
            val bottom = face.bottom * displayH + offsetY
            canvas.drawRect(left, top, right, bottom, boxPaint)

            drawPower(canvas, face.powerLevel, left, bottom)

            val label = labels.getOrNull(i)
            if (!label.isNullOrEmpty()) {
                drawLabel(canvas, label, left, top)
            }
        }
    }

    private fun drawPower(canvas: Canvas, power: Int, boxLeft: Float, boxBottom: Float) {
        val text = "战斗力：$power"
        val textHeight = powerPaint.textSize
        val textWidth = powerPaint.measureText(text)
        val padding = 8f * resources.displayMetrics.density

        val x = boxLeft.coerceAtLeast(0f)
        val y = (boxBottom + textHeight + padding).coerceAtMost(height - padding)

        canvas.drawRoundRect(
            x,
            y - textHeight,
            x + textWidth + padding * 2,
            y + padding,
            padding,
            padding,
            labelBackground
        )
        canvas.drawText(text, x + padding, y, powerPaint)
    }

    private fun drawLabel(canvas: Canvas, text: String, boxLeft: Float, boxTop: Float) {
        val textHeight = labelPaint.textSize
        val textWidth = labelPaint.measureText(text)
        val padding = 8f * resources.displayMetrics.density

        val x = boxLeft.coerceAtLeast(0f)
        val y = (boxTop - textHeight - padding).coerceAtLeast(textHeight + padding)

        canvas.drawRoundRect(
            x,
            y - textHeight,
            x + textWidth + padding * 2,
            y + padding,
            padding,
            padding,
            labelBackground
        )
        canvas.drawText(text, x + padding, y, labelPaint)
    }
}
