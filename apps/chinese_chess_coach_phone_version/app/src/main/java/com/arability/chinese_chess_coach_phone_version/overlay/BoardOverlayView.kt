package com.arability.chinese_chess_coach_phone_version.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.arability.chinese_chess_coach_phone_version.chess.BoardGeometry
import com.arability.chinese_chess_coach_phone_version.chess.BoardInfo
import com.arability.chinese_chess_coach_phone_version.chess.Homography
import com.arability.chinese_chess_coach_phone_version.chess.Move
import com.arability.chinese_chess_coach_phone_version.chess.Piece
import com.arability.chinese_chess_coach_phone_version.chess.PieceColor
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

    private val pieceFillRed = Paint().apply {
        color = 0xFFE53935.toInt()
        style = Paint.Style.FILL
    }

    private val pieceFillBlack = Paint().apply {
        color = 0xFF212121.toInt()
        style = Paint.Style.FILL
    }

    private val pieceRingPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val pieceTextPaint = Paint().apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private var boardInfo: BoardInfo? = null
    private var move: Move? = null
    private var pieces: Array<Array<Piece?>>? = null
    private var imageWidth = 0
    private var imageHeight = 0
    private var boardToSource: FloatArray? = null

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
        boardToSource = if (boardInfo != null && boardInfo.corners.size >= 8) {
            Homography.compute(BoardGeometry.physicalCorners, boardInfo.corners)
        } else {
            null
        }
        invalidate()
    }

    /** 设置识别出的棋子网格（[rank][file]），在每个交叉点上画红/黑中文名。 */
    fun setPieces(grid: Array<Array<Piece?>>?) {
        this.pieces = grid
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

        drawPieces(canvas, info)

        val m = move ?: return
        val fromSrc = intersection(info, m.fromFile, m.fromRank)
        val toSrc = intersection(info, m.toFile, m.toRank)
        val from = toView(fromSrc[0], fromSrc[1])
        val to = toView(toSrc[0], toSrc[1])
        drawArrow(canvas, from[0], from[1], to[0], to[1])
        val r = pointPaint.textSize.coerceAtLeast(14f)
        canvas.drawCircle(from[0], from[1], r, pointPaint)
        canvas.drawCircle(to[0], to[1], r, pointPaint)
    }

    /**
     * 将棋盘交叉点 (file, rank) 映射到源图像坐标。
     *
     * corners 为物理棋盘四角（SegmentationBoardDetector 输出），交叉点在 canonical 图中位于
     * (origin + file*cell, origin + rank*cell)。用单应矩阵（而非双线性插值）做精确透视映射，
     * 避免斜拍时网格交点偏离。
     */
    private fun intersection(info: BoardInfo, file: Int, rank: Int): FloatArray {
        val cx = (info.originX + file * info.cellSize).toFloat()
        val cy = (info.originY + rank * info.cellSize).toFloat()
        val h = boardToSource ?: Homography.compute(BoardGeometry.physicalCorners, info.corners)
        return Homography.transform(h, cx, cy)
    }

    /** 在每个交叉点上画棋子：红方红底、黑方黑底，白色中文名。 */
    private fun drawPieces(canvas: Canvas, info: BoardInfo) {
        val grid = pieces ?: return
        val r = cellRadius(info)
        if (r < 3f) return
        pieceTextPaint.textSize = r * 1.5f
        val fm = pieceTextPaint.fontMetrics
        val textOffsetY = -(fm.ascent + fm.descent) / 2f
        for (rank in 0..9) {
            for (file in 0..8) {
                val p = grid[rank][file] ?: continue
                val src = intersection(info, file, rank)
                val v = toView(src[0], src[1])
                val fill = if (p.color == PieceColor.RED) pieceFillRed else pieceFillBlack
                canvas.drawCircle(v[0], v[1], r, fill)
                canvas.drawCircle(v[0], v[1], r, pieceRingPaint)
                canvas.drawText(p.name, v[0], v[1] + textOffsetY, pieceTextPaint)
            }
        }
    }

    /** 棋盘交叉点间距（视图坐标）的一半略小，作为棋子圆的半径。 */
    private fun cellRadius(info: BoardInfo): Float {
        val a = intersection(info, 0, 0)
        val b = intersection(info, 1, 0)
        val va = toView(a[0], a[1])
        val vb = toView(b[0], b[1])
        val dx = vb[0] - va[0]
        val dy = vb[1] - va[1]
        return kotlin.math.sqrt((dx * dx + dy * dy).toDouble()).toFloat() * 0.42f
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
}
