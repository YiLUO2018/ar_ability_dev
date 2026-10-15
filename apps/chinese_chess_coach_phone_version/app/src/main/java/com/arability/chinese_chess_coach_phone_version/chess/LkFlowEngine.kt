package com.arability.chinese_chess_coach_phone_version.chess

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

/**
 * BoardTracker 的 OpenCV LK 执行层（纯逻辑在 [BoardTracker]，本类只做特征/光流；
 * PC 端 goodFeaturesToTrack + calcOpticalFlowPyrLK 前向+后向校验 的同款）。
 *
 * ⚠️ 仅真机/仪器测试可用（org.opencv 依赖）；JVM 单测不加载本类。
 * 调用方（MainActivity）负责灰度 Mat 的获取与生命周期：传入的 gray 会被 clone
 * 持有为参考帧，调用方可在帧回调结束后 recycle 自己的 Mat。
 */
class LkFlowEngine(
    private val maxPoints: Int = 250,
    private val minDistance: Double = 12.0,
    private val fbMax: Float = 2.5f,
) {
    private var prevGray: Mat? = null
    private var pts = FloatArray(0)   // 当前帧跟踪点（xy 交错，图像坐标）

    fun hasFeatures(): Boolean = pts.size / 2 >= BoardTracker.MIN_TRACK_PTS

    /** 按种子 ROI（棋盘四角外扩多边形）重提特征；返回特征数。 */
    fun extractFeatures(gray: Mat, roiCorners: FloatArray): Int {
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        val poly = MatOfPoint()
        val arr = ArrayList<org.opencv.core.Point>(4)
        for (i in 0 until 4) arr.add(org.opencv.core.Point(
            roiCorners[i * 2].toDouble(), roiCorners[i * 2 + 1].toDouble()))
        poly.fromList(arr)
        Imgproc.fillPoly(mask, java.util.Collections.singletonList(poly), org.opencv.core.Scalar(1.0))
        poly.release()
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(
            gray, corners, maxPoints, 0.01, minDistance.toDouble(), mask, 7, false, 0.04)
        val list = corners.toList()
        val out = FloatArray(list.size * 2)
        for (i in list.indices) {
            out[i * 2] = list[i].x.toFloat(); out[i * 2 + 1] = list[i].y.toFloat()
        }
        pts = out
        corners.release(); mask.release()
        prevGray?.release()
        prevGray = gray.clone()
        return out.size / 2
    }

    /**
     * 帧间前向+后向 LK。成功返回 (src, dst) 好点对（xy 交错，已过后向校验 + FB 闸
     * + 最少点数闸；点数不足时内部清特征并返回 null——PC track() 行为）。
     */
    fun track(gray: Mat): Pair<FloatArray, FloatArray>? {
        val prev = prevGray ?: return null
        if (pts.size / 2 < BoardTracker.MIN_TRACK_PTS) return null
        val prevPts = MatOfPoint2f()
        prevPts.fromArray(*toPointArray(pts))
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        val tc = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 30, 0.01)
        Video.calcOpticalFlowPyrLK(
            prev, gray, prevPts, nextPts, status, err, Size(31.0, 31.0), 3, tc)
        val backPts = MatOfPoint2f()
        val status2 = MatOfByte()
        val err2 = MatOfFloat()
        Video.calcOpticalFlowPyrLK(
            gray, prev, nextPts, backPts, status2, err2, Size(31.0, 31.0), 3, tc)
        val st = status.toArray()
        val st2 = status2.toArray()
        val srcArr = prevPts.toArray()
        val dstArr = nextPts.toArray()
        val backArr = backPts.toArray()
        val srcOut = ArrayList<Float>(srcArr.size * 2)
        val dstOut = ArrayList<Float>(srcArr.size * 2)
        for (i in srcArr.indices) {
            if (i >= st.size || st[i].toInt() != 1) continue
            if (i < st2.size && st2[i].toInt() != 1) continue
            val dx = backArr[i].x - srcArr[i].x
            val dy = backArr[i].y - srcArr[i].y
            if (dx * dx + dy * dy >= fbMax * fbMax) continue
            srcOut.add(srcArr[i].x.toFloat()); srcOut.add(srcArr[i].y.toFloat())
            dstOut.add(dstArr[i].x.toFloat()); dstOut.add(dstArr[i].y.toFloat())
        }
        prevPts.release(); nextPts.release(); status.release(); err.release()
        backPts.release(); status2.release(); err2.release()
        if (srcOut.size / 2 < BoardTracker.MIN_TRACK_PTS) {
            pts = FloatArray(0)
            return null
        }
        val src = FloatArray(srcOut.size); val dst = FloatArray(dstOut.size)
        for (i in srcOut.indices) { src[i] = srcOut[i]; dst[i] = dstOut[i] }
        pts = dst
        prevGray?.release()
        prevGray = gray.clone()
        return Pair(src, dst)
    }

    fun release() {
        prevGray?.release()
        prevGray = null
        pts = FloatArray(0)
    }

    private fun toPointArray(a: FloatArray): Array<org.opencv.core.Point> {
        val n = a.size / 2
        return Array(n) { org.opencv.core.Point(a[it * 2].toDouble(), a[it * 2 + 1].toDouble()) }
    }
}
