package com.arability.face_recognition_phone_version.face

import kotlin.math.max
import kotlin.random.Random

/**
 * 对连续帧的人脸框做时间平滑与稳定性过滤：
 * 1. EMA 平滑：消除逐帧抖动导致的闪烁；
 * 2. 确认迟滞：人脸需连续被检测到 [confirmFrames] 帧才会输出，过滤偶发误检；
 * 3. 丢失迟滞：已确认的人脸在消失 [maxLostFrames] 帧内仍保留，避免短暂丢失闪烁。
 *
 * 按 IoU 将当前帧人脸与上一帧轨迹匹配，匹配成功则加权融合。
 */
class FaceBoxSmoother(
    private val alpha: Float = 0.35f,
    private val matchThreshold: Float = 0.3f,
    private val confirmFrames: Int = 3,
    private val maxLostFrames: Int = 5
) {

    private class Track(
        var box: FaceBox,
        var seen: Int = 1,
        var lost: Int = 0
    )

    private var tracks: MutableList<Track> = mutableListOf()

    fun smooth(faces: List<FaceBox>): List<FaceBox> {
        if (faces.isEmpty()) {
            for (t in tracks) t.lost++
        } else {
            val originalCount = tracks.size
            val matched = BooleanArray(originalCount)

            for (face in faces) {
                var bestIndex = -1
                var bestIoU = 0f
                for (i in 0 until originalCount) {
                    if (matched[i]) continue
                    val iou = iou(face, tracks[i].box)
                    if (iou > bestIoU) {
                        bestIoU = iou
                        bestIndex = i
                    }
                }

                if (bestIndex >= 0 && bestIoU >= matchThreshold) {
                    matched[bestIndex] = true
                    val track = tracks[bestIndex]
                    track.box = blend(track.box, face)
                    track.seen++
                    track.lost = 0
                } else {
                    tracks.add(Track(face.copy(powerLevel = Random.nextInt(POWER_MAX + 1))))
                }
            }

            for (i in 0 until originalCount) {
                if (!matched[i]) tracks[i].lost++
            }
        }

        tracks.removeAll { it.lost > maxLostFrames }
        return tracks.filter { it.seen >= confirmFrames }.map { it.box }
    }

    fun reset() {
        tracks = mutableListOf()
    }

    private fun blend(prev: FaceBox, next: FaceBox): FaceBox {
        val blendedKeypoints = if (prev.keypoints.size == next.keypoints.size) {
            prev.keypoints.indices.map { i ->
                floatArrayOf(
                    alpha * next.keypoints[i][0] + (1 - alpha) * prev.keypoints[i][0],
                    alpha * next.keypoints[i][1] + (1 - alpha) * prev.keypoints[i][1]
                )
            }
        } else {
            next.keypoints
        }
        return FaceBox(
            left = alpha * next.left + (1 - alpha) * prev.left,
            top = alpha * next.top + (1 - alpha) * prev.top,
            right = alpha * next.right + (1 - alpha) * prev.right,
            bottom = alpha * next.bottom + (1 - alpha) * prev.bottom,
            score = next.score,
            keypoints = blendedKeypoints,
            powerLevel = prev.powerLevel
        )
    }

    private fun iou(a: FaceBox, b: FaceBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val iw = right - left
        val ih = bottom - top
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return inter / (areaA + areaB - inter)
    }

    companion object {
        /** 战斗力随机范围上限（0~10000） */
        private const val POWER_MAX = 10000
    }
}
