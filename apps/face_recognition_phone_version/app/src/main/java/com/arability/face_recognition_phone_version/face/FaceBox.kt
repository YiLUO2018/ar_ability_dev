package com.arability.face_recognition_phone_version.face

/**
 * 人脸检测结果，坐标均为归一化 [0,1]，与具体检测模型无关。
 *
 * @param keypoints 归一化 [0,1] 关键点列表，每个元素为 [x, y]。检测模型无关键点时为空。
 * @param powerLevel 战斗力显示值（0~10000），由平滑器在跟踪期间保持稳定。
 */
data class FaceBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    val keypoints: List<FloatArray> = emptyList(),
    val powerLevel: Int = 0
)
