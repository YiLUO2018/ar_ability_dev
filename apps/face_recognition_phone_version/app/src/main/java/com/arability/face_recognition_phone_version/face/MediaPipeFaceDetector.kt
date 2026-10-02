package com.arability.face_recognition_phone_version.face

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector as MpFaceDetector

/**
 * 基于 MediaPipe Tasks 的人脸检测实现（离线，不依赖 Google Play）。
 */
class MediaPipeFaceDetector(context: Context) : FaceDetector {

    private val detector: MpFaceDetector

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .build()
        val options = MpFaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setMinDetectionConfidence(MIN_DETECTION_CONFIDENCE)
            .build()
        detector = MpFaceDetector.createFromOptions(context, options)
    }

    override fun detect(bitmap: Bitmap): List<FaceBox> {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = detector.detect(mpImage)
        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()
        return result.detections().map { detection ->
            val box = detection.boundingBox()
            // MediaPipe 关键点顺序：0=右眼 1=左眼 2=鼻尖 3=嘴中心 4=右耳 5=左耳（归一化 [0,1]）
            val keypoints = detection.keypoints().map { kps ->
                kps.map { kp -> floatArrayOf(kp.x(), kp.y()) }
            }.orElse(emptyList())
            FaceBox(
                left = box.left / width,
                top = box.top / height,
                right = box.right / width,
                bottom = box.bottom / height,
                score = detection.categories().firstOrNull()?.score() ?: 0f,
                keypoints = keypoints
            )
        }
    }

    override fun close() {
        detector.close()
    }

    companion object {
        private const val MODEL_PATH = "models/blaze_face_short_range.tflite"

        /** 检测置信度阈值，低于该值的人脸被丢弃，用于抑制背景误检。 */
        private const val MIN_DETECTION_CONFIDENCE = 0.6f
    }
}
