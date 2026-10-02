package com.arability.face_recognition_phone_version.face

import android.content.Context

/**
 * 人脸引擎工厂。切换模型实现时只改这里，调用方无需改动。
 */
object FaceEngine {
    fun createDetector(context: Context): FaceDetector = MediaPipeFaceDetector(context)

    fun createEmbedder(context: Context): FaceEmbedder = TfliteFaceEmbedder(context)

    fun createDatabase(context: Context): FaceDatabase = FaceDatabase(context)
}
