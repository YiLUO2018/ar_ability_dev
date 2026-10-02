package com.arability.face_recognition_phone_version.face

import android.graphics.Bitmap

/**
 * 人脸检测器接口。上层只依赖本接口，不关心具体实现。
 * 后续替换为开源权重（TFLite）时，新增实现类并切换工厂即可。
 */
interface FaceDetector {
    fun detect(bitmap: Bitmap): List<FaceBox>
    fun close()
}
