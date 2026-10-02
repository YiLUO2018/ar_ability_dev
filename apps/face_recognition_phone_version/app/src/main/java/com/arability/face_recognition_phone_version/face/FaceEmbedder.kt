package com.arability.face_recognition_phone_version.face

import android.graphics.Bitmap

/**
 * 人脸特征提取器接口。输入一张裁剪后的人脸图，输出归一化特征向量。
 * 后续替换为开源权重时，新增实现类并切换工厂即可。
 */
interface FaceEmbedder {
    /**
     * @param faceBitmap 裁剪后的人脸图（调用方负责裁剪）
     * @return 已 L2 归一化的特征向量
     */
    fun embed(faceBitmap: Bitmap): FloatArray

    fun close()
}
