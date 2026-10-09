package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap

/**
 * 棋盘定位器接口。上层只依赖本接口，不关心具体实现。
 * 后续可替换为 OpenCV 实现，改工厂即可。
 */
interface BoardDetector {
    /**
     * 在图像中检测棋盘，返回棋盘四角与透视校正后的规整棋盘图。
     *
     * @return 检测结果；未检测到棋盘时返回 null。
     */
    fun detect(bitmap: Bitmap): BoardInfo?

    /**
     * 用外部几何源（如棋子质心反馈 [QuadRefiner]）精修后的四角重建检测结果。
     * 默认不支持（返回 null），调用方沿用原 [detect] 结果。
     */
    fun refineQuad(bitmap: Bitmap, refinedCorners: FloatArray): BoardInfo? = null

    /**
     * 释放定位器占用的资源。默认空实现，供需要清理的实现（如 TFLite 解释器）覆盖。
     */
    fun close() {}
}
