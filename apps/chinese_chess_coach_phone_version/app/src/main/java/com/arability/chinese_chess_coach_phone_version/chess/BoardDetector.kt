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
}
