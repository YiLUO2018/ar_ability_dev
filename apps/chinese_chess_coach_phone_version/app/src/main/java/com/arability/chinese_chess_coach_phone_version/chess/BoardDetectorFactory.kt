package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋盘定位器工厂：EdgeSAM 分割（ONNX，随 APK 内置）。
 *
 * 历史上的角点模型（Keypoint，合成域差）/ Hough（Grid，斜拍失效）兜底链已于
 * M1 后清理（2026-10-11，算法梳理-01 §6 C8）——两者均已被实证为死路。
 * 加载失败直接抛异常，由调用方显式提示用户。
 */
object BoardDetectorFactory {

    private const val TAG = "BoardDetectorFactory"

    /** @param useNnapi bench 模式下启用 NNAPI EP（实验，失败自动回退 CPU）。 */
    fun create(context: Context, debugDir: java.io.File? = null, useNnapi: Boolean = false): BoardDetector {
        return try {
            SegmentationBoardDetector.load(context, debugDir, useNnapi).also {
                Log.i(TAG, "使用 EdgeSAM 分割定位器（nnapi=$useNnapi）")
            }
        } catch (t: Throwable) {
            throw IllegalStateException("EdgeSAM 棋盘定位模型加载失败（内置资产损坏？）：${t.message}", t)
        }
    }
}
