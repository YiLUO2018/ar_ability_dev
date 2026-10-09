package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋盘定位器工厂。优先级：EdgeSAM 分割（[SegmentationBoardDetector]）→ 角点模型
 * （[KeypointBoardDetector]）→ 纯 Kotlin Hough（[GridBoardDetector]）。
 * 上一层加载失败时降级到下一层。调用方只依赖 [BoardDetector] 接口。
 */
object BoardDetectorFactory {

    private const val TAG = "BoardDetectorFactory"

    fun create(context: Context, debugDir: java.io.File? = null): BoardDetector {
        try {
            return SegmentationBoardDetector.load(context, debugDir).also {
                Log.i(TAG, "使用 EdgeSAM 分割定位器")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "EdgeSAM 分割不可用，尝试角点模型：${t.message}")
        }
        try {
            return KeypointBoardDetector.load(context, debugDir).also {
                Log.i(TAG, "使用 TFLite 角点模型定位器")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "TFLite 角点模型不可用，降级为 Hough 定位：${t.message}")
            return GridBoardDetector(debugDir = debugDir)
        }
    }
}
