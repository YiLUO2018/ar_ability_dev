package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋盘定位器工厂。优先加载角点模型（[KeypointBoardDetector]）；模型缺失/加载失败时
 * 降级为纯 Kotlin 的 Hough 定位（[GridBoardDetector]）。
 * 调用方只依赖 [BoardDetector] 接口。
 */
object BoardDetectorFactory {

    private const val TAG = "BoardDetectorFactory"

    fun create(context: Context, debugDir: java.io.File? = null): BoardDetector {
        return try {
            KeypointBoardDetector.load(context).also {
                Log.i(TAG, "使用 TFLite 角点模型定位器")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "TFLite 角点模型不可用，降级为 Hough 定位：${t.message}")
            GridBoardDetector(debugDir = debugDir)
        }
    }
}
