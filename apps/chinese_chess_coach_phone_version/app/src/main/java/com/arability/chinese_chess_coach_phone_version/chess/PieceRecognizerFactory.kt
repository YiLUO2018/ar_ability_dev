package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋子识别器工厂。优先加载 TFLite 模型；模型缺失/加载失败时降级为模板匹配。
 * 调用方只依赖 [PieceRecognizer] 接口。
 */
object PieceRecognizerFactory {

    private const val TAG = "PieceRecognizerFactory"

    fun create(context: Context): PieceRecognizer {
        return try {
            TFLitePieceRecognizer.load(context).also {
                Log.i(TAG, "使用 TFLite 棋子识别器")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "TFLite 模型不可用，降级为模板匹配：${t.message}")
            TemplatePieceRecognizer()
        }
    }
}
