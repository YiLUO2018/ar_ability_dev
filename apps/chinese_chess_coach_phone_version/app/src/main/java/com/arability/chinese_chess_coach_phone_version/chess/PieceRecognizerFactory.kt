package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋子识别器工厂。优先 YOLO 逐子检测（ONNX），其次 TFLite 整盘分类，最后降级模板匹配。
 * 调用方只依赖 [PieceRecognizer] 接口。
 */
object PieceRecognizerFactory {

    private const val TAG = "PieceRecognizerFactory"

    fun create(context: Context): PieceRecognizer {
        return try {
            YoloPieceDetector.load(context).also {
                Log.i(TAG, "使用 YOLO 逐子棋子检测器（ONNX）")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "YOLO 模型不可用，尝试 TFLite 整盘分类：${t.message}")
            try {
                TFLitePieceRecognizer.load(context).also {
                    Log.i(TAG, "使用 TFLite 整盘棋子识别器")
                }
            } catch (t2: Throwable) {
                Log.w(TAG, "TFLite 模型不可用，降级为模板匹配：${t2.message}")
                TemplatePieceRecognizer()
            }
        }
    }
}
