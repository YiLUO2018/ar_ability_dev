package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.util.Log

/**
 * 棋子识别器工厂：YOLO 逐子检测（ONNX，随 APK 内置）。
 *
 * 历史上的 TFLite 整盘分类 / 模板匹配兜底链已于 M1 后清理（2026-10-11，算法梳理-01 §6 C8）：
 * 兜底仅在「内置 assets 加载失败 = 构建损坏」时触发，且其真机质量不可用——
 * 静默降级到不可用识别器比显式报错更糟。加载失败直接抛异常，由调用方显式提示用户。
 */
object PieceRecognizerFactory {

    private const val TAG = "PieceRecognizerFactory"

    fun create(context: Context): PieceRecognizer {
        return try {
            YoloPieceDetector.load(context).also {
                Log.i(TAG, "使用 YOLO 逐子棋子检测器（ONNX）")
            }
        } catch (t: Throwable) {
            throw IllegalStateException("YOLO 棋子识别模型加载失败（内置资产损坏？）：${t.message}", t)
        }
    }
}
