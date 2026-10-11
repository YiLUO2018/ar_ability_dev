package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap

/**
 * 棋盘定位器接口。上层只依赖本接口，不关心具体实现。
 * 后续可替换为 OpenCV 实现，改工厂即可。
 */
interface BoardDetector {
    /**
     * 在图像中检测棋盘，返回棋盘四角与图像→canonical 单应。
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
     * 传入上一帧检出的棋子质心（图像像素 [x,y]），作为无历史（首帧/重锁）时
     * 候选甄别的先验（C7①：包含全部质心的最小四角 = 真棋盘）。默认空实现。
     */
    fun setReferenceCentroids(centroids: List<FloatArray>) {}

    /**
     * bench 模式（C3，实验）：对同一帧反复测 encoder/decoder 推理耗时，输出 p50/p95。
     * 结果打日志并写 debugDir/bench_<tag>.txt。默认空实现。
     */
    fun runBenchmark(bitmap: Bitmap, iters: Int, tag: String) {}

    /**
     * 清空时间跟踪状态（prevCorners/EMA/跳变闸基线）。上层在「锁定丢失、清空 UI」时
     * 调用，使下一次成功检测走「无历史」路径（C7① 质心先验仲裁 + EMA 重新起步），
     * 避免长时间失败后用陈旧基线拖影/误拒。默认空实现。
     */
    fun resetTrackingState() {}

    /**
     * 释放定位器占用的资源。默认空实现，供需要清理的实现（如 TFLite 解释器）覆盖。
     */
    fun close() {}
}
