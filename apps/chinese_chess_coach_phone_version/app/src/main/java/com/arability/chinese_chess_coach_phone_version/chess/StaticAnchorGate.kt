package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.hypot

/**
 * D1 静止锚闸（IMU 静止分支，`docs/01-现行设计/IMU静止分支设计方案.md`，2026-10-13 定稿）。
 *
 * 先验：实际对局中棋盘世界位姿恒定，动的只有相机。IMU 判「手机静止」时画面里棋盘
 * 四角不可能真移——相对锚漂移超过 0.2 格即伪成功（手遮挡使 SAM 掩码边界沿手轮廓
 * 走），拒绝且不污染 EMA/跳变闸基线，交上层迟滞保持。
 *
 * **静止态的挪盘兜底 = 双确认超时**（v1-v3 单条件仲裁均已证伪：6 连击一致性→手部
 * 漂移与真挪盘不可分；超时 30/60 裸重锚→vis4 全管线长遮挡段攒满阈值在漂移位重锚，
 * 6 次误仲裁→幽灵帧）。双确认 = 超时询问（连续拒绝 ≥60）+ 几何确认（四角整体平移
 * 每角 >0.5 格）+ 内容确认（上层用候选四角跑 YOLO：新位置棋子质心包含 ≥95% 且
 * ≥10 个——棋盘被挪则棋子随行）。三层全过才清锚重锁；否则继续保持。
 * 快路径仍唯一：IMU 离开静止（[clear]，碰手机秒级恢复）。
 *
 * 锚**固定不跟随**：只在建锚窗口取中位数，之后不随接受帧移动（否则手遮挡期的缓慢
 * 漂移会把锚慢慢带走，锚闸失效）。
 *
 * 纯逻辑、无 Android 依赖，JVM 可单测（`StaticAnchorGateTest`）。
 * 调用时序（与 PC 复刻 `_anchor_check/_anchor_feed` 一一对应）：
 *  1. [check]：detect 内、跳变闸**之前**——false = 拒绝本帧（调用方按漏检处理）；
 *  2. [feed]：跳变闸也通过后才喂锚（防坏帧污染建锚窗口）。
 */
class StaticAnchorGate(
    private val settleCount: Int = SETTLE_COUNT,
    private val settleJitterPx: Float = SETTLE_JITTER_PX,
    private val rejectCellRatio: Float = REJECT_CELL_RATIO,
    private val rollCount: Int = ROLL_COUNT,
) {
    /** 锚四角 TL,TR,BR,BL（8 float，图像像素）；null = 未建锚。 */
    var anchor: FloatArray? = null
        private set

    /** 累计锚拒绝次数（指标/落盘）。 */
    var rejectCount = 0
        private set

    /** 当前连续拒绝计数（挪盘探针触发条件用）。 */
    val rejectStreak: Int get() = rejectBuf.size

    private val settleBuf = ArrayList<FloatArray>()
    private val window = ArrayList<FloatArray>()     // 滚动锚窗口
    private val rejectBuf = ArrayList<FloatArray>()

    /** 离开静止态 / unlock 清屏时调用：锚与全部缓冲清空。 */
    fun clear() {
        anchor = null
        settleBuf.clear()
        window.clear()
        rejectBuf.clear()
    }

    /**
     * 锚闸前半：无锚放行；有锚时任一角位移 > [rejectCellRatio] 格 → 拒绝（计数）。
     * 静止态不自动重锚——唯一兜底是 [pendingTimeout] 双确认流程（见类注释）。
     */
    fun check(corners: FloatArray): Boolean {
        val a = anchor ?: return true
        val cell = cellSizeOf(a)
        if (maxCornerDisp(corners, a) <= rejectCellRatio * cell) {
            rejectBuf.clear()
            return true
        }
        rejectBuf.add(corners.copyOf())
        rejectCount++
        return false
    }

    /**
     * 双确认超时候选（兜底：静止手机下棋盘被人为挪动）：连续拒绝 ≥ [minRejects] 且
     * 最新被拒四角相对锚为**四角整体平移**（每角位移 > 0.5 格——真挪盘四角全动，
     * 手遮挡漂移只变形 1-2 角）。返回候选四角供上层做内容验证（新位置棋子质心
     * 包含率）；不满足返回 null（继续保持）。验证结果经 [resetRejects]/[clear] 回写。
     */
    fun pendingTimeout(minRejects: Int): FloatArray? {
        if (rejectBuf.size < minRejects) return null
        val a = anchor ?: return null
        val cell = cellSizeOf(a)
        val cand = rejectBuf.last()
        var minDisp = Float.MAX_VALUE
        for (i in 0 until 4) {
            val d = hypot(cand[i * 2] - a[i * 2], cand[i * 2 + 1] - a[i * 2 + 1])
            if (d < minDisp) minDisp = d
        }
        return if (minDisp > MOVE_ALL_CELL_RATIO * cell) cand else null
    }

    /** 内容验证失败：清零连续拒绝计数（重新攒 [minRejects] 才会再询问）。 */
    fun resetRejects() {
        rejectBuf.clear()
    }

    /**
     * 锚闸后半。两阶段：
     * 无锚 → 建锚：连续 [settleCount] 个互偏 < [settleJitterPx] → 锚 = 组中位数
     * （窗初始化为这组；贪心计数，中途不一致清零重数）。
     * 有锚 → **滚动维护**：被锚闸接受的检测进窗（cap [rollCount]），锚 = 窗中位数
     * ——跟随 SAM 慢漂（box 提示随棋子走子呼吸 ±20-40px 分钟级，固定锚会与稳态检测
     * 互斥卡死），中位数抗 ≤ 半窗-1 帧突变；**显示取锚**（恒定），慢漂不进显示。
     */
    fun feed(corners: FloatArray) {
        if (anchor == null) {
            if (settleBuf.isNotEmpty()) {
                val med = medianOf(settleBuf)
                if (meanCornerDisp(corners, med) > settleJitterPx) settleBuf.clear()
            }
            settleBuf.add(corners.copyOf())
            if (settleBuf.size >= settleCount) {
                window.clear()
                window.addAll(settleBuf)
                anchor = medianOf(window)
                settleBuf.clear()
            }
            return
        }
        window.add(corners.copyOf())
        if (window.size > rollCount) window.removeAt(0)
        anchor = medianOf(window)
    }

    companion object {
        const val SETTLE_COUNT = 5          // 连续成功且互偏 <5px 才建锚（检测 ~1Hz → ~5s）
        const val SETTLE_JITTER_PX = 5f     // S7 正常帧角点抖动中位 4px
        const val REJECT_CELL_RATIO = 0.2f  // 拒绝阈：任一角 vs 锚 >0.2 格（好帧抖动 p95≤11px
                                           // ＜0.2格；PC 实测静止视频单角 max 125→21px）
        const val MOVE_ALL_CELL_RATIO = 0.5f  // 双确认几何条件：每角位移 >0.5 格 = 整体平移
        const val ROLL_COUNT = 8            // 滚动锚窗口：中位数跟随慢漂、抗 ≤3 帧突变

        /** 画面格宽 = 四角对角线 × canonical 格/对角比（450×500 对角 672.68、格 50）。 */
        fun cellSizeOf(corners: FloatArray): Float {
            val diag = hypot(corners[4] - corners[0], corners[5] - corners[1])  // BR - TL
            val canonDiag = hypot(
                BoardGeometry.BOARD_W.toFloat(),
                BoardGeometry.BOARD_H.toFloat(),
            )
            return (diag * BoardGeometry.CELL / canonDiag).toFloat()
        }

        /** 任一角最大位移（px）。 */
        fun maxCornerDisp(q: FloatArray, ref: FloatArray): Float {
            var m = 0f
            for (i in 0 until 4) {
                val d = hypot(q[i * 2] - ref[i * 2], q[i * 2 + 1] - ref[i * 2 + 1])
                if (d > m) m = d
            }
            return m
        }

        /** 四角平均位移（px）。 */
        fun meanCornerDisp(q: FloatArray, ref: FloatArray): Float {
            var s = 0f
            for (i in 0 until 4) s += hypot(q[i * 2] - ref[i * 2], q[i * 2 + 1] - ref[i * 2 + 1])
            return s / 4f
        }

        /** 逐坐标中位数（建锚窗口 → 锚四角）。 */
        fun medianOf(buf: List<FloatArray>): FloatArray {
            val out = FloatArray(8)
            for (k in 0 until 8) {
                out[k] = buf.map { it[k] }.sorted()[buf.size / 2]
            }
            return out
        }
    }
}
