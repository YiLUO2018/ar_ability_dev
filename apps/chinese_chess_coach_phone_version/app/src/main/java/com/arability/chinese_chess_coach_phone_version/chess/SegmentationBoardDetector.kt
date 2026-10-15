package com.arability.chinese_chess_coach_phone_version.chess

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.round
import kotlin.math.sqrt

/** 掩码坐标系中的二维点。 */
internal class Pt(val x: Float, val y: Float)

/**
 * 基于 EdgeSAM（ONNX Runtime）的分割定位器：整图 → 图像编码 → 中心点提示 → 掩码 → 拟合四角 → 透视校正。
 *
 * EdgeSAM 官方只提供 ONNX（无 TFLite），本实现用 onnxruntime-android 跑 encoder + decoder 两个 ONNX：
 *  - encoder：输入 `image` [1,3,1024,1024] float32（NCHW，归一化 + 右下补零），输出 `image_embeddings` [1,256,64,64]。
 *  - decoder：输入 `image_embeddings` + `point_coords` [1,N,2] + `point_labels` [1,N]（0=负/1=正），
 *    输出 `scores` [1,C] 与 `masks` [1,C,256,256]（logits，阈值 0）。
 *
 * 预处理与 EdgeSAM 官方 `Sam.preprocess` 对齐：resize 长边→1024（保比例）→ 每通道 (v-mean)/std → 右下补零。
 * 后处理：取面积合法的候选掩码 → 二值化 → 最大连通域 → 凸包 →（外推复原或贪心减点）四顶点 →
 * 排序 TL,TR,BR,BL → 产出图像→canonical 单应（物理边界 [BoardGeometry.physicalCorners]）。
 */
class SegmentationBoardDetector(
    private val env: OrtEnvironment,
    private val encoder: OrtSession,
    private val decoder: OrtSession,
    private var promptX: Float,
    private var promptY: Float,
    private val debugDir: java.io.File? = null
) : BoardDetector {

    private val encoderInput: FloatBuffer =
        ByteBuffer.allocateDirect(3 * IMG_SIZE * IMG_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    private var logCounter = 0
    private var failCounter = 0
    private var maskShapeLogged = false
    private var prevCorners: FloatArray? = null
    private var jumpCandidate: FloatArray? = null
    private var jumpStreak = 0
    private var refineLogCounter = 0

    /** 连续 detect 失败次数（成功清零）。≥阈值时触发历史失效与中心提示救援。 */
    private var failStreak = 0

    /** 上一帧 YOLO 棋子质心（图像像素），无历史时候选甄别先验（C7①）。 */
    private var refCentroids: List<FloatArray> = emptyList()

    /** D1 静止锚闸状态（IMU 静止分支）；[imuStatic]/[imuMoving] 由上层 monitor 三态驱动。 */
    private var imuStatic = false

    /** MOVING 态：救援启用二级（无参考选择，移动重锁）；UNKNOWN 退回 S7 单级行为。 */
    private var imuMoving = false

    private val anchorGate = StaticAnchorGate()

    override fun detect(bitmap: Bitmap): BoardInfo? {
        val w = bitmap.width
        val h = bitmap.height
        if (w == 0 || h == 0) return null

        // 历史失效保护（移动重锁修复 A 配套，2026-10-11 移动实测教训）：连续失败 ≥2 次
        // 说明棋盘/相机已大幅移动，prevCorners 与跳变闸基线已陈旧——清空后走「无历史」
        // 路径（C7① 质心先验 + EMA 重新起步）。否则重锁结果会被跳变闸用陈旧基线误拒
        // （跳变闸要连续 3 帧一致才放行，移动中每帧都「跳」，永远凑不满）。
        if (failStreak >= HISTORY_INVALIDATE_AFTER_FAILS && prevCorners != null) {
            prevCorners = null
            jumpCandidate = null
            jumpStreak = 0
        }

        // 1. resize 长边到 IMG_SIZE（保比例），与 ResizeLongestSide 一致。
        val scale = IMG_SIZE.toFloat() / max(w, h)
        val newW = max(1, round(w * scale).toInt())
        val newH = max(1, round(h * scale).toInt())
        val resized = if (newW == w && newH == h) bitmap
            else Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        // 2. 归一化 + 右下补零 → NCHW 输入。
        fillEncoderInput(resized, newW, newH)

        // 3. encoder → image_embeddings。
        val (embeddings, embShape) = runEncoder()

        // 4. 中心点提示（归一化坐标 × resize 后尺寸 → 1024 坐标系）。
        val px = promptX * newW
        val py = promptY * newH

        // 5. decoder → scores + masks。S7 box+点混合提示（labels [2,3,1]）：有参考质心
        //    （≥MIN_REF_CENTROIDS）时用参考 bbox 外扩 5% 对角线当 box——暗对暗物理边
        //    场景点提示掩码会「改跟」内部高对比线（video_01 TR 内凹 40-150px 根因），
        //    box 直接约束物体范围；box 来源必须是物理参考（prev 四角会自我强化下陷，
        //    S4 实测 TR 粘滞游程 33 vs legacy 24）。无参考退回纯点提示。
        val box = boxHint(w, h)
        var out = if (box != null) {
            runDecoderBox(embeddings, embShape, box, floatArrayOf(px, py), w, h) ?: return null
        } else {
            runDecoder(embeddings, embShape, floatArrayOf(px, py), floatArrayOf(1f)) ?: return null
        }
        var chosen = selectCandidate(out, newW, newH, w, h, bitmap, useRef = true)

        // 5b. 帧内中心提示救援（移动重锁修复 A）：连续失败说明回灌提示点已陈旧——
        //     棋盘移走后提示点落在背景上，decoder 输出碎片噪声掩码（实测 fg 7.9% 但
        //     最大连通域 <52px）。decoder 仅 ~240ms，候选全拒且当前非中心提示时，
        //     用画面中心重跑一次 decoder 即可单帧自愈（PC 复刻已验证中心提示可检出）。
        //     分级救援（2026-10-12 修订 + 10-13 挪盘探针）：①带参考选（遮挡期参考仍
        //     有效，包含门挡残缺候选）；②无参考重选仅在两种情形——IMU=MOVING（板已
        //     移走、参考陈旧，移动重锁）或**静止挪盘探针**（锚拒绝攒满阈值：棋盘真被
        //     挪时中心提示掩码落在新位置、旧参考包不住，只有绕参考才能拿到新位置候选
        //     ——它会被锚闸拒绝进入超时候选，经双确认裁决；手遮挡期的探针垃圾候选过
        //     不了整体平移几何确认，无害）。
        if (chosen == null && failStreak >= PROMPT_RESCUE_AFTER_FAILS &&
            (promptX != 0.5f || promptY != 0.5f)
        ) {
            out = runDecoder(embeddings, embShape, floatArrayOf(newW / 2f, newH / 2f), floatArrayOf(1f)) ?: return null
            chosen = selectCandidate(out, newW, newH, w, h, null, useRef = true)
            val probe = imuStatic && anchorGate.rejectStreak >= REANCHOR_ASK_REJECTS
            if (chosen == null && (imuMoving || probe)) {
                chosen = selectCandidate(out, newW, newH, w, h, null, useRef = false)
            }
            if (chosen != null) {
                Log.i(TAG, "center-prompt rescue ok after $failStreak fails")
                // 救援成功 = 移动证据：历史与跳变闸基线已陈旧，清空走无历史路径。
                prevCorners = null
                jumpCandidate = null
                jumpStreak = 0
            }
        }

        // 所有候选都被拒 → 漏检，交给 MainActivity 的锁定迟滞保持上一帧。
        if (chosen == null) {
            failStreak++
            if (failCounter++ % 5 == 0) {
                Log.i(TAG, "detect fail streak=$failStreak")
            }
            return null
        }
        failStreak = 0

        val rawCorners = chosen.corners

        // 8a''. D1 静止锚闸（前半，跳变闸之前）：IMU 判静止 ⇒ 手机没动 ⇒ 画面里棋盘
        //       四角不可能真移——相对锚任一角漂移 >0.4 格 = 伪成功（手遮挡使掩码边界
        //       沿手轮廓走，vis2b f189/f321/f474 根因），拒绝且不污染 EMA/跳变闸基线，
        //       交上层迟滞保持显示。连续一致拒绝 → 棋盘移动仲裁（见 StaticAnchorGate）。
        if (imuStatic && !anchorGate.check(rawCorners)) {
            failStreak++
            if (failCounter++ % 5 == 0) {
                Log.i(TAG, "detect reject: static anchor gate (total=${anchorGate.rejectCount})")
            }
            return null
        }

        // 8b'. 时间一致性闸：已有锁定四角时，单帧「跳变四角」（面积/角点突偏）通常是掩码
        //      偶发溢出到桌面/阴影的坏帧——直接拒绝，交给锁定迟滞保持上一帧，避免 overlay
        //      忽大忽小再收敛。连续多帧一致的新四角 = 真实移动，放行重锁。
        if (prevCorners != null && isQuadJump(rawCorners, prevCorners!!)) {
            val consistent = jumpCandidate != null && !isQuadJump(rawCorners, jumpCandidate!!)
            jumpStreak = if (consistent) jumpStreak + 1 else 1
            jumpCandidate = rawCorners.copyOf()
            if (jumpStreak < JUMP_ACCEPT_STREAK) {
                if (failCounter++ % 5 == 0) {
                    Log.i(TAG, "detect reject: quad jump (streak=$jumpStreak) corners=[${rawCorners.joinToString { it.toInt().toString() }}]")
                }
                return null
            }
            jumpStreak = 0
            jumpCandidate = null
            Log.i(TAG, "quad jump accepted after streak (real movement)")
        } else {
            jumpStreak = 0
            jumpCandidate = null
        }

        // 8b''. D1 静止锚闸（后半）：跳变闸也通过的干净四角才喂锚（防坏帧污染建锚窗口）。
        if (imuStatic) anchorGate.feed(rawCorners)

        // 8c. 提示点回灌（归一化 [0,1]）。复原态四角质心可能在画面外，改用可见掩码质心。
        if (chosen.reconstructed && chosen.centroid != null) {
            val cx = chosen.centroid.x * (IMG_SIZE.toFloat() / out.maskW) * (w.toFloat() / newW)
            val cy = chosen.centroid.y * (IMG_SIZE.toFloat() / out.maskH) * (h.toFloat() / newH)
            promptX = (cx / w).coerceIn(0.02f, 0.98f)
            promptY = (cy / h).coerceIn(0.02f, 0.98f)
        } else {
            val centroidX = (rawCorners[0] + rawCorners[2] + rawCorners[4] + rawCorners[6]) / 4f
            val centroidY = (rawCorners[1] + rawCorners[3] + rawCorners[5] + rawCorners[7]) / 4f
            promptX = (centroidX / w).coerceIn(0.02f, 0.98f)
            promptY = (centroidY / h).coerceIn(0.02f, 0.98f)
        }

        // 8d. 四角自适应 EMA 平滑。静止锚显示模式（静止场景四角应完全一致标准）：
        //     显示=锚（恒定），检测只是锚的维护者（滚动中位数）——检测自身的分钟级
        //     慢漂（box 提示随棋子走子呼吸 ±20-40px）不进显示；prev 同步为锚。
        val corners = if (imuStatic && anchorGate.anchor != null) {
            val out = anchorGate.anchor!!.copyOf()
            prevCorners = out.copyOf()
            out
        } else {
            smoothCorners(rawCorners, w, h)
        }

        if (logCounter++ % 20 == 0) {
            Log.i(
                TAG,
                "corners=[${corners[0].toInt()},${corners[1].toInt()}] " +
                    "[${corners[2].toInt()},${corners[3].toInt()}] " +
                    "[${corners[4].toInt()},${corners[5].toInt()}] " +
                    "[${corners[6].toInt()},${corners[7].toInt()}] " +
                    "mask=${chosen.maskIdx} recon=${chosen.reconstructed} area=${chosen.area}"
            )
            saveDebugFrame(bitmap, logCounter)
        }

        // 9. 物理棋盘边 → canonical 单应（C1/C8：不再生成 warp 位图，主路径 YOLO 用原图+单应）。
        return buildInfo(corners)
    }

    /**
     * 候选级校验与选择（P3-a + C4/C7①，自 detect 提取以支持帧内救援重跑）：
     * 面积带过滤 → 逐候选提四角过校验（isValidQuad/形状闸/贴边/复原域）→
     * 先「全可见」后「外推复原」两轮 → 同优先级时间稳定性仲裁（无历史时质心先验）。
     * 溢出/碎块掩码四角逐帧乱跳，即使面积更大也会输给稳定的真棋盘掩码。
     *
     * @param bitmap 非空且无候选通过时做节流日志与失败诊断落盘（救援重跑传 null 免重复）。
     */
    private fun selectCandidate(
        out: DecoderOutput, newW: Int, newH: Int, w: Int, h: Int, bitmap: Bitmap?,
        useRef: Boolean = true,
    ): QuadChoice? {
        val maskSize = out.maskW * out.maskH
        val minArea = (maskSize * MIN_AREA_RATIO).toInt()
        val maxArea = (maskSize * MAX_AREA_RATIO).toInt()
        // 图像内容在 256 掩码画布中的边界（非方形输入时右/下是 letterbox 零填充）。
        val contentW = newW / 4f
        val contentH = newH / 4f
        val areas = IntArray(out.numMasks)
        val candIdx = ArrayList<Int>()
        val areaBuf = ByteArray(maskSize)
        for (i in 0 until out.numMasks) {
            val off = i * maskSize
            for (j in 0 until maskSize) areaBuf[j] = if (out.masks[off + j] > 0f) 1 else 0
            val a = largestComponentArea(areaBuf, out.maskW, out.maskH)
            areas[i] = a
            if (a >= minArea && a <= maxArea) candIdx.add(i)
        }

        if (!maskShapeLogged) {
            maskShapeLogged = true
            Log.i(
                TAG,
                "mask shape: numMasks=${out.numMasks} ${out.maskW}x${out.maskH} " +
                    "scores=${out.scores.joinToString { "%.3f".format(it) }} " +
                    "areas=${areas.joinToString()}"
            )
        }

        val sortedCands = candIdx.sortedByDescending { areas[it] }   // 面积降序 = 无历史时的优先序
        var chosen: QuadChoice? = null
        for (passRecon in booleanArrayOf(false, true)) {
            val acceptables = ArrayList<QuadChoice>()
            for (i in sortedCands) {
                val off = i * maskSize
                for (j in 0 until maskSize) areaBuf[j] = if (out.masks[off + j] > 0f) 1 else 0
                val mq = extractCornersEx(areaBuf, out.maskW, out.maskH, contentW, contentH)
                    ?: continue
                // 掩码坐标 → 原图坐标。掩码对应 1024 输入帧，真实图像占其左上 [newW,newH]。
                val raw = FloatArray(8)
                for (k in 0 until 4) {
                    val fx = mq.corners[k * 2] * (IMG_SIZE.toFloat() / out.maskW)
                    val fy = mq.corners[k * 2 + 1] * (IMG_SIZE.toFloat() / out.maskH)
                    raw[k * 2] = fx * (w.toFloat() / newW)
                    raw[k * 2 + 1] = fy * (h.toFloat() / newH)
                }
                // 8a. 非退化校验：掩码噪声/塌缩会产出两点重合或面积趋零的「四边形」。
                if (!isValidQuad(raw, w, h)) continue
                // 8a'. 形状闸（C4）：内角/对边比在宽松界内，杀掉溢出细条/尖菱形等「合法但非棋盘」四边形。
                if (!hasBoardShape(raw)) continue
                // 8a''. S7 包含门（常态过滤，非仅无历史仲裁）：残缺/内收候选包不住参考
                //       质心（<0.95）直接拒——手遮挡掩码沿手碎裂的候选在此被拦；0.85 会
                //       放过 TR 60-126px 中带候选（实测）。外扩型候选仍包住全部点，
                //       由 D1 静止锚闸拦。useRef=false 仅救援移动态（参考已陈旧）。
                if (useRef && refCentroids.size >= MIN_REF_CENTROIDS &&
                    containmentRatio(raw, refCentroids) < CENTROID_CONTAIN_MIN
                ) continue
                if (passRecon != mq.reconstructed) continue
                if (!mq.reconstructed && isCutOffQuad(raw, w, h)) continue
                if (mq.reconstructed && !withinExpandedBounds(raw, w, h)) continue
                acceptables.add(QuadChoice(raw, mq.reconstructed, mq.centroid, i, areas[i]))
            }
            if (acceptables.isNotEmpty()) {
                val prev = prevCorners
                // 无历史（首帧/重锁）：先试质心先验仲裁（C7①），无质心/无候选达标时回退最大面积。
                chosen = if (prev == null) pickByCentroids(acceptables) ?: acceptables[0]
                else acceptables.minByOrNull { meanCornerDisp(it.corners, prev) }
                break
            }
        }

        if (chosen == null && bitmap != null) {
            if (failCounter++ % 5 == 0) {
                Log.i(
                    TAG,
                    "detect fail: no valid candidate (min=$minArea max=$maxArea) " +
                        "areas=${areas.joinToString()} scores=${out.scores.joinToString { "%.3f".format(it) }}"
                )
            }
            if (failCounter % 10 == 0) {
                saveFailDiag(bitmap, out, sortedCands, failCounter)
            }
        }
        return chosen
    }

    /** 锁定丢失时清空时间跟踪状态（上层 unlock 清 UI 时调用）：下次成功走无历史路径。 */
    override fun resetTrackingState() {
        prevCorners = null
        jumpCandidate = null
        jumpStreak = 0
        anchorGate.clear()
    }

    /** D1/D2/救援分级的 IMU 三态输入（上层 StaticMotionMonitor 驱动）。离开静止即清锚。 */
    override fun onImuStateChanged(state: ImuMotion) {
        val isStatic = state == ImuMotion.STATIC
        if (imuStatic == isStatic && imuMoving == (state == ImuMotion.MOVING)) return
        if (imuStatic && !isStatic) anchorGate.clear()
        imuStatic = isStatic
        imuMoving = state == ImuMotion.MOVING
        Log.i(TAG, "imu state -> $state")
    }

    /** D1 挪盘兜底①：双确认超时候选（连续拒绝 ≥60 且四角整体平移 >0.5 格）。 */
    override fun reanchorCandidate(): FloatArray? =
        if (imuStatic) anchorGate.pendingTimeout(REANCHOR_ASK_REJECTS) else null

    /** D1 挪盘兜底②：上层内容验证（新位置棋子质心包含 ≥95% 且 ≥10 个）结果回写。 */
    override fun resolveReanchor(accept: Boolean) {
        if (accept) {
            Log.i(TAG, "reanchor accepted (dual-verified board move), total rejects=${anchorGate.rejectCount}")
            anchorGate.clear()
            prevCorners = null       // 旧位置历史/参考全部作废，走无历史重锁
            jumpCandidate = null
            jumpStreak = 0
            refCentroids = emptyList()
        } else {
            anchorGate.resetRejects()
        }
    }

    /** 由（已平滑的）物理四角构建 [BoardInfo]：仅计算图像→canonical 单应。 */
    private fun buildInfo(corners: FloatArray): BoardInfo {
        return BoardInfo(
            corners = corners,
            cellSize = BoardGeometry.CELL,
            originX = BoardGeometry.MARGIN,
            originY = BoardGeometry.MARGIN,
            imageToGrid = Homography.compute(corners, BoardGeometry.physicalCorners)
        )
    }

    /**
     * 棋子质心反馈的四角精修入口（P0-b）：把 [QuadRefiner] 算出的精修四角经校验后
     * 融入 EMA 状态并重建 [BoardInfo]。校验失败返回 null（调用方沿用原 info）。
     */
    override fun refineQuad(bitmap: Bitmap, refinedCorners: FloatArray): BoardInfo? {
        // 静止锚显示模式下显示=锚（恒定），精修结果不再进显示（检测只维护锚）。
        if (imuStatic && anchorGate.anchor != null) return null
        val w = bitmap.width
        val h = bitmap.height
        if (refinedCorners.size != 8 || !isValidQuad(refinedCorners, w, h)) return null
        if (isCutOffQuad(refinedCorners, w, h)) return null
        val prev = prevCorners
        if (prev != null && isQuadJump(refinedCorners, prev)) return null  // 精修不许跳变
        val smoothed = smoothCorners(refinedCorners, w, h)
        if (refineLogCounter++ % 10 == 0) {
            Log.i(TAG, "quad refined: [${smoothed.joinToString { it.toInt().toString() }}]")
        }
        return buildInfo(smoothed)
    }

    // ------------------------------------------------------------- 推理

    private fun fillEncoderInput(bitmap: Bitmap, w: Int, h: Int) {
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val plane = IMG_SIZE * IMG_SIZE
        // NCHW：按通道分平面写，补零区域保持 0。
        for (y in 0 until h) {
            val srcRow = y * w
            val dstRow = y * IMG_SIZE
            for (x in 0 until w) {
                val p = pixels[srcRow + x]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                val di = dstRow + x
                encoderInput.put(di, (r - PIXEL_MEAN[0]) / PIXEL_STD[0])
                encoderInput.put(plane + di, (g - PIXEL_MEAN[1]) / PIXEL_STD[1])
                encoderInput.put(2 * plane + di, (b - PIXEL_MEAN[2]) / PIXEL_STD[2])
            }
        }
    }

    private fun runEncoder(): Pair<FloatArray, LongArray> {
        encoderInput.rewind()
        val inputTensor = OnnxTensor.createTensor(
            env, encoderInput, longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        )
        try {
            val result = encoder.run(mapOf("image" to inputTensor))
            try {
                val out = result.get("image_embeddings").get() as OnnxTensor
                val shape = out.info.shape
                val fb = out.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return Pair(arr, shape)
            } finally {
                result.close()
            }
        } finally {
            inputTensor.close()
        }
    }

    /**
     * box 提示来源（优先级，2026-10-13 定稿）：①**静止锚 bbox**——锚是经建锚验证的
     * 固定物理位置，掩码被钉在锚附近 → 检测方差坍缩（质心 bbox 会随棋子移动/遮挡期
     * 崩塌 <10 个而漂移/消失，点提示掩码游走可达 195px，锚闸被逼成永久拒绝）。
     * 冻结锚非 S4 自强化：S4 的坑是 box 逐帧追掩码输出（prev/EMA），锚不追。
     * ②参考质心 bbox（移动态）。③null 退回纯点提示。均外扩 [BOX_EXPAND]×对角线。
     */
    private fun boxHint(w: Int, h: Int): FloatArray? {
        val anchorCorners = anchorGate.anchor
        val src: List<FloatArray> = when {
            imuStatic && anchorCorners != null ->
                List(4) { floatArrayOf(anchorCorners[it * 2], anchorCorners[it * 2 + 1]) }
            refCentroids.size >= MIN_REF_CENTROIDS -> refCentroids
            else -> return null
        }
        var x0 = Float.MAX_VALUE
        var y0 = Float.MAX_VALUE
        var x1 = -Float.MAX_VALUE
        var y1 = -Float.MAX_VALUE
        for (c in src) {
            if (c[0] < x0) x0 = c[0]
            if (c[0] > x1) x1 = c[0]
            if (c[1] < y0) y0 = c[1]
            if (c[1] > y1) y1 = c[1]
        }
        val expand = BOX_EXPAND * hypot(x1 - x0, y1 - y0)
        return floatArrayOf(x0 - expand, y0 - expand, x1 + expand, y1 + expand)
    }

    /** box+点混合提示（labels [2,3,1]）：box 约束物体范围，正点锚定 box 中心物体。
     *  box 为原图坐标，乘 1024/长边 进 1024 系（与点提示同系；padding 在右下无偏移）。 */
    private fun runDecoderBox(
        embeddings: FloatArray,
        embShape: LongArray,
        box: FloatArray,
        point: FloatArray,
        w: Int,
        h: Int,
    ): DecoderOutput? {
        val s = IMG_SIZE.toFloat() / max(w, h)
        val coords = floatArrayOf(
            box[0] * s, box[1] * s, box[2] * s, box[3] * s, point[0], point[1]
        )
        return runDecoder(embeddings, embShape, coords, floatArrayOf(2f, 3f, 1f))
    }

    private fun runDecoder(
        embeddings: FloatArray,
        embShape: LongArray,
        coords: FloatArray,
        labels: FloatArray
    ): DecoderOutput? {
        val embTensor = OnnxTensor.createTensor(env, directBufferOf(embeddings), embShape)
        val n = labels.size
        val coordsTensor = OnnxTensor.createTensor(env, directBufferOf(coords), longArrayOf(1, n.toLong(), 2))
        val labelsTensor = OnnxTensor.createTensor(env, directBufferOf(labels), longArrayOf(1, n.toLong()))
        try {
            val result = decoder.run(
                mapOf(
                    "image_embeddings" to embTensor,
                    "point_coords" to coordsTensor,
                    "point_labels" to labelsTensor
                )
            )
            try {
                val scoresTensor = result.get("scores").get() as OnnxTensor
                val masksTensor = result.get("masks").get() as OnnxTensor
                val scoresShape = scoresTensor.info.shape
                val masksShape = masksTensor.info.shape
                val numMasks = scoresShape[1].toInt()
                val maskH = masksShape[2].toInt()
                val maskW = masksShape[3].toInt()

                val sfb = scoresTensor.floatBuffer
                val scores = FloatArray(sfb.remaining())
                sfb.get(scores)
                val mfb = masksTensor.floatBuffer
                val masks = FloatArray(mfb.remaining())
                mfb.get(masks)
                return DecoderOutput(scores, masks, numMasks, maskW, maskH)
            } finally {
                result.close()
            }
        } finally {
            embTensor.close()
            coordsTensor.close()
            labelsTensor.close()
        }
    }

    private fun directBufferOf(arr: FloatArray): FloatBuffer {
        val bb = ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(arr)
        fb.rewind()
        return fb
    }

    private class DecoderOutput(
        val scores: FloatArray,
        val masks: FloatArray,
        val numMasks: Int,
        val maskW: Int,
        val maskH: Int
    )

    // ------------------------------------------------------------- 掩码后处理

    /** 计算二值掩码中最大连通域（4 邻接）的面积。 */
    private fun largestComponentArea(binary: ByteArray, w: Int, h: Int): Int {
        val visited = BooleanArray(w * h)
        val stack = IntArray(w * h)
        var best = 0
        for (start in 0 until w * h) {
            if (binary[start] == 0.toByte() || visited[start]) continue
            var sp = 0
            stack[sp++] = start
            visited[start] = true
            var area = 0
            while (sp > 0) {
                val idx = stack[--sp]
                area++
                val x = idx % w
                val y = idx / w
                if (x > 0 && binary[idx - 1] == 1.toByte() && !visited[idx - 1]) {
                    visited[idx - 1] = true; stack[sp++] = idx - 1
                }
                if (x < w - 1 && binary[idx + 1] == 1.toByte() && !visited[idx + 1]) {
                    visited[idx + 1] = true; stack[sp++] = idx + 1
                }
                if (y > 0 && binary[idx - w] == 1.toByte() && !visited[idx - w]) {
                    visited[idx - w] = true; stack[sp++] = idx - w
                }
                if (y < h - 1 && binary[idx + w] == 1.toByte() && !visited[idx + w]) {
                    visited[idx + w] = true; stack[sp++] = idx + w
                }
            }
            if (area > best) best = area
        }
        return best
    }

    /** 候选选择结果：四角（原图坐标）、是否外推复原、可见掩码质心（掩码坐标）、来源掩码序号、掩码面积。 */
    private class QuadChoice(
        val corners: FloatArray,
        val reconstructed: Boolean,
        val centroid: Pt?,
        val maskIdx: Int,
        val area: Int
    )

    /** 两四角平均角点位移（时间稳定性仲裁用）。 */
    private fun meanCornerDisp(q: FloatArray, ref: FloatArray): Float {
        var s = 0f
        for (i in 0 until 4) {
            val dx = q[i * 2] - ref[i * 2]
            val dy = q[i * 2 + 1] - ref[i * 2 + 1]
            s += hypot(dx, dy)
        }
        return s / 4f
    }

    /**
     * 无历史时的质心先验仲裁（首帧/重锁，C7①）：在包含率 ≥[CENTROID_CONTAIN_MIN] 的候选中
     * 取**面积最小**者——真棋盘是包住全部棋子的最紧四角；溢出候选（棋盘+桌面）同样包住但面积
     * 更大，碎块候选包不住。无候选达标返回 null（调用方回退最大面积优先）。
     */
    private fun pickByCentroids(cands: List<QuadChoice>): QuadChoice? {
        if (refCentroids.size < MIN_REF_CENTROIDS) return null
        var best: QuadChoice? = null
        for (c in cands) {
            if (containmentRatio(c.corners, refCentroids) < CENTROID_CONTAIN_MIN) continue
            if (best == null || quadAreaOf(c.corners) < quadAreaOf(best!!.corners)) best = c
        }
        return best
    }

    override fun setReferenceCentroids(centroids: List<FloatArray>) {
        refCentroids = centroids
    }

    /**
     * bench 模式（C3）：对同一帧反复测 encoder/decoder 耗时，p50/p95 打日志（tag=BenchEP）
     * 并写 debugDir/bench_<tag>.txt。adb 触发：
     * `am start -n .../.MainActivity --ez bench true --es ep cpu|nnapi`
     */
    override fun runBenchmark(bitmap: Bitmap, iters: Int, tag: String) {
        try {
            val w = bitmap.width
            val h = bitmap.height
            val scale = IMG_SIZE.toFloat() / max(w, h)
            val newW = max(1, round(w * scale).toInt())
            val newH = max(1, round(h * scale).toInt())
            val resized = if (newW == w && newH == h) bitmap
                else Bitmap.createScaledBitmap(bitmap, newW, newH, true)
            fillEncoderInput(resized, newW, newH)

            val encTimes = LongArray(iters)
            var embeddings = FloatArray(0)
            var embShape = longArrayOf()
            for (i in 0 until iters) {
                val t0 = android.os.SystemClock.elapsedRealtimeNanos()
                val r = runEncoder()
                encTimes[i] = (android.os.SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
                embeddings = r.first
                embShape = r.second
            }
            val decTimes = LongArray(iters)
            val prompt = floatArrayOf(newW / 2f, newH / 2f)
            for (i in 0 until iters) {
                val t0 = android.os.SystemClock.elapsedRealtimeNanos()
                runDecoder(embeddings, embShape, prompt, floatArrayOf(1f))
                decTimes[i] = (android.os.SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
            }
            val report = "ep=$tag iters=$iters " +
                "encoder_ms[p50=${percentile(encTimes, 0.50)}, p95=${percentile(encTimes, 0.95)}, " +
                "mean=${encTimes.average().toInt()}] " +
                "decoder_ms[p50=${percentile(decTimes, 0.50)}, p95=${percentile(decTimes, 0.95)}, " +
                "mean=${decTimes.average().toInt()}]"
            Log.i(BENCH_TAG, report)
            debugDir?.let { dir ->
                try {
                    if (!dir.exists()) dir.mkdirs()
                    java.io.File(dir, "bench_$tag.txt").writeText(report)
                } catch (t: Throwable) {
                    Log.w(BENCH_TAG, "bench 结果写文件失败", t)
                }
            }
        } catch (t: Throwable) {
            Log.w(BENCH_TAG, "bench 失败", t)
        }
    }

    private fun percentile(sorted: LongArray, q: Double): Long {
        val s = sorted.sorted()
        val idx = (s.size * q).toInt().coerceIn(0, s.size - 1)
        return s[idx]
    }

    /** 四角贴近画面边缘（棋盘被画面裁切，且无外推复原）→ 拒。 */
    private fun isCutOffQuad(c: FloatArray, w: Int, h: Int): Boolean =
        (0 until 8 step 2).any { c[it] < EDGE_MARGIN || c[it] > w - EDGE_MARGIN } ||
            (1 until 8 step 2).any { c[it] < EDGE_MARGIN || c[it] > h - EDGE_MARGIN }

    /** 复原态允许角点出画面，但不许远于半幅（防外推飞出）。 */
    private fun withinExpandedBounds(c: FloatArray, w: Int, h: Int): Boolean {
        val xLo = -w * 0.5f
        val xHi = w * 1.5f
        val yLo = -h * 0.5f
        val yHi = h * 1.5f
        for (i in 0 until 4) {
            if (c[i * 2] < xLo || c[i * 2] > xHi || c[i * 2 + 1] < yLo || c[i * 2 + 1] > yHi) {
                return false
            }
        }
        return true
    }

    /**
     * 判断新四角相对参考四角是否「跳变」：面积比超出 [JUMP_AREA_LO, JUMP_AREA_HI]
     * 或任一角位移超过参考对角线的 [JUMP_CORNER_RATIO]。掩码偶发溢出（变大）/
     * 塌缩（变小）都会命中；真实的连续移动逐帧差异小，不会命中。
     */
    private fun isQuadJump(q: FloatArray, ref: FloatArray): Boolean {
        val ratio = quadArea(q) / quadArea(ref).coerceAtLeast(1f)
        if (ratio !in JUMP_AREA_LO..JUMP_AREA_HI) return true
        val diag = max(
            hypot(ref[2] - ref[0], ref[3] - ref[1]),
            hypot(ref[4] - ref[6], ref[5] - ref[7]),
        )
        for (i in 0 until 4) {
            val d = hypot(q[i * 2] - ref[i * 2], q[i * 2 + 1] - ref[i * 2 + 1])
            if (d > diag * JUMP_CORNER_RATIO) return true
        }
        return false
    }

    private fun quadArea(c: FloatArray): Float {
        var s = 0f
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            s += c[i * 2] * c[j * 2 + 1] - c[j * 2] * c[i * 2 + 1]
        }
        return abs(s) / 2f
    }

    /**
     * 校验四角（TL,TR,BR,BL）是否构成非退化四边形：
     * 1) 任意两点不能过近（塌缩检测）；2) 面积不能过小（碎片/细条检测）。
     * 若通过校验说明四角可安全用于单应变换与 EMA。
     */
    private fun isValidQuad(c: FloatArray, w: Int, h: Int): Boolean {
        if (c.size != 8) return false
        val minDist = sqrt((w * w + h * h).toFloat()) * QUAD_MIN_DIST_RATIO
        for (i in 0 until 4) {
            for (j in i + 1 until 4) {
                val dx = c[i * 2] - c[j * 2]
                val dy = c[i * 2 + 1] - c[j * 2 + 1]
                if (sqrt(dx * dx + dy * dy) < minDist) return false
            }
        }
        // 四边形面积（鞋带公式）。
        var s = 0f
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            s += c[i * 2] * c[j * 2 + 1] - c[j * 2] * c[i * 2 + 1]
        }
        val area = abs(s) / 2f
        return area >= (w * h) * QUAD_MIN_AREA_RATIO
    }

    /**
     * 四角自适应 EMA 平滑：位移小→强平滑抑制抖动，位移大→快速跟随。
     * `α = clamp(dist / (diag·EMA_FOLLOW_RATIO), EMA_ALPHA_MIN, EMA_ALPHA_MAX)`。
     */
    private fun smoothCorners(raw: FloatArray, w: Int, h: Int): FloatArray {
        val prev = prevCorners
        if (prev == null || prev.size != 8) {
            prevCorners = raw.copyOf()
            return raw
        }
        val dx = ((raw[0] + raw[2] + raw[4] + raw[6]) - (prev[0] + prev[2] + prev[4] + prev[6])) / 4f
        val dy = ((raw[1] + raw[3] + raw[5] + raw[7]) - (prev[1] + prev[3] + prev[5] + prev[7])) / 4f
        val dist = sqrt(dx * dx + dy * dy)
        val diag = sqrt((w * w + h * h).toFloat())
        val alpha = (dist / (diag * EMA_FOLLOW_RATIO)).coerceIn(EMA_ALPHA_MIN, EMA_ALPHA_MAX)
        val out = FloatArray(8)
        for (i in 0 until 8) out[i] = alpha * raw[i] + (1f - alpha) * prev[i]
        prevCorners = out.copyOf()
        return out
    }

    private fun saveDebugFrame(bitmap: Bitmap, idx: Int) {
        val dir = debugDir ?: return
        try {
            if (!dir.exists()) dir.mkdirs()
            val file = java.io.File(dir, "seg_$idx.png")
            java.io.FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "保存调试帧失败", t)
        }
    }

    /** 失败诊断落盘：失败帧 JPEG + 各带内候选掩码 PNG（外推复原离线调试用）。 */
    private fun saveFailDiag(bitmap: Bitmap, out: DecoderOutput, cands: List<Int>, n: Int) {
        val dir = debugDir ?: return
        try {
            if (!dir.exists()) dir.mkdirs()
            java.io.File(dir, "fail_$n.jpg").outputStream().use { fo ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fo)
            }
            val maskSize = out.maskW * out.maskH
            for (i in cands) {
                val binary = ByteArray(maskSize)
                val off = i * maskSize
                for (j in 0 until maskSize) binary[j] = if (out.masks[off + j] > 0f) 1 else 0
                saveMaskPng(binary, out.maskW, out.maskH, "fail_${n}_m$i")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "保存失败诊断失败", t)
        }
    }

    /** 把二值掩码放大 4 倍（最近邻）保存为 PNG，便于目视/离线复现诊断。 */
    private fun saveMaskPng(binary: ByteArray, w: Int, h: Int, name: String) {
        val dir = debugDir ?: return
        try {
            if (!dir.exists()) dir.mkdirs()
            val up = 4
            val bmp = Bitmap.createBitmap(w * up, h * up, Bitmap.Config.ARGB_8888)
            val px = IntArray(w * up * h * up)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val c = if (binary[y * w + x] == 1.toByte()) -1 else -0x1000000
                    for (dy in 0 until up) {
                        for (dx in 0 until up) {
                            px[(y * up + dy) * (w * up) + (x * up + dx)] = c
                        }
                    }
                }
            }
            bmp.setPixels(px, 0, w * up, 0, 0, w * up, h * up)
            java.io.FileOutputStream(java.io.File(dir, "$name.png")).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            bmp.recycle()
        } catch (t: Throwable) {
            Log.w(TAG, "保存掩码失败", t)
        }
    }

    override fun close() {
        try { encoder.close() } catch (t: Throwable) { Log.w(TAG, "关闭 encoder 失败", t) }
        try { decoder.close() } catch (t: Throwable) { Log.w(TAG, "关闭 decoder 失败", t) }
        try { env.close() } catch (t: Throwable) { Log.w(TAG, "关闭 env 失败", t) }
    }

    companion object {
        private const val TAG = "SegmentationBoardDetector"
        private const val BENCH_TAG = "BenchEP"
        const val ENCODER_PATH = "models/edge_sam_3x_encoder.onnx"
        const val DECODER_PATH = "models/edge_sam_3x_decoder.onnx"
        const val IMG_SIZE = 1024

        /** 最小掩码面积占比（相对掩码图），低于视为无棋盘。 */
        private const val MIN_AREA_RATIO = 0.02f

        /** 最大掩码面积占比（相对掩码图），高于视为误分割到背景/整幅画面。 */
        private const val MAX_AREA_RATIO = 0.35f

        /** 四角距画面边缘小于该像素视为棋盘被裁切（部分出画），判为漏检。 */
        private const val EDGE_MARGIN = 8f

        /** 四角任意两点最小距离（相对画面对角线），低于视为塌缩的退化四边形。 */
        private const val QUAD_MIN_DIST_RATIO = 0.02f

        /** 四边形最小面积（相对画面），低于视为碎片/细条（非棋盘）。 */
        private const val QUAD_MIN_AREA_RATIO = 0.01f

        /** 自适应 EMA：小位移（抖动）时的最小平滑系数。 */
        private const val EMA_ALPHA_MIN = 0.15f

        /** 自适应 EMA：大位移（快速移动）时的最大跟随系数。 */
        private const val EMA_ALPHA_MAX = 0.9f

        /** 质心位移达到画面对角线的该比例时 α 接近最大，实现快速跟随。 */
        private const val EMA_FOLLOW_RATIO = 0.08f

        /** 时间一致性闸：面积比超出该区间视为跳变（掩码偶发溢出/塌缩）。 */
        private const val JUMP_AREA_LO = 0.75f
        private const val JUMP_AREA_HI = 1.35f

        /** 任一角位移超过参考对角线的该比例视为跳变。 */
        private const val JUMP_CORNER_RATIO = 0.22f

        /** 连续 N 帧一致的「跳变」四角 = 真实移动，放行重锁。 */
        private const val JUMP_ACCEPT_STREAK = 3

        /** 贴边判据（掩码坐标 px）：顶点距内容边界 ≤ 此值视为贴边（被画面裁切痕迹）。
         *  解码器上采样掩码距内容边界有 1~3px 内缩，须宽容差（真机 fail 掩码实测内缩 2px）。 */
        private const val BORDER_TOL = 4.0f

        /** 连续失败 ≥ 此值时判定历史（prevCorners/跳变闸基线）失效，清空走无历史路径。
         *  失效 v2（2026-10-12 S7）：原 2 在手遮挡场景误清历史（恢复帧走无历史路径
         *  直接采纳坏候选，放大跳变）——抬高到 5 = 真移动长丢失才失效。 */
        private const val HISTORY_INVALIDATE_AFTER_FAILS = 5

        /** 连续失败 ≥ 此值后，候选全拒时启用画面中心提示救援（帧内重跑 decoder）。 */
        private const val PROMPT_RESCUE_AFTER_FAILS = 1

        // ---------------- C4 形状闸 / C7① 质心先验（纯几何，JVM 可单测） ----------------

        /** 形状闸：内角下界（度）。低于视为细条/尖菱形（非棋盘投影）。 */
        internal const val SHAPE_MIN_ANGLE_DEG = 50f

        /** 形状闸：内角上界（度）。透视下矩形投影内角可偏离 90°，但不会过远。 */
        internal const val SHAPE_MAX_ANGLE_DEG = 130f

        /** 形状闸：对边长度比界（上下边、左右边各查一次，单向比值即可双向覆盖）。 */
        internal val SHAPE_EDGE_RATIO = 0.35f..2.9f

        /** 质心先验仲裁所需的最少参考质心数（不足则视为不可靠，回退面积优先）。 */
        internal const val MIN_REF_CENTROIDS = 10

        /** 质心包含率阈值（S7 起为**常态过滤**门 + 无历史仲裁共同阈值）：候选至少包住
         *  该比例的参考质心。0.95 = 好候选实测下界（err<15px ⇒ cont=1.00）；0.85 会
         *  放过手遮挡 TR 60-126px 中带候选。 */
        internal const val CENTROID_CONTAIN_MIN = 0.95f

        /** S7 box 提示外扩比例（相对参考 bbox 对角线）。 */
        private const val BOX_EXPAND = 0.05f

        /** D1 挪盘兜底的超时询问阈值：连续拒绝 ≥ 此值才触发双确认（手遮挡段 3-10 次检测
         *  远达不到；vis4 的长遮挡段 ~60+ 会达到但过不了几何+内容确认）。 */
        private const val REANCHOR_ASK_REJECTS = 60

        /** C4 形状闸：四角内角 ∈ [50°,130°] 且对边比在宽松界内。杀掉通过面积带/非退化校验的溢出细条。 */
        internal fun hasBoardShape(c: FloatArray): Boolean {
            if (c.size != 8) return false
            for (i in 0 until 4) {
                val p = (i + 3) % 4
                val n = (i + 1) % 4
                val v1x = c[p * 2] - c[i * 2]
                val v1y = c[p * 2 + 1] - c[i * 2 + 1]
                val v2x = c[n * 2] - c[i * 2]
                val v2y = c[n * 2 + 1] - c[i * 2 + 1]
                val l1 = hypot(v1x, v1y)
                val l2 = hypot(v2x, v2y)
                if (l1 < 1e-6f || l2 < 1e-6f) return false
                val cosA = ((v1x * v2x + v1y * v2y) / (l1 * l2)).coerceIn(-1f, 1f)
                val deg = acos(cosA) * (180f / Math.PI.toFloat())
                if (deg < SHAPE_MIN_ANGLE_DEG || deg > SHAPE_MAX_ANGLE_DEG) return false
            }
            val top = hypot(c[2] - c[0], c[3] - c[1])
            val bottom = hypot(c[6] - c[4], c[7] - c[5])
            val left = hypot(c[6] - c[0], c[7] - c[1])
            val right = hypot(c[4] - c[2], c[5] - c[3])
            if (bottom <= 0f || right <= 0f) return false
            if (top / bottom !in SHAPE_EDGE_RATIO) return false
            if (left / right !in SHAPE_EDGE_RATIO) return false
            return true
        }

        /** 点是否在凸四边形内（TL,TR,BR,BL 序，顺/逆时针均可；叉积同号判定）。 */
        internal fun pointInQuad(q: FloatArray, x: Float, y: Float): Boolean {
            var pos = false
            var neg = false
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                val cr = (q[j * 2] - q[i * 2]) * (y - q[i * 2 + 1]) -
                    (q[j * 2 + 1] - q[i * 2 + 1]) * (x - q[i * 2])
                if (cr > 0f) pos = true else if (cr < 0f) neg = true
                if (pos && neg) return false
            }
            return true
        }

        /** 质心包含率：落在四边形内的质心比例。 */
        internal fun containmentRatio(q: FloatArray, cents: List<FloatArray>): Float {
            if (cents.isEmpty()) return 0f
            var n = 0
            for (c in cents) if (pointInQuad(q, c[0], c[1])) n++
            return n.toFloat() / cents.size
        }

        /** 鞋带面积（非负）。 */
        internal fun quadAreaOf(c: FloatArray): Float {
            var s = 0f
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                s += c[i * 2] * c[j * 2 + 1] - c[j * 2] * c[i * 2 + 1]
            }
            return abs(s) / 2f
        }

        /** 掩码四角提取结果：四角 TL,TR,BR,BL（掩码坐标）、是否外推复原、最大连通域质心。 */
        internal class MaskQuad(val corners: FloatArray, val reconstructed: Boolean, val centroid: Pt?)

        /**
         * 二值掩码 → 四角。先试外推复原（角点出画面：贴边段以外的可见边线延长求交），
         * 无贴边痕迹时走凸包 + 贪心减点（全可见路径，与 P3-a 前行为一致）。
         */
        internal fun extractCornersEx(
            binary: ByteArray, w: Int, h: Int, contentW: Float, contentH: Float
        ): MaskQuad? {
            // 1. 4 邻接连通域标记，取最大连通域。
            val labels = IntArray(w * h) { -1 }
            val stack = IntArray(w * h)
            var bestLabel = -1
            var bestArea = 0
            var nextLabel = 0
            for (start in 0 until w * h) {
                if (binary[start] == 0.toByte() || labels[start] != -1) continue
                var sp = 0
                stack[sp++] = start
                labels[start] = nextLabel
                var area = 0
                while (sp > 0) {
                    val idx = stack[--sp]
                    area++
                    val x = idx % w
                    val y = idx / w
                    if (x > 0 && binary[idx - 1] == 1.toByte() && labels[idx - 1] == -1) {
                        labels[idx - 1] = nextLabel; stack[sp++] = idx - 1
                    }
                    if (x < w - 1 && binary[idx + 1] == 1.toByte() && labels[idx + 1] == -1) {
                        labels[idx + 1] = nextLabel; stack[sp++] = idx + 1
                    }
                    if (y > 0 && binary[idx - w] == 1.toByte() && labels[idx - w] == -1) {
                        labels[idx - w] = nextLabel; stack[sp++] = idx - w
                    }
                    if (y < h - 1 && binary[idx + w] == 1.toByte() && labels[idx + w] == -1) {
                        labels[idx + w] = nextLabel; stack[sp++] = idx + w
                    }
                }
                if (area > bestArea) {
                    bestArea = area
                    bestLabel = nextLabel
                }
                nextLabel++
            }
            if (bestLabel < 0 || bestArea < w * h * MIN_AREA_RATIO) return null

            // 2. 提取最大连通域边界像素 + 质心（复原态提示点用）。
            val boundary = ArrayList<Pt>(bestArea / 2)
            var sx = 0f
            var sy = 0f
            for (idx in 0 until w * h) {
                if (labels[idx] != bestLabel) continue
                sx += idx % w
                sy += idx / w
                val x = idx % w
                val y = idx / w
                val isBoundary = (x == 0 || labels[idx - 1] != bestLabel) ||
                    (x == w - 1 || labels[idx + 1] != bestLabel) ||
                    (y == 0 || labels[idx - w] != bestLabel) ||
                    (y == h - 1 || labels[idx + w] != bestLabel)
                if (isBoundary) boundary.add(Pt(x.toFloat(), y.toFloat()))
            }
            if (boundary.size < 4) return null
            val centroid = Pt(sx / bestArea, sy / bestArea)

            // 3. 凸包 → 4'. 外推复原（优先）或 4. 贪心减点 → 5. 排序 TL,TR,BR,BL。
            val hull = convexHull(boundary)
            if (hull.size < 4) return null
            val rec = reconstructClipped(hull, contentW, contentH)
            if (rec != null) return MaskQuad(rec, true, centroid)
            val quad = reduceToQuad(hull)
            if (quad.size != 4) return null
            return MaskQuad(orderQuad(quad), false, centroid)
        }

        /**
         * 角点出画面时的外推复原：凸包环中「两端点都贴内容边界」的段 = 掩码被画面裁切的
         * 痕迹（沿边框走线），整段丢弃；只剩一端贴边的段保留（贴边端点 = 棋盘边与边框的
         * 交点，在真实边线上）。剩余段按环序为棋盘可见边，近共线链合并（栅格阶梯毛刺）
         * 后须恰 4 条边线，相邻边线延长求交 = 四角（允许出画面）。
         * 覆盖 1 角出画 / 2 对角出画；整条边出画无解返回 null。全可见（无贴边顶点）返回 null。
         */
        internal fun reconstructClipped(hull: List<Pt>, contentW: Float, contentH: Float): FloatArray? {
            val n = hull.size
            if (n < 4) return null
            val ring = simplifyRing(hull)
            val tol = BORDER_TOL

            fun onBorder(p: Pt) =
                p.x <= tol || p.x >= contentW - 1f - tol ||
                    p.y <= tol || p.y >= contentH - 1f - tol

            val borderV = BooleanArray(ring.size) { onBorder(ring[it]) }
            if (!borderV.any { it }) return null

            fun collinear(pa: Pt, pb: Pt, c: Pt, d: Pt): Boolean {
                val v1x = pb.x - pa.x
                val v1y = pb.y - pa.y
                val v2x = d.x - c.x
                val v2y = d.y - c.y
                val l1 = hypot(v1x, v1y)
                val l2 = hypot(v2x, v2y)
                if (l1 < 1e-6f || l2 < 1e-6f) return false
                return abs((v1x * v2x + v1y * v2y) / (l1 * l2)) > 0.94f
            }

            val board = ArrayList<Pair<Pt, Pt>>()
            for (i in ring.indices) {
                if (!(borderV[i] && borderV[(i + 1) % ring.size])) {
                    board.add(ring[i] to ring[(i + 1) % ring.size])
                }
            }
            if (board.isEmpty()) return null

            val chains = ArrayList<Pair<Pt, Pt>>()
            for (s in board) {
                val last = chains.lastOrNull()
                if (last != null && collinear(last.first, last.second, s.first, s.second)) {
                    chains[chains.size - 1] = last.first to s.second
                } else {
                    chains.add(s)
                }
            }
            if (chains.size >= 2) {
                val lastC = chains.last()
                val firstC = chains.first()
                if (collinear(lastC.first, lastC.second, firstC.first, firstC.second)) {
                    chains[0] = lastC.first to firstC.second
                    chains.removeAt(chains.size - 1)
                }
            }
            if (chains.size != 4) return null

            fun lineIntersect(s1: Pair<Pt, Pt>, s2: Pair<Pt, Pt>): Pt? {
                val dx1 = s1.second.x - s1.first.x
                val dy1 = s1.second.y - s1.first.y
                val dx2 = s2.second.x - s2.first.x
                val dy2 = s2.second.y - s2.first.y
                val d = dx1 * dy2 - dy1 * dx2
                if (abs(d) < 1e-9f) return null
                val t = ((s2.first.x - s1.first.x) * dy2 - (s2.first.y - s1.first.y) * dx2) / d
                return Pt(s1.first.x + t * dx1, s1.first.y + t * dy1)
            }

            val pts = ArrayList<Pt>(4)
            for (i in 0 until 4) {
                pts.add(lineIntersect(chains[i], chains[(i + 1) % 4]) ?: return null)
            }
            return orderQuad(pts)
        }

        /**
         * 保轮廓简化：贪心删除到邻点弦垂距 < 2.5px 的顶点（栅格阶梯毛刺），
         * 真角点垂距大得以保留。凸环上与 RDP 等效（PC 复刻 _simplify_ring）。
         */
        private fun simplifyRing(ring: List<Pt>, eps: Float = 2.5f): List<Pt> {
            val poly = ArrayList(ring)
            while (poly.size > 4) {
                val n = poly.size
                var bestIdx = -1
                var bestCost = Float.MAX_VALUE
                for (i in 0 until n) {
                    val prev = poly[(i - 1 + n) % n]
                    val cur = poly[i]
                    val next = poly[(i + 1) % n]
                    val cost = pointSegDistance(cur, prev, next)
                    if (cost < bestCost) {
                        bestCost = cost
                        bestIdx = i
                    }
                }
                if (bestIdx < 0 || bestCost >= eps) break
                poly.removeAt(bestIdx)
            }
            return poly
        }

        private fun convexHull(ptsIn: List<Pt>): List<Pt> {
            val pts = ptsIn.distinctBy { it.x * 1_000_000f + it.y }
                .sortedWith(compareBy({ it.x }, { it.y }))
            if (pts.size <= 3) return pts

            val lower = ArrayList<Pt>()
            for (p in pts) {
                while (lower.size >= 2 && cross(lower[lower.size - 2], lower[lower.size - 1], p) <= 0f) {
                    lower.removeAt(lower.size - 1)
                }
                lower.add(p)
            }
            val upper = ArrayList<Pt>()
            for (p in pts.asReversed()) {
                while (upper.size >= 2 && cross(upper[upper.size - 2], upper[upper.size - 1], p) <= 0f) {
                    upper.removeAt(upper.size - 1)
                }
                upper.add(p)
            }
            lower.removeAt(lower.size - 1)
            upper.removeAt(upper.size - 1)
            lower.addAll(upper)
            return lower
        }

        private fun cross(o: Pt, a: Pt, b: Pt): Float =
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

        private fun reduceToQuad(hull: List<Pt>): List<Pt> {
            val poly = ArrayList(hull)
            while (poly.size > 4) {
                val n = poly.size
                var bestIdx = -1
                var bestCost = Float.MAX_VALUE
                for (i in 0 until n) {
                    val prev = poly[(i - 1 + n) % n]
                    val cur = poly[i]
                    val next = poly[(i + 1) % n]
                    val cost = pointSegDistance(cur, prev, next)
                    if (cost < bestCost) {
                        bestCost = cost
                        bestIdx = i
                    }
                }
                if (bestIdx < 0) break
                poly.removeAt(bestIdx)
            }
            return poly
        }

        private fun pointSegDistance(p: Pt, a: Pt, b: Pt): Float {
            val dx = b.x - a.x
            val dy = b.y - a.y
            val len2 = dx * dx + dy * dy
            if (len2 == 0f) return dist(p, a)
            val t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2
            val tc = t.coerceIn(0f, 1f)
            val projX = a.x + tc * dx
            val projY = a.y + tc * dy
            return dist(p, Pt(projX, projY))
        }

        private fun dist(a: Pt, b: Pt): Float {
            val dx = a.x - b.x
            val dy = a.y - b.y
            return sqrt(dx * dx + dy * dy)
        }

        private fun orderQuad(pts: List<Pt>): FloatArray {
            var tl = 0; var tr = 0; var br = 0; var bl = 0
            var minSum = Float.MAX_VALUE; var maxSum = -Float.MAX_VALUE
            var minDiff = Float.MAX_VALUE; var maxDiff = -Float.MAX_VALUE
            for (i in 0 until 4) {
                val s = pts[i].x + pts[i].y
                val d = pts[i].y - pts[i].x
                if (s < minSum) { minSum = s; tl = i }
                if (s > maxSum) { maxSum = s; br = i }
                if (d < minDiff) { minDiff = d; tr = i }
                if (d > maxDiff) { maxDiff = d; bl = i }
            }
            return floatArrayOf(
                pts[tl].x, pts[tl].y,
                pts[tr].x, pts[tr].y,
                pts[br].x, pts[br].y,
                pts[bl].x, pts[bl].y
            )
        }

        /** 精修生效日志的节流计数器上限内取样间隔。 */
        private const val REFINE_LOG_EVERY = 10

        // RGB 归一化参数（与 EdgeSAM Sam.preprocess 一致）。
        private val PIXEL_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val PIXEL_STD = floatArrayOf(58.395f, 57.12f, 57.375f)

        /**
         * 从 assets 加载 encoder/decoder ONNX 并创建定位器；模型缺失/损坏时抛出异常（显式失败，无降级链）。
         *
         * @param useNnapi 实验开关：尝试给 encoder/decoder 会话启用 NNAPI EP（bench 模式用）。
         *                 启用失败（设备/算子不支持）自动回退纯 CPU，不影响可用性。
         */
        fun load(context: Context, debugDir: java.io.File? = null, useNnapi: Boolean = false): SegmentationBoardDetector {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                val cores = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
                setIntraOpNumThreads(cores)
                if (useNnapi) {
                    try {
                        addNnapi()
                        Log.i(TAG, "NNAPI EP 已启用（算子不支持部分自动回退 CPU）")
                    } catch (t: Throwable) {
                        Log.w(TAG, "NNAPI EP 启用失败，回退纯 CPU：${t.message}")
                    }
                }
            }
            val encoder = env.createSession(readAsset(context, ENCODER_PATH), opts)
            val decoder = env.createSession(readAsset(context, DECODER_PATH), opts)
            return SegmentationBoardDetector(env, encoder, decoder, 0.5f, 0.5f, debugDir)
        }

        private fun readAsset(context: Context, path: String): ByteArray =
            context.assets.open(path).use { it.readBytes() }
    }
}
