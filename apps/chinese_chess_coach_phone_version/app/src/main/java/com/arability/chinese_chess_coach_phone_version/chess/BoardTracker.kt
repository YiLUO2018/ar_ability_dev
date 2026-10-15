package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * C9-T1 BoardTracker 帧间几何跟踪（replay_pipeline.py BoardTracker 移植，2026-10-14）。
 *
 * 前验：对局中棋盘世界位姿恒定、动的只有相机 → 帧间只解相机位姿 = 图像间单应：
 * 每帧 H_g2i ← H_lk @ H_g2i（LK 好点对 RANSAC），四角 = H ∙ PHYSICAL_CORNERS。
 *
 * 分层（OpenCV 只在 [LkFlowEngine]，本类 100% 纯 Kotlin、JVM 可单测）：
 *  - [accumulate]：LK 好点对 → RANSAC 单应 → 累积 + 质量闸（内点/退化/四角边界）；
 *  - [pieceCorrect]：棋锚纠偏——ref_board（棋盘系质心真值）↔ 当前 YOLO 质心，
 *    两级匹配（就近 ≤0.75 格 / PCA 免对应）+ 旋转一致闸 → RANSAC 钉住 H。
 *
 * video_03 定稿基线（v03_final，用户看图裁决）：**单种子 + 棋锚纠偏**——SAM 只在
 * 好几何时供种（内部包含门保持开启），斜视角段由纠偏承担传播精度（~3.5px/10帧）。
 */
class BoardTracker {

    companion object {
        /** 跟踪健康时定周期重锚（帧）。 */
        const val REANCHOR_EVERY = 15
        /** 重锚失败后的重试间隔（跟踪仍活着时不背靠背烧 SAM）。 */
        const val RECHECK_AFTER_FAIL = 5
        /** 传播陈旧逃生：超过此帧数未过准入闸且无棋锚背书时，无条件信 SAM。 */
        const val STALE_AGE = 75
        /** 棋锚纠偏周期（帧）。 */
        const val CORRECT_EVERY = 10
        /** 近期纠偏成功过的窗口（帧）：此窗口内传播有棋子背书，不走陈旧逃生。 */
        const val CORRECT_BACKING = 30
        /** 准入闸 v3：棋锚背书时 SAM 与传播四角的逐角最大容差（格）。 */
        const val AGREE_CELL = 1.5f
        /** PCA 二级拟合的旋转一致闸（度）：标准开局云 180° 旋转对称，翻转组合
         *  count 满分且 det>0 闸拦不住——用「纠偏周期内棋盘不可能转 40°+」拦。 */
        const val MAX_CORRECT_ROT_DEG = 40f

        const val MIN_TRACK_PTS = 25
        const val MIN_LK_INLIERS = 35
        const val LK_RANSAC_REPROJ = 3.0f
        const val MIN_CORRECT_REFS = 12
        const val MIN_CORRECT_INLIERS = 10
        const val CORRECT_INLIER_CELL = 0.35f
        const val MATCH_CELL = 0.75f
        const val PCA_MATCH_CELL = 1.2f

        private val BOARD_DIAG = hypot(
            BoardGeometry.BOARD_W.toFloat(), BoardGeometry.BOARD_H.toFloat())

        /** 格宽自适应：四角对角线 × CELL/棋盘对角线（PC _cell_size 同位，D×50/672.7）。 */
        fun cellSizeOf(corners: FloatArray): Float {
            var dx = 0f; var dy = 0f
            for (i in 0 until 2) {
                dx = maxOf(dx, kotlin.math.abs(corners[(i + 1) * 2] - corners[i * 2]))
                dy = maxOf(dy, kotlin.math.abs(corners[(i + 1) * 2 + 1] - corners[i * 2 + 1]))
            }
            val d = hypot(corners[4] - corners[0], corners[5] - corners[1])
                .coerceAtLeast(hypot(corners[6] - corners[2], corners[7] - corners[3]))
            return d * BoardGeometry.CELL / BOARD_DIAG
        }

        /** 逐角最大位移（准入闸 v3 一致性判据）。 */
        fun maxCornerDisp(a: FloatArray, b: FloatArray): Float {
            var m = 0f
            for (i in 0 until 4) {
                m = maxOf(m, hypot(a[i * 2] - b[i * 2], a[i * 2 + 1] - b[i * 2 + 1]))
            }
            return m
        }

        /** 点在凸四边形（TL,TR,BR,BL）内（同侧法）。 */
        fun pointInQuad(quad: FloatArray, x: Float, y: Float): Boolean {
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                val cross = (quad[j * 2] - quad[i * 2]) * (y - quad[i * 2 + 1]) -
                    (quad[j * 2 + 1] - quad[i * 2 + 1]) * (x - quad[i * 2])
                if (cross < 0f) return false
            }
            return true
        }

        /** 参考点被四边形包含的比例（PC _containment 同位）。 */
        fun containmentRatio(quad: FloatArray, pts: FloatArray): Float {
            val n = pts.size / 2
            if (n == 0) return 1f
            var inside = 0
            for (i in 0 until n) {
                if (pointInQuad(quad, pts[i * 2], pts[i * 2 + 1])) inside++
            }
            return inside.toFloat() / n
        }

        /** 点到线段距离。 */
        private fun segDist(px: Float, py: Float, ax: Float, ay: Float, bx: Float, by: Float): Float {
            val dx = bx - ax; val dy = by - ay
            val len2 = dx * dx + dy * dy
            val t = if (len2 < 1e-12f) 0f
            else (((px - ax) * dx + (py - ay) * dy) / len2).coerceIn(0f, 1f)
            return hypot(px - (ax + t * dx), py - (ay + t * dy))
        }

        /** 参考点到四边形 4 边的最小距离（PC _min_edge_dist 同位，外扩检测用）。 */
        fun minEdgeDist(quad: FloatArray, pts: FloatArray): Float {
            var minD = Float.MAX_VALUE
            for (i in pts.indices step 2) {
                var d = Float.MAX_VALUE
                for (e in 0 until 4) {
                    val j = (e + 1) % 4
                    d = minOf(d, segDist(pts[i], pts[i + 1],
                        quad[e * 2], quad[e * 2 + 1], quad[j * 2], quad[j * 2 + 1]))
                }
                minD = minOf(minD, d)
            }
            return if (minD == Float.MAX_VALUE) 0f else minD
        }

        /**
         * 重锚准入闸 v2+v3（replay run() 的闸逻辑，PC 定稿行为）：
         *  - **传播已死（propagationAlive=false）→ 直接 ACCEPT**（PC 同款：闸只在
         *    prop_info 存在即传播活着的帧生效——否则 LK 死亡后背书永不过期
         *    （age 冻结）→ SAM 全被 v3 拦死 → 无法重播种的死锁，2026-10-15 真机教训）；
         *  - v2：包含率 ≥0.9（拦内收）且 最小边距 ≤1.0 格（拦外扩）；
         *  - v3：棋锚背书时 SAM 四角逐角一致 ≤[AGREE_CELL] 格（拦形变/膨胀——
         *    551-1644px 坏候选靠 0.90+ 包含率混入投毒 refBoard 的教训）；
         *  - 陈旧逃生：age ≥[STALE_AGE] 且无背书 → 无条件信 SAM（防快甩后走歪
         *    卡死；STALE_RESET 由调用方清 refBoard 后重建）。
         */
        fun admitReanchor(
            detectedCorners: FloatArray,
            tracker: BoardTracker,
            refImg: FloatArray?,
            propagationAlive: Boolean,
        ): ReanchorDecision {
            if (!propagationAlive) return ReanchorDecision.ACCEPT
            val hasRef = refImg != null && refImg.size >= 24   // ≥12 点
            if (hasRef) {
                val cont = containmentRatio(detectedCorners, refImg!!)
                val cell = cellSizeOf(detectedCorners)
                val medCell = minEdgeDist(detectedCorners, refImg) / cell
                if (cont < 0.9f || medCell > 1.0f) {
                    return if (tracker.age < STALE_AGE || tracker.isBacked) {
                        ReanchorDecision.HOLD
                    } else {
                        ReanchorDecision.STALE_RESET
                    }
                }
            }
            if (tracker.isBacked) {
                val prop = tracker.corners()
                if (prop != null && maxCornerDisp(detectedCorners, prop) >
                    AGREE_CELL * cellSizeOf(detectedCorners)
                ) {
                    return ReanchorDecision.HOLD
                }
            }
            return ReanchorDecision.ACCEPT
        }
    }

    /** 棋盘系(canonical) → 当前图像的累积单应（行优先 9 float）。 */
    var H: FloatArray? = null
        private set
    /** 距上次成功重锚的帧数。 */
    var age = 0
        private set
    /** 最近一次棋锚纠偏成功时的 age；-1 = 无。 */
    var lastCorrectAge = -1
        private set
    var lastInliers = 0
        private set
    /** 最近一次 accumulate 拟合的帧间单应 H_lk（T2 零偏在线估计用）。 */
    var lastHlk: FloatArray? = null
        private set
    var nTrackFrames = 0
        private set
    var nFail = 0
        private set

    /** 特征需按新种子 ROI 重提（LkFlowEngine 消费后置回 [onFeaturesFresh]）。 */
    var featuresStale = false
        private set

    /** 种子 ROI 四角（棋盘四角外扩 ~0.35 格，特征提取掩码用）。 */
    var roiCorners: FloatArray? = null
        private set

    /** 近期棋锚纠偏成功过 = 传播有棋子真值背书。 */
    val isBacked: Boolean
        get() = lastCorrectAge >= 0 && age - lastCorrectAge <= CORRECT_BACKING

    /** EdgeSAM 重锚后播种：H 由检测四角计算；特征按新 ROI 重提。 */
    fun seed(corners: FloatArray) {
        H = Homography.compute(BoardGeometry.physicalCorners, corners)
        age = 0
        lastCorrectAge = -1
        featuresStale = true
        // ROI = 四角多边形外扩 ~0.35 格（棋盘内纹理为主，略含边框；PC seed 同式）
        val c = FloatArray(8) { corners[it] }
        val midX = (c[0] + c[2] + c[4] + c[6]) / 4f
        val midY = (c[1] + c[3] + c[5] + c[7]) / 4f
        val cell = cellSizeOf(c)
        var maxR = 1f
        for (i in 0 until 4) {
            maxR = maxOf(maxR, hypot(c[i * 2] - midX, c[i * 2 + 1] - midY))
        }
        val grow = 1f + 2f * 0.35f * cell / maxR
        val roi = FloatArray(8)
        for (i in 0 until 4) {
            roi[i * 2] = (c[i * 2] - midX) * grow + midX
            roi[i * 2 + 1] = (c[i * 2 + 1] - midY) * grow + midY
        }
        roiCorners = roi
    }

    fun onFeaturesFresh() { featuresStale = false }

    /** 当前传播四角（无 H 返回 null）。 */
    fun corners(): FloatArray? =
        H?.let { HomographyMath.apply(it, BoardGeometry.physicalCorners) }

    /** 棋盘系参考点 → 当前图像系（移动场景包含门/box 提示/精修参考的关键）。 */
    fun refToImage(refBoard: FloatArray): FloatArray? =
        H?.let { HomographyMath.apply(it, refBoard) }

    /**
     * LK 好点对 → RANSAC 单应 → 累积传播（PC track() 的数学部分）。
     * 质量闸：内点 ≥[MIN_LK_INLIERS]、det∈[0.25,4]、有限、四角在画面 ±0.5/1.5 幅内。
     * 失败计 [nFail]（四角越界同时清 H——走重锚/全新重锁路径）。
     */
    fun accumulate(src: FloatArray, dst: FloatArray, nPairs: Int, imgW: Int, imgH: Int): Boolean {
        val h0 = H ?: return false
        if (nPairs < MIN_TRACK_PTS) { nFail++; return false }
        val fit = HomographyMath.ransacHomography(
            src, dst, nPairs, LK_RANSAC_REPROJ, MIN_LK_INLIERS)
        if (fit == null) { nFail++; return false }
        val hlk = fit.first
        val d = HomographyMath.det2(hlk)
        if (d < 0.25f || d > 4f || !HomographyMath.finite(hlk)) { nFail++; return false }
        val hNew = HomographyMath.mul(hlk, h0)
        if (!HomographyMath.finite(hNew)) { nFail++; return false }
        val c = HomographyMath.apply(hNew, BoardGeometry.physicalCorners)
        if (c[0] < -imgW * 0.5f || c[2] < -imgW * 0.5f || c[4] < -imgW * 0.5f || c[6] < -imgW * 0.5f ||
            c[0] > imgW * 1.5f || c[2] > imgW * 1.5f || c[4] > imgW * 1.5f || c[6] > imgW * 1.5f ||
            c[1] < -imgH * 0.5f || c[3] < -imgH * 0.5f || c[5] < -imgH * 0.5f || c[7] < -imgH * 0.5f ||
            c[1] > imgH * 1.5f || c[3] > imgH * 1.5f || c[5] > imgH * 1.5f || c[7] > imgH * 1.5f
        ) {
            H = null
            nFail++
            return false
        }
        H = hNew
        age++
        lastInliers = fit.second
        lastHlk = hlk
        nTrackFrames++
        return true
    }

    fun markTrackFail() { nFail++ }

    /**
     * 棋锚纠偏（PC piece_correct 同款）：ref_board ↔ 当前 YOLO 质心。
     *
     * 两级匹配：① 就近唯一（≤0.75 格，小漂移快速路径——捕获范围仅此，大漂移靠②）；
     * ② PCA 粗对齐（免对应，8 组合按匹配数降序试拟合；镜像组合 det2<0 拒、
     * 180° 翻转组合由旋转一致闸拒——对称点云的翻转 count 也可满分）。
     * 成功：H := 拟合、[lastCorrectAge] := age，返回纠偏量 px；失败返回 null（保持传播 H）。
     *
     * @param detPts YOLO 棋子质心（xy 交错，图像坐标；<12 个时调用方应用外扩裁剪重跑）。
     */
    fun pieceCorrect(
        refBoard: FloatArray,
        detPts: FloatArray,
        debug: ((String) -> Unit)? = null,
    ): Float? {
        val h0 = H ?: return null
        val nRef = refBoard.size / 2
        if (nRef < MIN_CORRECT_REFS || detPts.size / 2 < MIN_CORRECT_REFS) return null
        val corners = corners() ?: return null
        val cell = cellSizeOf(corners)
        val proj = HomographyMath.apply(h0, refBoard)

        val fits = ArrayList<Pair<FloatArray, FloatArray>>()   // (src, dst)
        val near = HomographyMath.uniqueNearest(proj, detPts, MATCH_CELL * cell)
        val nNear = near.size / 2
        debug?.invoke("cell=%.1f nNear=%d stage=%s".format(cell, nNear,
            if (nNear >= MIN_CORRECT_REFS) "near" else "pca"))
        if (nNear >= MIN_CORRECT_REFS) {
            val src = FloatArray(nNear * 2); val dst = FloatArray(nNear * 2)
            for (k in 0 until nNear) {
                val i = near[k * 2]; val j = near[k * 2 + 1]
                src[k * 2] = refBoard[i * 2]; src[k * 2 + 1] = refBoard[i * 2 + 1]
                dst[k * 2] = detPts[j * 2]; dst[k * 2 + 1] = detPts[j * 2 + 1]
            }
            fits.add(Pair(src, dst))
        } else {
            for (set in HomographyMath.pcaPairSets(refBoard, detPts, PCA_MATCH_CELL * cell)) {
                fits.add(Pair(set.src, set.dst))
            }
        }
        val cx = BoardGeometry.BOARD_W / 2f
        val cy = BoardGeometry.BOARD_H / 2f
        val maxRot = Math.toRadians(MAX_CORRECT_ROT_DEG.toDouble()).toFloat()
        for ((src, dst) in fits) {
            val n = src.size / 2
            val fit = HomographyMath.ransacHomography(
                src, dst, n, CORRECT_INLIER_CELL * cell, MIN_CORRECT_INLIERS)
            if (fit == null) {
                debug?.invoke("fit fail n=%d reproj=%.1f".format(n, CORRECT_INLIER_CELL * cell))
                continue
            }
            val hf = fit.first
            // 尺度闸（分辨率无关）：相对当前 H 的行列式比 ∈[0.5,2]——纠偏周期内棋盘
            // 显示尺度不可能变 1.4×+。绝对闸（PC 的 [0.25,4]）是 1080p 大棋盘标定，
            // 480p 手机上棋盘 ~200px 时合法 det≈0.1 会被误杀（2026-10-15 真机教训）。
            val d = HomographyMath.det2(hf)
            val d0 = HomographyMath.det2(h0)
            if (!(d0 > 0f && d / d0 in 0.5f..2.0f) || !HomographyMath.finite(hf)) {
                debug?.invoke("gate fail det=%.3f d0=%.3f".format(d, d0))
                continue
            }
            // 旋转一致闸（翻转组合拦截；PC 2026-10-14 同款）
            val rot = kotlin.math.abs(HomographyMath.axisAngleBetween(hf, h0, cx, cy))
            if (rot > maxRot) {
                debug?.invoke("rot fail %.0f°".format(Math.toDegrees(rot.toDouble())))
                continue
            }
            val newCorners = HomographyMath.apply(hf, BoardGeometry.physicalCorners)
            var shift = 0f
            for (i in 0 until 4) {
                shift += hypot(newCorners[i * 2] - corners[i * 2],
                               newCorners[i * 2 + 1] - corners[i * 2 + 1])
            }
            shift /= 4f
            H = hf
            lastCorrectAge = age
            return shift
        }
        return null
    }

    /** 传播彻底失效（unlock / 挪盘重锁）——与 PC run() unlock 路径同款。 */
    fun reset() {
        H = null
        age = 0
        lastCorrectAge = -1
        featuresStale = false
        roiCorners = null
    }

    /**
     * T2 外部旋转单应传播（LK 失效时 RotationPredictor 的 H_rot=K·ΔR·K⁻¹ 补位；
     * 平移小增量忽略）。仅累积 H 与 age，不做质量闸（调用方自证 ΔR 有限性）。
     */
    fun propagateExternal(hRot: FloatArray) {
        val h0 = H ?: return
        if (!HomographyMath.finite(hRot)) return
        H = HomographyMath.mul(hRot, h0)
        age++
    }
}

/** 准入闸裁决（[BoardTracker.admitReanchor]）。 */
enum class ReanchorDecision {
    /** 过闸：播种 tracker 并重建 refBoard。 */
    ACCEPT,
    /** 拦下：显示继续走传播（MOVING）/锚（STATIC）。 */
    HOLD,
    /** 陈旧逃生：无条件信 SAM，refBoard 先清空再重建。 */
    STALE_RESET,
}
