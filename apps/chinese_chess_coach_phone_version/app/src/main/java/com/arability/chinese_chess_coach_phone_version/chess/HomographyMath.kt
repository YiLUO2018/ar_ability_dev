package com.arability.chinese_chess_coach_phone_version.chess

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * BoardTracker 纯 Kotlin 数学层（replay_pipeline.py 的 HomographyMath 部分，2026-10-14 移植）。
 *
 * 全部不依赖 OpenCV → JVM 可单测；OpenCV 只承担 LK（goodFeaturesToTrack /
 * calcOpticalFlowPyrLK）与灰度 Mat 管理（见 [BoardTracker]）。
 *
 * 来源：video_03 定稿基线（v03_final = 单种子 + 棋锚纠偏）：
 *  - [ransacHomography]：棋锚纠偏拟合（PC cv2.findHomography RANSAC 同位）；
 *  - [uniqueNearest]：就近唯一匹配（一级，捕获范围 ≤0.75 格）；
 *  - [pcaPairSets]：PCA 粗对齐候选对（二级，免对应、大漂移恢复——track5 教训：
 *    一级捕获范围只有 0.75 格，拉近段 300px 漂移恰好让它最需要时失灵）。
 */
object HomographyMath {

    /** 3×3 矩阵乘（行优先 9 float）。 */
    fun mul(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0f
            for (k in 0 until 3) s += a[i * 3 + k] * b[k * 3 + j]
            out[i * 3 + j] = s
        }
        return out
    }

    /** 3×3 矩阵逆（伴随矩阵法；奇异返回 null）。 */
    fun invert(h: FloatArray): FloatArray? {
        val c00 = h[4] * h[8] - h[5] * h[7]
        val c01 = h[5] * h[6] - h[3] * h[8]
        val c02 = h[3] * h[7] - h[4] * h[6]
        val det = h[0] * c00 + h[1] * c01 + h[2] * c02
        if (abs(det) < 1e-12f) return null
        val inv = 1f / det
        return floatArrayOf(
            c00 * inv, (h[2] * h[7] - h[1] * h[8]) * inv, (h[1] * h[5] - h[2] * h[4]) * inv,
            c01 * inv, (h[0] * h[8] - h[2] * h[6]) * inv, (h[2] * h[3] - h[0] * h[5]) * inv,
            c02 * inv, (h[1] * h[6] - h[0] * h[7]) * inv, (h[0] * h[4] - h[1] * h[3]) * inv,
        )
    }

    /** 批量变换点（xy 交错数组）通过单应 h。 */
    fun apply(h: FloatArray, pts: FloatArray): FloatArray {
        val out = FloatArray(pts.size)
        for (i in pts.indices step 2) {
            val x = pts[i]; val y = pts[i + 1]
            val w = h[6] * x + h[7] * y + h[8]
            if (abs(w) < 1e-12f) { out[i] = Float.NaN; out[i + 1] = Float.NaN; continue }
            out[i] = (h[0] * x + h[1] * y + h[2]) / w
            out[i + 1] = (h[3] * x + h[4] * y + h[5]) / w
        }
        return out
    }

    /** H 左上 2×2 行列式（PC 端 det 闸同位：棋锚拟合 ∈[0.25,4.0]、LK ∈[0.25,4.0]）。 */
    fun det2(h: FloatArray): Float = h[0] * h[4] - h[1] * h[3]

    fun finite(h: FloatArray): Boolean = h.all { it.isFinite() }

    /**
     * RANSAC 单应拟合（PC cv2.findHomography(src, dst, RANSAC, reproj) 同位）。
     *
     * 匹配对已由调用方做过唯一化（uniqueNearest / pcaPairSets 都是一对一），无随机
     * 组合采样退化陷阱；仍保留 det 闸与有限性闸（PC 同款）。迭代内取内点最多且
     * 通过闸的解，最后用全体内点最小二乘不再重解（4 点 DLT 精度足够——PC 端同款）。
     *
     * @return (H, 内点数) 或 null（内点 < [minInliers] / 退化）。
     */
    fun ransacHomography(
        src: FloatArray, dst: FloatArray, pairCount: Int,
        reprojPx: Float, minInliers: Int, iterations: Int = 2000,
        seed: Long = 42L,
    ): Pair<FloatArray, Int>? {
        if (pairCount < 4) return null
        val rnd = kotlin.random.Random(seed)
        val sx = FloatArray(pairCount); val sy = FloatArray(pairCount)
        val dx = FloatArray(pairCount); val dy = FloatArray(pairCount)
        for (i in 0 until pairCount) {
            sx[i] = src[i * 2]; sy[i] = src[i * 2 + 1]
            dx[i] = dst[i * 2]; dy[i] = dst[i * 2 + 1]
        }
        var bestH: FloatArray? = null
        var bestInliers = -1
        val reprojSq = reprojPx * reprojPx
        repeat(iterations) {
            val i0 = rnd.nextInt(pairCount)
            var i1 = rnd.nextInt(pairCount); if (i1 == i0) i1 = (i1 + 1) % pairCount
            var i2 = rnd.nextInt(pairCount); while (i2 == i0 || i2 == i1) i2 = (i2 + 1) % pairCount
            var i3 = rnd.nextInt(pairCount); while (i3 == i0 || i3 == i1 || i3 == i2) i3 = (i3 + 1) % pairCount
            val h = try {
                Homography.compute(
                    floatArrayOf(sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2], sx[i3], sy[i3]),
                    floatArrayOf(dx[i0], dy[i0], dx[i1], dy[i1], dx[i2], dy[i2], dx[i3], dy[i3]),
                )
            } catch (e: Exception) { null } ?: return@repeat
            if (!finite(h)) return@repeat
            val d = det2(h)
            if (d < 0.05f || d > 20f) return@repeat   // 粗闸（细闸由调用方按场景判）
            var inliers = 0
            for (i in 0 until pairCount) {
                val w = h[6] * sx[i] + h[7] * sy[i] + h[8]
                if (abs(w) < 1e-9f) continue
                val ex = (h[0] * sx[i] + h[1] * sy[i] + h[2]) / w - dx[i]
                val ey = (h[3] * sx[i] + h[4] * sy[i] + h[5]) / w - dy[i]
                if (ex * ex + ey * ey <= reprojSq) inliers++
            }
            if (inliers > bestInliers) { bestInliers = inliers; bestH = h }
        }
        val h = bestH ?: return null
        return if (bestInliers >= minInliers) Pair(h, bestInliers) else null
    }

    /**
     * 就近唯一匹配（PC _unique_nearest 同位）：每个参考点找最近质心（≤radius），
     * 距离升序贪心一对一。返回匹配下标对 (refIdx, detIdx) 交错数组。
     */
    fun uniqueNearest(
        proj: FloatArray, detPts: FloatArray, radius: Float,
    ): IntArray {
        val nRef = proj.size / 2
        val nDet = detPts.size / 2
        val cands = ArrayList<Triple<Float, Int, Int>>(nRef)
        for (i in 0 until nRef) {
            val px = proj[i * 2]; val py = proj[i * 2 + 1]
            var bestJ = -1; var bestD2 = radius * radius
            for (j in 0 until nDet) {
                val ex = detPts[j * 2] - px; val ey = detPts[j * 2 + 1] - py
                val d2 = ex * ex + ey * ey
                if (d2 <= bestD2) { bestD2 = d2; bestJ = j }
            }
            if (bestJ >= 0) cands.add(Triple(sqrt(bestD2), i, bestJ))
        }
        cands.sortBy { it.first }
        val usedRef = HashSet<Int>(); val usedDet = HashSet<Int>()
        val out = IntArray(cands.size * 2)
        var n = 0
        for ((_, i, j) in cands) {
            if (i in usedRef || j in usedDet) continue
            usedRef.add(i); usedDet.add(j)
            out[n * 2] = i; out[n * 2 + 1] = j; n++
        }
        return out.copyOf(n * 2)
    }

    /**
     * PCA 粗对齐候选对（PC piece_correct 二级 / video_label_yoloassist.pca_putative 同款）。
     *
     * 两团点主轴（2×2 协方差特征向量）的 8 种符号/换轴组合各估相似变换（旋转 +
     * 逐轴尺度 + 平移），对齐后按预测位置吸附质心（≤radius 一对一）→ 候选对集合，
     * 按匹配数降序。免对应：漂移任意大（旋转/缩放/平移）都能恢复。
     * 镜像/翻转组合由调用方 det>0 闸拒绝（反射 H det2<0）。
     *
     * @return List of (匹配数, srcPts, dstPts, refIdx, detIdx)，按匹配数降序。
     */
    fun pcaPairSets(
        ref: FloatArray, detPts: FloatArray, radius: Float,
    ): List<PcaPairSet> {
        val nRef = ref.size / 2
        val nDet = detPts.size / 2
        if (nRef < 6 || nDet < 6) return emptyList()
        val (ms, vs, ws) = pca2(ref)
        val (md, vd, wd) = pca2(detPts)
        val sets = ArrayList<PcaPairSet>()
        for (swap in 0..1) {
            // swap：主轴/次轴互换（覆盖 90° 旋转歧义）
            for (f0 in floatArrayOf(1f, -1f)) for (f1 in floatArrayOf(1f, -1f)) {
                // R = Vd · diag(f0,f1) · Vs2ᵀ（2×2）
                val s0 = if (swap == 0) 0 else 1
                val s1 = if (swap == 0) 1 else 0
                // vs2 列 = vs 的 [s0, s1] 顺序
                val vs00 = vs[s0 * 2]; val vs01 = vs[s0 * 2 + 1]
                val vs10 = vs[s1 * 2]; val vs11 = vs[s1 * 2 + 1]
                // A = diag(f0,f1)·Vs2ᵀ → 行 i = f_i · (vs 第 s_i 行)
                val a00 = f0 * vs00; val a01 = f0 * vs01
                val a10 = f1 * vs10; val a11 = f1 * vs11
                // R = Vd · A
                val r00 = vd[0] * a00 + vd[1] * a10
                val r01 = vd[0] * a01 + vd[1] * a11
                val r10 = vd[2] * a00 + vd[3] * a10
                val r11 = vd[2] * a01 + vd[3] * a11
                // 逐轴尺度：输出轴不动（wd 不换），源轴随 swap 换序（ws[s0]/ws[s1]）
                val scale0 = sqrt((if (wd[0] > 1e-6f) wd[0] else 1e-6f) /
                    (if (ws[s0] > 1e-6f) ws[s0] else 1e-6f))
                val scale1 = sqrt((if (wd[1] > 1e-6f) wd[1] else 1e-6f) /
                    (if (ws[s1] > 1e-6f) ws[s1] else 1e-6f))
                // proj = md + R·(ref−ms)·scale（numpy 的 (s−ms) @ R.T 等价于 R·v，逐轴尺度）
                val proj = FloatArray(nRef * 2)
                for (i in 0 until nRef) {
                    val rx = ref[i * 2] - ms[0]
                    val ry = ref[i * 2 + 1] - ms[1]
                    proj[i * 2] = md[0] + (rx * r00 + ry * r01) * scale0
                    proj[i * 2 + 1] = md[1] + (rx * r10 + ry * r11) * scale1
                }
                val pairs = uniqueNearest(proj, detPts, radius)
                if (pairs.size >= 12) {
                    val m = pairs.size / 2
                    val src = FloatArray(m * 2); val dst = FloatArray(m * 2)
                    for (k in 0 until m) {
                        val i = pairs[k * 2]; val j = pairs[k * 2 + 1]
                        src[k * 2] = ref[i * 2]; src[k * 2 + 1] = ref[i * 2 + 1]
                        dst[k * 2] = detPts[j * 2]; dst[k * 2 + 1] = detPts[j * 2 + 1]
                    }
                    sets.add(PcaPairSet(m, src, dst, pairs))
                }
            }
        }
        sets.sortByDescending { it.count }
        return sets
    }

    /** PCA 候选对集合（[pcaPairSets] 返回项）。 */
    class PcaPairSet(
        val count: Int,
        val src: FloatArray,
        val dst: FloatArray,
        /** 匹配下标对 (refIdx, detIdx) 交错。 */
        val pairs: IntArray,
    )

    /** 点团 PCA：均值、特征向量（列主序 [v0x,v0y,v1x,v1y]，按特征值降序）、特征值（降序）。 */
    private fun pca2(pts: FloatArray): Triple<FloatArray, FloatArray, FloatArray> {
        val n = pts.size / 2
        var mx = 0f; var my = 0f
        for (i in 0 until n) { mx += pts[i * 2]; my += pts[i * 2 + 1] }
        mx /= n; my /= n
        var cxx = 0f; var cxy = 0f; var cyy = 0f
        for (i in 0 until n) {
            val x = pts[i * 2] - mx; val y = pts[i * 2 + 1] - my
            cxx += x * x; cxy += x * y; cyy += y * y
        }
        cxx /= n; cxy /= n; cyy /= n
        // 2×2 对称阵特征值闭式解
        val tr = cxx + cyy
        val diff = cxx - cyy
        val disc = sqrt(diff * diff / 4f + cxy * cxy)
        val l0 = tr / 2f + disc
        val l1 = tr / 2f - disc
        // 特征向量：l0 对应 (cxy, l0-cxx)（cxy≈0 时取轴对齐基）
        var v0x: Float; var v0y: Float
        if (abs(cxy) > 1e-9f) {
            v0x = cxy; v0y = l0 - cxx
            val norm = hypot(v0x, v0y).coerceAtLeast(1e-12f)
            v0x /= norm; v0y /= norm
        } else {
            v0x = if (cxx >= cyy) 1f else 0f; v0y = if (cxx >= cyy) 0f else 1f
        }
        // v1 = v0 旋转 90°
        val v1x = -v0y; val v1y = v0x
        return Triple(
            floatArrayOf(mx, my),
            floatArrayOf(v0x, v0y, v1x, v1y),
            floatArrayOf(l0, l1),
        )
    }

    /** 两轴夹角（弧度），诊断用。 */
    fun angleBetween(a: FloatArray, b: FloatArray): Float {
        val dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        return atan2(dot.coerceIn(-1f, 1f), 0f)
    }

    /**
     * 旋转一致闸（PC 端 2026-10-14 同款）：两个单应在棋盘中心处 x 轴方向像的夹角
     * （带符号弧度）。标准开局点云 180° 旋转对称 → PCA 180° 组合 count 满分且
     * det>0 闸拦不住（180° 是正常旋转）——用「棋盘纠偏周期内不可能转 40°+」拦。
     */
    fun axisAngleBetween(hA: FloatArray, hB: FloatArray, cx: Float, cy: Float): Float {
        val a0 = apply(hA, floatArrayOf(cx, cy))
        val a1 = apply(hA, floatArrayOf(cx + 100f, cy))
        val b0 = apply(hB, floatArrayOf(cx, cy))
        val b1 = apply(hB, floatArrayOf(cx + 100f, cy))
        val dax = a1[0] - a0[0]; val day = a1[1] - a0[1]
        val dbx = b1[0] - b0[0]; val dby = b1[1] - b0[1]
        val na = hypot(dax, day).coerceAtLeast(1e-9f)
        val nb = hypot(dbx, dby).coerceAtLeast(1e-9f)
        return atan2((dax * dby - day * dbx) / (na * nb), (dax * dbx + day * dby) / (na * nb))
    }
}
