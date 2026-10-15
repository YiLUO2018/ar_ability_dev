package com.arability.gpu_on_phone.bench

import kotlin.math.sqrt

/**
 * 延迟统计（单位 ms）。
 *
 * 百分位采用最近邻秩法（nearest-rank）：排序后取第 ceil(p*n) 个（1-based），
 * 即索引 ceil(p*n)-1 —— 与 Excel PERCENTILE.INC 的 inclusive 口径一致。
 * 方差为样本方差（n-1）。
 */
data class Stats(
    val n: Int,
    val median: Double,
    val mean: Double,
    val min: Double,
    val max: Double,
    val p90: Double,
    val p99: Double,
    val variance: Double,
) {
    val std: Double get() = sqrt(variance)
    val cv: Double get() = if (mean == 0.0) 0.0 else std / mean

    companion object {
        fun of(latenciesMs: LongArray): Stats = of(latenciesMs.map { it.toDouble() }.toDoubleArray())

        fun of(latenciesMs: DoubleArray): Stats {
            require(latenciesMs.isNotEmpty()) { "latencies must not be empty" }
            val sorted = latenciesMs.clone().also { it.sort() }
            val n = sorted.size
            val median = if (n % 2 == 1) sorted[n / 2].toDouble()
            else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            val mean = latenciesMs.sum() / n.toDouble()
            val variance = if (n == 1) 0.0 else
                latenciesMs.sumOf { (it - mean) * (it - mean) } / (n - 1)
            return Stats(
                n = n,
                median = median,
                mean = mean,
                min = sorted.first(),
                max = sorted.last(),
                p90 = percentile(sorted, 0.90),
                p99 = percentile(sorted, 0.99),
                variance = variance,
            )
        }

        private fun percentile(sorted: DoubleArray, p: Double): Double {
            val rank = Math.ceil(p * sorted.size).toInt().coerceIn(1, sorted.size)
            return sorted[rank - 1]
        }
    }
}
