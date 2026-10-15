package com.arability.gpu_on_phone.bench

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class StatsTest {

    @Test
    fun medianOddCount() {
        assertEquals(3.0, Stats.of(longArrayOf(5, 1, 3)).median, 1e-9)
    }

    @Test
    fun medianEvenCount() {
        assertEquals(2.5, Stats.of(longArrayOf(4, 3, 2, 1)).median, 1e-9)
    }

    @Test
    fun percentilesOnOneToHundred() {
        val s = Stats.of(LongArray(100) { (it + 1).toLong() })
        assertEquals(50.5, s.median, 1e-9)
        assertEquals(90.0, s.p90, 1e-9)
        assertEquals(99.0, s.p99, 1e-9)
        assertEquals(1.0, s.min, 1e-9)
        assertEquals(100.0, s.max, 1e-9)
        assertEquals(50.5, s.mean, 1e-9)
    }

    @Test
    fun percentileClampsToMaxOnTinyArray() {
        val s = Stats.of(longArrayOf(7, 2))
        assertEquals(2.0, s.min, 1e-9)
        assertEquals(7.0, s.p90, 1e-9)
        assertEquals(7.0, s.p99, 1e-9)
        // 样本方差 n=2：(7-4.5)^2 + (2-4.5)^2 = 12.5
        assertEquals(12.5, s.variance, 1e-9)
    }

    @Test
    fun singleElementHasZeroVariance() {
        val s = Stats.of(longArrayOf(42))
        assertEquals(42.0, s.median, 1e-9)
        assertEquals(0.0, s.variance, 1e-9)
        assertEquals(0.0, s.cv, 1e-9)
    }

    @Test
    fun cvIsStdOverMean() {
        val s = Stats.of(longArrayOf(10, 20, 30))
        val expectedCv = kotlin.math.sqrt(s.variance) / s.mean
        assertEquals(expectedCv, s.cv, 1e-12)
        assertTrue(s.cv > 0)
    }

    @Test
    fun emptyInputThrows() {
        try {
            Stats.of(longArrayOf())
            fail("expected IllegalArgumentException")
        } catch (_: IllegalArgumentException) {
            // expected
        }
    }

    @Test
    fun unsortedInputIsHandled() {
        val s = Stats.of(longArrayOf(100, 1, 50, 25, 75))
        assertEquals(50.0, s.median, 1e-9)
        assertEquals(1.0, s.min, 1e-9)
        assertEquals(100.0, s.max, 1e-9)
    }
}
