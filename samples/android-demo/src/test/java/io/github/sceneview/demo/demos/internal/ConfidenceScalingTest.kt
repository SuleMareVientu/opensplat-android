package io.github.sceneview.demo.demos.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ConfidenceScalingTest {

    private val config = PriorConditionedSplatConfig()

    @Test
    fun testInitialOpacityMapping() {
        // c = 0.0 -> opacityMin (0.05)
        assertEquals(0.05f, config.computeInitialOpacity(0.0f), 1e-5f)

        // c = 1.0 -> opacityMax (0.70)
        assertEquals(0.70f, config.computeInitialOpacity(1.0f), 1e-5f)

        // c = 0.5 -> 0.05 + 0.65 * 0.5 = 0.375
        assertEquals(0.375f, config.computeInitialOpacity(0.5f), 1e-5f)

        // Clamping checks
        assertEquals(0.05f, config.computeInitialOpacity(-0.5f), 1e-5f)
        assertEquals(0.70f, config.computeInitialOpacity(1.5f), 1e-5f)
    }

    @Test
    fun testOpacityLogitConversion() {
        // logit(0.5) = ln(0.5 / 0.5) = ln(1.0) = 0.0
        assertEquals(0.0f, config.computeOpacityLogit(0.5f), 1e-5f)

        val logitLow = config.computeOpacityLogit(0.05f)
        val logitMid = config.computeOpacityLogit(0.375f)
        val logitHigh = config.computeOpacityLogit(0.70f)

        assertTrue("Logit must be strictly monotonic: low < mid", logitLow < logitMid)
        assertTrue("Logit must be strictly monotonic: mid < high", logitMid < logitHigh)
        assertTrue("Logit for alpha < 0.5 must be negative", logitLow < 0f)
        assertTrue("Logit for alpha > 0.5 must be positive", logitHigh > 0f)

        // Verify clamping doesn't blow up with +/- Infinity
        val logitZero = config.computeOpacityLogit(0.0f)
        val logitOne = config.computeOpacityLogit(1.0f)
        assertTrue("Clamped logit must be finite", !logitZero.isInfinite() && !logitZero.isNaN())
        assertTrue("Clamped logit must be finite", !logitOne.isInfinite() && !logitOne.isNaN())
    }

    @Test
    fun testDepthNoiseModel() {
        // sigma_z(z) = 0.0015 * z^2 + 0.0010
        val noise1m = config.computeDepthNoise(1.0f)
        val expected1m = 0.0015f * 1.0f + 0.0010f // 0.0025 m = 2.5 mm
        assertEquals(expected1m, noise1m, 1e-6f)

        val noise2m = config.computeDepthNoise(2.0f)
        val expected2m = 0.0015f * 4.0f + 0.0010f // 0.0070 m = 7.0 mm
        assertEquals(expected2m, noise2m, 1e-6f)

        assertTrue("Depth noise must increase quadratically with distance", noise2m > noise1m)
    }

    @Test
    fun testPresets() {
        val room = PriorConditionedSplatConfig.roomScale()
        val tabletop = PriorConditionedSplatConfig.tabletopScale()

        assertEquals(0.015f, room.voxelSizeMeters, 1e-5f)
        assertEquals(0.005f, tabletop.voxelSizeMeters, 1e-5f)
        assertEquals(16, room.knnK)
        assertEquals(12, tabletop.knnK)
        assertEquals(0.10f, room.depthSearchRangeFraction, 1e-5f)
        assertEquals(0.05f, tabletop.depthSearchRangeFraction, 1e-5f)
    }
}
