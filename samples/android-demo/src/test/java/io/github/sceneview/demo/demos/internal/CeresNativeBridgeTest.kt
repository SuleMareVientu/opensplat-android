package io.github.sceneview.demo.demos.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin
import kotlin.math.cos

class CeresNativeBridgeTest {

    private fun createPoseMatrix(tx: Float, ty: Float, tz: Float): FloatArray {
        return floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            tx, ty, tz, 1f
        )
    }

    @Test
    fun testSafeFallback_whenNativeLibNotLoaded() {
        // When running on host JVM (Windows x86_64), libceres_ba.so (arm64-v8a) is not loaded.
        // runBundleAdjustment must return gracefully with success = false and zero exceptions.
        val kf0 = KeyframeView(0, createPoseMatrix(0f, 0f, 0f), 500f, 500f, 250f, 250f, width = 500, height = 500)
        val kf1 = KeyframeView(1, createPoseMatrix(0.2f, 0f, 0f), 500f, 500f, 250f, 250f, width = 500, height = 500)
        val pt = GeometricPoint(0f, 0f, -1.0f, r = 128, g = 128, b = 128, confidence = 0.5f, depthZ = 1.0f, nx = 0f, ny = 0f, nz = 1f)

        val result = CeresNativeBridge.runBundleAdjustment(
            keyframes = listOf(kf0, kf1),
            candidatePoints = listOf(pt),
            targetAnchors = 10,
            maxIterations = 5
        )

        assertNotNull(result)
        if (!CeresNativeBridge.isAvailable) {
            assertFalse("Should report false when native lib not loaded on host", result.success)
            assertEquals("Should return input keyframes on failure", 2, result.updatedKeyframes.size)
            assertTrue("Should return empty anchors on failure", result.updatedAnchorPoints.isEmpty())
        }
    }

    @Test
    fun testEmptyInput_returnsSafeResult() {
        val result = CeresNativeBridge.runBundleAdjustment(
            keyframes = emptyList(),
            candidatePoints = emptyList()
        )
        assertFalse(result.success)
        assertEquals(0, result.numAnchorPoints)
        assertEquals(0, result.numObservations)
    }

    @Test
    fun testSingleKeyframe_returnsSafeResult() {
        val kf0 = KeyframeView(0, createPoseMatrix(0f, 0f, 0f), 500f, 500f, 250f, 250f, width = 500, height = 500)
        val pt = GeometricPoint(0f, 0f, -1.0f, r = 128, g = 128, b = 128, confidence = 0.5f, depthZ = 1.0f, nx = 0f, ny = 0f, nz = 1f)

        val result = CeresNativeBridge.runBundleAdjustment(
            keyframes = listOf(kf0),
            candidatePoints = listOf(pt)
        )
        assertFalse("At least 2 keyframes required for BA", result.success)
    }
}
