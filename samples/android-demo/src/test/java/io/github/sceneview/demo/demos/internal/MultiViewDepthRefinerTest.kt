package io.github.sceneview.demo.demos.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.cos

class MultiViewDepthRefinerTest {

    private val config = PriorConditionedSplatConfig.roomScale()
    private val refiner = MultiViewDepthRefiner(config)

    /**
     * Create a standard 4x4 OpenGL camera-to-world matrix at position (tx, ty, tz)
     * looking down -Z.
     */
    private fun createPoseMatrix(tx: Float, ty: Float, tz: Float): FloatArray {
        return floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            tx, ty, tz, 1f
        )
    }

    @Test
    fun testSelectKeyframes_prefersOptimalParallaxAndFiltersDegenerate() {
        val refView = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 500f, fy = 500f, cx = 250f, cy = 250f,
            width = 500, height = 500
        )

        // Point 1 meter in front of refView (OpenGL camera looks down -Z)
        val pt = GeometricPoint(
            x = 0f, y = 0f, z = -1.0f,
            r = 128, g = 128, b = 128,
            confidence = 0.8f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1.0f // Facing camera at z >= 0
        )

        // Candidate 1: baseline angle ~18 deg
        val angle18Rad = Math.toRadians(18.0).toFloat()
        val candGood = KeyframeView(
            index = 1,
            cameraToWorld = createPoseMatrix(sin(angle18Rad), 0f, cos(angle18Rad) - 1.0f),
            fx = 500f, fy = 500f, cx = 250f, cy = 250f,
            width = 500, height = 500
        )

        // Candidate 2: too large baseline angle (60 deg > 40 deg limit)
        val angle60Rad = Math.toRadians(60.0).toFloat()
        val candTooWide = KeyframeView(
            index = 2,
            cameraToWorld = createPoseMatrix(sin(angle60Rad), 0f, cos(angle60Rad) - 1.0f),
            fx = 500f, fy = 500f, cx = 250f, cy = 250f,
            width = 500, height = 500
        )

        // Candidate 3: looking from behind surface (incidence cos < 0)
        val candBehind = KeyframeView(
            index = 3,
            cameraToWorld = createPoseMatrix(0f, 0f, -2.0f),
            fx = 500f, fy = 500f, cx = 250f, cy = 250f,
            width = 500, height = 500
        )

        val selected = refiner.selectKeyframes(pt, refView, listOf(candGood, candTooWide, candBehind))

        assertEquals("Should select exactly 1 qualifying keyframe", 1, selected.size)
        assertEquals("Selected view must be the 18 deg optimal baseline view", 1, selected[0].index)
    }

    @Test
    fun testLowCoverageFallback_doesNotDropPoint() {
        val refView = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 500f, fy = 500f, cx = 250f, cy = 250f,
            width = 500, height = 500
        )

        val pt = GeometricPoint(
            x = 0f, y = 0f, z = -1.0f,
            r = 128, g = 128, b = 128,
            confidence = 0.8f,
            depthZ = 1.0f
        )

        val refLum = ArrayLuminanceProvider(500, 500, ByteArray(500 * 500) { 100.toByte() })

        // No qualifying neighbor views available
        val result = refiner.refinePoint(
            point = pt,
            refView = refView,
            refLum = refLum,
            candidateViews = emptyList(),
            luminanceLookup = { null }
        )

        assertNotNull("Low coverage point must NOT be dropped (null)", result)
        assertFalse("Point must be marked as not multi-view verified", result!!.isMultiViewVerified)
        assertEquals(
            "Confidence must receive 0.7x soft penalty",
            0.8f * 0.7f,
            result.point.confidence,
            1e-4f
        )
        assertEquals("Point position must remain initial d0", -1.0f, result.point.z, 1e-4f)
    }

    @Test
    fun testZNCC_identicalPatchesYieldCorrelationNearOne() {
        val width = 100
        val height = 100

        // Create texture pattern with distinct gradient
        val data = ByteArray(width * height) { idx ->
            val x = idx % width
            val y = idx / width
            ((x * 2 + y * 3) % 255).toByte()
        }

        val lumA = ArrayLuminanceProvider(width, height, data)
        val lumB = ArrayLuminanceProvider(width, height, data)

        val viewA = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )

        val viewB = KeyframeView(
            index = 1,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f), // Identical pose for direct correlation test
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )

        // Point at center (u=50, v=50) -> x=0, y=0, z=-1.0f (in front of camera)
        val zncc = refiner.computeZNCC(
            pWorldX = 0f, pWorldY = 0f, pWorldZ = -1.0f,
            refView = viewA, refLum = lumA,
            targetView = viewB, targetLum = lumB
        )

        assertTrue("Identical patches must yield ZNCC close to 1.0, was: $zncc", zncc > 0.95f)
    }

    @Test
    fun testZNCC_corruptDiscordantPatchYieldsLowCorrelation() {
        val width = 100
        val height = 100

        val dataA = ByteArray(width * height) { idx -> ((idx % width) * 5).toByte() }
        val dataB = ByteArray(width * height) { idx -> (((width - (idx % width))) * 5).toByte() } // Inverted gradient

        val lumA = ArrayLuminanceProvider(width, height, dataA)
        val lumB = ArrayLuminanceProvider(width, height, dataB)

        val viewA = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )
        val viewB = KeyframeView(
            index = 1,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )

        val zncc = refiner.computeZNCC(
            pWorldX = 0f, pWorldY = 0f, pWorldZ = -1.0f,
            refView = viewA, refLum = lumA,
            targetView = viewB, targetLum = lumB
        )

        assertTrue("Opposing gradients must yield negative or low ZNCC, was: $zncc", zncc < config.znccThreshold)
    }

    @Test
    fun testTextureVarianceGating_texturelessPatchBypassesPruning() {
        val width = 100
        val height = 100

        // Completely flat/homogeneous texture (zero variance)
        val flatData = ByteArray(width * height) { 128.toByte() }
        val lumA = ArrayLuminanceProvider(width, height, flatData)
        val lumB = ArrayLuminanceProvider(width, height, flatData)

        val angle18Rad = Math.toRadians(18.0).toFloat()
        val angleMinus18Rad = Math.toRadians(-18.0).toFloat()
        val viewA = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )
        val viewB = KeyframeView(
            index = 1,
            cameraToWorld = createPoseMatrix(sin(angle18Rad), 0f, cos(angle18Rad) - 1.0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )
        val viewC = KeyframeView(
            index = 2,
            cameraToWorld = createPoseMatrix(sin(angleMinus18Rad), 0f, cos(angleMinus18Rad) - 1.0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )

        val pt = GeometricPoint(
            x = 0f, y = 0f, z = -1.0f,
            r = 128, g = 128, b = 128,
            confidence = 0.8f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f
        )

        val result = refiner.refinePoint(
            point = pt,
            refView = viewA,
            refLum = lumA,
            candidateViews = listOf(viewB, viewC),
            luminanceLookup = { if (it.index == 1) lumB else lumB }
        )

        assertNotNull("Textureless patch must NOT be pruned on ZNCC", result)
        assertFalse("Textureless patch must be marked as not verified", result!!.isMultiViewVerified)
        assertEquals("Depth must be preserved", -1.0f, result.point.z, 1e-4f)
    }

    @Test
    fun testTextureVarianceGating_texturedPatchPrunesOnDiscordance() {
        val width = 100
        val height = 100

        // Strong horizontal stripe pattern with variance >> 8.0:
        // lumA has high value on even y, low on odd y
        val dataA = ByteArray(width * height) { idx ->
            val y = idx / width
            if (y % 2 == 0) 200.toByte() else 50.toByte()
        }
        // Inverted horizontal stripe pattern on target view:
        // lumB has low value on even y, high on odd y -> ZNCC = -1.0 along entire epipolar line
        val dataB = ByteArray(width * height) { idx ->
            val y = idx / width
            if (y % 2 == 0) 50.toByte() else 200.toByte()
        }

        val lumA = ArrayLuminanceProvider(width, height, dataA)
        val lumB = ArrayLuminanceProvider(width, height, dataB)

        val angle18Rad = Math.toRadians(18.0).toFloat()
        val angleMinus18Rad = Math.toRadians(-18.0).toFloat()
        val viewA = KeyframeView(
            index = 0,
            cameraToWorld = createPoseMatrix(0f, 0f, 0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )
        val viewB = KeyframeView(
            index = 1,
            cameraToWorld = createPoseMatrix(sin(angle18Rad), 0f, cos(angle18Rad) - 1.0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )
        val viewC = KeyframeView(
            index = 2,
            cameraToWorld = createPoseMatrix(sin(angleMinus18Rad), 0f, cos(angleMinus18Rad) - 1.0f),
            fx = 100f, fy = 100f, cx = 50f, cy = 50f,
            width = width, height = height
        )

        val pt = GeometricPoint(
            x = 0f, y = 0f, z = -1.0f,
            r = 128, g = 128, b = 128,
            confidence = 0.5f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f
        )

        val result = refiner.refinePoint(
            point = pt,
            refView = viewA,
            refLum = lumA,
            candidateViews = listOf(viewB, viewC),
            luminanceLookup = { if (it.index == 1 || it.index == 2) lumB else null }
        )

        assertNull("Textured patch with discordant ZNCC must be pruned (null)", result)
    }
}
