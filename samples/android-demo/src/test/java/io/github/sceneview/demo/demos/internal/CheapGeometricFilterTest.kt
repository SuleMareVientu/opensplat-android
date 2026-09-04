package io.github.sceneview.demo.demos.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class CheapGeometricFilterTest {

    private val config = PriorConditionedSplatConfig.roomScale()

    @Test
    fun testInLoopEdgeFilter_planarRegionRetainsConfidence() {
        val width = 10
        val height = 10
        val depthArray = ShortArray(width * height) { 1000.toShort() } // flat 1000 mm plane
        val confByte = 200 // ~0.784
        val gradient = 5f  // low texture gradient

        val conf = CheapGeometricFilter.filterEdgeAndNormalizeConfidence(
            u = 5,
            v = 5,
            rawDepthMm = 1000,
            confByte = confByte,
            depthWidth = width,
            depthHeight = height,
            rawDepthArray = depthArray,
            luminanceGradient = gradient,
            config = config
        )

        assertTrue("Planar point with high raw confidence should yield high normalized confidence", conf > 0.70f)
        assertTrue("Confidence must not exceed 1.0", conf <= 1.0f)
    }

    @Test
    fun testInLoopEdgeFilter_depthDiscontinuityAndEdgeRejection() {
        val width = 10
        val height = 10
        val depthArray = ShortArray(width * height) { 1000.toShort() }
        // Create depth jump on right neighbor
        depthArray[5 * width + 6] = 1600.toShort() // 60% depth jump

        val conf = CheapGeometricFilter.filterEdgeAndNormalizeConfidence(
            u = 5,
            v = 5,
            rawDepthMm = 1000,
            confByte = 250,
            depthWidth = width,
            depthHeight = height,
            rawDepthArray = depthArray,
            luminanceGradient = 60f, // high luminance edge
            config = config
        )

        assertEquals("Point on depth jump and texture edge must be rejected (0.0)", 0.0f, conf, 1e-6f)
    }

    @Test
    fun testComputeSmallestEigenvector_horizontalPlaneNormal() {
        // Covariance for XY spread with negligible Z variance:
        // Cxx = 1.0, Cyy = 1.0, Czz = 0.0001
        val normal = CheapGeometricFilter.computeSmallestEigenvector(
            c00 = 1.0f, c01 = 0.0f, c02 = 0.0f,
            c11 = 1.0f, c12 = 0.0f,
            c22 = 0.0001f
        )

        assertEquals(0.0f, normal[0], 0.01f)
        assertEquals(0.0f, normal[1], 0.01f)
        assertEquals(1.0f, abs(normal[2]), 0.01f)
    }

    @Test
    fun testPlaneFitResidualFilter_removesFlyingPixelOutlier() {
        val points = ArrayList<GeometricPoint>()

        // Create a 5x5 grid of points on the Z=1.0 plane at depth = 1.0m
        for (ix in -2..2) {
            for (iy in -2..2) {
                points.add(
                    GeometricPoint(
                        x = ix * 0.02f,
                        y = iy * 0.02f,
                        z = 1.0f,
                        r = 128, g = 128, b = 128,
                        confidence = 0.8f,
                        depthZ = 1.0f
                    )
                )
            }
        }

        // Insert a flying pixel outlier point sticking 5 cm out of the plane
        val outlier = GeometricPoint(
            x = 0.01f,
            y = 0.01f,
            z = 1.05f, // 5 cm off plane (threshold at 1.0m is 3 * (0.0015*1 + 0.001) = 0.0075m = 7.5mm)
            r = 128, g = 128, b = 128,
            confidence = 0.8f,
            depthZ = 1.0f
        )
        points.add(outlier)

        val filtered = CheapGeometricFilter.filterPointSet(points, config)

        assertFalse("Outlier 5 cm off plane must be pruned by residual filter", filtered.contains(outlier))
        assertTrue("In-plane points should largely survive", filtered.size >= 20)
    }

    @Test
    fun testStatisticalOutlierRemoval_removesIsolatedCluster() {
        val points = ArrayList<GeometricPoint>()

        // Dense cluster of 20 points around origin
        for (i in 0 until 25) {
            points.add(
                GeometricPoint(
                    x = (i % 5) * 0.01f,
                    y = (i / 5) * 0.01f,
                    z = 1.0f,
                    r = 200, g = 200, b = 200,
                    confidence = 0.9f,
                    depthZ = 1.0f
                )
            )
        }

        // Single isolated point far away (50 cm away)
        val isolated = GeometricPoint(
            x = 0.50f,
            y = 0.50f,
            z = 1.0f,
            r = 200, g = 200, b = 200,
            confidence = 0.5f,
            depthZ = 1.0f
        )
        points.add(isolated)

        val filtered = CheapGeometricFilter.filterPointSet(points, config)

        assertFalse("Isolated distant point must be removed by SOR / KNN lookup", filtered.contains(isolated))
    }
}
