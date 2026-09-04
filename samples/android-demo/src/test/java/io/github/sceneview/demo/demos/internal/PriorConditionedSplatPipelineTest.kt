package io.github.sceneview.demo.demos.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln

class PriorConditionedSplatPipelineTest {

    private val config = PriorConditionedSplatConfig.roomScale()

    @Test
    fun testFusePoints_confidenceWeightedAveraging() {
        // Two points in the same voxel (voxelSize = 0.015m)
        val pHigh = GeometricPoint(
            x = 0.002f, y = 0.002f, z = 1.0f,
            r = 200, g = 0, b = 0,
            confidence = 0.9f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f
        )
        val pLow = GeometricPoint(
            x = 0.010f, y = 0.010f, z = 1.0f,
            r = 0, g = 200, b = 0,
            confidence = 0.1f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f
        )

        val fused = PriorConditionedSplatPipeline.fusePoints(listOf(pHigh, pLow), config)

        assertEquals("Should fuse into single point in 1.5 cm voxel", 1, fused.size)
        val f = fused[0]

        // High confidence weight should dominate (w ~ 0.9 vs 0.1)
        // x expected ~ (0.002 * 0.9 + 0.010 * 0.1) / 1.0 = 0.0028
        assertTrue("Fused X should be closer to high confidence point (0.002)", f.x < 0.004f)
        assertTrue("Cumulative confidence must be higher than individual: 1 - 0.1*0.9 = 0.91", f.confidence > 0.90f)
        assertTrue("Color should be dominated by red", f.r > 150 && f.g < 50)
    }

    @Test
    fun testParameterizeGaussians_discLikeCovarianceAndNormalAlignment() {
        val p = GeometricPoint(
            x = 0f, y = 0f, z = 1f,
            r = 128, g = 128, b = 128,
            confidence = 0.8f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f // Normal = (0, 0, 1) -> Canonical +Z
        )

        val gaussians = PriorConditionedSplatPipeline.parameterizeGaussians(listOf(p), config)
        assertEquals(1, gaussians.size)
        val g = gaussians[0]

        // Quaternion for (0, 0, 1) normal should be identity: w=1, x=0, y=0, z=0
        assertEquals(1.0f, g.rot0, 1e-4f) // w
        assertEquals(0.0f, g.rot1, 1e-4f) // x
        assertEquals(0.0f, g.rot2, 1e-4f) // y
        assertEquals(0.0f, g.rot3, 1e-4f) // z

        // Tangential vs normal scale: s3 = 0.1 * s1 => ln(s3) = ln(s1) - ln(10)
        val s1 = exp(g.scale0)
        val s2 = exp(g.scale1)
        val s3 = exp(g.scale2)

        assertEquals("Tangential scales s1 and s2 must be equal", s1, s2, 1e-5f)
        assertEquals("Normal scale s3 must be 0.1 * s1", 0.1f * s1, s3, 1e-5f)
    }

    @Test
    fun testRotationQuaternion_surfaceTilted() {
        // Tilted normal along +X: (1, 0, 0)
        val q = PriorConditionedSplatPipeline.computeRotationQuaternion(1f, 0f, 0f)

        // Rotate +Z (0, 0, 1) using q:
        // q * (0, 0, 1) * q^-1 must yield (1, 0, 0)
        val w = q[0]; val x = q[1]; val y = q[2]; val z = q[3]

        // v' = v + 2*r x (r x v + w*v)
        // Here v = (0, 0, 1)
        val rx = y; val ry = -x; val rz = 0f // (r x v) = (x, y, z) x (0, 0, 1) = (y, -x, 0)
        val termX = rx + w * 0f
        val termY = ry + w * 0f
        val termZ = rz + w * 1f

        // r x term = (x, y, z) x (termX, termY, termZ)
        val cx = y * termZ - z * termY
        val cy = z * termX - x * termZ
        val cz = x * termY - y * termX

        val rotatedX = 0f + 2f * cx
        val rotatedY = 0f + 2f * cy
        val rotatedZ = 1f + 2f * cz

        assertEquals(1.0f, rotatedX, 1e-3f)
        assertEquals(0.0f, rotatedY, 1e-3f)
        assertEquals(0.0f, rotatedZ, 1e-3f)
    }

    @Test
    fun testWriteBinaryPly_matchesExact56ByteStrideAndParses() {
        val tempFile = File.createTempFile("splat_test_", ".ply")
        tempFile.deleteOnExit()

        val splats = listOf(
            GaussianSplat(
                x = 1.0f, y = 2.0f, z = 3.0f,
                scale0 = -3.0f, scale1 = -3.0f, scale2 = -5.3f,
                opacityLogit = 0.5f,
                rot0 = 1.0f, rot1 = 0.0f, rot2 = 0.0f, rot3 = 0.0f,
                fDc0 = 0.1f, fDc1 = 0.2f, fDc2 = 0.3f
            ),
            GaussianSplat(
                x = -1.0f, y = -2.0f, z = -3.0f,
                scale0 = -2.5f, scale1 = -2.5f, scale2 = -4.8f,
                opacityLogit = -0.5f,
                rot0 = 0.7071f, rot1 = 0.7071f, rot2 = 0.0f, rot3 = 0.0f,
                fDc0 = -0.1f, fDc1 = -0.2f, fDc2 = -0.3f
            )
        )

        PriorConditionedSplatPipeline.writeBinaryPly(splats, tempFile)

        assertTrue("Generated PLY file must exist and be non-empty", tempFile.exists() && tempFile.length() > 0)

        // Read bytes and verify
        val bytes = tempFile.readBytes()
        val headerEndStr = "end_header\n"
        val headerEndIdx = String(bytes, Charsets.US_ASCII).indexOf(headerEndStr) + headerEndStr.length
        assertTrue("PLY header must be present", headerEndIdx > headerEndStr.length)

        val binarySize = bytes.size - headerEndIdx
        val expectedBinarySize = splats.size * 56 // 14 floats * 4 bytes = 56 bytes per vertex
        assertEquals("Binary section must have exactly 56 bytes per vertex", expectedBinarySize, binarySize)

        // Verify first vertex data read as little-endian floats
        val buffer = ByteBuffer.wrap(bytes, headerEndIdx, 56).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1.0f, buffer.getFloat(), 1e-6f) // x
        assertEquals(2.0f, buffer.getFloat(), 1e-6f) // y
        assertEquals(3.0f, buffer.getFloat(), 1e-6f) // z
        assertEquals(-3.0f, buffer.getFloat(), 1e-6f) // scale0
        assertEquals(-3.0f, buffer.getFloat(), 1e-6f) // scale1
        assertEquals(-5.3f, buffer.getFloat(), 1e-6f) // scale2
        assertEquals(0.5f, buffer.getFloat(), 1e-6f) // opacityLogit
        assertEquals(1.0f, buffer.getFloat(), 1e-6f) // rot0
        assertEquals(0.0f, buffer.getFloat(), 1e-6f) // rot1
        assertEquals(0.0f, buffer.getFloat(), 1e-6f) // rot2
        assertEquals(0.0f, buffer.getFloat(), 1e-6f) // rot3
        assertEquals(0.1f, buffer.getFloat(), 1e-6f) // fDc0
        assertEquals(0.2f, buffer.getFloat(), 1e-6f) // fDc1
        assertEquals(0.3f, buffer.getFloat(), 1e-6f) // fDc2
    }

    @Test
    fun testProcessAndExportPoints_endToEndPipeline() {
        val tempFile = File.createTempFile("e2e_splat_", ".ply")
        tempFile.deleteOnExit()

        // Create 2 keyframe views observing a plane at Z = -1.0
        val eye1 = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f // Camera 1 at origin
        )
        val eye2 = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0.3f, 0f, 0f, 1f // Camera 2 shifted 0.3m right
        )

        val views = listOf(
            KeyframeView(0, eye1, fx = 500f, fy = 500f, cx = 320f, cy = 240f, width = 640, height = 480),
            KeyframeView(1, eye2, fx = 500f, fy = 500f, cx = 320f, cy = 240f, width = 640, height = 480)
        )

        // Create a 5x5 planar grid of points at Z = -1.0m (25 points)
        val points = ArrayList<GeometricPoint>()
        for (i in -2..2) {
            for (j in -2..2) {
                points.add(
                    GeometricPoint(
                        x = i * 0.02f,
                        y = j * 0.02f,
                        z = -1.0f,
                        r = 180, g = 120, b = 60,
                        confidence = 0.8f,
                        depthZ = 1.0f,
                        nx = 0f, ny = 0f, nz = 1f,
                        frameIndex = 0
                    )
                )
            }
        }

        val exportedCount = PriorConditionedSplatPipeline.processAndExportPoints(
            points = points,
            keyframeViews = views,
            outputPlyFile = tempFile,
            config = config
        )

        assertTrue("Should export valid number of Gaussians", exportedCount > 0)
        assertTrue("Output file must exist", tempFile.exists() && tempFile.length() > 0)

        val bytes = tempFile.readBytes()
        val headerEndStr = "end_header\n"
        val headerEndIdx = String(bytes, Charsets.US_ASCII).indexOf(headerEndStr) + headerEndStr.length
        val binarySize = bytes.size - headerEndIdx
        assertEquals("Binary payload must be exactly count * 56 bytes", exportedCount * 56, binarySize)
    }

    @Test
    fun testProcessAndExportPointsProgressReporting() {
        val tempFile = File.createTempFile("test_splat_progress", ".ply")
        tempFile.deleteOnExit()

        val config = PriorConditionedSplatConfig.roomScale()
        val eye1 = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f
        )
        val views = listOf(
            KeyframeView(0, eye1, fx = 500f, fy = 500f, cx = 320f, cy = 240f, width = 640, height = 480)
        )

        val points = listOf(
            GeometricPoint(
                x = 0f, y = 0f, z = -1.0f,
                r = 100, g = 100, b = 100,
                confidence = 0.9f,
                depthZ = 1.0f,
                nx = 0f, ny = 0f, nz = 1f,
                frameIndex = 0
            )
        )

        val progressValues = mutableListOf<Float>()
        PriorConditionedSplatPipeline.processAndExportPoints(
            points = points,
            keyframeViews = views,
            outputPlyFile = tempFile,
            config = config,
            onProgress = { p -> progressValues.add(p) }
        )

        assertTrue("Progress updates should have been captured", progressValues.isNotEmpty())
        assertTrue("Initial progress should be <= 0.25f", progressValues.first() <= 0.25f)
        assertEquals("Final progress milestone should be 0.85f", 0.85f, progressValues.last(), 1e-4f)

        for (i in 0 until progressValues.size - 1) {
            assertTrue(
                "Progress should be non-decreasing: ${progressValues[i]} <= ${progressValues[i + 1]}",
                progressValues[i] <= progressValues[i + 1]
            )
        }
    }

    @Test
    fun testParameterizeGaussians_nonPlanarIsotropicSplats() {
        val pNonPlanar = GeometricPoint(
            x = 0.1f, y = 0.2f, z = 1.0f,
            r = 100, g = 150, b = 200,
            confidence = 0.7f,
            depthZ = 1.0f,
            nx = 0f, ny = 0f, nz = 1f,
            isPlanar = false
        )

        val splats = PriorConditionedSplatPipeline.parameterizeGaussians(listOf(pNonPlanar), config)
        assertEquals(1, splats.size)
        val s = splats[0]

        val s1 = exp(s.scale0)
        val s2 = exp(s.scale1)
        val s3 = exp(s.scale2)

        assertEquals("Non-planar splats must have isotropic scale: s1 == s2", s1, s2, 1e-4f)
        assertEquals("Non-planar splats must have isotropic scale: s1 == s3", s1, s3, 1e-4f)

        assertEquals("Non-planar splats must have identity rotation w=1", 1.0f, s.rot0, 1e-4f)
        assertEquals("Non-planar splats must have identity rotation x=0", 0.0f, s.rot1, 1e-4f)
        assertEquals("Non-planar splats must have identity rotation y=0", 0.0f, s.rot2, 1e-4f)
        assertEquals("Non-planar splats must have identity rotation z=0", 0.0f, s.rot3, 1e-4f)
    }

    @Test
    fun testAdaptiveVoxelSize_scaling() {
        // Close range (0.3m) with high texture -> should clamp to min 4mm
        val vCloseHighTex = config.computeAdaptiveVoxelSize(depthZ = 0.3f, textureVariance = 50f)
        assertEquals(0.004f, vCloseHighTex, 1e-4f)

        // Mid range (1.5m) with medium texture
        val vMidMedTex = config.computeAdaptiveVoxelSize(depthZ = 1.5f, textureVariance = 20f)
        assertTrue("Mid range should be between 10mm and 20mm, was $vMidMedTex", vMidMedTex in 0.010f..0.020f)

        // Far range (3.0m) with low texture -> should reach or clamp to max 25mm
        val vFarLowTex = config.computeAdaptiveVoxelSize(depthZ = 3.0f, textureVariance = 5f)
        assertEquals(0.025f, vFarLowTex, 1e-4f)
    }
}

