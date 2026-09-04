package io.github.sceneview.demo.demos.internal

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Prior-conditioned 3D Gaussian Splat representation matching Brush's exact 14 float layout:
 * x, y, z, scale_0, scale_1, scale_2, opacity, rot_0, rot_1, rot_2, rot_3, f_dc_0, f_dc_1, f_dc_2.
 * Exactly 56 bytes per vertex, little-endian binary PLY.
 */
data class GaussianSplat(
    val x: Float,
    val y: Float,
    val z: Float,
    val scale0: Float, // ln(s_x)
    val scale1: Float, // ln(s_y)
    val scale2: Float, // ln(s_z)
    val opacityLogit: Float,
    val rot0: Float,   // w (scalar-first)
    val rot1: Float,   // x
    val rot2: Float,   // y
    val rot3: Float,   // z
    val fDc0: Float,   // SH DC red
    val fDc1: Float,   // SH DC green
    val fDc2: Float    // SH DC blue
)

object PriorConditionedSplatPipeline {

    private const val SH_C0 = 0.28209479177387814f

    /**
     * Phase 5 Temporal Fusion:
     * Fuses multi-observation points in spatial voxels via confidence-weighted spatial averaging:
     * p_fused = sum(w_i * p_i) / sum(w_i), where w_i = c_i / (sigma_z(z_i)^2).
     */
    fun fusePoints(
        points: List<GeometricPoint>,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale()
    ): List<GeometricPoint> {
        if (points.isEmpty()) return emptyList()

        val voxelMap = HashMap<Long, VoxelAccumulator>()
        val mask = 0x1FFFFFL

        for (p in points) {
            val voxelSize = config.computeAdaptiveVoxelSize(p.depthZ)
            val xi = (p.x / voxelSize).toInt().toLong()
            val yi = (p.y / voxelSize).toInt().toLong()
            val zi = (p.z / voxelSize).toInt().toLong()
            val key = ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
            val acc = voxelMap.getOrPut(key) { VoxelAccumulator() }

            val sigmaZ = config.computeDepthNoise(p.depthZ)
            val w = (p.confidence / (sigmaZ * sigmaZ)).coerceIn(1e-4f, 1e6f)

            acc.add(p, w)
        }

        val fusedList = ArrayList<GeometricPoint>(voxelMap.size)
        for (acc in voxelMap.values) {
            val fused = acc.toGeometricPoint()
            if (fused != null) {
                fusedList.add(fused)
            }
        }
        return fusedList
    }

    private class VoxelAccumulator {
        var sumW = 0.0
        var sumX = 0.0; var sumY = 0.0; var sumZ = 0.0
        var sumNx = 0.0; var sumNy = 0.0; var sumNz = 0.0
        var sumR = 0.0; var sumG = 0.0; var sumB = 0.0
        var sumDepthZ = 0.0
        var count = 0
        var planarCount = 0
        var prodNotConf = 1.0

        fun add(p: GeometricPoint, weight: Float) {
            val w = weight.toDouble()
            sumW += w
            sumX += p.x * w
            sumY += p.y * w
            sumZ += p.z * w

            sumNx += p.nx * w
            sumNy += p.ny * w
            sumNz += p.nz * w

            sumR += p.r * w
            sumG += p.g * w
            sumB += p.b * w

            sumDepthZ += p.depthZ * w

            val c = p.confidence.toDouble().coerceIn(0.0, 0.999)
            prodNotConf *= (1.0 - c)

            if (p.isPlanar) planarCount++
            count++
        }

        fun toGeometricPoint(): GeometricPoint? {
            if (count == 0 || sumW <= 0.0) return null

            val fx = (sumX / sumW).toFloat()
            val fy = (sumY / sumW).toFloat()
            val fz = (sumZ / sumW).toFloat()

            var fnx = (sumNx / sumW).toFloat()
            var fny = (sumNy / sumW).toFloat()
            var fnz = (sumNz / sumW).toFloat()
            val nLen = sqrt(fnx * fnx + fny * fny + fnz * fnz)
            if (nLen > 1e-4f) {
                fnx /= nLen; fny /= nLen; fnz /= nLen
            } else {
                fnx = 0f; fny = 0f; fnz = 1f
            }

            val fr = (sumR / sumW).roundToInt().coerceIn(0, 255)
            val fg = (sumG / sumW).roundToInt().coerceIn(0, 255)
            val fb = (sumB / sumW).roundToInt().coerceIn(0, 255)

            val fDepthZ = (sumDepthZ / sumW).toFloat()
            val fConf = (1.0 - prodNotConf).toFloat().coerceIn(0f, 1f)

            return GeometricPoint(
                x = fx, y = fy, z = fz,
                r = fr, g = fg, b = fb,
                confidence = fConf,
                depthZ = fDepthZ,
                nx = fnx, ny = fny, nz = fnz,
                isPlanar = (planarCount >= (count + 1) / 2)
            )
        }
    }

    /**
     * Operator I: Prior-conditioned 3D Gaussian Splat parameterization:
     * - Tangential scale from K_s=3 nearest neighbors: s_1 = s_2 = d_knn3
     * - Normal scale: s_3 = 0.1 * s_1
     * - Quaternion aligning canonical +Z to normal n (scalar-first w, x, y, z)
     * - RGB to spherical harmonics DC color
     * - Initial opacity to logit
     */
    fun parameterizeGaussians(
        fusedPoints: List<GeometricPoint>,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale(),
        onProgress: ((progress: Float) -> Unit)? = null
    ): List<GaussianSplat> {
        if (fusedPoints.isEmpty()) return emptyList()

        val n = fusedPoints.size
        val kNeighbors = min(config.knnScaleK, max(1, n - 1))

        // Simple spatial grid to estimate local KNN scale
        val cellSize = config.voxelSizeMeters * 3f
        val grid = HashMap<Long, MutableList<Int>>()
        val mask = 0x1FFFFFL

        fun hash(x: Float, y: Float, z: Float): Long {
            val xi = (x / cellSize).toInt().toLong()
            val yi = (y / cellSize).toInt().toLong()
            val zi = (z / cellSize).toInt().toLong()
            return ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
        }

        for (i in fusedPoints.indices) {
            val p = fusedPoints[i]
            grid.getOrPut(hash(p.x, p.y, p.z)) { ArrayList(8) }.add(i)
        }

        val result = ArrayList<GaussianSplat>(n)
        val reportInterval = max(250, n / 20)

        for (i in fusedPoints.indices) {
            if (onProgress != null && (i % reportInterval == 0 || i == n - 1)) {
                onProgress((i + 1).toFloat() / n)
            }
            val p = fusedPoints[i]

            // Find local nearest neighbors
            val xi = (p.x / cellSize).toInt()
            val yi = (p.y / cellSize).toInt()
            val zi = (p.z / cellSize).toInt()

            val dists = ArrayList<Float>(16)
            for (dx in -1..1) {
                for (dy in -1..1) {
                    for (dz in -1..1) {
                        val key = (((xi + dx).toLong() and mask) shl 42) or
                                (((yi + dy).toLong() and mask) shl 21) or
                                ((zi + dz).toLong() and mask)
                        val cellPoints = grid[key] ?: continue
                        for (idx in cellPoints) {
                            if (idx == i) continue
                            val np = fusedPoints[idx]
                            val d = sqrt((np.x - p.x) * (np.x - p.x) + (np.y - p.y) * (np.y - p.y) + (np.z - p.z) * (np.z - p.z))
                            dists.add(d)
                        }
                    }
                }
            }

            dists.sort()
            val meanDist = if (dists.isNotEmpty()) {
                val takeK = min(kNeighbors, dists.size)
                var sum = 0f
                for (k in 0 until takeK) sum += dists[k]
                sum / takeK
            } else {
                config.computeAdaptiveVoxelSize(p.depthZ)
            }.coerceIn(0.002f, 0.08f)

            val s1: Float
            val s2: Float
            val s3: Float
            val q: FloatArray

            if (p.isPlanar) {
                // Disc-like covariance: s1 = s2 = meanDist, s3 = 0.1 * s1
                s1 = meanDist
                s2 = meanDist
                s3 = 0.1f * s1
                q = computeRotationQuaternion(p.nx, p.ny, p.nz)
            } else {
                // Non-planar geometry (edges, curves, thin structures): isotropic sphere with identity quaternion
                val sIso = 0.8f * meanDist
                s1 = sIso
                s2 = sIso
                s3 = sIso
                q = floatArrayOf(1f, 0f, 0f, 0f)
            }

            val scale0 = ln(s1)
            val scale1 = ln(s2)
            val scale2 = ln(s3)

            // RGB to Spherical Harmonics DC
            val fDc0 = (p.r / 255f - 0.5f) / SH_C0
            val fDc1 = (p.g / 255f - 0.5f) / SH_C0
            val fDc2 = (p.b / 255f - 0.5f) / SH_C0

            // Continuous confidence to opacity logit
            val opacity = config.computeInitialOpacity(p.confidence)
            val opacityLogit = config.computeOpacityLogit(opacity)

            result.add(
                GaussianSplat(
                    x = p.x, y = p.y, z = p.z,
                    scale0 = scale0, scale1 = scale1, scale2 = scale2,
                    opacityLogit = opacityLogit,
                    rot0 = q[0], rot1 = q[1], rot2 = q[2], rot3 = q[3],
                    fDc0 = fDc0, fDc1 = fDc1, fDc2 = fDc2
                )
            )
        }

        return result
    }

    /**
     * Compute unit quaternion (w, x, y, z) rotating canonical +Z (0, 0, 1) to target normal (nx, ny, nz).
     */
    internal fun computeRotationQuaternion(nx: Float, ny: Float, nz: Float): FloatArray {
        // Dot product with (0, 0, 1) is nz
        val dot = nz.coerceIn(-1f, 1f)

        if (dot > 0.9999f) {
            // Identity rotation
            return floatArrayOf(1f, 0f, 0f, 0f)
        }
        if (dot < -0.9999f) {
            // 180 deg rotation about X axis
            return floatArrayOf(0f, 1f, 0f, 0f)
        }

        // Half-angle formula: w = cos(theta/2) = sqrt((1 + dot) / 2)
        val w = sqrt((1f + dot) / 2f)
        // Vector part = (z_axis x n) / (2 * w) = (-ny, nx, 0) / (2 * w)
        val inv2w = 1f / (2f * w)
        val x = -ny * inv2w
        val y = nx * inv2w
        val z = 0f

        val len = sqrt(w * w + x * x + y * y + z * z)
        return if (len > 1e-6f) {
            floatArrayOf(w / len, x / len, y / len, z / len)
        } else {
            floatArrayOf(1f, 0f, 0f, 0f)
        }
    }

    /**
     * Serialize 3D Gaussians into binary little-endian PLY matching Brush's exact 14-float layout:
     * 56 bytes per vertex.
     */
    fun writeBinaryPly(gaussians: List<GaussianSplat>, outputFile: File) {
        val count = gaussians.size
        val header = "ply\n" +
                "format binary_little_endian 1.0\n" +
                "element vertex $count\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property float scale_0\n" +
                "property float scale_1\n" +
                "property float scale_2\n" +
                "property float opacity\n" +
                "property float rot_0\n" +
                "property float rot_1\n" +
                "property float rot_2\n" +
                "property float rot_3\n" +
                "property float f_dc_0\n" +
                "property float f_dc_1\n" +
                "property float f_dc_2\n" +
                "end_header\n"

        val bufferSize = 56 * 1024 // 1024 vertices per chunk (56 KB)
        val byteBuffer = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN)

        BufferedOutputStream(FileOutputStream(outputFile), 64 * 1024).use { out ->
            out.write(header.toByteArray(Charsets.US_ASCII))

            for (g in gaussians) {
                if (byteBuffer.remaining() < 56) {
                    out.write(byteBuffer.array(), 0, byteBuffer.position())
                    byteBuffer.clear()
                }

                byteBuffer.putFloat(g.x)
                byteBuffer.putFloat(g.y)
                byteBuffer.putFloat(g.z)
                byteBuffer.putFloat(g.scale0)
                byteBuffer.putFloat(g.scale1)
                byteBuffer.putFloat(g.scale2)
                byteBuffer.putFloat(g.opacityLogit)
                byteBuffer.putFloat(g.rot0)
                byteBuffer.putFloat(g.rot1)
                byteBuffer.putFloat(g.rot2)
                byteBuffer.putFloat(g.rot3)
                byteBuffer.putFloat(g.fDc0)
                byteBuffer.putFloat(g.fDc1)
                byteBuffer.putFloat(g.fDc2)
            }

            if (byteBuffer.position() > 0) {
                out.write(byteBuffer.array(), 0, byteBuffer.position())
            }
        }
    }

    /**
     * Complete multi-stage execution on points:
     * Phase 1 Geometric Filter -> Phase 3/4 Multi-View Depth Refiner -> Phase 5 Temporal Fusion -> Parameterize Gaussians -> Binary PLY.
     */
    fun processAndExportPoints(
        points: List<GeometricPoint>,
        keyframeViews: List<KeyframeView>,
        outputPlyFile: File,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale(),
        luminanceLookup: (KeyframeView) -> LuminanceProvider? = { null },
        onKeyframesUpdated: ((List<KeyframeView>) -> Unit)? = null,
        onProgress: ((progress: Float) -> Unit)? = null
    ): Int {
        var lastProgress = 0f
        val emitProgress: (Float) -> Unit = { p ->
            val v = max(lastProgress, p).coerceIn(0f, 1f)
            lastProgress = v
            onProgress?.invoke(v)
        }

        if (points.isEmpty()) {
            emitProgress(0.85f)
            writeBinaryPly(emptyList(), outputPlyFile)
            return 0
        }

        emitProgress(0.05f)

        // Pre-decimate raw input points if very large (> 60,000) using distance-adaptive voxelization
        val inputPoints = if (points.size > 60000) {
            val bestInVoxel = HashMap<Long, GeometricPoint>(60000)
            val mask = 0x1FFFFFL
            for (p in points) {
                val voxelSize = config.computeAdaptiveVoxelSize(p.depthZ)
                val xi = (p.x / voxelSize).toInt().toLong()
                val yi = (p.y / voxelSize).toInt().toLong()
                val zi = (p.z / voxelSize).toInt().toLong()
                val key = ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
                val existing = bestInVoxel[key]
                if (existing == null || p.confidence > existing.confidence) {
                    bestInVoxel[key] = p
                }
            }
            bestInVoxel.values.toList()
        } else {
            points
        }

        // Phase 1: Cheap Geometric Filtering (plane residual + SOR)
        val geometricFiltered = CheapGeometricFilter.filterPointSet(
            points = inputPoints,
            config = config,
            onProgress = { current, total ->
                val frac = if (total > 0) current.toFloat() / total else 1f
                emitProgress(min(0.20f, 0.05f + 0.15f * frac))
            }
        )
        emitProgress(0.20f)
        if (geometricFiltered.isEmpty()) {
            emitProgress(0.85f)
            writeBinaryPly(emptyList(), outputPlyFile)
            return 0
        }

        // Phase 3 & 4: Multi-View Consistency and Narrow-Range Depth Refinement
        val refiner = MultiViewDepthRefiner(config)
        val refinedPoints = refiner.refinePointSet(
            points = geometricFiltered,
            keyframeViews = keyframeViews,
            luminanceLookup = luminanceLookup,
            onProgress = { current, total ->
                val frac = if (total > 0) current.toFloat() / total else 1f
                emitProgress(min(0.50f, 0.20f + 0.30f * frac))
            }
        )
        emitProgress(0.50f)
        if (refinedPoints.isEmpty()) {
            emitProgress(0.85f)
            writeBinaryPly(emptyList(), outputPlyFile)
            return 0
        }

        // Phase 5: Temporal Fusion
        val fusedPoints = fusePoints(refinedPoints, config)
        emitProgress(0.55f)
        if (fusedPoints.isEmpty()) {
            emitProgress(0.85f)
            writeBinaryPly(emptyList(), outputPlyFile)
            return 0
        }

        // Phase B: Ceres Solver Native Bundle Adjustment
        var baRefinedPoints = fusedPoints
        if (CeresNativeBridge.isAvailable && keyframeViews.size >= 2) {
            val baResult = CeresNativeBridge.runBundleAdjustment(
                keyframes = keyframeViews,
                candidatePoints = fusedPoints,
                targetAnchors = config.ceresAnchorPoints,
                maxIterations = 25,
                onProgress = { baFrac ->
                    emitProgress(min(0.72f, 0.55f + 0.17f * baFrac))
                }
            )
            if (baResult.success) {
                onKeyframesUpdated?.invoke(baResult.updatedKeyframes)
                if (baResult.updatedAnchorPoints.isNotEmpty()) {
                    if (baResult.updatedAnchorIndices.size == baResult.updatedAnchorPoints.size) {
                        val updatedList = fusedPoints.toMutableList()
                        for (i in baResult.updatedAnchorIndices.indices) {
                            val targetIdx = baResult.updatedAnchorIndices[i]
                            if (targetIdx in updatedList.indices) {
                                updatedList[targetIdx] = baResult.updatedAnchorPoints[i]
                            }
                        }
                        baRefinedPoints = updatedList
                    } else {
                        val mask = 0x1FFFFFL
                        val voxelSize = config.voxelSizeMeters
                        val anchorMap = HashMap<Long, GeometricPoint>(baResult.updatedAnchorPoints.size)
                        for (a in baResult.updatedAnchorPoints) {
                            val xi = (a.x / voxelSize).toInt().toLong()
                            val yi = (a.y / voxelSize).toInt().toLong()
                            val zi = (a.z / voxelSize).toInt().toLong()
                            val key = ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
                            anchorMap[key] = a
                        }
                        baRefinedPoints = fusedPoints.map { p ->
                            val xi = (p.x / voxelSize).toInt().toLong()
                            val yi = (p.y / voxelSize).toInt().toLong()
                            val zi = (p.z / voxelSize).toInt().toLong()
                            val key = ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
                            anchorMap[key] ?: p
                        }
                    }
                }
            }
        }
        emitProgress(0.72f)

        // Prior-conditioned Gaussian parameterization (PocketGS Operator I)
        val gaussians = parameterizeGaussians(
            fusedPoints = baRefinedPoints,
            config = config,
            onProgress = { paramFrac ->
                emitProgress(min(0.83f, 0.72f + 0.11f * paramFrac))
            }
        )
        emitProgress(0.83f)

        // Serialize 56-byte binary little-endian PLY matching Brush's exact Splats layout
        writeBinaryPly(gaussians, outputPlyFile)
        emitProgress(0.85f)
        return gaussians.size
    }

    /**
     * Unprojects raw points from captured frames, executes the full pipeline, and writes binary PLY.
     */
    fun processAndExport(
        frames: List<io.github.sceneview.demo.demos.CapturedFrameData>,
        tempDir: File,
        outputPlyFile: File,
        imageWidth: Int = 0,
        imageHeight: Int = 0,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale(),
        luminanceLoader: ((File) -> LuminanceProvider?)? = null,
        onKeyframesUpdated: ((List<KeyframeView>) -> Unit)? = null,
        onProgress: ((progress: Float) -> Unit)? = null
    ): Int {
        onProgress?.invoke(0.01f)
        val keyframeViews = ArrayList<KeyframeView>(frames.size)
        val rawPoints = ArrayList<GeometricPoint>()

        for ((frameIdx, frameData) in frames.withIndex()) {
            if (frames.isNotEmpty() && onProgress != null && (frameIdx % 5 == 0 || frameIdx == frames.size - 1)) {
                onProgress(min(0.05f, 0.01f + 0.04f * (frameIdx + 1) / frames.size))
            }
            val anchorMatrix = FloatArray(16)
            frameData.anchor.pose.toMatrix(anchorMatrix, 0)
            val imgFile = File(tempDir, frameData.filePath)
            keyframeViews.add(
                KeyframeView(
                    index = frameIdx,
                    cameraToWorld = anchorMatrix,
                    fx = frameData.fl_x,
                    fy = frameData.fl_y,
                    cx = frameData.cx,
                    cy = frameData.cy,
                    imageFile = imgFile,
                    width = imageWidth,
                    height = imageHeight
                )
            )

            val pts = frameData.localPoints
            for (i in 0 until pts.size) {
                val i3 = i * 3
                val px = pts.xyz[i3]
                val py = pts.xyz[i3 + 1]
                val pz = pts.xyz[i3 + 2]
                val pr = pts.rgb[i3].toInt() and 0xFF
                val pg = pts.rgb[i3 + 1].toInt() and 0xFF
                val pb = pts.rgb[i3 + 2].toInt() and 0xFF
                val conf = pts.confidence[i]
                val dist = pts.distance[i]

                val xWorld = anchorMatrix[0] * px + anchorMatrix[4] * py + anchorMatrix[8] * pz + anchorMatrix[12]
                val yWorld = anchorMatrix[1] * px + anchorMatrix[5] * py + anchorMatrix[9] * pz + anchorMatrix[13]
                val zWorld = anchorMatrix[2] * px + anchorMatrix[6] * py + anchorMatrix[10] * pz + anchorMatrix[14]

                if (!xWorld.isFinite() || !yWorld.isFinite() || !zWorld.isFinite() || dist <= 0.05f) {
                    continue
                }

                rawPoints.add(
                    GeometricPoint(
                        x = xWorld,
                        y = yWorld,
                        z = zWorld,
                        r = pr,
                        g = pg,
                        b = pb,
                        confidence = conf,
                        depthZ = dist,
                        frameIndex = frameIdx
                    )
                )
            }
        }

        val loader = luminanceLoader ?: { file -> MultiViewDepthRefiner.loadLuminanceFromFile(file) }
        val lruCache = LruLuminanceCache(maxResident = config.maxResidentLuminanceFrames, loader = loader)

        return processAndExportPoints(
            points = rawPoints,
            keyframeViews = keyframeViews,
            outputPlyFile = outputPlyFile,
            config = config,
            luminanceLookup = { view -> view.imageFile?.let { lruCache.get(it) } },
            onKeyframesUpdated = onKeyframesUpdated,
            onProgress = onProgress
        )
    }
}
