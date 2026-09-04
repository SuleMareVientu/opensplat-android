package io.github.sceneview.demo.demos.internal

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Geometric point representation carrying continuous confidence, color, depth, and normal.
 */
data class GeometricPoint(
    val x: Float,
    val y: Float,
    val z: Float,
    val r: Int,
    val g: Int,
    val b: Int,
    val confidence: Float,
    val depthZ: Float,
    var nx: Float = 0f,
    var ny: Float = 0f,
    var nz: Float = 1f,
    var residual: Float = 0f,
    var frameIndex: Int = -1,
    var isPlanar: Boolean = true
)

/**
 * Phase 1 Cheap Geometric Filtering:
 * 1. In-loop edge detection & continuous confidence normalization.
 * 2. Export-phase spatial neighbor search, K=16 local plane-fit residual filter, and SOR.
 */
object CheapGeometricFilter {

    /**
     * In-loop filter: checks depth discontinuity across neighbor raw-depth pixels
     * and normalizes raw confidence byte into continuous [0.0, 1.0] confidence.
     *
     * @return continuous confidence in [0f, 1f], or 0f if rejected as an edge artifact.
     */
    fun filterEdgeAndNormalizeConfidence(
        u: Int,
        v: Int,
        rawDepthMm: Int,
        confByte: Int,
        depthWidth: Int,
        depthHeight: Int,
        rawDepthArray: ShortArray,
        luminanceGradient: Float,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale()
    ): Float {
        if (rawDepthMm <= 0 || confByte <= 0) return 0f

        val dCenter = rawDepthMm.toFloat()
        var maxNeighborDiff = 0f
        var validNeighbors = 0

        // Check 4-connected spatial neighbors in raw depth map
        val offsets = intArrayOf(-1, 0, 1, 0, 0, -1, 0, 1)
        for (i in 0 until 4) {
            val nu = u + offsets[i * 2]
            val nv = v + offsets[i * 2 + 1]
            if (nu in 0 until depthWidth && nv in 0 until depthHeight) {
                val nDepth = rawDepthArray[nv * depthWidth + nu].toInt() and 0xFFFF
                if (nDepth > 0) {
                    val diff = abs(nDepth - dCenter)
                    if (diff > maxNeighborDiff) {
                        maxNeighborDiff = diff
                    }
                    validNeighbors++
                }
            }
        }

        // Relative depth jump: J = max(|d_neighbor - d|) / d
        val jump = if (dCenter > 0f) maxNeighborDiff / dCenter else 0f

        // High relative depth discontinuity with high luminance gradient indicates a flying pixel
        val isDepthDiscontinuity = jump > config.depthJumpThreshold
        val isHighLuminanceEdge = luminanceGradient > config.luminanceGradientSigma * 1.5f

        // Extreme jump (> 15%) or combined depth jump + edge -> reject completely
        if (jump > 0.15f || (isDepthDiscontinuity && isHighLuminanceEdge)) {
            return 0f
        }

        // Soft edge down-weighting: w = exp(-||grad||^2 / (2 * sigma^2))
        val sigma = config.luminanceGradientSigma
        val wEdge = exp(-(luminanceGradient * luminanceGradient) / (2f * sigma * sigma)).coerceIn(0.1f, 1.0f)

        // Continuous confidence in [0.0, 1.0]
        val rawNorm = (confByte.coerceIn(0, 255) / 255f)
        return (rawNorm * wEdge).coerceIn(0f, 1f)
    }

    /**
     * Fallback confidence when raw depth is absent but smoothed depth is available.
     * Assigned a lower base confidence (0.3) scaled by luminance edge weight.
     */
    fun fallbackSmoothedConfidence(
        smoothedDepthMm: Int,
        luminanceGradient: Float,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale()
    ): Float {
        if (smoothedDepthMm <= 0) return 0f
        val sigma = config.luminanceGradientSigma
        val wEdge = exp(-(luminanceGradient * luminanceGradient) / (2f * sigma * sigma)).coerceIn(0.1f, 1.0f)
        return (0.30f * wEdge).coerceIn(0f, 1f)
    }

    /**
     * Spatial hash grid for fast neighbor lookup during export.
     */
    /**
     * Spatial hash grid for fast neighbor lookup during export.
     */
    private class SpatialGrid(val cellSize: Float) {
        private val grid = HashMap<Long, MutableList<Int>>()

        private fun hash(x: Float, y: Float, z: Float): Long {
            val xi = (x / cellSize).toInt().toLong()
            val yi = (y / cellSize).toInt().toLong()
            val zi = (z / cellSize).toInt().toLong()
            val mask = 0x1FFFFFL
            return ((xi and mask) shl 42) or ((yi and mask) shl 21) or (zi and mask)
        }

        fun insert(index: Int, x: Float, y: Float, z: Float) {
            val key = hash(x, y, z)
            val list = grid.getOrPut(key) { ArrayList(8) }
            if (list.size < 32) {
                list.add(index)
            }
        }

        fun findNeighbors(
            qx: Float,
            qy: Float,
            qz: Float,
            points: List<GeometricPoint>,
            maxK: Int
        ): IntArray {
            val xi = (qx / cellSize).toInt()
            val yi = (qy / cellSize).toInt()
            val zi = (qz / cellSize).toInt()
            val mask = 0x1FFFFFL

            val bestIndices = IntArray(maxK)
            val bestSqDists = FloatArray(maxK)
            var count = 0

            // Search 3x3x3 adjacent neighborhood
            for (dx in -1..1) {
                for (dy in -1..1) {
                    for (dz in -1..1) {
                        val key = (((xi + dx).toLong() and mask) shl 42) or
                                (((yi + dy).toLong() and mask) shl 21) or
                                ((zi + dz).toLong() and mask)
                        val cellPoints = grid[key] ?: continue
                        val cSize = cellPoints.size
                        for (i in 0 until cSize) {
                            val idx = cellPoints[i]
                            val p = points[idx]
                            val d2 = (p.x - qx) * (p.x - qx) + (p.y - qy) * (p.y - qy) + (p.z - qz) * (p.z - qz)

                            if (count < maxK) {
                                var insertPos = count
                                while (insertPos > 0 && bestSqDists[insertPos - 1] > d2) {
                                    bestSqDists[insertPos] = bestSqDists[insertPos - 1]
                                    bestIndices[insertPos] = bestIndices[insertPos - 1]
                                    insertPos--
                                }
                                bestSqDists[insertPos] = d2
                                bestIndices[insertPos] = idx
                                count++
                            } else if (d2 < bestSqDists[maxK - 1]) {
                                var insertPos = maxK - 1
                                while (insertPos > 0 && bestSqDists[insertPos - 1] > d2) {
                                    bestSqDists[insertPos] = bestSqDists[insertPos - 1]
                                    bestIndices[insertPos] = bestIndices[insertPos - 1]
                                    insertPos--
                                }
                                bestSqDists[insertPos] = d2
                                bestIndices[insertPos] = idx
                            }
                        }
                    }
                }
            }

            return if (count == maxK) bestIndices else bestIndices.copyOf(count)
        }
    }

    /**
     * Export-phase pipeline:
     * 1. Fits local plane across K=16 neighbors, computes surface normal and residual.
     * 2. Prunes points where residual r > beta * sigma_z(depthZ).
     * 3. Statistical Outlier Removal (SOR) based on mean distance to neighbors.
     */
    fun filterPointSet(
        points: List<GeometricPoint>,
        config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale(),
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): List<GeometricPoint> {
        if (points.size < config.knnK) return points

        val grid = SpatialGrid(cellSize = config.voxelSizeMeters * 3f)
        for (i in points.indices) {
            val p = points[i]
            grid.insert(i, p.x, p.y, p.z)
        }

        // Pass 1: Local plane fitting and residual thresholding
        val total = points.size
        val planeFiltered = ArrayList<GeometricPoint>(total)
        val meanDistances = FloatArray(total)
        var meanCount = 0

        for (i in points.indices) {
            if (onProgress != null && (i % 250 == 0 || i == total - 1)) {
                onProgress(i + 1, total)
            }
            val p = points[i]
            val neighborIndices = grid.findNeighbors(p.x, p.y, p.z, points, config.knnK)
            if (neighborIndices.size < 4) {
                // Isolated noise floater (fewer than 4 points in a 13.5 cm neighborhood)
                continue
            }

            val k = neighborIndices.size
            var cx = 0f; var cy = 0f; var cz = 0f
            var distSum = 0f
            for (idx in neighborIndices) {
                val np = points[idx]
                cx += np.x; cy += np.y; cz += np.z
                val d = sqrt((np.x - p.x) * (np.x - p.x) + (np.y - p.y) * (np.y - p.y) + (np.z - p.z) * (np.z - p.z))
                distSum += d
            }
            cx /= k; cy /= k; cz /= k

            // 3x3 Covariance matrix
            var c00 = 0f; var c01 = 0f; var c02 = 0f
            var c11 = 0f; var c12 = 0f; var c22 = 0f
            for (idx in neighborIndices) {
                val np = points[idx]
                val dx = np.x - cx; val dy = np.y - cy; val dz = np.z - cz
                c00 += dx * dx; c01 += dx * dy; c02 += dx * dz
                c11 += dy * dy; c12 += dy * dz
                c22 += dz * dz
            }
            c00 /= k; c01 /= k; c02 /= k
            c11 /= k; c12 /= k; c22 /= k

            // Minimum eigenvector is plane normal
            val normal = computeSmallestEigenvector(c00, c01, c02, c11, c12, c22)

            // Residual r_i = |n . (p - c)|
            val rx = p.x - cx; val ry = p.y - cy; val rz = p.z - cz
            val residual = abs(normal[0] * rx + normal[1] * ry + normal[2] * rz)

            // Sensor noise threshold
            val sigmaZ = config.computeDepthNoise(p.depthZ)
            val threshold = config.planeResidualBeta * sigmaZ

            p.nx = normal[0]
            p.ny = normal[1]
            p.nz = normal[2]
            p.residual = residual
            p.isPlanar = (residual <= threshold)

            planeFiltered.add(p)
            meanDistances[meanCount++] = distSum / k
        }

        if (planeFiltered.size < 4 || meanCount == 0) return planeFiltered

        // Pass 2: Statistical Outlier Removal (SOR) on surviving points
        var sumD = 0.0
        for (i in 0 until meanCount) {
            sumD += meanDistances[i]
        }
        val muD = sumD / meanCount

        var sumSqDiff = 0.0
        for (i in 0 until meanCount) {
            val diff = meanDistances[i] - muD
            sumSqDiff += diff * diff
        }
        val stdD = sqrt(sumSqDiff / meanCount)
        val sorThreshold = (muD + config.sorStdDevMultiplier * stdD).toFloat()

        val finalPoints = ArrayList<GeometricPoint>(planeFiltered.size)
        for (i in planeFiltered.indices) {
            if (meanDistances[i] <= sorThreshold) {
                finalPoints.add(planeFiltered[i])
            }
        }

        return finalPoints
    }

    /**
     * Compute eigenvector corresponding to smallest eigenvalue of 3x3 symmetric matrix:
     * [ c00  c01  c02 ]
     * [ c01  c11  c12 ]
     * [ c02  c12  c22 ]
     *
     * Using Jacobi rotation method (converges in <= 8 sweeps for 3x3).
     */
    internal fun computeSmallestEigenvector(
        c00: Float, c01: Float, c02: Float,
        c11: Float, c12: Float,
        c22: Float
    ): FloatArray {
        // A matrix
        var a00 = c00.toDouble(); var a01 = c01.toDouble(); var a02 = c02.toDouble()
        var a11 = c11.toDouble(); var a12 = c12.toDouble()
        var a22 = c22.toDouble()

        // V matrix (eigenvectors, initially identity)
        var v00 = 1.0; var v01 = 0.0; var v02 = 0.0
        var v10 = 0.0; var v11 = 1.0; var v12 = 0.0
        var v20 = 0.0; var v21 = 0.0; var v22 = 1.0

        for (sweep in 0 until 10) {
            val offDiag = abs(a01) + abs(a02) + abs(a12)
            if (offDiag < 1e-12) break

            // Rotate (0, 1)
            if (abs(a01) > 1e-14) {
                val theta = 0.5 * (a11 - a00) / a01
                val t = if (theta >= 0.0) 1.0 / (theta + sqrt(theta * theta + 1.0)) else -1.0 / (-theta + sqrt(theta * theta + 1.0))
                val c = 1.0 / sqrt(t * t + 1.0)
                val s = t * c
                val a00New = a00 - t * a01
                val a11New = a11 + t * a01
                val a02New = c * a02 - s * a12
                val a12New = s * a02 + c * a12
                a00 = a00New; a11 = a11New; a02 = a02New; a12 = a12New; a01 = 0.0

                val v00N = c * v00 - s * v01; val v01N = s * v00 + c * v01
                val v10N = c * v10 - s * v11; val v11N = s * v10 + c * v11
                val v20N = c * v20 - s * v21; val v21N = s * v20 + c * v21
                v00 = v00N; v01 = v01N; v10 = v10N; v11 = v11N; v20 = v20N; v21 = v21N
            }

            // Rotate (0, 2)
            if (abs(a02) > 1e-14) {
                val theta = 0.5 * (a22 - a00) / a02
                val t = if (theta >= 0.0) 1.0 / (theta + sqrt(theta * theta + 1.0)) else -1.0 / (-theta + sqrt(theta * theta + 1.0))
                val c = 1.0 / sqrt(t * t + 1.0)
                val s = t * c
                val a00New = a00 - t * a02
                val a22New = a22 + t * a02
                val a01New = c * a01 - s * a12
                val a12New = s * a01 + c * a12
                a00 = a00New; a22 = a22New; a01 = a01New; a12 = a12New; a02 = 0.0

                val v00N = c * v00 - s * v02; val v02N = s * v00 + c * v02
                val v10N = c * v10 - s * v12; val v12N = s * v10 + c * v12
                val v20N = c * v20 - s * v22; val v22N = s * v20 + c * v22
                v00 = v00N; v02 = v02N; v10 = v10N; v12 = v12N; v20 = v20N; v22 = v22N
            }

            // Rotate (1, 2)
            if (abs(a12) > 1e-14) {
                val theta = 0.5 * (a22 - a11) / a12
                val t = if (theta >= 0.0) 1.0 / (theta + sqrt(theta * theta + 1.0)) else -1.0 / (-theta + sqrt(theta * theta + 1.0))
                val c = 1.0 / sqrt(t * t + 1.0)
                val s = t * c
                val a11New = a11 - t * a12
                val a22New = a22 + t * a12
                val a01New = c * a01 - s * a02
                val a02New = s * a01 + c * a02
                a11 = a11New; a22 = a22New; a01 = a01New; a02 = a02New; a12 = 0.0

                val v01N = c * v01 - s * v02; val v02N = s * v01 + c * v02
                val v11N = c * v11 - s * v12; val v12N = s * v11 + c * v12
                val v21N = c * v21 - s * v22; val v22N = s * v21 + c * v22
                v01 = v01N; v02 = v02N; v11 = v11N; v12 = v12N; v21 = v21N; v22 = v22N
            }
        }

        // The diagonal elements are the eigenvalues: lambda0 = a00, lambda1 = a11, lambda2 = a22
        // Column 0 is (v00, v10, v20), Column 1 is (v01, v11, v21), Column 2 is (v02, v12, v22)
        val minIdx = when {
            a00 <= a11 && a00 <= a22 -> 0
            a11 <= a00 && a11 <= a22 -> 1
            else -> 2
        }

        val nx = when (minIdx) { 0 -> v00; 1 -> v01; else -> v02 }.toFloat()
        val ny = when (minIdx) { 0 -> v10; 1 -> v11; else -> v12 }.toFloat()
        val nz = when (minIdx) { 0 -> v20; 1 -> v21; else -> v22 }.toFloat()

        val len = sqrt(nx * nx + ny * ny + nz * nz)
        return if (len > 1e-6f) {
            floatArrayOf(nx / len, ny / len, nz / len)
        } else {
            floatArrayOf(0f, 0f, 1f)
        }
    }
}
