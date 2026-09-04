package io.github.sceneview.demo.demos.internal

import java.io.File
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Metadata for a captured keyframe needed for multi-view stereo geometric matching.
 */
data class KeyframeView(
    val index: Int,
    val cameraToWorld: FloatArray, // 16-element column-major matrix
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val imageFile: File? = null,
    val width: Int = 0,
    val height: Int = 0
) {
    // Camera center in world space = translation column of cameraToWorld
    val cameraX: Float get() = cameraToWorld[12]
    val cameraY: Float get() = cameraToWorld[13]
    val cameraZ: Float get() = cameraToWorld[14]

    /**
     * Transform world point (xw, yw, zw) into camera coordinates (xc, yc, zc)
     * using rigid matrix inverse: R^T * (p - t).
     */
    fun worldToCamera(xw: Float, yw: Float, zw: Float, out: FloatArray) {
        val dx = xw - cameraToWorld[12]
        val dy = yw - cameraToWorld[13]
        val dz = zw - cameraToWorld[14]

        // R^T in column-major:
        // row0: [0], [1], [2]
        // row1: [4], [5], [6]
        // row2: [8], [9], [10]
        out[0] = cameraToWorld[0] * dx + cameraToWorld[1] * dy + cameraToWorld[2] * dz
        out[1] = cameraToWorld[4] * dx + cameraToWorld[5] * dy + cameraToWorld[6] * dz
        out[2] = cameraToWorld[8] * dx + cameraToWorld[9] * dy + cameraToWorld[10] * dz
    }

    /**
     * Project camera coordinates (xc, yc, zc) to pixel coordinates (u, v).
     * OpenGL right-handed camera space: zCam = -zc.
     * Returns true if in front of camera and within patch margin [margin, width - margin].
     */
    fun project(xc: Float, yc: Float, zc: Float, outPix: FloatArray, margin: Int = 2): Boolean {
        val zCam = -zc
        if (zCam <= 0.05f) return false

        val u = fx * (xc / zCam) + cx
        val v = cy - fy * (yc / zCam)

        outPix[0] = u
        outPix[1] = v

        return if (width > 0 && height > 0) {
            u >= margin && u <= (width - 1 - margin) &&
                    v >= margin && v <= (height - 1 - margin)
        } else {
            true
        }
    }
    /**
     * Directly project world point (xw, yw, zw) to pixel coordinates (u, v)
     * without intermediate matrix/vector allocations.
     */
    fun projectWorld(xw: Float, yw: Float, zw: Float, outPix: FloatArray, margin: Int = 2): Boolean {
        val dx = xw - cameraToWorld[12]
        val dy = yw - cameraToWorld[13]
        val dz = zw - cameraToWorld[14]

        val zc = cameraToWorld[8] * dx + cameraToWorld[9] * dy + cameraToWorld[10] * dz
        val zCam = -zc
        if (zCam <= 0.05f) return false

        val xc = cameraToWorld[0] * dx + cameraToWorld[1] * dy + cameraToWorld[2] * dz
        val yc = cameraToWorld[4] * dx + cameraToWorld[5] * dy + cameraToWorld[6] * dz

        val u = fx * (xc / zCam) + cx
        val v = cy - fy * (yc / zCam)

        outPix[0] = u
        outPix[1] = v

        return if (width > 0 && height > 0) {
            u >= margin && u <= (width - 1 - margin) &&
                    v >= margin && v <= (height - 1 - margin)
        } else {
            true
        }
    }
}

/**
 * Interface providing 8-bit grayscale luminance values with bilinear interpolation.
 */
interface LuminanceProvider {
    val width: Int
    val height: Int

    /**
     * Sample continuous sub-pixel luminance at (u, v) using bilinear interpolation.
     */
    fun sampleBilinear(u: Float, v: Float): Float
}

/**
 * In-memory flat ByteArray luminance provider (1 byte per pixel).
 */
class ArrayLuminanceProvider(
    override val width: Int,
    override val height: Int,
    private val data: ByteArray
) : LuminanceProvider {

    override fun sampleBilinear(u: Float, v: Float): Float {
        val uClamped = u.coerceIn(0f, (width - 1).toFloat())
        val vClamped = v.coerceIn(0f, (height - 1).toFloat())

        val u0 = uClamped.toInt()
        val v0 = vClamped.toInt()
        val u1 = min(u0 + 1, width - 1)
        val v1 = min(v0 + 1, height - 1)

        val wu = uClamped - u0
        val wv = vClamped - v0

        val i00 = data[v0 * width + u0].toInt() and 0xFF
        val i10 = data[v0 * width + u1].toInt() and 0xFF
        val i01 = data[v1 * width + u0].toInt() and 0xFF
        val i11 = data[v1 * width + u1].toInt() and 0xFF

        val top = (1f - wu) * i00 + wu * i10
        val bottom = (1f - wu) * i01 + wu * i11
        return (1f - wv) * top + wv * bottom
    }
}

/**
 * Result of multi-view depth refinement for a point.
 */
data class RefinedPoint(
    val point: GeometricPoint,
    val isMultiViewVerified: Boolean,
    val znccScore: Float
)

/**
 * Phase 3 & 4 Multi-View Consistency and Narrow-Range Depth Refiner.
 */
class MultiViewDepthRefiner(
    val config: PriorConditionedSplatConfig = PriorConditionedSplatConfig.roomScale()
) {

    /**
     * Select top qualifying neighbor keyframes for a 3D candidate point.
     * Evaluates baseline parallax angle (scored with Gaussian centered at 18 deg),
     * surface normal incidence (cos > minIncidenceCosine), and scale ratio.
     */
    fun selectKeyframes(
        p: GeometricPoint,
        refView: KeyframeView,
        candidateViews: List<KeyframeView>
    ): List<KeyframeView> {
        // Ray from reference camera to point
        val rxRef = p.x - refView.cameraX
        val ryRef = p.y - refView.cameraY
        val rzRef = p.z - refView.cameraZ
        val distRef = sqrt(rxRef * rxRef + ryRef * ryRef + rzRef * rzRef)
        if (distRef < 1e-4f) return emptyList()

        val uxRef = rxRef / distRef
        val uyRef = ryRef / distRef
        val uzRef = rzRef / distRef

        val searchViews = if (candidateViews.size > 10) {
            candidateViews.filter { it.index != refView.index }
                .sortedBy { v ->
                    val dx = v.cameraX - refView.cameraX
                    val dy = v.cameraY - refView.cameraY
                    val dz = v.cameraZ - refView.cameraZ
                    dx * dx + dy * dy + dz * dz
                }.take(8)
        } else {
            candidateViews
        }

        var top1View: KeyframeView? = null; var top1Score = -1f
        var top2View: KeyframeView? = null; var top2Score = -1f
        var top3View: KeyframeView? = null; var top3Score = -1f
        val scratchPix = FloatArray(2)

        for (cand in searchViews) {
            if (cand.index == refView.index) continue

            val rx = p.x - cand.cameraX
            val ry = p.y - cand.cameraY
            val rz = p.z - cand.cameraZ
            val distCand = sqrt(rx * rx + ry * ry + rz * rz)
            if (distCand < 1e-4f) continue

            val ux = rx / distCand
            val uy = ry / distCand
            val uz = rz / distCand

            // 1. Scale ratio check
            val scaleRatio = distCand / distRef
            if (scaleRatio < config.minScaleRatio || scaleRatio > config.maxScaleRatio) continue

            // 2. Parallax baseline angle
            val cosTheta = (uxRef * ux + uyRef * uy + uzRef * uz).coerceIn(-1f, 1f)
            val thetaDeg = (acos(cosTheta) * (180f / Math.PI.toFloat()))
            if (thetaDeg < config.minParallaxDeg || thetaDeg > config.maxParallaxDeg) continue

            // 3. Surface normal incidence: ray points toward surface front
            // Ray from surface to camera = -ux, -uy, -uz
            val cosIncidence = p.nx * (-ux) + p.ny * (-uy) + p.nz * (-uz)
            if (cosIncidence < config.minIncidenceCosine) continue

            // 4. Frustum visibility check without object allocation
            if (!cand.projectWorld(p.x, p.y, p.z, scratchPix, margin = config.patchRadius)) continue

            // Score: Gaussian centered at optimalParallaxDeg (18 deg)
            val angleDiff = thetaDeg - config.optimalParallaxDeg
            val score = exp(-(angleDiff * angleDiff) / 128f) * cosIncidence

            if (score > top1Score) {
                top3View = top2View; top3Score = top2Score
                top2View = top1View; top2Score = top1Score
                top1View = cand; top1Score = score
            } else if (score > top2Score) {
                top3View = top2View; top3Score = top2Score
                top2View = cand; top2Score = score
            } else if (score > top3Score) {
                top3View = cand; top3Score = score
            }
        }

        val result = ArrayList<KeyframeView>(3)
        if (top1View != null) result.add(top1View)
        if (top2View != null) result.add(top2View)
        if (top3View != null && config.targetKeyframes >= 3) result.add(top3View)
        return result
    }

    /**
     * Compute Zero-Normalized Cross-Correlation (ZNCC) between 5x5 patches in refView and targetView.
     * Zero-allocation implementation operating directly on cached byte/float arrays.
     */
    fun computeZNCC(
        pWorldX: Float, pWorldY: Float, pWorldZ: Float,
        refView: KeyframeView, refLum: LuminanceProvider,
        targetView: KeyframeView, targetLum: LuminanceProvider
    ): Float {
        val pixRef = FloatArray(2)
        if (!refView.projectWorld(pWorldX, pWorldY, pWorldZ, pixRef, margin = config.patchRadius)) return -1f

        val pixTgt = FloatArray(2)
        if (!targetView.projectWorld(pWorldX, pWorldY, pWorldZ, pixTgt, margin = config.patchRadius)) return -1f

        val r = config.patchRadius
        val n = (2 * r + 1) * (2 * r + 1)
        val uRef = pixRef[0]; val vRef = pixRef[1]
        val uTgt = pixTgt[0]; val vTgt = pixTgt[1]

        var sumRef = 0f; var sumTgt = 0f
        for (dy in -r..r) {
            for (dx in -r..r) {
                sumRef += refLum.sampleBilinear(uRef + dx, vRef + dy)
                sumTgt += targetLum.sampleBilinear(uTgt + dx, vTgt + dy)
            }
        }
        val meanRef = sumRef / n
        val meanTgt = sumTgt / n

        var cov = 0f; var varRef = 0f; var varTgt = 0f
        for (dy in -r..r) {
            for (dx in -r..r) {
                val dr = refLum.sampleBilinear(uRef + dx, vRef + dy) - meanRef
                val dt = targetLum.sampleBilinear(uTgt + dx, vTgt + dy) - meanTgt
                cov += dr * dt
                varRef += dr * dr
                varTgt += dt * dt
            }
        }

        val denom = sqrt(varRef * varTgt)
        return if (denom > 1e-4f) {
            (cov / denom).coerceIn(-1f, 1f)
        } else {
            0.5f // Flat texture patch default
        }
    }

    /**
     * Compute pixel intensity variance for a patch around (pWorldX, pWorldY, pWorldZ) in view.
     * Used to detect textureless surfaces where ZNCC is mathematically ill-conditioned.
     */
    fun computePatchVariance(
        pWorldX: Float, pWorldY: Float, pWorldZ: Float,
        view: KeyframeView, lum: LuminanceProvider
    ): Float {
        val pix = FloatArray(2)
        if (!view.projectWorld(pWorldX, pWorldY, pWorldZ, pix, margin = config.patchRadius)) return 0f

        val r = config.patchRadius
        val n = (2 * r + 1) * (2 * r + 1)
        val u = pix[0]; val v = pix[1]

        var sum = 0f
        for (dy in -r..r) {
            for (dx in -r..r) {
                sum += lum.sampleBilinear(u + dx, v + dy)
            }
        }
        val mean = sum / n

        var sumSqDiff = 0f
        for (dy in -r..r) {
            for (dx in -r..r) {
                val diff = lum.sampleBilinear(u + dx, v + dy) - mean
                sumSqDiff += diff * diff
            }
        }
        return sumSqDiff / n
    }

    /**
     * Refine a candidate point using multi-view ZNCC consistency and narrow-range ray search.
     * Returns RefinedPoint or null if pruned due to photometric inconsistency.
     */
    fun refinePoint(
        point: GeometricPoint,
        refView: KeyframeView,
        refLum: LuminanceProvider,
        candidateViews: List<KeyframeView>,
        luminanceLookup: (KeyframeView) -> LuminanceProvider?
    ): RefinedPoint? {
        val qualifyingViews = selectKeyframes(point, refView, candidateViews)

        // Low-coverage fallback path: if < minKeyframesForVerification, do not prune!
        if (qualifyingViews.size < config.minKeyframesForVerification) {
            val penalizedPoint = point.copy(
                confidence = (point.confidence * config.insufficientCoveragePenalty).coerceIn(0f, 1f)
            )
            return RefinedPoint(penalizedPoint, isMultiViewVerified = false, znccScore = 0.5f)
        }

        // Gather valid luminance providers for qualifying views
        val activeViews = ArrayList<Pair<KeyframeView, LuminanceProvider>>()
        for (v in qualifyingViews) {
            val lum = luminanceLookup(v)
            if (lum != null) activeViews.add(Pair(v, lum))
        }

        if (activeViews.size < config.minKeyframesForVerification) {
            val penalizedPoint = point.copy(
                confidence = (point.confidence * config.insufficientCoveragePenalty).coerceIn(0f, 1f)
            )
            return RefinedPoint(penalizedPoint, isMultiViewVerified = false, znccScore = 0.5f)
        }

        // Texture-variance gating: if the reference patch is flat/textureless (e.g. blank wall or desk),
        // ZNCC denominator approaches zero and correlation becomes noise.
        // Skip ZNCC ray search and pruning, preserving the point via the low-coverage fallback path.
        val refVariance = computePatchVariance(point.x, point.y, point.z, refView, refLum)
        if (refVariance < config.minPatchVariance) {
            val penalizedPoint = point.copy(
                confidence = (point.confidence * config.insufficientCoveragePenalty).coerceIn(0f, 1f)
            )
            return RefinedPoint(penalizedPoint, isMultiViewVerified = false, znccScore = 0.5f)
        }

        fun evaluateAvgZNCC(x: Float, y: Float, z: Float): Float {
            var sum = 0f
            for ((view, lum) in activeViews) {
                sum += computeZNCC(x, y, z, refView, refLum, view, lum)
            }
            return sum / activeViews.size
        }

        // Ray from reference camera to point
        val rayX = point.x - refView.cameraX
        val rayY = point.y - refView.cameraY
        val rayZ = point.z - refView.cameraZ
        val initialDist = sqrt(rayX * rayX + rayY * rayY + rayZ * rayZ)
        if (initialDist < 1e-4f) return null

        val dirX = rayX / initialDist
        val dirY = rayY / initialDist
        val dirZ = rayZ / initialDist

        // Compute budget optimization: if confidence is already authoritative (>= 0.85),
        // evaluate ZNCC only at d0 to confirm persistence and bypass 13-step search.
        val znccInitial = evaluateAvgZNCC(point.x, point.y, point.z)
        if (point.confidence >= 0.85f && point.residual <= config.computeDepthNoise(point.depthZ)) {
            if (znccInitial < config.znccThreshold) {
                return null // Pruned as inconsistent
            }
            return RefinedPoint(point, isMultiViewVerified = true, znccScore = znccInitial)
        }

        // Narrow-range 1D search: S steps in [d0 * (1 - range), d0 * (1 + range)]
        val steps = config.depthSearchSteps
        val halfSteps = steps / 2
        val rangeFrac = config.depthSearchRangeFraction
        val costs = FloatArray(steps)
        val znccScores = FloatArray(steps)

        var minCost = Float.MAX_VALUE
        var bestIdx = halfSteps

        for (s in 0 until steps) {
            val offsetFrac = rangeFrac * ((s - halfSteps).toFloat() / halfSteps.toFloat())
            val dHyp = initialDist * (1.0f + offsetFrac)
            val hx = refView.cameraX + dirX * dHyp
            val hy = refView.cameraY + dirY * dHyp
            val hz = refView.cameraZ + dirZ * dHyp

            val zncc = evaluateAvgZNCC(hx, hy, hz)
            znccScores[s] = zncc
            val cost = 1.0f - zncc
            costs[s] = cost

            if (cost < minCost) {
                minCost = cost
                bestIdx = s
            }
        }

        // Consistency check: best ZNCC must meet threshold
        val bestZNCC = znccScores[bestIdx]
        if (bestZNCC < config.znccThreshold) {
            return null // Pruned as inconsistent
        }

        // Sub-pixel 1D parabola fit around discrete minimum
        var deltaStep = 0.0f
        if (bestIdx > 0 && bestIdx < steps - 1) {
            val yMinus = costs[bestIdx - 1]
            val yZero = costs[bestIdx]
            val yPlus = costs[bestIdx + 1]
            val denom = 2f * (yMinus - 2f * yZero + yPlus)
            if (abs(denom) > 1e-5f) {
                deltaStep = ((yMinus - yPlus) / denom).coerceIn(-0.5f, 0.5f)
            }
        }

        val optStepFloat = (bestIdx - halfSteps) + deltaStep
        val optFrac = rangeFrac * (optStepFloat / halfSteps.toFloat())
        val refinedDist = initialDist * (1.0f + optFrac)

        val refinedX = refView.cameraX + dirX * refinedDist
        val refinedY = refView.cameraY + dirY * refinedDist
        val refinedZ = refView.cameraZ + dirZ * refinedDist

        val boostedConfidence = max(point.confidence, bestZNCC).coerceIn(0f, 1f)
        val refinedPt = point.copy(
            x = refinedX,
            y = refinedY,
            z = refinedZ,
            confidence = boostedConfidence
        )

        return RefinedPoint(refinedPt, isMultiViewVerified = true, znccScore = bestZNCC)
    }

    /**
     * Batch refinement of points across keyframe views.
     */
    fun refinePointSet(
        points: List<GeometricPoint>,
        keyframeViews: List<KeyframeView>,
        luminanceLookup: (KeyframeView) -> LuminanceProvider?,
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): List<GeometricPoint> {
        if (points.isEmpty()) return emptyList()

        val candidates = if (points.size > 60000) {
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

        val sortedCandidates = candidates.sortedBy { it.frameIndex }
        val total = sortedCandidates.size
        val result = ArrayList<GeometricPoint>(total)

        for (idx in sortedCandidates.indices) {
            val p = sortedCandidates[idx]
            if (onProgress != null && (idx % 250 == 0 || idx == total - 1)) {
                onProgress(idx + 1, total)
            }
            val fIdx = p.frameIndex
            if (fIdx in keyframeViews.indices) {
                val refView = keyframeViews[fIdx]

                // Orient normal towards reference camera
                val dx = refView.cameraX - p.x
                val dy = refView.cameraY - p.y
                val dz = refView.cameraZ - p.z
                if (p.nx * dx + p.ny * dy + p.nz * dz < 0f) {
                    p.nx = -p.nx
                    p.ny = -p.ny
                    p.nz = -p.nz
                }

                val refLum = luminanceLookup(refView)
                if (refLum == null) {
                    val penalized = p.copy(
                        confidence = (p.confidence * config.insufficientCoveragePenalty).coerceIn(0f, 1f)
                    )
                    result.add(penalized)
                } else {
                    val refined = refinePoint(p, refView, refLum, keyframeViews, luminanceLookup)
                    if (refined != null) {
                        result.add(refined.point)
                    }
                }
            } else {
                val penalized = p.copy(
                    confidence = (p.confidence * config.insufficientCoveragePenalty).coerceIn(0f, 1f)
                )
                result.add(penalized)
            }
        }

        return result
    }

    companion object {
        /**
         * Load uncompressed 8-bit grayscale luminance from image file.
         */
        fun loadLuminanceFromFile(file: File): LuminanceProvider? {
            if (!file.exists()) return null
            try {
                val bmp = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
                if (bmp != null) {
                    val w = bmp.width
                    val h = bmp.height
                    val pixels = IntArray(w * h)
                    bmp.getPixels(pixels, 0, w, 0, 0, w, h)
                    bmp.recycle()
                    val gray = ByteArray(w * h)
                    for (i in pixels.indices) {
                        val p = pixels[i]
                        val r = (p shr 16) and 0xFF
                        val g = (p shr 8) and 0xFF
                        val b = p and 0xFF
                        gray[i] = ((299 * r + 587 * g + 114 * b + 500) / 1000).toByte()
                    }
                    return ArrayLuminanceProvider(w, h, gray)
                }
            } catch (_: Throwable) {
            }

            return null
        }
    }
}

/**
 * Bounded-memory LRU cache storing uncompressed 8-bit grayscale luminance frames.
 * Keeps at most maxResident frames in memory (~3.6 MB for 4 x 720p frames).
 */
class LruLuminanceCache(
    private val maxResident: Int = 16,
    private val loader: (File) -> LuminanceProvider?
) {
    private val cache = object : java.util.LinkedHashMap<File, LuminanceProvider>(maxResident, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<File, LuminanceProvider>?): Boolean {
            return size > maxResident
        }
    }

    @Synchronized
    fun get(file: File?): LuminanceProvider? {
        if (file == null || !file.exists()) return null
        cache[file]?.let { return it }
        val provider = loader(file)
        if (provider != null) {
            cache[file] = provider
        }
        return provider
    }
}
