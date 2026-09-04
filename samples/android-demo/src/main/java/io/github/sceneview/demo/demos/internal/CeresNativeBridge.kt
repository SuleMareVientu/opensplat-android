package io.github.sceneview.demo.demos.internal

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Result of the Ceres Solver native Bundle Adjustment optimization.
 */
data class CeresBAResult(
    val success: Boolean,
    val numAnchorPoints: Int,
    val numObservations: Int,
    val updatedKeyframes: List<KeyframeView>,
    val updatedAnchorPoints: List<GeometricPoint>,
    val updatedAnchorIndices: IntArray = IntArray(0)
)

private data class AnchorCandidate(
    val origIndex: Int,
    val point: GeometricPoint,
    val observations: List<Pair<Int, FloatArray>>
)

/**
 * High-performance JNI bridge to Ceres Solver native Bundle Adjustment module.
 * Jointly refines 6-DoF camera poses and 3D anchor points using Levenberg-Marquardt
 * with Dense Schur linear solver and Huber loss outlier rejection.
 */
object CeresNativeBridge {
    private const val TAG = "CeresNativeBridge"

    val isAvailable: Boolean = try {
        System.loadLibrary("ceres_ba")
        true
    } catch (_: Throwable) {
        false
    }

    private fun logI(msg: String) {
        try {
            android.util.Log.i(TAG, msg)
        } catch (_: Throwable) {
            // Ignored on host JVM unit tests
        }
    }

    private fun logW(msg: String) {
        try {
            android.util.Log.w(TAG, msg)
        } catch (_: Throwable) {
            // Ignored on host JVM unit tests
        }
    }

    @JvmStatic
    external fun nativeRunBundleAdjustment(
        numCameras: Int,
        camerasToWorld: FloatArray,
        cameraIntrinsics: FloatArray,
        numPoints: Int,
        pointCoords: FloatArray,
        numObservations: Int,
        obsPointIndices: IntArray,
        obsCameraIndices: IntArray,
        obsPixels: FloatArray,
        maxIterations: Int
    ): Boolean

    /**
     * Run joint camera pose and anchor point Bundle Adjustment.
     *
     * @param keyframes Captured keyframe views.
     * @param candidatePoints Cleaned 3D candidate points from geometric filtering / depth refinement.
     * @param targetAnchors Desired number of spatially bucketed ground-truth anchor points (default 8,000).
     * @param maxIterations Maximum Ceres iterations (default 25).
     * @return CeresBAResult containing updated keyframe poses and ground-truth tagged anchor points (c = 1.0).
     */
    fun runBundleAdjustment(
        keyframes: List<KeyframeView>,
        candidatePoints: List<GeometricPoint>,
        targetAnchors: Int = 8000,
        maxIterations: Int = 25,
        onProgress: ((progress: Float) -> Unit)? = null
    ): CeresBAResult {
        if (!isAvailable || keyframes.size < 2 || candidatePoints.isEmpty()) {
            return CeresBAResult(
                success = false,
                numAnchorPoints = 0,
                numObservations = 0,
                updatedKeyframes = keyframes,
                updatedAnchorPoints = emptyList()
            )
        }

        val scratchPix = FloatArray(2)

        // 1. Filter candidate points observed across >= 3 keyframes
        val qualifying = ArrayList<AnchorCandidate>(candidatePoints.size)
        val reportInterval = max(100, candidatePoints.size / 20)
        for ((candIdx, pt) in candidatePoints.withIndex()) {
            if (onProgress != null && (candIdx % reportInterval == 0 || candIdx == candidatePoints.size - 1)) {
                val frac = (candIdx + 1).toFloat() / candidatePoints.size
                onProgress(0.70f * frac)
            }
            val obsList = ArrayList<Pair<Int, FloatArray>>(keyframes.size)
            for (v in keyframes) {
                // Ray from camera to point
                val rx = pt.x - v.cameraX
                val ry = pt.y - v.cameraY
                val rz = pt.z - v.cameraZ
                val dist = sqrt(rx * rx + ry * ry + rz * rz)
                if (dist < 0.1f || dist > 6.0f) continue

                val ux = rx / dist
                val uy = ry / dist
                val uz = rz / dist

                // Surface incidence check: only enforce for planar surfaces.
                // Non-planar points (edges, curves, thin structures) are isotropic 3D features.
                if (pt.isPlanar) {
                    val cosIncidence = pt.nx * (-ux) + pt.ny * (-uy) + pt.nz * (-uz)
                    if (cosIncidence < 0.25f) continue
                }

                val pix = FloatArray(2)
                if (v.projectWorld(pt.x, pt.y, pt.z, pix, margin = 2)) {
                    obsList.add(Pair(v.index, pix))
                }
            }
            if (obsList.size >= 3) {
                qualifying.add(AnchorCandidate(candIdx, pt, obsList))
            }
        }

        if (qualifying.isEmpty()) {
            logW("No candidate points qualified for Bundle Adjustment (need >= 3 view tracks)")
            return CeresBAResult(
                success = false,
                numAnchorPoints = 0,
                numObservations = 0,
                updatedKeyframes = keyframes,
                updatedAnchorPoints = emptyList()
            )
        }

        // 2. Spatial bucketing: uniformly decimate to targetAnchors
        val bucketSize = 0.03f // 3 cm spatial grid
        val bucketMap = HashMap<Long, AnchorCandidate>()
        for (item in qualifying) {
            val pt = item.point
            val bx = (pt.x / bucketSize).toInt()
            val by = (pt.y / bucketSize).toInt()
            val bz = (pt.z / bucketSize).toInt()
            val key = (bx.toLong() and 0xFFFFFL) or
                    ((by.toLong() and 0xFFFFFL) shl 20) or
                    ((bz.toLong() and 0xFFFFFL) shl 40)

            val existing = bucketMap[key]
            if (existing == null || item.observations.size > existing.observations.size ||
                (item.observations.size == existing.observations.size && pt.confidence > existing.point.confidence)) {
                bucketMap[key] = item
            }
        }

        val selectedAnchors = if (bucketMap.size > targetAnchors) {
            bucketMap.values.sortedByDescending { it.observations.size * 10f + it.point.confidence }.take(targetAnchors)
        } else {
            bucketMap.values.toList()
        }

        val numCameras = keyframes.size
        val numPoints = selectedAnchors.size

        // Build flat array buffers for JNI
        val camerasToWorld = FloatArray(16 * numCameras)
        val cameraIntrinsics = FloatArray(4 * numCameras)

        val keyframeIndexToSlot = HashMap<Int, Int>(numCameras)
        for (slot in 0 until numCameras) {
            val kf = keyframes[slot]
            keyframeIndexToSlot[kf.index] = slot
            System.arraycopy(kf.cameraToWorld, 0, camerasToWorld, 16 * slot, 16)
            cameraIntrinsics[4 * slot + 0] = kf.fx
            cameraIntrinsics[4 * slot + 1] = kf.fy
            cameraIntrinsics[4 * slot + 2] = kf.cx
            cameraIntrinsics[4 * slot + 3] = kf.cy
        }

        val pointCoords = FloatArray(3 * numPoints)
        var totalObservations = 0
        for (slot in 0 until numPoints) {
            val pt = selectedAnchors[slot].point
            pointCoords[3 * slot + 0] = pt.x
            pointCoords[3 * slot + 1] = pt.y
            pointCoords[3 * slot + 2] = pt.z
            totalObservations += selectedAnchors[slot].observations.size
        }

        val obsPointIndices = IntArray(totalObservations)
        val obsCameraIndices = IntArray(totalObservations)
        val obsPixels = FloatArray(2 * totalObservations)

        var obsCursor = 0
        for (ptSlot in 0 until numPoints) {
            val obsList = selectedAnchors[ptSlot].observations
            for ((kfIndex, pix) in obsList) {
                val camSlot = keyframeIndexToSlot[kfIndex] ?: continue
                obsPointIndices[obsCursor] = ptSlot
                obsCameraIndices[obsCursor] = camSlot
                obsPixels[2 * obsCursor + 0] = pix[0]
                obsPixels[2 * obsCursor + 1] = pix[1]
                obsCursor++
            }
        }

        onProgress?.invoke(0.75f)
        logI("Starting Ceres Bundle Adjustment: $numCameras cameras, $numPoints anchor points, $obsCursor observations")

        onProgress?.invoke(0.80f)
        val success = nativeRunBundleAdjustment(
            numCameras = numCameras,
            camerasToWorld = camerasToWorld,
            cameraIntrinsics = cameraIntrinsics,
            numPoints = numPoints,
            pointCoords = pointCoords,
            numObservations = obsCursor,
            obsPointIndices = obsPointIndices,
            obsCameraIndices = obsCameraIndices,
            obsPixels = obsPixels,
            maxIterations = maxIterations
        )
        onProgress?.invoke(0.95f)

        if (!success) {
            logW("Ceres Bundle Adjustment optimization failed or did not converge")
            return CeresBAResult(
                success = false,
                numAnchorPoints = numPoints,
                numObservations = obsCursor,
                updatedKeyframes = keyframes,
                updatedAnchorPoints = emptyList()
            )
        }

        // 3. Unpack updated camera poses
        val updatedKeyframes = ArrayList<KeyframeView>(numCameras)
        for (slot in 0 until numCameras) {
            val original = keyframes[slot]
            val updatedMat = FloatArray(16)
            System.arraycopy(camerasToWorld, 16 * slot, updatedMat, 0, 16)
            updatedKeyframes.add(original.copy(cameraToWorld = updatedMat))
        }

        // 4. Unpack updated 3D coordinates and tag confidence with 1.0f
        val updatedAnchorPoints = ArrayList<GeometricPoint>(numPoints)
        val updatedAnchorIndices = IntArray(numPoints)
        for (slot in 0 until numPoints) {
            val candidate = selectedAnchors[slot]
            val original = candidate.point
            updatedAnchorIndices[slot] = candidate.origIndex
            updatedAnchorPoints.add(
                original.copy(
                    x = pointCoords[3 * slot + 0],
                    y = pointCoords[3 * slot + 1],
                    z = pointCoords[3 * slot + 2],
                    confidence = 1.0f // Maximal ground-truth confidence tag
                )
            )
        }

        logI("Ceres Bundle Adjustment successfully updated $numCameras camera poses and $numPoints anchor points")
        onProgress?.invoke(1.0f)

        return CeresBAResult(
            success = true,
            numAnchorPoints = numPoints,
            numObservations = obsCursor,
            updatedKeyframes = updatedKeyframes,
            updatedAnchorPoints = updatedAnchorPoints,
            updatedAnchorIndices = updatedAnchorIndices
        )
    }
}
