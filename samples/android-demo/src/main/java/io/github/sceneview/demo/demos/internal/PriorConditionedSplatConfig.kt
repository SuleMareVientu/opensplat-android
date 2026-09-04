package io.github.sceneview.demo.demos.internal

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Unified configuration and hyperparameter specification for the Prior-Conditioned
 * Gaussian Splatting pipeline (PocketGS) and ARCore raw-depth refinement.
 *
 * All mathematical constants and thresholds are declared here to ensure deterministic
 * behavior and easy device-tier / scene-scale calibration.
 */
data class PriorConditionedSplatConfig(
    // --- Spatial Binning & Scale ---
    val voxelSizeMeters: Float = 0.015f,           // 1.5 cm for room-scale; override to 0.005f for tabletop
    val knnK: Int = 16,                            // Neighbors for local plane fit and SOR
    val knnScaleK: Int = 3,                        // Neighbors for tangential scale estimation (PocketGS)

    // --- Phase 1: In-Loop Edge & Geometric Filtering ---
    val depthJumpThreshold: Float = 0.05f,         // 5% relative jump triggers depth discontinuity
    val luminanceGradientSigma: Float = 25.0f,     // Scale for soft edge weight: exp(-||grad||^2 / (2 * sigma^2))
    val planeResidualBeta: Float = 3.0f,           // Plane residual threshold: r_i > beta * sigma_z(z_i)
    val sorStdDevMultiplier: Float = 1.5f,         // SOR threshold: mean_d + alpha * std_d
    val depthNoiseCoeffA: Float = 0.0015f,         // Quadratic depth sensor noise: sigma_z(z) = a * z^2 + b
    val depthNoiseCoeffB: Float = 0.0010f,

    // --- Phase 2: Continuous Confidence & Opacity Mapping ---
    val opacityMin: Float = 0.05f,                 // Minimum initial opacity for low-confidence splats
    val opacityMax: Float = 0.70f,                 // Maximum initial opacity for high-confidence splats
    val opacityGamma: Float = 1.0f,                // Power-law exponent: alpha(c) = alpha_min + (alpha_max - alpha_min) * c^gamma

    // --- Phase 3 & 4: Multi-View Consistency & Refinement ---
    val minKeyframesForVerification: Int = 2,      // Required neighbor views for multi-view gating
    val targetKeyframes: Int = 3,                  // Top-N neighbor views selected by pose score
    val optimalParallaxDeg: Float = 18.0f,         // Baseline angle sweet spot (Gaussian center)
    val minParallaxDeg: Float = 5.0f,              // Parallax lower bound
    val maxParallaxDeg: Float = 40.0f,             // Parallax upper bound
    val minIncidenceCosine: Float = 0.5f,          // 60 deg max angle between surface normal and camera ray
    val minScaleRatio: Float = 0.7f,               // Baseline distance scale bounds
    val maxScaleRatio: Float = 1.4f,
    val patchRadius: Int = 2,                      // 5x5 patch (radius = 2)
    val znccThreshold: Float = 0.40f,              // Consistency acceptance threshold
    val depthSearchRangeFraction: Float = 0.10f,   // +/- 10% search around initial depth d0
    val depthSearchSteps: Int = 13,                // Odd number of sample steps along camera ray

    // --- Low-Coverage Fallback ---
    val insufficientCoveragePenalty: Float = 0.7f, // Confidence penalty when qualifying views < minKeyframesForVerification

    // --- Texture Variance Gating ---
    val minPatchVariance: Float = 8.0f,            // Reference patch variance threshold: below this, surface is textureless (skips ZNCC pruning)

    // --- Adaptive Spatial Voxelization (Distance & Texture Scaled) ---
    val adaptiveVoxelMinMeters: Float = 0.004f,     // 4 mm minimum voxel resolution for close/textured surfaces
    val adaptiveVoxelMaxMeters: Float = 0.025f,     // 2.5 cm maximum voxel resolution for distant/flat surfaces

    // --- Bundle Adjustment Anchors ---
    val ceresAnchorPoints: Int = 8000,             // Target number of ground-truth bundle adjustment anchor points

    // --- Memory-Safe Luminance Cache ---
    val maxResidentLuminanceFrames: Int = 16       // Maximum uncompressed 8-bit grayscale frames in RAM (~14 MB)
) {

    /**
     * Compute adaptive spatial voxel size based on distance and texture variance:
     * High texture / close range -> fine resolution (down to 4 mm).
     * Low texture / far range -> coarser resolution (up to 25 mm).
     */
    fun computeAdaptiveVoxelSize(depthZ: Float, textureVariance: Float = 20f): Float {
        val z = max(0.2f, depthZ)
        val textureMultiplier = if (textureVariance < 10f) 2.5f else if (textureVariance < 30f) 1.5f else 1.0f
        return (z * 0.008f * textureMultiplier).coerceIn(adaptiveVoxelMinMeters, adaptiveVoxelMaxMeters)
    }

    /**
     * Empirical depth noise estimate sigma_z(z) = a * z^2 + b.
     * Note: default is calibrated for hardware ToF. On software-depth devices,
     * a higher coeffA (e.g. 0.0030) can be configured.
     */
    fun computeDepthNoise(depthZ: Float): Float {
        val z = max(0.1f, depthZ)
        return depthNoiseCoeffA * z * z + depthNoiseCoeffB
    }

    /**
     * Map continuous confidence c in [0, 1] to initial Gaussian opacity alpha_init in [opacityMin, opacityMax].
     */
    fun computeInitialOpacity(confidence: Float): Float {
        val c = confidence.coerceIn(0f, 1f)
        val factor = if (opacityGamma == 1.0f) c else c.pow(opacityGamma)
        return opacityMin + (opacityMax - opacityMin) * factor
    }

    /**
     * Convert initial opacity alpha into raw unconstrained logit for 3DGS PLY format:
     * logit(alpha) = ln(alpha / (1 - alpha)).
     * Clamped to [0.0001, 0.9999] to prevent +/- infinity.
     */
    fun computeOpacityLogit(opacity: Float): Float {
        val a = opacity.coerceIn(0.0001f, 0.9999f)
        return ln(a / (1f - a))
    }

    companion object {
        fun roomScale() = PriorConditionedSplatConfig()

        fun tabletopScale() = PriorConditionedSplatConfig(
            voxelSizeMeters = 0.005f,
            knnK = 12,
            depthSearchRangeFraction = 0.05f
        )
    }
}
