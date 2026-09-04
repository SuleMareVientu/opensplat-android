#include "CeresBundleAdjuster.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <algorithm>
#include <android/log.h>

#define LOG_TAG "CeresBundleAdjuster"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace opensplat {

/**
 * Reprojection error functor for OpenGL right-handed camera coordinates:
 * Points in world coordinates X_w are mapped to camera coordinates:
 *   P_c = R_w2c * X_w + t_w2c
 * In OpenGL camera coordinates, camera looks down -Z:
 *   zCam = -P_c.z
 * Perspective projection:
 *   u = fx * (P_c.x / zCam) + cx
 *   v = cy - fy * (P_c.y / zCam)
 */
struct ReprojectionError {
    ReprojectionError(double observed_u, double observed_v,
                      double fx, double fy, double cx, double cy)
        : observed_u_(observed_u), observed_v_(observed_v),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const camera_rot,   // angle-axis (3)
                    const T* const camera_trans, // translation (3)
                    const T* const point,        // 3D point (3)
                    T* residuals) const {
        // Rotate point by angle-axis
        T p_rot[3];
        ceres::AngleAxisRotatePoint(camera_rot, point, p_rot);

        // Add translation to obtain camera-space coordinates [xc, yc, zc]
        T xc = p_rot[0] + camera_trans[0];
        T yc = p_rot[1] + camera_trans[1];
        T zc = p_rot[2] + camera_trans[2];

        // In OpenGL camera space, camera looks along -Z
        T zCam = -zc;
        if (zCam < T(0.001)) {
            zCam = T(0.001);
        }

        T invZ = T(1.0) / zCam;
        T u = T(fx_) * (xc * invZ) + T(cx_);
        T v = T(cy_) - T(fy_) * (yc * invZ);

        residuals[0] = u - T(observed_u_);
        residuals[1] = v - T(observed_v_);

        return true;
    }

    static ceres::CostFunction* Create(double observed_u, double observed_v,
                                       double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
            new ReprojectionError(observed_u, observed_v, fx, fy, cx, cy));
    }

    double observed_u_;
    double observed_v_;
    double fx_, fy_, cx_, cy_;
};

BundleAdjustmentResult CeresBundleAdjuster::RunBundleAdjustment(
    int numCameras,
    float* camerasToWorld,
    const float* cameraIntrinsics,
    int numPoints,
    float* pointCoords,
    int numObservations,
    const int* obsPointIndices,
    const int* obsCameraIndices,
    const float* obsPixels,
    int maxIterations
) {
    BundleAdjustmentResult result;
    if (numCameras <= 0 || numPoints <= 0 || numObservations <= 0) {
        LOGE("Invalid dimensions: cameras=%d, points=%d, obs=%d",
             numCameras, numPoints, numObservations);
        return result;
    }

    // 1. Unpack camera poses from 4x4 column-major matrices to angle-axis and translation
    std::vector<double> camera_rotations(3 * numCameras);
    std::vector<double> camera_translations(3 * numCameras);

    for (int j = 0; j < numCameras; ++j) {
        const float* c2w = &camerasToWorld[16 * j];

        // Column-major 4x4:
        // Col 0: [0], [1], [2]
        // Col 1: [4], [5], [6]
        // Col 2: [8], [9], [10]
        // Col 3: [12], [13], [14] (C_j)
        // R_w2c = R_c2w^T
        // Column-major R_w2c:
        double R_w2c[9];
        // Col 0 of R_w2c is Row 0 of R_c2w:
        R_w2c[0] = c2w[0]; R_w2c[1] = c2w[4]; R_w2c[2] = c2w[8];
        // Col 1 of R_w2c is Row 1 of R_c2w:
        R_w2c[3] = c2w[1]; R_w2c[4] = c2w[5]; R_w2c[5] = c2w[9];
        // Col 2 of R_w2c is Row 2 of R_c2w:
        R_w2c[6] = c2w[2]; R_w2c[7] = c2w[6]; R_w2c[8] = c2w[10];

        double Cx = c2w[12], Cy = c2w[13], Cz = c2w[14];
        // t_w2c = - R_w2c * C
        double t_w2c[3];
        t_w2c[0] = -(R_w2c[0] * Cx + R_w2c[3] * Cy + R_w2c[6] * Cz);
        t_w2c[1] = -(R_w2c[1] * Cx + R_w2c[4] * Cy + R_w2c[7] * Cz);
        t_w2c[2] = -(R_w2c[2] * Cx + R_w2c[5] * Cy + R_w2c[8] * Cz);

        double angleAxis[3];
        ceres::RotationMatrixToAngleAxis(R_w2c, angleAxis);

        camera_rotations[3 * j + 0] = angleAxis[0];
        camera_rotations[3 * j + 1] = angleAxis[1];
        camera_rotations[3 * j + 2] = angleAxis[2];

        camera_translations[3 * j + 0] = t_w2c[0];
        camera_translations[3 * j + 1] = t_w2c[1];
        camera_translations[3 * j + 2] = t_w2c[2];
    }

    // 2. Unpack points
    std::vector<double> points(3 * numPoints);
    for (int i = 0; i < 3 * numPoints; ++i) {
        points[i] = static_cast<double>(pointCoords[i]);
    }

    // 3. Build Ceres problem
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.5);
    bool loss_function_added = false;

    for (int k = 0; k < numObservations; ++k) {
        int ptIdx = obsPointIndices[k];
        int camIdx = obsCameraIndices[k];

        if (ptIdx < 0 || ptIdx >= numPoints || camIdx < 0 || camIdx >= numCameras) {
            continue;
        }

        double u_obs = static_cast<double>(obsPixels[2 * k + 0]);
        double v_obs = static_cast<double>(obsPixels[2 * k + 1]);

        double fx = static_cast<double>(cameraIntrinsics[4 * camIdx + 0]);
        double fy = static_cast<double>(cameraIntrinsics[4 * camIdx + 1]);
        double cx = static_cast<double>(cameraIntrinsics[4 * camIdx + 2]);
        double cy = static_cast<double>(cameraIntrinsics[4 * camIdx + 3]);

        ceres::CostFunction* cost_function =
            ReprojectionError::Create(u_obs, v_obs, fx, fy, cx, cy);

        problem.AddResidualBlock(
            cost_function,
            loss_function,
            &camera_rotations[3 * camIdx],
            &camera_translations[3 * camIdx],
            &points[3 * ptIdx]
        );
        loss_function_added = true;
    }

    if (!loss_function_added) {
        delete loss_function;
        LOGE("No valid residual blocks added to Bundle Adjustment problem!");
        return result;
    }

    // 4. Pin reference camera 0 (gauge freedom constraint)
    problem.SetParameterBlockConstant(&camera_rotations[0]);
    problem.SetParameterBlockConstant(&camera_translations[0]);

    // 5. Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = maxIterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOGI("Bundle Adjustment completed in %d iterations (initial cost: %.4f, final cost: %.4f)",
         static_cast<int>(summary.iterations.size()),
         summary.initial_cost,
         summary.final_cost);

    result.success = summary.IsSolutionUsable();
    result.numIterations = static_cast<int>(summary.iterations.size());
    result.initialCost = summary.initial_cost;
    result.finalCost = summary.final_cost;

    if (!result.success) {
        LOGE("Bundle Adjustment solution not usable!");
        return result;
    }

    // 6. Pack updated camera poses back into column-major 4x4 matrices
    for (int j = 0; j < numCameras; ++j) {
        float* c2w = &camerasToWorld[16 * j];

        double R_w2c_opt[9];
        ceres::AngleAxisToRotationMatrix(&camera_rotations[3 * j], R_w2c_opt);
        const double* t_w2c_opt = &camera_translations[3 * j];

        // R_c2w = R_w2c^T
        // Col 0 of R_c2w is Row 0 of R_w2c:
        c2w[0] = static_cast<float>(R_w2c_opt[0]);
        c2w[1] = static_cast<float>(R_w2c_opt[3]);
        c2w[2] = static_cast<float>(R_w2c_opt[6]);
        c2w[3] = 0.0f;

        // Col 1 of R_c2w is Row 1 of R_w2c:
        c2w[4] = static_cast<float>(R_w2c_opt[1]);
        c2w[5] = static_cast<float>(R_w2c_opt[4]);
        c2w[6] = static_cast<float>(R_w2c_opt[7]);
        c2w[7] = 0.0f;

        // Col 2 of R_c2w is Row 2 of R_w2c:
        c2w[8] = static_cast<float>(R_w2c_opt[2]);
        c2w[9] = static_cast<float>(R_w2c_opt[5]);
        c2w[10] = static_cast<float>(R_w2c_opt[8]);
        c2w[11] = 0.0f;

        // C = - R_c2w * t_w2c:
        double Cx_opt = -(c2w[0] * t_w2c_opt[0] + c2w[4] * t_w2c_opt[1] + c2w[8] * t_w2c_opt[2]);
        double Cy_opt = -(c2w[1] * t_w2c_opt[0] + c2w[5] * t_w2c_opt[1] + c2w[9] * t_w2c_opt[2]);
        double Cz_opt = -(c2w[2] * t_w2c_opt[0] + c2w[6] * t_w2c_opt[1] + c2w[10] * t_w2c_opt[2]);

        c2w[12] = static_cast<float>(Cx_opt);
        c2w[13] = static_cast<float>(Cy_opt);
        c2w[14] = static_cast<float>(Cz_opt);
        c2w[15] = 1.0f;
    }

    // 7. Pack updated 3D points
    for (int i = 0; i < 3 * numPoints; ++i) {
        pointCoords[i] = static_cast<float>(points[i]);
    }

    return result;
}

} // namespace opensplat
