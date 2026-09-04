#include "bundle_adjuster.h"
#include <ceres/rotation.h>
#include <android/log.h>

#define LOG_TAG "BundleAdjuster"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

struct SfmReprojectionError {
    SfmReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera_r,
                    const T* const camera_t,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera_r, point, p);
        p[0] += camera_t[0];
        p[1] += camera_t[1];
        p[2] += camera_t[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y,
                                       double fx, double fy, double cx, double cy) {
        return (new ceres::AutoDiffCostFunction<SfmReprojectionError, 2, 3, 3, 3>(
            new SfmReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }

    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;
};

BundleAdjuster::BundleAdjuster() {}
BundleAdjuster::~BundleAdjuster() {}

bool BundleAdjuster::optimize(
    std::vector<CameraPose>& poses,
    std::vector<Track>& tracks,
    int max_time_seconds) {

    if (poses.empty() || tracks.empty()) return false;

    ceres::Problem problem;

    std::vector<std::vector<double>> camera_r(poses.size(), std::vector<double>(3));
    std::vector<std::vector<double>> camera_t(poses.size(), std::vector<double>(3));

    for (size_t i = 0; i < poses.size(); ++i) {
        double R_arr[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R_arr[r * 3 + c] = static_cast<double>(poses[i].R.at<float>(r, c));
            }
        }
        ceres::RotationMatrixToAngleAxis(R_arr, camera_r[i].data());
        camera_t[i][0] = static_cast<double>(poses[i].t.at<float>(0, 0));
        camera_t[i][1] = static_cast<double>(poses[i].t.at<float>(1, 0));
        camera_t[i][2] = static_cast<double>(poses[i].t.at<float>(2, 0));
    }

    std::vector<std::vector<double>> points_3d(tracks.size(), std::vector<double>(3));
    for (size_t i = 0; i < tracks.size(); ++i) {
        points_3d[i][0] = static_cast<double>(tracks[i].pt3d.x);
        points_3d[i][1] = static_cast<double>(tracks[i].pt3d.y);
        points_3d[i][2] = static_cast<double>(tracks[i].pt3d.z);
    }

    // Fix first camera pose to anchor gauge freedom
    problem.AddParameterBlock(camera_r[0].data(), 3);
    problem.AddParameterBlock(camera_t[0].data(), 3);
    problem.SetParameterBlockConstant(camera_r[0].data());
    problem.SetParameterBlockConstant(camera_t[0].data());

    int num_residuals = 0;
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (!tracks[i].valid) continue;

        for (const auto& obs : tracks[i].observations) {
            if (obs.camera_idx >= poses.size()) continue;

            const auto& pose = poses[obs.camera_idx];
            double fx = pose.K.at<float>(0, 0);
            double fy = pose.K.at<float>(1, 1);
            double cx = pose.K.at<float>(0, 2);
            double cy = pose.K.at<float>(1, 2);

            ceres::CostFunction* cost_function = SfmReprojectionError::Create(
                obs.pt2d.x, obs.pt2d.y, fx, fy, cx, cy);

            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

            problem.AddResidualBlock(
                cost_function,
                loss_function,
                camera_r[obs.camera_idx].data(),
                camera_t[obs.camera_idx].data(),
                points_3d[i].data());
            num_residuals++;
        }
    }

    if (num_residuals == 0) return false;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 50;
    options.max_solver_time_in_seconds = static_cast<double>(max_time_seconds);
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOGI("Bundle Adjustment finished: %s", summary.BriefReport().c_str());

    // Update optimized poses
    for (size_t i = 0; i < poses.size(); ++i) {
        double R_arr[9];
        ceres::AngleAxisToRotationMatrix(camera_r[i].data(), R_arr);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                poses[i].R.at<float>(r, c) = static_cast<float>(R_arr[r * 3 + c]);
            }
        }
        poses[i].t.at<float>(0, 0) = static_cast<float>(camera_t[i][0]);
        poses[i].t.at<float>(1, 0) = static_cast<float>(camera_t[i][1]);
        poses[i].t.at<float>(2, 0) = static_cast<float>(camera_t[i][2]);
    }

    // Update optimized 3D points
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (!tracks[i].valid) continue;
        tracks[i].pt3d.x = static_cast<float>(points_3d[i][0]);
        tracks[i].pt3d.y = static_cast<float>(points_3d[i][1]);
        tracks[i].pt3d.z = static_cast<float>(points_3d[i][2]);
    }

    return true;
}
