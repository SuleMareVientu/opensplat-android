#include "triangulator.h"
#include <cmath>
#include <android/log.h>

#define LOG_TAG "Triangulator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

Triangulator::Triangulator() {}
Triangulator::~Triangulator() {}

bool Triangulator::triangulateTrack(Track& track, const std::vector<CameraPose>& poses) {
    if (track.observations.size() < 2) return false;

    cv::Mat A(static_cast<int>(track.observations.size() * 2), 4, CV_32F);

    for (size_t i = 0; i < track.observations.size(); ++i) {
        const auto& obs = track.observations[i];
        if (obs.camera_idx >= poses.size()) return false;
        const auto& pose = poses[obs.camera_idx];

        cv::Mat Rt;
        cv::hconcat(pose.R, pose.t, Rt); // 3x4
        cv::Mat P = pose.K * Rt; // 3x4

        float u = obs.pt2d.x;
        float v = obs.pt2d.y;

        cv::Mat row0 = u * P.row(2) - P.row(0);
        cv::Mat row1 = v * P.row(2) - P.row(1);

        row0.copyTo(A.row(static_cast<int>(i * 2)));
        row1.copyTo(A.row(static_cast<int>(i * 2 + 1)));
    }

    cv::Mat X;
    cv::SVD::solveZ(A, X); // X is 4x1

    float w = X.at<float>(3, 0);
    if (std::abs(w) < 1e-6f) return false;

    track.pt3d = cv::Point3f(
        X.at<float>(0, 0) / w,
        X.at<float>(1, 0) / w,
        X.at<float>(2, 0) / w
    );

    // Cheirality test (point must be in front of all cameras)
    for (const auto& obs : track.observations) {
        const auto& pose = poses[obs.camera_idx];
        cv::Mat pt3d_mat = (cv::Mat_<float>(3, 1) << track.pt3d.x, track.pt3d.y, track.pt3d.z);
        cv::Mat pt_cam = pose.R * pt3d_mat + pose.t;
        if (pt_cam.at<float>(2, 0) <= 0.01f) {
            return false;
        }
    }

    float parallax = computeParallax(track, poses);
    if (parallax < min_parallax_deg_) return false;

    float reproj_err = computeReprojectionError(track, poses);
    if (reproj_err > max_reproj_error_) return false;

    track.valid = true;
    return true;
}

float Triangulator::computeParallax(const Track& track, const std::vector<CameraPose>& poses) {
    if (track.observations.size() < 2) return 0.0f;

    cv::Point3f pt = track.pt3d;
    float max_parallax = 0.0f;

    for (size_t i = 0; i < track.observations.size(); ++i) {
        cv::Mat c1 = -poses[track.observations[i].camera_idx].R.t() * poses[track.observations[i].camera_idx].t;
        cv::Point3f cam1(c1.at<float>(0, 0), c1.at<float>(1, 0), c1.at<float>(2, 0));
        cv::Point3f ray1 = pt - cam1;
        float norm1 = cv::norm(ray1);
        if (norm1 < 1e-6f) continue;
        ray1 *= (1.0f / norm1);

        for (size_t j = i + 1; j < track.observations.size(); ++j) {
            cv::Mat c2 = -poses[track.observations[j].camera_idx].R.t() * poses[track.observations[j].camera_idx].t;
            cv::Point3f cam2(c2.at<float>(0, 0), c2.at<float>(1, 0), c2.at<float>(2, 0));
            cv::Point3f ray2 = pt - cam2;
            float norm2 = cv::norm(ray2);
            if (norm2 < 1e-6f) continue;
            ray2 *= (1.0f / norm2);

            float dot = std::max(-1.0f, std::min(1.0f, ray1.dot(ray2)));
            float angle_rad = std::acos(dot);
            float angle_deg = angle_rad * 180.0f / 3.1415926535f;
            if (angle_deg > max_parallax) {
                max_parallax = angle_deg;
            }
        }
    }

    return max_parallax;
}

float Triangulator::computeReprojectionError(const Track& track, const std::vector<CameraPose>& poses) {
    if (track.observations.empty()) return 999.0f;

    float total_err = 0.0f;
    cv::Mat pt3d_mat = (cv::Mat_<float>(3, 1) << track.pt3d.x, track.pt3d.y, track.pt3d.z);

    for (const auto& obs : track.observations) {
        const auto& pose = poses[obs.camera_idx];
        cv::Mat pt_cam = pose.R * pt3d_mat + pose.t;
        cv::Mat pt_proj = pose.K * pt_cam;

        float z = pt_proj.at<float>(2, 0);
        if (std::abs(z) < 1e-6f) return 999.0f;

        float u_proj = pt_proj.at<float>(0, 0) / z;
        float v_proj = pt_proj.at<float>(1, 0) / z;

        float du = u_proj - obs.pt2d.x;
        float dv = v_proj - obs.pt2d.y;
        total_err += std::sqrt(du * du + dv * dv);
    }

    return total_err / static_cast<float>(track.observations.size());
}
