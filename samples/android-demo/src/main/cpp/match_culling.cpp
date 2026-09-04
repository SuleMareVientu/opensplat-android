#include "match_culling.h"
#include <android/log.h>

#define LOG_TAG "MatchCuller"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

MatchCuller::MatchCuller() {}
MatchCuller::~MatchCuller() {}

float MatchCuller::computeBaseline(const CameraPose& a, const CameraPose& b) {
    // Assuming R, t are World-to-Camera (Extrinsics):
    // C = -R^T * t
    cv::Mat c_a = -a.R.t() * a.t;
    cv::Mat c_b = -b.R.t() * b.t;
    return static_cast<float>(cv::norm(c_a - c_b));
}

float MatchCuller::computeLookVectorDot(const CameraPose& a, const CameraPose& b) {
    // In OpenCV, camera looks down +Z axis.
    // The look vector in world space is the 3rd row of R (or R^T * [0, 0, 1]^T)
    cv::Mat look_a = a.R.row(2).t();
    cv::Mat look_b = b.R.row(2).t();
    
    // Normalize just in case
    look_a /= cv::norm(look_a);
    look_b /= cv::norm(look_b);
    
    return static_cast<float>(look_a.dot(look_b));
}

std::vector<std::pair<int, int>> MatchCuller::getValidPairs(const std::vector<CameraPose>& poses) {
    std::vector<std::pair<int, int>> valid_pairs;
    int n = poses.size();
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float baseline = computeBaseline(poses[i], poses[j]);
            float look_dot = computeLookVectorDot(poses[i], poses[j]);
            
            if (baseline < min_baseline_) {
                // Too close, insufficient parallax
                continue;
            }
            if (baseline > max_baseline_) {
                // Too far, unlikely to have good overlapping features
                continue;
            }
            if (look_dot < min_look_dot_) {
                // Cameras are not looking in the same general direction
                continue;
            }
            
            valid_pairs.push_back({i, j});
        }
    }
    
    LOGI("Generated %zu valid pairs out of %d total possible pairs", valid_pairs.size(), (n * (n - 1)) / 2);
    return valid_pairs;
}
