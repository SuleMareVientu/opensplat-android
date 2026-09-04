#include "feature_matcher.h"
#include <android/log.h>

#define LOG_TAG "FeatureMatcher"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

FeatureMatcher::FeatureMatcher() {}
FeatureMatcher::~FeatureMatcher() {}

float FeatureMatcher::computeBaseline(const CameraPose& a, const CameraPose& b) {
    cv::Mat c_a = -a.R.t() * a.t;
    cv::Mat c_b = -b.R.t() * b.t;
    return static_cast<float>(cv::norm(c_a - c_b));
}

cv::Mat FeatureMatcher::computeFundamentalMatrix(const CameraPose& pose_a, const CameraPose& pose_b, const cv::Mat& K_a, const cv::Mat& K_b) {
    // Relative pose from A to B
    cv::Mat R_ab = pose_b.R * pose_a.R.t();
    cv::Mat t_ab = pose_b.t - R_ab * pose_a.t;

    // Essential matrix: E = [t]_x R
    cv::Mat tx = (cv::Mat_<float>(3, 3) << 
                   0, -t_ab.at<float>(2), t_ab.at<float>(1),
                   t_ab.at<float>(2), 0, -t_ab.at<float>(0),
                  -t_ab.at<float>(1), t_ab.at<float>(0), 0);
    cv::Mat E = tx * R_ab;

    // Fundamental matrix: F = K_b^-T E K_a^-1
    cv::Mat K_a_inv = K_a.inv();
    cv::Mat K_b_inv = K_b.inv();
    cv::Mat F = K_b_inv.t() * E * K_a_inv;
    return F;
}

std::vector<FeatureMatch> FeatureMatcher::matchMNN(const std::vector<Keypoint>& kpts_a, const std::vector<Keypoint>& kpts_b) {
    std::vector<FeatureMatch> matches;
    if (kpts_a.empty() || kpts_b.empty()) return matches;
    
    int Na = kpts_a.size();
    int Nb = kpts_b.size();
    
    cv::Mat desc_a(Na, 64, CV_32F);
    for (int i = 0; i < Na; ++i) {
        std::memcpy(desc_a.ptr<float>(i), kpts_a[i].descriptor.data(), 64 * sizeof(float));
    }
    
    cv::Mat desc_b(Nb, 64, CV_32F);
    for (int j = 0; j < Nb; ++j) {
        std::memcpy(desc_b.ptr<float>(j), kpts_b[j].descriptor.data(), 64 * sizeof(float));
    }
    
    // Matrix multiplication: A * B^T
    cv::Mat scores = desc_a * desc_b.t();
    
    // For each point in A, find best in B
    std::vector<int> best_in_b(Na, -1);
    for (int i = 0; i < Na; ++i) {
        float best_score = -1.0f;
        int best_idx = -1;
        const float* row = scores.ptr<float>(i);
        for (int j = 0; j < Nb; ++j) {
            if (row[j] > best_score) {
                best_score = row[j];
                best_idx = j;
            }
        }
        if (best_score > 0.82f) { // Threshold for matching
            best_in_b[i] = best_idx;
        }
    }
    
    // For each point in B, find best in A (Mutual Nearest Neighbor)
    for (int j = 0; j < Nb; ++j) {
        float best_score = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < Na; ++i) {
            float s = scores.at<float>(i, j);
            if (s > best_score) {
                best_score = s;
                best_idx = i;
            }
        }
        
        // Check mutual
        if (best_score > 0.82f && best_idx >= 0) {
            if (best_in_b[best_idx] == j) {
                FeatureMatch m;
                m.idx_a = best_idx;
                m.idx_b = j;
                m.distance = best_score;
                matches.push_back(m);
            }
        }
    }
    
    return matches;
}

std::vector<FeatureMatch> FeatureMatcher::filterEpipolar(
    const std::vector<FeatureMatch>& matches,
    const std::vector<Keypoint>& kpts_a,
    const std::vector<Keypoint>& kpts_b,
    const CameraPose& pose_a,
    const CameraPose& pose_b) {
    
    std::vector<FeatureMatch> filtered_matches;
    if (matches.empty()) return filtered_matches;

    cv::Mat F = computeFundamentalMatrix(pose_a, pose_b, pose_a.K, pose_b.K);
    float baseline = computeBaseline(pose_a, pose_b);

    // Adaptive threshold: scale from 5px for narrow baseline to 3px for wide baseline
    // (example heuristic: if baseline < 0.1m threshold = 5, if > 1.0m threshold = 2)
    float threshold = 5.0f - (baseline - 0.1f) * (3.0f / 0.9f);
    threshold = std::max(2.0f, std::min(5.0f, threshold));

    for (const auto& match : matches) {
        cv::Point2f pt_a = kpts_a[match.idx_a].pt;
        cv::Point2f pt_b = kpts_b[match.idx_b].pt;

        cv::Mat p1 = (cv::Mat_<float>(3, 1) << pt_a.x, pt_a.y, 1.0f);
        cv::Mat p2 = (cv::Mat_<float>(3, 1) << pt_b.x, pt_b.y, 1.0f);

        // Epipolar line in image B: l2 = F * p1
        cv::Mat l2 = F * p1;
        float a = l2.at<float>(0);
        float b_line = l2.at<float>(1);
        float c = l2.at<float>(2);

        // Distance from point p2 to line l2: |p2^T F p1| / sqrt(a^2 + b^2)
        float distance = std::abs(p2.dot(l2)) / std::sqrt(a*a + b_line*b_line);

        if (distance <= threshold) {
            filtered_matches.push_back(match);
        }
    }

    LOGI("Epipolar filter: kept %zu / %zu matches (thresh=%.2f px)", filtered_matches.size(), matches.size(), threshold);
    return filtered_matches;
}
