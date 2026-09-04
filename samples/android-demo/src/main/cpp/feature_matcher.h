#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "feature_extractor.h"
#include "match_culling.h"

struct FeatureMatch {
    int idx_a;
    int idx_b;
    float distance;
};

class FeatureMatcher {
public:
    FeatureMatcher();
    ~FeatureMatcher();

    // Perform Mutual Nearest Neighbor (MNN) matching using descriptor distances
    std::vector<FeatureMatch> matchMNN(const std::vector<Keypoint>& kpts_a, const std::vector<Keypoint>& kpts_b);

    // Filter matches using Epipolar geometry (Fundamental Matrix computed from poses)
    std::vector<FeatureMatch> filterEpipolar(
        const std::vector<FeatureMatch>& matches,
        const std::vector<Keypoint>& kpts_a,
        const std::vector<Keypoint>& kpts_b,
        const CameraPose& pose_a,
        const CameraPose& pose_b);

private:
    float computeBaseline(const CameraPose& a, const CameraPose& b);
    cv::Mat computeFundamentalMatrix(const CameraPose& pose_a, const CameraPose& pose_b, const cv::Mat& K_a, const cv::Mat& K_b);
};
