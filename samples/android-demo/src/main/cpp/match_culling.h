#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct CameraPose {
    cv::Mat R; // 3x3 rotation matrix
    cv::Mat t; // 3x1 translation vector
    cv::Mat K; // 3x3 intrinsics matrix
    int w;
    int h;
};

class MatchCuller {
public:
    MatchCuller();
    ~MatchCuller();

    // Returns a list of pairs (indices) that should be matched
    std::vector<std::pair<int, int>> getValidPairs(const std::vector<CameraPose>& poses);

private:
    float computeBaseline(const CameraPose& a, const CameraPose& b);
    float computeLookVectorDot(const CameraPose& a, const CameraPose& b);
    
    // Thresholds
    float min_baseline_ = 0.05f; // min distance (e.g. 5cm)
    float max_baseline_ = 2.0f;  // max distance (e.g. 2m)
    float min_look_dot_ = 0.5f;  // approx 60 degrees max angle difference
};
