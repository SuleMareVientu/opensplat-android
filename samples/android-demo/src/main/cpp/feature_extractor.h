#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"

struct Keypoint {
    cv::Point2f pt;
    float response;
    std::vector<float> descriptor; // Example if we extract descriptors later
};

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();

    bool initialize(const std::string& modelPath);
    std::vector<Keypoint> extractFeatures(const cv::Mat& image);

private:
    TfLiteModel* model_ = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteInterpreterOptions* options_ = nullptr;
    TfLiteDelegate* xnnpack_delegate_ = nullptr;
    TfLiteDelegate* nnapi_delegate_ = nullptr;
};
