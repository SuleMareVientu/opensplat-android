#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "match_culling.h"
#include "triangulator.h"

class Exporter {
public:
    Exporter();
    ~Exporter();

    // Export PLY and transforms.json
    // images: original images to sample color
    // image_names: filenames of the images
    // poses: optimized camera poses
    // tracks: optimized 3D points
    // intrinsics: {fx, fy, cx, cy, width, height}
    // output_dir: directory to write files
    bool exportNerfstudio(
        const std::vector<cv::Mat>& images,
        const std::vector<std::string>& image_names,
        const std::vector<CameraPose>& poses,
        const std::vector<Track>& tracks,
        const std::string& output_dir);

private:
    cv::Vec3b computeTrackColor(const Track& track, const std::vector<cv::Mat>& images);
    cv::Vec3b sampleColor(const cv::Mat& image, const cv::Point2f& pt2d);
};
