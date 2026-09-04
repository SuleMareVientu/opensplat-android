#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "match_culling.h" // For CameraPose

struct TrackObservation {
    int camera_idx;
    cv::Point2f pt2d;
};

struct Track {
    std::vector<TrackObservation> observations;
    cv::Point3f pt3d;
    bool valid;
};

class Triangulator {
public:
    Triangulator();
    ~Triangulator();

    // Triangulate a single track and apply geometric gates
    bool triangulateTrack(
        Track& track, 
        const std::vector<CameraPose>& poses);

private:
    float computeParallax(const Track& track, const std::vector<CameraPose>& poses);
    float computeReprojectionError(const Track& track, const std::vector<CameraPose>& poses);

    float min_parallax_deg_ = 2.0f;
    float max_reproj_error_ = 2.0f; // pixels
};
