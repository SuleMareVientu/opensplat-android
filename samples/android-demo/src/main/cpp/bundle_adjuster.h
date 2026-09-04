#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include "match_culling.h"
#include "triangulator.h"

class BundleAdjuster {
public:
    BundleAdjuster();
    ~BundleAdjuster();

    // Perform Bundle Adjustment on poses and tracks
    // camera_intrinsics: {fx, fy, cx, cy}
    bool optimize(
        std::vector<CameraPose>& poses,
        std::vector<Track>& tracks,
        int max_time_seconds = 20);
};
