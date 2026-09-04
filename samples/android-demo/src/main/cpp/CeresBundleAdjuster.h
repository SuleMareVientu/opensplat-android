#pragma once

#include <vector>

namespace opensplat {

struct BundleAdjustmentResult {
    bool success = false;
    int numIterations = 0;
    double initialCost = 0.0;
    double finalCost = 0.0;
};

class CeresBundleAdjuster {
public:
    static BundleAdjustmentResult RunBundleAdjustment(
        int numCameras,
        float* camerasToWorld,         // 16 * numCameras (in-out)
        const float* cameraIntrinsics, // 4 * numCameras (fx, fy, cx, cy)
        int numPoints,
        float* pointCoords,            // 3 * numPoints (in-out)
        int numObservations,
        const int* obsPointIndices,    // numObservations
        const int* obsCameraIndices,   // numObservations
        const float* obsPixels,        // 2 * numObservations (u, v)
        int maxIterations = 25
    );
};

} // namespace opensplat
