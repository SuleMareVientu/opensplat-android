#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>

// 1. Pose Parsing & Extrinsics Conversion
void test_pose_conversion() {
    std::cout << "--- test_pose_conversion ---\n";
    float raw_pose[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 2, 3, 1
    };

    cv::Mat pose_mat(4, 4, CV_32F);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            pose_mat.at<float>(r, c) = raw_pose[c * 4 + r];
        }
    }

    cv::Mat R_gl = pose_mat(cv::Rect(0, 0, 3, 3));
    cv::Mat t_gl = pose_mat(cv::Rect(3, 0, 1, 3));

    cv::Mat S = (cv::Mat_<float>(3, 3) << 
        1, 0, 0, 
        0, -1, 0, 
        0, 0, -1);
    
    cv::Mat R_cv = S * R_gl.t();
    cv::Mat t_cv = -R_cv * t_gl;

    std::cout << "R_cv:\n" << R_cv << "\n";
    std::cout << "t_cv:\n" << t_cv << "\n";

    cv::Mat p_world = (cv::Mat_<float>(3, 1) << 1, 2, 2);
    cv::Mat p_cam = R_cv * p_world + t_cv;
    std::cout << "p_cam (Expected: 0, 0, 1):\n" << p_cam << "\n";

    cv::Mat c2w_R = R_cv.t();
    cv::Mat c2w_t = -c2w_R * t_cv;
    cv::Mat flip = (cv::Mat_<float>(3, 3) << 
        1, 0, 0,
        0, -1, 0,
        0, 0, -1);
    cv::Mat c2w_R_gl = c2w_R * flip;

    std::cout << "Export c2w_R_gl (Expected: Identity):\n" << c2w_R_gl << "\n";
    std::cout << "Export c2w_t (Expected: 1, 2, 3):\n" << c2w_t << "\n";
}

void test_bundle_adjustment() {
    std::cout << "--- test_bundle_adjustment ---\n";
    std::cout << "To prevent scale drift, we must fix the translation of the second camera in addition to the first.\n";
    std::cout << "Camera 0 is fixed (translation and rotation) -> fixes 6-DOF gauge.\n";
    std::cout << "Camera 1's translation is fixed -> fixes scale gauge.\n";
}

int main() {
    test_pose_conversion();
    test_bundle_adjustment();
    std::cout << "All tests complete.\n";
    return 0;
}
