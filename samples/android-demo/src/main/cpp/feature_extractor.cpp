#include "feature_extractor.h"
#include <android/log.h>

#define LOG_TAG "FeatureExtractor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

FeatureExtractor::FeatureExtractor() {}

FeatureExtractor::~FeatureExtractor() {
    if (interpreter_ != nullptr) {
        TfLiteInterpreterDelete(interpreter_);
    }
    if (model_ != nullptr) {
        TfLiteModelDelete(model_);
    }
    if (options_ != nullptr) {
        TfLiteInterpreterOptionsDelete(options_);
    }
}

bool FeatureExtractor::initialize(const std::string& modelPath) {
    model_ = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!model_) {
        LOGE("Failed to mmap model: %s", modelPath.c_str());
        return false;
    }

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 4);
    
    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (!interpreter_) {
        LOGE("Failed to build interpreter");
        return false;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
        LOGE("Failed to allocate tensors");
        return false;
    }

    return true;
}

std::vector<Keypoint> FeatureExtractor::extractFeatures(const cv::Mat& image) {
    std::vector<Keypoint> global_keypoints;
    if (image.empty() || !interpreter_) {
        LOGE("Image empty or interpreter not initialized");
        return global_keypoints;
    }

    // Convert to Grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image.clone();
    }

    // Convert to Float32
    cv::Mat gray_f32;
    gray.convertTo(gray_f32, CV_32F, 1.0f / 255.0f);

    int img_w = gray_f32.cols;
    int img_h = gray_f32.rows;

    struct Tile {
        cv::Rect rect;
    };
    
    std::vector<Tile> tiles;
    if (img_w >= img_h) {
        int step_w = img_w / 3;
        tiles.push_back({cv::Rect(0, 0, step_w, img_h)});
        tiles.push_back({cv::Rect(step_w, 0, step_w, img_h)});
        tiles.push_back({cv::Rect(step_w * 2, 0, img_w - step_w * 2, img_h)});
    } else {
        int step_h = img_h / 3;
        tiles.push_back({cv::Rect(0, 0, img_w, step_h)});
        tiles.push_back({cv::Rect(0, step_h, img_w, step_h)});
        tiles.push_back({cv::Rect(0, step_h * 2, img_w, img_h - step_h * 2)});
    }

    const int MODEL_W = 640;
    const int MODEL_H = 480;

    for (const auto& tile : tiles) {
        cv::Mat roi = gray_f32(tile.rect);
        
        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(MODEL_W, MODEL_H));
        
        // Instance Normalization (mean=0, std=1)
        cv::Scalar mean, stddev;
        cv::meanStdDev(resized, mean, stddev);
        float m = static_cast<float>(mean[0]);
        float s = static_cast<float>(stddev[0]);
        if (s < 1e-6f) s = 1.0f; // avoid div by zero
        
        resized = (resized - m) / s;

        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
        if (!input_tensor) {
            LOGE("Input tensor is null!");
            continue;
        }
        
        if (TfLiteTensorCopyFromBuffer(input_tensor, resized.data, MODEL_W * MODEL_H * sizeof(float)) != kTfLiteOk) {
            LOGE("Failed to copy input data");
            continue;
        }

        if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk) {
            LOGE("Failed to invoke interpreter!");
            continue;
        }

        const float* descriptors = nullptr;
        const float* keypoints = nullptr;
        const float* reliability = nullptr;
        
        int num_outputs = TfLiteInterpreterGetOutputTensorCount(interpreter_);
        for(int o=0; o<num_outputs; ++o) {
            const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter_, o);
            int channels = TfLiteTensorDim(tensor, 1);
            if (channels == 64) {
                descriptors = reinterpret_cast<const float*>(TfLiteTensorData(tensor));
            } else if (channels == 65) {
                keypoints = reinterpret_cast<const float*>(TfLiteTensorData(tensor));
            } else if (channels == 1) {
                reliability = reinterpret_cast<const float*>(TfLiteTensorData(tensor));
            }
        }
        
        if (!descriptors || !keypoints || !reliability) {
            LOGE("Failed to find all required output tensors");
            continue;
        }
        
        int H_feat = 60;
        int W_feat = 80;
        float threshold = 0.1f;
        
        std::vector<Keypoint> local_kpts;
        
        for (int y = 0; y < H_feat; ++y) {
            for (int x = 0; x < W_feat; ++x) {
                float rel_score = reliability[y * W_feat + x];
                if (rel_score < threshold) continue;
                
                float max_val = 0.0f;
                int max_c = -1;
                for (int c = 0; c < 64; ++c) {
                    float val = keypoints[(c * H_feat + y) * W_feat + x];
                    if (val > max_val) {
                        max_val = val;
                        max_c = c;
                    }
                }
                
                if (max_val > 0.0f && max_c >= 0) {
                    int dy = max_c / 8;
                    int dx = max_c % 8;
                    
                    float px = static_cast<float>(x * 8 + dx);
                    float py = static_cast<float>(y * 8 + dy);
                    
                    // Scale from MODEL_W x MODEL_H to tile.rect
                    float scale_x = static_cast<float>(tile.rect.width) / MODEL_W;
                    float scale_y = static_cast<float>(tile.rect.height) / MODEL_H;
                    
                    Keypoint kp;
                    kp.pt = cv::Point2f(px * scale_x + tile.rect.x, py * scale_y + tile.rect.y);
                    kp.response = rel_score * max_val;
                    
                    kp.descriptor.resize(64);
                    float norm = 0.0f;
                    for (int c = 0; c < 64; ++c) {
                        float v = descriptors[(c * H_feat + y) * W_feat + x];
                        kp.descriptor[c] = v;
                        norm += v * v;
                    }
                    norm = std::sqrt(norm) + 1e-6f;
                    for (int c = 0; c < 64; ++c) {
                        kp.descriptor[c] /= norm;
                    }
                    
                    local_kpts.push_back(kp);
                }
            }
        }
        
        // NMS per tile
        std::sort(local_kpts.begin(), local_kpts.end(), [](const Keypoint& a, const Keypoint& b) {
            return a.response > b.response;
        });
        
        float nms_dist_sq = 4.0f * 4.0f;
        std::vector<bool> keep(local_kpts.size(), true);
        for (size_t i = 0; i < local_kpts.size(); ++i) {
            if (!keep[i]) continue;
            global_keypoints.push_back(local_kpts[i]);
            
            for (size_t j = i + 1; j < local_kpts.size(); ++j) {
                if (!keep[j]) continue;
                float dx = local_kpts[i].pt.x - local_kpts[j].pt.x;
                float dy = local_kpts[i].pt.y - local_kpts[j].pt.y;
                if (dx*dx + dy*dy < nms_dist_sq) {
                    keep[j] = false;
                }
            }
        }
    } // End of tiles loop
    
    // Sort globally by response
    std::sort(global_keypoints.begin(), global_keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
        return a.response > b.response;
    });
    
    // Cap at 4096 keypoints total (standard for SfM pipelines)
    if (global_keypoints.size() > 4096) {
        global_keypoints.resize(4096);
    }
    
    LOGI("Extracted %zu keypoints from frame", global_keypoints.size());
    return global_keypoints;
}
