#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <android/log.h>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <jni.h>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "libyuv.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define LOG_TAG "SplatCaptureJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct Voxel {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float weight = 0.0f;
  bool occupied = false;
};

struct Chunk {
  std::array<Voxel, 4096> voxels;
};

struct ExportFrame {
  std::string file_path;
  float fx;
  float fy;
  float cx;
  float cy;
  std::array<float, 16> pose; // column-major
};

struct FrameTask {
  std::vector<uint8_t> y_data;
  std::vector<uint8_t> u_data;
  std::vector<uint8_t> v_data;
  int y_stride, u_stride, u_pixel_stride, v_stride, v_pixel_stride;
  int width, height;

  std::vector<uint16_t> depth_data;
  int depth_width, depth_height;

  std::vector<uint8_t> confidence_data;

  std::array<float, 16> pose_matrix;
  float fx, fy, cx, cy;

  std::string image_file_path;
  std::string image_relative_path;
};

struct Anchor {
  float u_ai;
  float v_ai;
  float z_ai_linear;
  float z_metric;
};

struct RansacResult {
  float s = 1.0f;
  float t = 0.0f;
  bool success = false;
};

class SplatCapturePipeline {
public:
  SplatCapturePipeline(const std::string &model_path)
      : model_path_(model_path) {
    worker_thread_ = std::thread(&SplatCapturePipeline::WorkerLoop, this);
  }

  ~SplatCapturePipeline() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_worker_ = true;
    }
    queue_cv_.notify_one();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  void EnqueueFrame(FrameTask &&task) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
  }

  int GetPointCount() {
    return total_point_count_.load(std::memory_order_relaxed);
  }

  void ExportDataset(const std::string &output_dir) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // 1. Write points.ply
    std::string ply_path = output_dir + "/points.ply";
    std::ofstream ply_file(ply_path, std::ios::binary);
    if (!ply_file.is_open()) {
      LOGE("Failed to open PLY file for writing: %s", ply_path.c_str());
      return;
    }

    std::vector<Voxel> occupied_voxels;
    for (const auto &pair : voxel_grid_) {
      for (const auto &voxel : pair.second.voxels) {
        if (voxel.occupied) {
          occupied_voxels.push_back(voxel);
        }
      }
    }

    ply_file << "ply\n"
             << "format binary_little_endian 1.0\n"
             << "element vertex " << occupied_voxels.size() << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "property uchar red\n"
             << "property uchar green\n"
             << "property uchar blue\n"
             << "end_header\n";

    for (const auto &v : occupied_voxels) {
      float x = v.x;
      float y = v.y;
      float z = v.z;
      uint8_t r = static_cast<uint8_t>(std::clamp(v.r, 0.0f, 255.0f));
      uint8_t g = static_cast<uint8_t>(std::clamp(v.g, 0.0f, 255.0f));
      uint8_t b = static_cast<uint8_t>(std::clamp(v.b, 0.0f, 255.0f));

      ply_file.write(reinterpret_cast<const char *>(&x), sizeof(float));
      ply_file.write(reinterpret_cast<const char *>(&y), sizeof(float));
      ply_file.write(reinterpret_cast<const char *>(&z), sizeof(float));
      ply_file.write(reinterpret_cast<const char *>(&r), sizeof(uint8_t));
      ply_file.write(reinterpret_cast<const char *>(&g), sizeof(uint8_t));
      ply_file.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
    }
    ply_file.close();
    LOGI("Successfully exported %zu points to points.ply",
         occupied_voxels.size());

    // 2. Write transforms.json
    std::string json_path = output_dir + "/transforms.json";
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
      LOGE("Failed to open transforms.json for writing: %s", json_path.c_str());
      return;
    }

    json_file << "{\n"
              << "  \"camera_model\": \"PERSPECTIVE\",\n"
              << "  \"w\": 504,\n"
              << "  \"h\": 896,\n"
              << "  \"ply_file_path\": \"points.ply\",\n"
              << "  \"frames\": [\n";

    for (size_t i = 0; i < export_frames_.size(); ++i) {
      const auto &frame = export_frames_[i];

      // Apply 90-degree Z-axis rotation to align landscape sensor pose with
      // portrait image coordinate frame. Rotated Pose = M_pose * T_p_to_s Where
      // T_p_to_s is:
      //   [ 0  -1   0   0 ]
      //   [ 1   0   0   0 ]
      //   [ 0   0   1   0 ]
      //   [ 0   0   0   1 ]
      // Column-major multiplication results in Row-major:
      // Row 0: [  M[4], -M[0], M[8],  M[12] ]
      // Row 1: [  M[5], -M[1], M[9],  M[13] ]
      // Row 2: [  M[6], -M[2], M[10], M[14] ]
      // Row 3: [  M[7], -M[3], M[11], M[15] ]
      const auto &m = frame.pose;
      std::array<float, 4> r0 = {m[4], -m[0], m[8], m[12]};
      std::array<float, 4> r1 = {m[5], -m[1], m[9], m[13]};
      std::array<float, 4> r2 = {m[6], -m[2], m[10], m[14]};
      std::array<float, 4> r3 = {m[7], -m[3], m[11], m[15]};

      json_file << "    {\n"
                << "      \"file_path\": \"" << frame.file_path << "\",\n"
                << "      \"fl_x\": " << frame.fx << ",\n"
                << "      \"fl_y\": " << frame.fy << ",\n"
                << "      \"cx\": " << frame.cx << ",\n"
                << "      \"cy\": " << frame.cy << ",\n"
                << "      \"transform_matrix\": [\n"
                << "        [" << r0[0] << ", " << r0[1] << ", " << r0[2]
                << ", " << r0[3] << "],\n"
                << "        [" << r1[0] << ", " << r1[1] << ", " << r1[2]
                << ", " << r1[3] << "],\n"
                << "        [" << r2[0] << ", " << r2[1] << ", " << r2[2]
                << ", " << r2[3] << "],\n"
                << "        [" << r3[0] << ", " << r3[1] << ", " << r3[2]
                << ", " << r3[3] << "]\n"
                << "      ]\n"
                << "    }" << (i == export_frames_.size() - 1 ? "" : ",")
                << "\n";
    }

    json_file << "  ]\n"
              << "}\n";
    json_file.close();
    LOGI("Successfully exported transforms.json");
  }

  void StartCompute() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      start_compute_ = true;
    }
    queue_cv_.notify_one();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock_q(queue_mutex_);
    std::lock_guard<std::mutex> lock_d(data_mutex_);
    
    std::queue<FrameTask> empty_q;
    std::swap(task_queue_, empty_q);
    
    voxel_grid_.clear();
    export_frames_.clear();
    start_compute_ = false;
    total_point_count_.store(0, std::memory_order_relaxed);
  }

  int GetPendingFramesCount() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(task_queue_.size());
  }

  int GetProcessedFramesCount() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return static_cast<int>(export_frames_.size());
  }

  int is_gpu_enabled_ = 0; // 0 = unknown, 1 = GPU, -1 = CPU

private:
  void WorkerLoop() {
    // Initialize LiteRT strictly inside the background thread to prevent GPU
    // delegate context issues
    if (LiteRtCreateEnvironment(0, nullptr, &env_) != kLiteRtStatusOk) {
      LOGE("Failed to create LiteRT environment.");
      return;
    }

    if (LiteRtCreateModelFromFile(env_, model_path_.c_str(), &model_) != kLiteRtStatusOk) {
      LOGE("Failed to load LiteRT model from path: %s", model_path_.c_str());
      return;
    }

    if (LiteRtCreateOptions(&options_) != kLiteRtStatusOk) {
      LOGE("Failed to create LiteRT options.");
      return;
    }

    // Try GPU acceleration first
    if (LiteRtSetOptionsHardwareAccelerators(options_, kLiteRtHwAcceleratorGpu) == kLiteRtStatusOk) {
      if (LiteRtCreateCompiledModel(env_, model_, options_, &compiled_model_) == kLiteRtStatusOk) {
        bool fully_accelerated = false;
        LiteRtCompiledModelIsFullyAccelerated(compiled_model_, &fully_accelerated);
        if (fully_accelerated) {
          is_gpu_enabled_ = 1;
          LOGI("LiteRT GPU acceleration successfully enabled and fully accelerated.");
        } else {
          is_gpu_enabled_ = 1;
          LOGI("LiteRT GPU acceleration partially enabled (some ops on CPU).");
        }
      } else {
        compiled_model_ = nullptr;
      }
    }
    
    // Fallback to CPU if GPU failed
    if (!compiled_model_) {
      LOGE("Failed to compile with GPU. Falling back to CPU.");
      LiteRtSetOptionsHardwareAccelerators(options_, kLiteRtHwAcceleratorCpu);
      if (LiteRtCreateCompiledModel(env_, model_, options_, &compiled_model_) != kLiteRtStatusOk) {
        LOGE("Failed to compile LiteRT model even on CPU.");
        return;
      }
      is_gpu_enabled_ = -1;
      LOGI("LiteRT CPU execution successfully initialized.");
    }

    LOGI("LiteRT native thread successfully initialized.");

    while (true) {
      FrameTask task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() {
          return (!task_queue_.empty() && start_compute_) || stop_worker_;
        });

        if (stop_worker_ && task_queue_.empty()) {
          break;
        }

        if (task_queue_.empty() || !start_compute_) {
          continue;
        }

        task = std::move(task_queue_.front());
        task_queue_.pop();
      }

      ProcessFrameInternal(task);
    }

    // Clean up resources on the same thread
    if (compiled_model_)
      LiteRtDestroyCompiledModel(compiled_model_);
    if (options_)
      LiteRtDestroyOptions(options_);
    if (model_)
      LiteRtDestroyModel(model_);
    if (env_)
      LiteRtDestroyEnvironment(env_);

    LOGI("LiteRT native thread successfully terminated.");
  }

  void ProcessFrameInternal(const FrameTask &task) {
    int W = task.width;
    int H = task.height;

    // 1. Android420ToI420 planar alignment using libyuv
    int stride_y = W;
    int stride_u = W / 2;
    int stride_v = W / 2;

    std::vector<uint8_t> contiguous_y(W * H);
    std::vector<uint8_t> contiguous_u(W * H / 4);
    std::vector<uint8_t> contiguous_v(W * H / 4);

    libyuv::Android420ToI420(
        task.y_data.data(), task.y_stride, task.u_data.data(), task.u_stride,
        task.v_data.data(), task.v_stride,
        task.u_pixel_stride, // pixel_stride of uv (assuming u and v strides are
                             // symmetric)
        contiguous_y.data(), stride_y, contiguous_u.data(), stride_u,
        contiguous_v.data(), stride_v, W, H);

    // 2. Crop to 16:9 ratio
    int W_cropped = W;
    int H_cropped = H;
    int offset_x = 0;
    int offset_y = 0;

    double aspect_ratio = (double)W / H;
    double target_aspect = 16.0 / 9.0;

    if (aspect_ratio >= target_aspect) {
      W_cropped = static_cast<int>(std::round(H * target_aspect)) & ~1;
      offset_x = ((W - W_cropped) / 2) & ~1;
    } else {
      H_cropped = static_cast<int>(std::round(W / target_aspect)) & ~1;
      offset_y = ((H - H_cropped) / 2) & ~1;
    }

    const uint8_t *crop_y =
        contiguous_y.data() + offset_y * stride_y + offset_x;
    const uint8_t *crop_u =
        contiguous_u.data() + (offset_y / 2) * stride_u + (offset_x / 2);
    const uint8_t *crop_v =
        contiguous_v.data() + (offset_y / 2) * stride_v + (offset_x / 2);

    // 3. Resize/Scale to exactly 896x504 landscape
    std::vector<uint8_t> scaled_y(896 * 504);
    std::vector<uint8_t> scaled_u(448 * 252);
    std::vector<uint8_t> scaled_v(448 * 252);

    libyuv::I420Scale(crop_y, stride_y, crop_u, stride_u, crop_v, stride_v,
                      W_cropped, H_cropped, scaled_y.data(), 896,
                      scaled_u.data(), 448, scaled_v.data(), 448, 896, 504,
                      libyuv::kFilterBox);

    // 4. Rotate 90 degrees clockwise (kRotate90) to yield 504x896 portrait
    std::vector<uint8_t> rotated_y(504 * 896);
    std::vector<uint8_t> rotated_u(252 * 448);
    std::vector<uint8_t> rotated_v(252 * 448);

    libyuv::I420Rotate(scaled_y.data(), 896, scaled_u.data(), 448,
                       scaled_v.data(), 448, rotated_y.data(), 504,
                       rotated_u.data(), 252, rotated_v.data(), 252, 896, 504,
                       libyuv::kRotate90);

    // 5. Convert I420 to RGB (outputs true R-G-B byte order in RAW format)
    std::vector<uint8_t> rgb_data(504 * 896 * 3);
    libyuv::I420ToRAW(rotated_y.data(), 504, rotated_u.data(), 252,
                      rotated_v.data(), 252, rgb_data.data(), 504 * 3, 504,
                      896);

    // Save rotated RGB image to output file path using stb_image_write
    stbi_write_jpg(task.image_file_path.c_str(), 504, 896, 3, rgb_data.data(),
                   85);

    // ---------------------------------------------------------
    // REPLACEMENT FOR STEP 6 & 7: Dynamic Tensor Formatting
    // ---------------------------------------------------------
    LiteRtTensorBufferRequirements input_reqs;
    if (LiteRtGetCompiledModelInputBufferRequirements(compiled_model_, 0, 0, &input_reqs) != kLiteRtStatusOk) {
      LOGE("Failed to get input buffer requirements.");
      return;
    }

    LiteRtLayout input_layout;
    if (LiteRtGetCompiledModelInputTensorLayout(compiled_model_, 0, 0, &input_layout) != kLiteRtStatusOk) {
      LOGE("Failed to get input layout.");
      return;
    }

    // Dynamically check if the model wants NHWC (dim 3 is channels)
    bool is_nhwc = (input_layout.rank == 4 && input_layout.dimensions[3] == 3);

    std::vector<float> input_tensor(3 * 896 * 504);
    for (int y = 0; y < 896; ++y) {
      for (int x = 0; x < 504; ++x) {
        int pixel_idx = (y * 504 + x) * 3;
        // ImageNet Normalize
        float r = (rgb_data[pixel_idx + 0] / 255.0f - 0.485f) / 0.229f;
        float g = (rgb_data[pixel_idx + 1] / 255.0f - 0.456f) / 0.224f;
        float b = (rgb_data[pixel_idx + 2] / 255.0f - 0.406f) / 0.225f;

        if (is_nhwc) {
          input_tensor[pixel_idx + 0] = r;
          input_tensor[pixel_idx + 1] = g;
          input_tensor[pixel_idx + 2] = b;
        } else {
          input_tensor[0 * 896 * 504 + y * 504 + x] = r;
          input_tensor[1 * 896 * 504 + y * 504 + x] = g;
          input_tensor[2 * 896 * 504 + y * 504 + x] = b;
        }
      }
    }

    LiteRtTensorBuffer input_buffer = nullptr;
    LiteRtRankedTensorType input_tensor_type;
    input_tensor_type.element_type = kLiteRtElementTypeFloat32;
    input_tensor_type.layout = input_layout;

    if (LiteRtCreateTensorBufferFromHostMemory(&input_tensor_type, input_tensor.data(),
                                               input_tensor.size() * sizeof(float), nullptr,
                                               &input_buffer) != kLiteRtStatusOk) {
      LOGE("Failed to create input tensor buffer.");
      return;
    }

    LiteRtTensorBufferRequirements output_reqs;
    if (LiteRtGetCompiledModelOutputBufferRequirements(compiled_model_, 0, 0, &output_reqs) != kLiteRtStatusOk) {
      LOGE("Failed to get output buffer requirements.");
      LiteRtDestroyTensorBuffer(input_buffer);
      return;
    }

    LiteRtLayout output_layout;
    if (LiteRtGetCompiledModelOutputTensorLayouts(compiled_model_, 0, 1, &output_layout, false) != kLiteRtStatusOk) {
      LOGE("Failed to get output layout.");
      LiteRtDestroyTensorBuffer(input_buffer);
      return;
    }

    LiteRtRankedTensorType output_tensor_type;
    output_tensor_type.element_type = kLiteRtElementTypeFloat32;
    output_tensor_type.layout = output_layout;

    LiteRtTensorBuffer output_buffer = nullptr;
    if (LiteRtCreateManagedTensorBufferFromRequirements(env_, &output_tensor_type, output_reqs, &output_buffer) != kLiteRtStatusOk) {
      LOGE("Failed to create output tensor buffer.");
      LiteRtDestroyTensorBuffer(input_buffer);
      return;
    }

    if (LiteRtRunCompiledModel(compiled_model_, 0, 1, &input_buffer, 1, &output_buffer) != kLiteRtStatusOk) {
      LOGE("Inference execution failed.");
      LiteRtDestroyTensorBuffer(input_buffer);
      LiteRtDestroyTensorBuffer(output_buffer);
      return;
    }

    void* output_data_ptr = nullptr;
    if (LiteRtLockTensorBuffer(output_buffer, &output_data_ptr, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) {
      LOGE("Failed to lock output buffer.");
      LiteRtDestroyTensorBuffer(input_buffer);
      LiteRtDestroyTensorBuffer(output_buffer);
      return;
    }

    std::vector<float> output_ai(896 * 504);

    // LiteRT handles the FP16 to FP32 conversion natively if we created the buffer as Float32.
    // However, if the underlying buffer exposes FP16 anyway, we need to check its actual type.
    LiteRtRankedTensorType actual_output_type;
    if (LiteRtGetTensorBufferTensorType(output_buffer, &actual_output_type) == kLiteRtStatusOk &&
        actual_output_type.element_type == kLiteRtElementTypeFloat16) {
      uint16_t* fp16_data = reinterpret_cast<uint16_t*>(output_data_ptr);

      // Fast FP16 to FP32 conversion using standard bit-shifting
      for (int i = 0; i < 896 * 504; ++i) {
        uint16_t h = fp16_data[i];
        int sign = (h >> 15) & 0x00000001;
        int exp = (h >> 10) & 0x0000001F;
        int frac = h & 0x000003FF;
        float f32;
        if (exp == 0) {
          f32 =
              (sign ? -1.0f : 1.0f) * std::pow(2.0f, -14.0f) * (frac / 1024.0f);
        } else if (exp == 31) {
          f32 = frac == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
        } else {
          f32 = (sign ? -1.0f : 1.0f) * std::pow(2.0f, exp - 15.0f) *
                (1.0f + frac / 1024.0f);
        }
        output_ai[i] = f32;
      }
    } else {
      // Standard FP32 extraction
      float* fp32_data = reinterpret_cast<float*>(output_data_ptr);
      std::copy(fp32_data, fp32_data + (896 * 504), output_ai.begin());
    }

    LiteRtUnlockTensorBuffer(output_buffer);
    LiteRtDestroyTensorBuffer(input_buffer);
    LiteRtDestroyTensorBuffer(output_buffer);

    // The exp() decode is already baked into the TFLite graph (DualDPT output)
    std::vector<float> &z_ai_linear = output_ai;

    // 9. C++ 2D Coordinate Mapping (Optical Center Math)
    std::vector<Anchor> anchors;
    int depth_w = task.depth_width;
    int depth_h = task.depth_height;

    // Isotropic scale based on shared horizontal FOV
    float scale = 896.0f / (float)depth_w;

    for (int v = 0; v < depth_h; ++v) {
      for (int u = 0; u < depth_w; ++u) {
        int depth_idx = v * depth_w + u;
        uint8_t confidence = task.confidence_data[depth_idx];

        if (confidence >= 240) {
          float z_metric = task.depth_data[depth_idx] / 1000.0f;

          // Map from center of Depth to center of Landscape Tensor
          float u_land = (u - depth_w / 2.0f) * scale + 448.0f;
          float v_land = (v - depth_h / 2.0f) * scale + 252.0f;

          // Rotate 90 degrees clockwise to Portrait Tensor
          float u_ai = 504.0f - 1.0f - v_land;
          float v_ai = u_land;

          int tx = static_cast<int>(std::round(u_ai));
          int ty = static_cast<int>(std::round(v_ai));

          if (tx >= 0 && tx < 504 && ty >= 0 && ty < 896) {
            float z_ai_lin = z_ai_linear[ty * 504 + tx];
            anchors.push_back(Anchor{u_ai, v_ai, z_ai_lin, z_metric});
          }
        }
      }
    }

    // 10. RANSAC Optimization (2-parameter affine model: metric = s * linear_ai
    // + t)
    RansacResult ransac = RunRansac(anchors);
    if (!ransac.success) {
      LOGI("RANSAC failed: Discarding frame due to lack of anchors or metric "
           "scale alignment.");
      return;
    }

    // 11. Densification & Matrix Unprojection and Voxel Fusion
    std::lock_guard<std::mutex> lock(data_mutex_);

    float s = ransac.s;
    float t = ransac.t;

    for (int y = 0; y < 896; ++y) {
      for (int x = 0; x < 504; ++x) {
        int tensor_idx = y * 504 + x;
        float z_linear_val = z_ai_linear[tensor_idx];
        float z_true = s * z_linear_val + t;

        // Unproject into Portrait camera coordinates
        float x_p = (x - task.cx) * z_true / task.fx;
        float y_p = -(y - task.cy) * z_true / task.fy;
        float z_p = -z_true;

        // Rotation matching: Portrait camera coordinates to Landscape sensor
        // coordinate frame
        float x_s = -y_p;
        float y_s = x_p;
        float z_s = z_p;

        // Multiply by camera pose matrix
        float x_w = task.pose_matrix[0] * x_s + task.pose_matrix[4] * y_s +
                    task.pose_matrix[8] * z_s + task.pose_matrix[12];
        float y_w = task.pose_matrix[1] * x_s + task.pose_matrix[5] * y_s +
                    task.pose_matrix[9] * z_s + task.pose_matrix[13];
        float z_w = task.pose_matrix[2] * x_s + task.pose_matrix[6] * y_s +
                    task.pose_matrix[10] * z_s + task.pose_matrix[14];

        // Divide space into 10cm chunk coordinates
        int64_t cx_chunk = static_cast<int64_t>(std::floor(x_w / 0.1f));
        int64_t cy_chunk = static_cast<int64_t>(std::floor(y_w / 0.1f));
        int64_t cz_chunk = static_cast<int64_t>(std::floor(z_w / 0.1f));

        // Bit-pack 3D chunk coordinates into a lossless uint64_t key
        uint64_t key = (((uint64_t)cx_chunk & 0x1FFFFF) << 42) |
                       (((uint64_t)cy_chunk & 0x1FFFFF) << 21) |
                       ((uint64_t)cz_chunk & 0x1FFFFF);

        Chunk &chunk = voxel_grid_[key];

        // Voxel coordinate index within the chunk at 6.25mm resolution
        int64_t gx = static_cast<int64_t>(std::floor(x_w / 0.00625f));
        int64_t gy = static_cast<int64_t>(std::floor(y_w / 0.00625f));
        int64_t gz = static_cast<int64_t>(std::floor(z_w / 0.00625f));

        int vx = static_cast<int>(gx - 16 * cx_chunk);
        int vy = static_cast<int>(gy - 16 * cy_chunk);
        int vz = static_cast<int>(gz - 16 * cz_chunk);

        int voxel_index = vx * 256 + vy * 16 + vz;

        // Running average update on the voxel grid
        float r = rgb_data[tensor_idx * 3 + 0];
        float g = rgb_data[tensor_idx * 3 + 1];
        float b = rgb_data[tensor_idx * 3 + 2];

        if (vx >= 0 && vx < 16 && vy >= 0 && vy < 16 && vz >= 0 && vz < 16) {
          UpdateVoxel(chunk.voxels[voxel_index], x_w, y_w, z_w, r, g, b, 1.0f);
        }
      }
    }

    // 12. Save frame data to export list
    export_frames_.push_back(ExportFrame{task.image_relative_path, task.fx,
                                         task.fy, task.cx, task.cy,
                                         task.pose_matrix});

    LOGI("Processed frame successfully: %s. Total frames: %zu",
         task.image_relative_path.c_str(), export_frames_.size());
  }

  RansacResult RunRansac(const std::vector<Anchor> &anchors) {
    RansacResult result;
    if (anchors.size() < 2)
      return result;

    int best_inliers = -1;
    float best_s = 1.0f;
    float best_t = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, anchors.size() - 1);

    for (int iter = 0; iter < 500; ++iter) {
      size_t idx1 = dist(gen);
      size_t idx2 = dist(gen);
      if (idx1 == idx2)
        continue;

      const auto &a1 = anchors[idx1];
      const auto &a2 = anchors[idx2];

      float diff_ai = a2.z_ai_linear - a1.z_ai_linear;

      // CRITICAL FIX: Lowered from 0.1f to 1e-4f
      if (std::abs(diff_ai) < 1e-4f)
        continue;

      float s = (a2.z_metric - a1.z_metric) / diff_ai;

      // CRITICAL FIX: Explicitly reject negative scales!
      if (s <= 1e-5f)
        continue;

      float t = a1.z_metric - s * a1.z_ai_linear;

      int inliers = 0;
      for (const auto &a : anchors) {
        float z_pred = s * a.z_ai_linear + t;
        float tol = std::max(0.05f, a.z_metric * 0.05f);
        if (std::abs(z_pred - a.z_metric) < tol) {
          inliers++;
        }
      }

      if (inliers > best_inliers) {
        best_inliers = inliers;
        best_s = s;
        best_t = t;
      }
    }

    if (best_inliers >= 2) {
      result.s = best_s;
      result.t = best_t;
      result.success = true;
    }
    return result;
  }

  void UpdateVoxel(Voxel &v, float x, float y, float z, float r, float g,
                   float b, float weight) {
    if (!v.occupied) {
      v.x = x;
      v.y = y;
      v.z = z;
      v.r = r;
      v.g = g;
      v.b = b;
      v.weight = weight;
      v.occupied = true;
      total_point_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
      float new_weight = v.weight + weight;
      v.x = (v.x * v.weight + x * weight) / new_weight;
      v.y = (v.y * v.weight + y * weight) / new_weight;
      v.z = (v.z * v.weight + z * weight) / new_weight;
      v.r = (v.r * v.weight + r * weight) / new_weight;
      v.g = (v.g * v.weight + g * weight) / new_weight;
      v.b = (v.b * v.weight + b * weight) / new_weight;
      v.weight = new_weight;
    }
  }

  std::string model_path_;

  // Threading
  std::thread worker_thread_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<FrameTask> task_queue_;
  bool stop_worker_ = false;
  bool start_compute_ = false;

  // Voxel grid and frame list
  std::mutex data_mutex_;
  std::unordered_map<uint64_t, Chunk> voxel_grid_;
  std::atomic<int> total_point_count_{0};
  std::vector<ExportFrame> export_frames_;

  // LiteRT
  LiteRtEnvironment env_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtOptions options_ = nullptr;
  LiteRtCompiledModel compiled_model_ = nullptr;
};

// Global Pipeline Reference
static std::unique_ptr<SplatCapturePipeline> g_pipeline = nullptr;

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_initPipeline(
    JNIEnv *env, jobject thiz, jstring model_path) {
  const char *path_chars = env->GetStringUTFChars(model_path, nullptr);
  std::string model_path_str(path_chars);
  env->ReleaseStringUTFChars(model_path, path_chars);

  g_pipeline = std::make_unique<SplatCapturePipeline>(model_path_str);
  return reinterpret_cast<jlong>(g_pipeline.get());
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_freePipeline(
    JNIEnv *env, jobject thiz, jlong handle) {
  if (g_pipeline) {
    g_pipeline.reset();
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_processFrame(
    JNIEnv *env, jobject thiz, jlong handle, jobject y_buf, jint y_row_stride,
    jobject u_buf, jint u_row_stride, jint u_pixel_stride, jobject v_buf,
    jint v_row_stride, jint v_pixel_stride, jint width, jint height,
    jobject depth_buf, jint depth_width, jint depth_height, jobject conf_buf,
    jfloatArray pose_matrix, jfloat fx, jfloat fy, jfloat cx, jfloat cy,
    jstring image_file_path, jstring image_relative_path) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return JNI_FALSE;

  // Synchronously copy all direct buffers into std::vectors
  FrameTask task;
  task.width = width;
  task.height = height;
  task.y_stride = y_row_stride;
  task.u_stride = u_row_stride;
  task.u_pixel_stride = u_pixel_stride;
  task.v_stride = v_row_stride;
  task.v_pixel_stride = v_pixel_stride;

  uint8_t *y_addr =
      reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(y_buf));
  uint8_t *u_addr =
      reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(u_buf));
  uint8_t *v_addr =
      reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(v_buf));

  // Android YUV420 Image sizes: Y size = width * height, U/V sizes depend on
  // strides and layout. Copy the raw plane buffers safely based on layout
  jlong y_cap = env->GetDirectBufferCapacity(y_buf);
  task.y_data.assign(y_addr, y_addr + y_cap);

  // Since U/V planes are typically sub-sampled by 2, their row height is height
  // / 2. Stride extends to cover row padding.
  jlong u_cap = env->GetDirectBufferCapacity(u_buf);
  task.u_data.assign(u_addr, u_addr + u_cap);

  jlong v_cap = env->GetDirectBufferCapacity(v_buf);
  task.v_data.assign(v_addr, v_addr + v_cap);

  // Copy Raw Depth & Confidence data
  task.depth_width = depth_width;
  task.depth_height = depth_height;
  uint16_t *depth_addr =
      reinterpret_cast<uint16_t *>(env->GetDirectBufferAddress(depth_buf));
  task.depth_data.assign(depth_addr, depth_addr + depth_width * depth_height);

  uint8_t *conf_addr =
      reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(conf_buf));
  task.confidence_data.assign(conf_addr,
                              conf_addr + depth_width * depth_height);

  // Copy Pose Matrix (16 float array)
  jfloat *pose_elements = env->GetFloatArrayElements(pose_matrix, nullptr);
  std::copy(pose_elements, pose_elements + 16, task.pose_matrix.begin());
  env->ReleaseFloatArrayElements(pose_matrix, pose_elements, JNI_ABORT);

  task.fx = fx;
  task.fy = fy;
  task.cx = cx;
  task.cy = cy;

  const char *path_chars = env->GetStringUTFChars(image_file_path, nullptr);
  task.image_file_path = std::string(path_chars);
  env->ReleaseStringUTFChars(image_file_path, path_chars);

  const char *rel_chars = env->GetStringUTFChars(image_relative_path, nullptr);
  task.image_relative_path = std::string(rel_chars);
  env->ReleaseStringUTFChars(image_relative_path, rel_chars);

  pipeline->EnqueueFrame(std::move(task));
  return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getPointCount(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return 0;
  return pipeline->GetPointCount();
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_exportDataset(
    JNIEnv *env, jobject thiz, jlong handle, jstring output_dir) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return;

  const char *dir_chars = env->GetStringUTFChars(output_dir, nullptr);
  std::string output_dir_str(dir_chars);
  env->ReleaseStringUTFChars(output_dir, dir_chars);

  pipeline->ExportDataset(output_dir_str);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_startDepthGeneration(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return;
  pipeline->StartCompute();
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_clearPipeline(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return;
  pipeline->Clear();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getPendingFrames(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return 0;
  return pipeline->GetPendingFramesCount();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getProcessedFrames(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return 0;
  return pipeline->GetProcessedFramesCount();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getGpuStatus(
    JNIEnv *env, jobject thiz, jlong handle) {
  auto *pipeline = reinterpret_cast<SplatCapturePipeline *>(handle);
  if (!pipeline)
    return 0;
  return pipeline->is_gpu_enabled_;
}
