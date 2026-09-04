#include <android/log.h>
#include <jni.h>
#include <memory>
#include <string>

#define LOG_TAG "SplatCaptureJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include "feature_extractor.h"
#include "match_culling.h"
#include "feature_matcher.h"
#include "triangulator.h"
#include "bundle_adjuster.h"
#include "exporter.h"
#include <fstream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <thread>
#include <atomic>

class SplatCapturePipeline {
public:
  SplatCapturePipeline(const std::string &model_path) {
      extractor_ = std::make_unique<FeatureExtractor>();
      if (!extractor_->initialize(model_path)) {
          LOGE("Failed to initialize FeatureExtractor");
      }
      culler_ = std::make_unique<MatchCuller>();
      matcher_ = std::make_unique<FeatureMatcher>();
      triangulator_ = std::make_unique<Triangulator>();
      bundle_adjuster_ = std::make_unique<BundleAdjuster>();
      exporter_ = std::make_unique<Exporter>();
  }
  
  ~SplatCapturePipeline() {}

  void ProcessDataset(const std::string &manifest_path) {
      std::lock_guard<std::mutex> lock(process_mutex_);
      current_phase_ = 0; // Starting
      cancel_ = false;
      LOGI("ProcessDataset called with manifest: %s", manifest_path.c_str());
      
      std::ifstream infile(manifest_path);
      if (!infile.is_open()) {
          LOGE("Failed to open manifest");
          return;
      }
      
      std::string line;
      std::getline(infile, line);
      std::stringstream ss(line);
      double fx, fy, cx, cy, w, h;
      std::string output_dir;
      ss >> output_dir;
      
      // Skip the count line
      std::getline(infile, line);
      
      std::vector<CameraPose> poses;
      std::vector<cv::Mat> images;
      std::vector<std::string> image_names;
      
      while (std::getline(infile, line)) {
          if (line.empty()) continue;
          std::stringstream ss2(line);
          std::string img_path;
          ss2 >> img_path;
          
          float fx, fy, cx, cy;
          int w, h;
          ss2 >> fx >> fy >> cx >> cy >> w >> h;
          
          cv::Mat pose_mat(4, 4, CV_32F);
          // Correct column-major parsing
          for (int c = 0; c < 4; ++c) {
              for (int r = 0; r < 4; ++r) {
                  ss2 >> pose_mat.at<float>(r, c);
              }
          }
          
          CameraPose pose;
          cv::Mat R_gl = pose_mat(cv::Rect(0, 0, 3, 3));
          cv::Mat t_gl = pose_mat(cv::Rect(3, 0, 1, 3));

          // Convert from ARCore Camera-to-World (OpenGL) to OpenCV World-to-Camera
          cv::Mat S = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, -1);
          pose.R = S * R_gl.t();
          pose.t = -pose.R * t_gl;
          pose.w = w;
          pose.h = h;
          pose.K = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
          
          poses.push_back(pose);
          
          cv::Mat img = cv::imread(img_path);
          images.push_back(img);
          
          size_t slash_pos = img_path.find_last_of('/');
          image_names.push_back((slash_pos != std::string::npos) ? img_path.substr(slash_pos + 1) : img_path);
      }
      
      LOGI("Loaded %zu frames", poses.size());
      
      // Feature Extraction
      current_phase_ = 1;
      auto t0 = std::chrono::steady_clock::now();
      std::vector<std::vector<Keypoint>> all_keypoints(poses.size());
      for (size_t i = 0; i < images.size(); ++i) {
          if (cancel_) return;
          all_keypoints[i] = extractor_->extractFeatures(images[i]);
      }
      auto t1 = std::chrono::steady_clock::now();
      LOGI("Phase 1 (Extraction) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
      
      // Match Culling
      current_phase_ = 2;
      auto pairs = culler_->getValidPairs(poses);
      auto t2 = std::chrono::steady_clock::now();
      LOGI("Phase 2 (Culling) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
      
      current_phase_ = 3;
          auto t_match_start = std::chrono::steady_clock::now();
          
          // Setup Union-Find
          std::vector<int> cam_starts(images.size(), 0);
          int total_kpts = 0;
          for (size_t i = 0; i < all_keypoints.size(); ++i) {
              cam_starts[i] = total_kpts;
              total_kpts += all_keypoints[i].size();
          }
          
          std::vector<int> parent(total_kpts);
          for (int i = 0; i < total_kpts; ++i) parent[i] = i;
          
          auto find_set = [&](int i) {
              int root = i;
              while (root != parent[root]) root = parent[root];
              int curr = i;
              while (curr != root) {
                  int nxt = parent[curr];
                  parent[curr] = root;
                  curr = nxt;
              }
              return root;
          };
          auto union_set = [&](int i, int j) {
              int root_i = find_set(i);
              int root_j = find_set(j);
              if (root_i != root_j) parent[root_j] = root_i;
          };
          
          // Parallel Matching
          std::vector<std::vector<FeatureMatch>> thread_matches(pairs.size());
          
          int num_threads = std::thread::hardware_concurrency();
          if (num_threads == 0) num_threads = 4;
          std::vector<std::thread> workers;
          std::atomic<int> current_idx{0};
          for (int t = 0; t < num_threads; ++t) {
              workers.emplace_back([&]() {
                  while (!cancel_) {
                      int i = current_idx.fetch_add(1);
                      if (i >= pairs.size()) break;
                      if (cancel_) return;
                      const auto& pair = pairs[i];
                      auto matches = matcher_->matchMNN(all_keypoints[pair.first], all_keypoints[pair.second]);
                      thread_matches[i] = matcher_->filterEpipolar(matches, all_keypoints[pair.first], all_keypoints[pair.second], poses[pair.first], poses[pair.second]);
                  }
              });
          }
          for (auto& w : workers) { w.join(); }
          
          if (cancel_) return;
          for (size_t i = 0; i < pairs.size(); ++i) {
              const auto& pair = pairs[i];
              for (const auto& m : thread_matches[i]) {
                  int global_a = cam_starts[pair.first] + m.idx_a;
                  int global_b = cam_starts[pair.second] + m.idx_b;
                  union_set(global_a, global_b);
              }
          }
          
          auto t_match_end = std::chrono::steady_clock::now();
          LOGI("Phase 3 (Matching) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_match_end - t_match_start).count());
          
          // Collect Tracks
          auto t_tri_start = std::chrono::steady_clock::now();
          std::map<int, Track> track_map;
          for (size_t c = 0; c < all_keypoints.size(); ++c) {
              for (size_t k = 0; k < all_keypoints[c].size(); ++k) {
                  int global_id = cam_starts[c] + k;
                  int root = find_set(global_id);
                  track_map[root].observations.push_back({static_cast<int>(c), all_keypoints[c][k].pt});
              }
          }
          
          std::vector<Track> flat_tracks;
          flat_tracks.reserve(track_map.size());
          for (auto& kv : track_map) {
              flat_tracks.push_back(std::move(kv.second));
          }
          
          // Parallel Triangulation
          std::vector<bool> valid_flags(flat_tracks.size(), false);
          
          int num_threads2 = std::thread::hardware_concurrency();
          if (num_threads2 == 0) num_threads2 = 4;
          std::vector<std::thread> workers2;
          std::atomic<int> current_idx2{0};
          for (int t = 0; t < num_threads2; ++t) {
              workers2.emplace_back([&]() {
                  while (!cancel_) {
                      int i = current_idx2.fetch_add(1);
                      if (i >= flat_tracks.size()) break;
                      if (cancel_) return;
                      Track& t = flat_tracks[i];
                      
                      std::set<int> seen_cams;
                      bool valid = true;
                      for (const auto& obs : t.observations) {
                          if (seen_cams.count(obs.camera_idx)) {
                              valid = false;
                              break;
                          }
                          seen_cams.insert(obs.camera_idx);
                      }
                      
                      if (valid && t.observations.size() >= 2) {
                          if (triangulator_->triangulateTrack(t, poses)) {
                              valid_flags[i] = true;
                          }
                      }
                  }
              });
          }
          for (auto& w : workers2) { w.join(); }
          
          if (cancel_) return;
          std::vector<Track> tracks;
          tracks.reserve(flat_tracks.size());
          for (size_t i = 0; i < flat_tracks.size(); ++i) {
              if (valid_flags[i]) {
                  tracks.push_back(std::move(flat_tracks[i]));
              }
          }
          
          auto t_tri_end = std::chrono::steady_clock::now();
          LOGI("Phase 3 (Triangulation) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_tri_end - t_tri_start).count());
          
          // Bundle Adjustment
          current_phase_ = 4;
          auto t_ba_start = std::chrono::steady_clock::now();
          bundle_adjuster_->optimize(poses, tracks, 20);
          auto t_ba_end = std::chrono::steady_clock::now();
          LOGI("Phase 4 (BA) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_ba_end - t_ba_start).count());
          
      point_count_ = tracks.size();
      
      // Export
      if (cancel_) return;
      current_phase_ = 5;
      auto t_export_start = std::chrono::steady_clock::now();
      exporter_->exportNerfstudio(images, image_names, poses, tracks, output_dir);
      auto t_export_end = std::chrono::steady_clock::now();
      LOGI("Phase 5 (Export) took %lld ms", (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_export_end - t_export_start).count());
      
            current_phase_ = 6; // Complete
  }

  void Clear() { cancel_ = true; }
  int GetPendingFramesCount() { return 0; }
  int GetProcessedFramesCount() { return 0; }
  int GetPointCount() { return point_count_; }
  int GetProcessingPhase() { return current_phase_.load(); }

private:
  std::atomic<bool> cancel_{false};
  std::atomic<int> current_phase_{0};
  std::unique_ptr<FeatureExtractor> extractor_;
  std::unique_ptr<MatchCuller> culler_;
  std::unique_ptr<FeatureMatcher> matcher_;
  std::unique_ptr<Triangulator> triangulator_;
  std::unique_ptr<BundleAdjuster> bundle_adjuster_;
  std::unique_ptr<Exporter> exporter_;
  std::atomic<int> point_count_{0};
  std::mutex process_mutex_;
};

// Removed g_pipeline
extern "C" JNIEXPORT jlong JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_initPipeline(
    JNIEnv *env, jobject thiz, jstring model_path) {
  const char *path_chars = env->GetStringUTFChars(model_path, nullptr);
  std::string model_path_str(path_chars);
  env->ReleaseStringUTFChars(model_path, path_chars);

  SplatCapturePipeline* pipeline = new SplatCapturePipeline(model_path_str);
  return reinterpret_cast<jlong>(pipeline);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_freePipeline(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) {
    delete pipeline;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_processDataset(
    JNIEnv *env, jobject thiz, jlong handle, jstring manifest_path) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (!pipeline) return;
  const char *path_chars = env->GetStringUTFChars(manifest_path, nullptr);
  std::string path_str(path_chars);
  env->ReleaseStringUTFChars(manifest_path, path_chars);
  
  pipeline->ProcessDataset(path_str);
}

// Temporary processFrame to keep it compiling while Kotlin is updated
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_processFrame(
    JNIEnv *env, jobject thiz, jlong handle, jobject y_buf, jint y_row_stride,
    jobject u_buf, jint u_row_stride, jint u_pixel_stride, jobject v_buf,
    jint v_row_stride, jint v_pixel_stride, jint width, jint height,
    jobject depth_buf, jint depth_width, jint depth_height, jobject conf_buf,
    jobject point_cloud_buf, jint point_count, jfloatArray pose_matrix,
    jfloat fx, jfloat fy, jfloat cx, jfloat cy, jstring image_file_path,
    jstring image_relative_path) {
  // OBSOLETE: To be replaced by processDataset
  return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getPointCount(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) return pipeline->GetPointCount();
  return 0;
}

// Removed exportDataset

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_startDepthGeneration(
    JNIEnv *env, jobject thiz, jlong handle) {
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_clearPipeline(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) pipeline->Clear();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getPendingFrames(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) return pipeline->GetPendingFramesCount();
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getProcessedFrames(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) return pipeline->GetProcessedFramesCount();
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getGpuStatus(
    JNIEnv *env, jobject thiz, jlong handle) {
  return 1; // Fake GPU status
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_sceneview_demo_demos_SplatCapturePipeline_getProcessingPhase(
    JNIEnv *env, jobject thiz, jlong handle) {
  SplatCapturePipeline* pipeline = reinterpret_cast<SplatCapturePipeline*>(handle);
  if (pipeline) return pipeline->GetProcessingPhase();
  return 0;
}
