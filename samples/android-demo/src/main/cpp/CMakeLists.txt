cmake_minimum_required(VERSION 3.22.1)
project(splat_capture_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Enable Neon optimizations for ARM architectures
if(ANDROID_ABI MATCHES "^(arm64-v8a|armeabi-v7a)$")
    add_definitions(-DLIBYUV_NEON)
endif()

set(BUILD_TESTING OFF CACHE BOOL "Disable testing globally" FORCE)

include(FetchContent)



# 2. Fetch LiteRT Core AAR (as a zip archive)
FetchContent_Declare(
  tflite_aar
  URL https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.6/litert-2.1.6.aar
)
FetchContent_MakeAvailable(tflite_aar)

# Fetch TensorFlow Lite Headers
FetchContent_Declare(
  tflite_headers
  URL https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.1.zip
)
FetchContent_MakeAvailable(tflite_headers)

# Define imported shared libraries for LiteRT
add_library(tensorflow-lite SHARED IMPORTED)
set_target_properties(tensorflow-lite PROPERTIES
    IMPORTED_LOCATION ${tflite_aar_SOURCE_DIR}/jni/${ANDROID_ABI}/libLiteRt.so
)

# 3. Fetch opencv-mobile
FetchContent_Declare(
  opencv_mobile
  URL https://github.com/nihui/opencv-mobile/releases/download/v36/opencv-mobile-5.0.0-android.zip
)
FetchContent_MakeAvailable(opencv_mobile)
set(OpenCV_DIR ${opencv_mobile_SOURCE_DIR}/sdk/native/jni)
find_package(OpenCV REQUIRED)

# 4. Fetch Eigen3 (Header-only dependency for Ceres)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
# Since Eigen is header-only, we don't need to build it.
set(Eigen3_DIR ${eigen_SOURCE_DIR}/cmake)

# 5. Fetch Ceres Solver
FetchContent_Declare(
  ceres
  GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver
  GIT_TAG 2.2.0
)
set(MINIGLOG ON CACHE BOOL "Use miniglog" FORCE)
set(GFLAGS OFF CACHE BOOL "Disable gflags" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Disable suitesparse" FORCE)
set(CXSPARSE OFF CACHE BOOL "Disable cxsparse" FORCE)
set(EIGENSPARSE ON CACHE BOOL "Enable eigensparse" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable testing" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable uninstall to prevent clashes" FORCE)
set(Eigen3_DIR ${eigen_BINARY_DIR})
FetchContent_MakeAvailable(ceres)

# Add our JNI library
add_library(splat_capture_jni SHARED
    splat_capture_jni.cpp
    feature_extractor.cpp
    match_culling.cpp
    feature_matcher.cpp
    triangulator.cpp
    bundle_adjuster.cpp
    exporter.cpp
)

# Set include directories
target_include_directories(splat_capture_jni PRIVATE
    ${tflite_headers_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${eigen_SOURCE_DIR}
)

# Link splat_capture_jni against yuv, TFLite core, OpenCV, Ceres, and Android log/libraries
target_link_libraries(splat_capture_jni
    tensorflow-lite
    ${OpenCV_LIBS}
    ceres
    android
    log
)
