cmake_minimum_required(VERSION 3.22.1)
project(splat_capture_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Enable Neon optimizations for ARM architectures
if(ANDROID_ABI MATCHES "^(arm64-v8a|armeabi-v7a)$")
    add_definitions(-DLIBYUV_NEON)
endif()

include(FetchContent)

# 1. Fetch Chromium libyuv from Git repository
FetchContent_Declare(
  libyuv
  GIT_REPOSITORY https://chromium.googlesource.com/libyuv/libyuv.git
  GIT_TAG        afff0cf6
)

# Configure libyuv build options
set(libyuv_disable_jpeg ON CACHE BOOL "Disable JPEG support in libyuv" FORCE)
set(libyuv_disable_tests ON CACHE BOOL "Disable tests in libyuv" FORCE)

FetchContent_MakeAvailable(libyuv)

# 2. Fetch LiteRT Core AAR (as a zip archive)
FetchContent_Declare(
  tflite_aar
  URL https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.6/litert-2.1.6.aar
)
FetchContent_MakeAvailable(tflite_aar)

# Define imported shared libraries for LiteRT
add_library(tensorflow-lite SHARED IMPORTED)
set_target_properties(tensorflow-lite PROPERTIES
    IMPORTED_LOCATION ${tflite_aar_SOURCE_DIR}/jni/${ANDROID_ABI}/libLiteRt.so
)

# Add our JNI library
add_library(splat_capture_jni SHARED
    splat_capture_jni.cpp
)

# Set include directories
target_include_directories(splat_capture_jni PRIVATE
    litert_github
    tflite_include
    ${libyuv_SOURCE_DIR}/include
)

# Link splat_capture_jni against yuv, TFLite core, and Android log/libraries
target_link_libraries(splat_capture_jni
    yuv
    tensorflow-lite
    android
    log
)
