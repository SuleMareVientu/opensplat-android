cmake_minimum_required(VERSION 3.22.1)
project(ceres_bundle_adjuster LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ceres
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ceres/internal/miniglog
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
)

add_library(ceres STATIC IMPORTED)
set_target_properties(ceres PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/${ANDROID_ABI}/libceres.a
)

add_library(ceres_ba SHARED
    CeresBundleAdjuster.cpp
    CeresNativeBridge.cpp
)

target_link_options(ceres_ba PRIVATE
    "-Wl,-z,max-page-size=16384"
)

target_link_libraries(ceres_ba
    ceres
    log
    m
)
