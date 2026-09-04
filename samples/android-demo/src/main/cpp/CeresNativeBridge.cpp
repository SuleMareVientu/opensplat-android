#include <jni.h>
#include "CeresBundleAdjuster.h"

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_sceneview_demo_demos_internal_CeresNativeBridge_nativeRunBundleAdjustment(
    JNIEnv* env,
    jclass /* clazz */,
    jint numCameras,
    jfloatArray camerasToWorld,
    jfloatArray cameraIntrinsics,
    jint numPoints,
    jfloatArray pointCoords,
    jint numObservations,
    jintArray obsPointIndices,
    jintArray obsCameraIndices,
    jfloatArray obsPixels,
    jint maxIterations
) {
    if (!camerasToWorld || !cameraIntrinsics || !pointCoords ||
        !obsPointIndices || !obsCameraIndices || !obsPixels) {
        return JNI_FALSE;
    }

    jfloat* c2wPtr = env->GetFloatArrayElements(camerasToWorld, nullptr);
    jfloat* intrinsicsPtr = env->GetFloatArrayElements(cameraIntrinsics, nullptr);
    jfloat* ptsPtr = env->GetFloatArrayElements(pointCoords, nullptr);
    jint* ptIdxPtr = env->GetIntArrayElements(obsPointIndices, nullptr);
    jint* camIdxPtr = env->GetIntArrayElements(obsCameraIndices, nullptr);
    jfloat* pixPtr = env->GetFloatArrayElements(obsPixels, nullptr);

    if (!c2wPtr || !intrinsicsPtr || !ptsPtr || !ptIdxPtr || !camIdxPtr || !pixPtr) {
        if (c2wPtr) env->ReleaseFloatArrayElements(camerasToWorld, c2wPtr, JNI_ABORT);
        if (intrinsicsPtr) env->ReleaseFloatArrayElements(cameraIntrinsics, intrinsicsPtr, JNI_ABORT);
        if (ptsPtr) env->ReleaseFloatArrayElements(pointCoords, ptsPtr, JNI_ABORT);
        if (ptIdxPtr) env->ReleaseIntArrayElements(obsPointIndices, ptIdxPtr, JNI_ABORT);
        if (camIdxPtr) env->ReleaseIntArrayElements(obsCameraIndices, camIdxPtr, JNI_ABORT);
        if (pixPtr) env->ReleaseFloatArrayElements(obsPixels, pixPtr, JNI_ABORT);
        return JNI_FALSE;
    }

    opensplat::BundleAdjustmentResult result = opensplat::CeresBundleAdjuster::RunBundleAdjustment(
        numCameras,
        c2wPtr,
        intrinsicsPtr,
        numPoints,
        ptsPtr,
        numObservations,
        ptIdxPtr,
        camIdxPtr,
        pixPtr,
        maxIterations
    );

    // Release and copy back modified arrays (0 mode commits and frees)
    env->ReleaseFloatArrayElements(camerasToWorld, c2wPtr, 0);
    env->ReleaseFloatArrayElements(pointCoords, ptsPtr, 0);

    // Release unmodified input arrays (JNI_ABORT mode frees without copying back)
    env->ReleaseFloatArrayElements(cameraIntrinsics, intrinsicsPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(obsPointIndices, ptIdxPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(obsCameraIndices, camIdxPtr, JNI_ABORT);
    env->ReleaseFloatArrayElements(obsPixels, pixPtr, JNI_ABORT);

    return result.success ? JNI_TRUE : JNI_FALSE;
}
