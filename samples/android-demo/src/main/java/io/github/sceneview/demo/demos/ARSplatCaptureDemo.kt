package io.github.sceneview.demo.demos

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.LaunchedEffect
import kotlinx.coroutines.isActive
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.arcore.cameraImage
import io.github.sceneview.demo.DemoScaffold
import io.github.sceneview.demo.R
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberMaterialLoader
import io.github.sceneview.rememberModelLoader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.io.BufferedOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.sqrt
import kotlin.math.roundToInt
import android.app.Activity
import android.content.Context
import android.content.pm.ActivityInfo

object SplatCapturePipeline {
    init {
        System.loadLibrary("splat_capture_jni")
    }

    external fun initPipeline(modelPath: String): Long
    external fun freePipeline(handle: Long)
    external fun processFrame(
        handle: Long,
        yBuf: java.nio.ByteBuffer, yRowStride: Int,
        uBuf: java.nio.ByteBuffer, uRowStride: Int, uPixelStride: Int,
        vBuf: java.nio.ByteBuffer, vRowStride: Int, vPixelStride: Int,
        width: Int, height: Int,
        depthBuf: java.nio.ByteBuffer, depthWidth: Int, depthHeight: Int,
        confBuf: java.nio.ByteBuffer,
        pointCloudBuf: java.nio.FloatBuffer, pointCount: Int,
        poseMatrix: FloatArray,
        fx: Float, fy: Float, cx: Float, cy: Float,
        imageFilePath: String, imageRelativePath: String
    ): Boolean

    external fun getPointCount(handle: Long): Int
    external fun exportDataset(handle: Long, outputDir: String)
    external fun startDepthGeneration(handle: Long)
    external fun clearPipeline(handle: Long)
    external fun getPendingFrames(handle: Long): Int
    external fun getProcessedFrames(handle: Long): Int
    external fun getGpuStatus(handle: Long): Int
}

class CaptureContext {
    var lastPose: Pose? = null
}

private fun copyAssetToFile(context: Context, assetPath: String, outFile: File) {
    if (outFile.exists() && outFile.length() > 0) return
    context.assets.open(assetPath).use { inputStream ->
        FileOutputStream(outFile).use { outputStream ->
            inputStream.copyTo(outputStream)
        }
    }
}

@Composable
fun ArSplatCaptureDemo(onBack: () -> Unit) {
    val context = LocalContext.current
    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine)
    val materialLoader = rememberMaterialLoader(engine)
    val coroutineScope = rememberCoroutineScope()

    val activity = context as? Activity
    DisposableEffect(activity) {
        val originalOrientation = activity?.requestedOrientation ?: ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        onDispose {
            activity?.requestedOrientation = originalOrientation
        }
    }

    var isCapturing by remember { mutableStateOf(false) }
    var warmupFrameCount by remember { mutableIntStateOf(0) }
    var isGenerating by remember { mutableStateOf(false) }
    var isExporting by remember { mutableStateOf(false) }
    var isAutoFocus by remember { mutableStateOf(true) }
    var showDiscardDialog by remember { mutableStateOf(false) }
    var arSession by remember { mutableStateOf<com.google.ar.core.Session?>(null) }

    val internalFrameCount = remember { java.util.concurrent.atomic.AtomicInteger(0) }
    var displayFrameCount by remember { mutableIntStateOf(0) }
    var displayPointCount by remember { mutableIntStateOf(0) }
    var pendingFrames by remember { mutableIntStateOf(0) }
    var processedFrames by remember { mutableIntStateOf(0) }
    var totalFramesToProcess by remember { mutableIntStateOf(0) }
    var gpuStatus by remember { mutableIntStateOf(0) }

    val tempDir = remember {
        File(context.cacheDir, "splat_capture").apply {
            deleteRecursively()
            mkdirs()
            File(this, "images").mkdirs()
        }
    }

    val modelFile = remember { File(context.cacheDir, "da3_small_gpu_fp16.tflite") }
    var pipelineHandle by remember { mutableStateOf(0L) }
    var initError by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                copyAssetToFile(context, "models/da3_small_gpu_fp16.tflite", modelFile)
                val handle = SplatCapturePipeline.initPipeline(modelFile.absolutePath)
                if (handle == 0L) {
                    throw RuntimeException("JNI pipeline handle initialization failed.")
                }
                withContext(Dispatchers.Main) {
                    pipelineHandle = handle
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    initError = "Error loading model or initializing JNI: ${e.message}\nEnsure 'da3_small_gpu_fp16.tflite' exists in your assets/models folder."
                }
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            if (pipelineHandle != 0L) {
                SplatCapturePipeline.freePipeline(pipelineHandle)
                pipelineHandle = 0L
            }
        }
    }

    LaunchedEffect(isAutoFocus, arSession) {
        arSession?.let { session ->
            val config = session.config
            val newMode = if (isAutoFocus) Config.FocusMode.AUTO else Config.FocusMode.FIXED
            if (config.focusMode != newMode) {
                config.focusMode = newMode
                session.configure(config)
            }
        }
    }

    LaunchedEffect(isCapturing) {
        if (isCapturing) {
            warmupFrameCount = 0
        }
    }

    LaunchedEffect(isCapturing, isGenerating, pipelineHandle) {
        while (isActive) {
            displayFrameCount = internalFrameCount.get()
            if (pipelineHandle != 0L) {
                pendingFrames = SplatCapturePipeline.getPendingFrames(pipelineHandle)
                processedFrames = SplatCapturePipeline.getProcessedFrames(pipelineHandle)
                gpuStatus = SplatCapturePipeline.getGpuStatus(pipelineHandle)
                if (isGenerating && pendingFrames == 0) {
                    isGenerating = false
                }
                if (!isCapturing) {
                    displayPointCount = SplatCapturePipeline.getPointCount(pipelineHandle)
                } else {
                    displayPointCount = 0
                }
            }
            kotlinx.coroutines.delay(200)
        }
    }

    val captureContext = remember { CaptureContext() }

    var lastW by remember { mutableIntStateOf(0) }
    var lastH by remember { mutableIntStateOf(0) }

    val zipLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) {
            coroutineScope.launch(Dispatchers.IO) {
                val zipFile = File(context.cacheDir, "export.zip")
                if (zipFile.exists()) {
                    context.contentResolver.openOutputStream(uri)?.use { outStream ->
                        zipFile.inputStream().use { inStream ->
                            inStream.copyTo(outStream)
                        }
                    }
                }
            }
        }
    }

    DemoScaffold(
        title = stringResource(R.string.demo_ar_splat_capture_title),
        onBack = onBack,
        controls = {
            Text(
                text = "Capture a dataset for Gaussian Splatting. Move the camera slowly around an object. " +
                    "Frames are automatically captured when you move 10cm or rotate 15 degrees.",
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                androidx.compose.material3.Switch(
                    checked = isAutoFocus,
                    onCheckedChange = { isAutoFocus = it },
                    enabled = !isCapturing
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text("Auto Focus", style = MaterialTheme.typography.labelLarge)
                    if (isAutoFocus) {
                        Text("Warning: Not recommended for Gaussian Splats!", style = MaterialTheme.typography.labelSmall, color = Color.Red)
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))


            Spacer(modifier = Modifier.height(8.dp))

            if (gpuStatus == -1) {
                Text(
                    text = "Running on CPU (Slow). GPU Delegate initialization failed.",
                    color = Color.Red,
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }

            if (initError != null) {
                Text(
                    text = initError ?: "",
                    color = Color.Red,
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (isCapturing) {
                    Button(
                        onClick = { isCapturing = false },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.ar_splat_capture_stop))
                    }
                } else if (internalFrameCount.get() == 0) {
                    Button(
                        onClick = {
                            isCapturing = true
                            isGenerating = false
                            internalFrameCount.set(0)
                            displayFrameCount = 0
                            displayPointCount = 0
                            captureContext.lastPose = null
                            tempDir.deleteRecursively()
                            tempDir.mkdirs()
                            File(tempDir, "images").mkdirs()
                        },
                        enabled = pipelineHandle != 0L,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                    ) {
                        Icon(Icons.Default.FiberManualRecord, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.ar_splat_capture_start))
                    }
                } else {
                    // Left button: New Capture (ALWAYS present here)
                    Button(
                        onClick = {
                            if (pendingFrames == 0 && processedFrames > 0) {
                                showDiscardDialog = true
                            } else {
                                if (pipelineHandle != 0L) {
                                    SplatCapturePipeline.clearPipeline(pipelineHandle)
                                }
                                internalFrameCount.set(0)
                                displayFrameCount = 0
                                displayPointCount = 0
                                captureContext.lastPose = null
                                tempDir.deleteRecursively()
                                tempDir.mkdirs()
                                File(tempDir, "images").mkdirs()
                                isGenerating = false
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                    ) {
                        Text("New Capture")
                    }

                    // Right button: Generate OR Export
                    if (pendingFrames == 0 && processedFrames > 0) {
                        // Export Button
                        Button(
                            onClick = {
                                if (isExporting || pipelineHandle == 0L) return@Button
                                isExporting = true
                                coroutineScope.launch(Dispatchers.IO) {
                                    try {
                                        SplatCapturePipeline.exportDataset(pipelineHandle, tempDir.absolutePath)
                                        val zipFile = File(context.cacheDir, "export.zip")
                                        if (zipFile.exists()) {
                                            zipFile.delete()
                                        }
                                        zip(tempDir, zipFile)
                                        withContext(Dispatchers.Main) {
                                            zipLauncher.launch("splat_dataset.zip")
                                        }
                                    } catch (e: Exception) {
                                    } finally {
                                        withContext(Dispatchers.Main) { isExporting = false }
                                    }
                                }
                            },
                            enabled = !isExporting,
                            modifier = Modifier.weight(1f)
                        ) {
                            if (isExporting) {
                                androidx.compose.material3.CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = androidx.compose.material3.MaterialTheme.colorScheme.onPrimary,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Exporting...")
                            } else {
                                Icon(Icons.Default.Save, contentDescription = null)
                                Spacer(Modifier.width(8.dp))
                                Text(stringResource(R.string.ar_splat_capture_export))
                            }
                        }
                    } else {
                        // Generate Button
                        Button(
                            onClick = {
                                isGenerating = true
                                totalFramesToProcess = pendingFrames
                                SplatCapturePipeline.startDepthGeneration(pipelineHandle)
                            },
                            enabled = !isGenerating && pipelineHandle != 0L && pendingFrames > 0,
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            if (isGenerating) {
                                Text("Generating $processedFrames / $totalFramesToProcess")
                            } else {
                                Text("Generate Depth Maps")
                            }
                        }
                    }
                }
            }
        }
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            ARSceneView(
                modifier = Modifier.fillMaxSize(),
                engine = engine,
                modelLoader = modelLoader,
                materialLoader = materialLoader,
                sessionCameraConfig = { session ->
                    val configs = session.getSupportedCameraConfigs(com.google.ar.core.CameraConfigFilter(session))
                    val p1080_60 = configs.firstOrNull { it.imageSize.height >= 1080 && it.fpsRange.upper >= 60 }
                    val p720_60 = configs.firstOrNull { it.imageSize.height >= 720 && it.fpsRange.upper >= 60 }
                    p1080_60 ?: p720_60 ?: session.cameraConfig
                },
                    sessionConfiguration = { session, config ->
                        config.focusMode = if (isAutoFocus) Config.FocusMode.AUTO else Config.FocusMode.FIXED
                        config.depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                    },
                    onSessionUpdated = { session, frame ->
                        if (arSession != session) {
                            arSession = session
                        }
                        
                        if (isCapturing && frame.camera.trackingState == TrackingState.TRACKING && pipelineHandle != 0L) {
                            if (warmupFrameCount < 60) {
                                warmupFrameCount++
                                return@ARSceneView
                            }
                            
                            val currentPose = frame.camera.pose
                            val lastPose = captureContext.lastPose
                            
                            val shouldCapture = lastPose == null || run {
                                val dist = distance(lastPose, currentPose)
                                val angle = angleBetween(lastPose, currentPose)
                                dist > 0.10f || angle > 15.0f
                            }
                            
                            if (shouldCapture) {
                                captureContext.lastPose = currentPose
                                
                                var cameraImage: Image? = null
                                var rawDepthImage: Image? = null
                                var confidenceImage: Image? = null
                                
                                try {
                                    cameraImage = frame.cameraImage()
                                    rawDepthImage = frame.acquireRawDepthImage16Bits()
                                    confidenceImage = frame.acquireRawDepthConfidenceImage()
                                    var pointCloud: com.google.ar.core.PointCloud? = null
                                    
                                    try {
                                        pointCloud = frame.acquirePointCloud()
                                        if (cameraImage != null && rawDepthImage != null && confidenceImage != null && pointCloud != null) {
                                        val yPlane = cameraImage.planes[0]
                                        val uPlane = cameraImage.planes[1]
                                        val vPlane = cameraImage.planes[2]
                                        
                                        val depthPlane = rawDepthImage.planes[0]
                                        val confidencePlane = confidenceImage.planes[0]
                                        
                                        val poseMatrix = FloatArray(16)
                                        frame.camera.pose.toMatrix(poseMatrix, 0)
                                        
                                        val intrinsics = frame.camera.imageIntrinsics
                                        val camW = intrinsics.imageDimensions[0]
                                        val camH = intrinsics.imageDimensions[1]
                                        var fx = intrinsics.focalLength[0]
                                        var fy = intrinsics.focalLength[1]
                                        var cx = intrinsics.principalPoint[0]
                                        var cy = intrinsics.principalPoint[1]
                                        
                                        var focusDiopters = 0.0f
                                        var physicalFocalLength = 0.0f
                                        try {
                                            val metadata = frame.imageMetadata
                                            focusDiopters = metadata.getFloat(com.google.ar.core.ImageMetadata.LENS_FOCUS_DISTANCE)
                                            physicalFocalLength = metadata.getFloat(com.google.ar.core.ImageMetadata.LENS_FOCAL_LENGTH)
                                        } catch (e: Exception) {}
                                        
                                        if (focusDiopters > 0f && physicalFocalLength > 0f) {
                                            val multiplier = 1.0f / (1.0f - (focusDiopters * (physicalFocalLength / 1000.0f)))
                                            if (!multiplier.isNaN() && !multiplier.isInfinite()) {
                                                fx *= multiplier
                                                fy *= multiplier
                                            }
                                        }
                                        
                                        val aspect_ratio = camW.toFloat() / camH.toFloat()
                                        val target_aspect = 16.0f / 9.0f
                                        var W_cropped = camW
                                        var H_cropped = camH
                                        var offset_x = 0f
                                        var offset_y = 0f

                                        if (aspect_ratio >= target_aspect) {
                                            W_cropped = (camH * target_aspect).roundToInt() and 1.inv()
                                            offset_x = (((camW - W_cropped) / 2) and 1.inv()).toFloat()
                                        } else {
                                            H_cropped = (camW / target_aspect).roundToInt() and 1.inv()
                                            offset_y = (((camH - H_cropped) / 2) and 1.inv()).toFloat()
                                        }
                                        
                                        val cx_cropped = cx - offset_x
                                        val cy_cropped = cy - offset_y
                                        
                                        val fx_rot = fy
                                        val fy_rot = fx
                                        val cx_rot = H_cropped.toFloat() - cy_cropped
                                        val cy_rot = cx_cropped
                                        
                                        val scale_x = 504.0f / H_cropped.toFloat()
                                        val scale_y = 896.0f / W_cropped.toFloat()
                                        
                                        val fx_final = fx_rot * scale_x
                                        val fy_final = fy_rot * scale_y
                                        val cx_final = cx_rot * scale_x
                                        val cy_final = cy_rot * scale_y
                                        
                                        val timestamp = System.currentTimeMillis()
                                        val imageRelPath = "images/frame_${timestamp}.jpg"
                                        val imageFile = File(tempDir, imageRelPath)
                                        
                                        val success = SplatCapturePipeline.processFrame(
                                            pipelineHandle,
                                            yPlane.buffer, yPlane.rowStride,
                                            uPlane.buffer, uPlane.rowStride, uPlane.pixelStride,
                                            vPlane.buffer, vPlane.rowStride, vPlane.pixelStride,
                                            cameraImage.width, cameraImage.height,
                                            depthPlane.buffer, rawDepthImage.width, rawDepthImage.height,
                                            confidencePlane.buffer,
                                            pointCloud.points, pointCloud.points.capacity() / 4,
                                            poseMatrix,
                                            fx_final, fy_final, cx_final, cy_final,
                                            imageFile.absolutePath, imageRelPath
                                        )
                                        
                                        if (success) {
                                            internalFrameCount.incrementAndGet()
                                        }
                                    }
                                        } catch (e: Exception) {
                                        } finally {
                                            pointCloud?.close()
                                        }
                                    } catch (e: Exception) {
                                    } finally {
                                        cameraImage?.close()
                                        rawDepthImage?.close()
                                        confidenceImage?.close()
                                    }
                            }
                        }
                    }
                )

            Card(
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)),
                shape = RoundedCornerShape(24.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = stringResource(R.string.ar_splat_capture_frames, displayFrameCount), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    }
                    if (!isCapturing) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(text = stringResource(R.string.ar_splat_capture_points, displayPointCount), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                        }
                    }
                    if (isCapturing) {
                        Icon(Icons.Default.FiberManualRecord, null, tint = Color.Red, modifier = Modifier.size(12.dp))
                    } else if (isGenerating) {
                        androidx.compose.material3.CircularProgressIndicator(
                            modifier = Modifier.size(12.dp),
                            color = MaterialTheme.colorScheme.primary,
                            strokeWidth = 2.dp
                        )
                    }
                }
            }
        }
    }

    if (showDiscardDialog) {
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { showDiscardDialog = false },
            title = { Text("Discard Capture?") },
            text = { Text("Are you sure you want to discard this generated capture? All depth maps and points will be lost.") },
            confirmButton = {
                Button(
                    onClick = {
                        showDiscardDialog = false
                        if (pipelineHandle != 0L) {
                            SplatCapturePipeline.clearPipeline(pipelineHandle)
                        }
                        internalFrameCount.set(0)
                        displayFrameCount = 0
                        displayPointCount = 0
                        captureContext.lastPose = null
                        tempDir.deleteRecursively()
                        tempDir.mkdirs()
                        File(tempDir, "images").mkdirs()
                        isGenerating = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                ) {
                    Text("Discard")
                }
            },
            dismissButton = {
                Button(onClick = { showDiscardDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

private fun distance(p1: Pose, p2: Pose): Float {
    val dx = p1.tx() - p2.tx(); val dy = p1.ty() - p2.ty(); val dz = p1.tz() - p2.tz()
    return sqrt(dx * dx + dy * dy + dz * dz)
}

private fun angleBetween(p1: Pose, p2: Pose): Float {
    val dot = p1.qx() * p2.qx() + p1.qy() * p2.qy() + p1.qz() * p2.qz() + p1.qw() * p2.qw()
    return Math.toDegrees(2.0 * Math.acos(Math.min(1.0, Math.abs(dot).toDouble()))).toFloat()
}

private fun zip(directory: File, zipFile: File) {
    ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
        directory.walkTopDown().forEach { file ->
            val zipFileName = file.absolutePath.removePrefix(directory.absolutePath).removePrefix(File.separator)
            if (zipFileName.isNotEmpty()) {
                val entry = ZipEntry("$zipFileName${if (file.isDirectory) "/" else ""}")
                zos.putNextEntry(entry)
                if (file.isFile) {
                    file.inputStream().use { it.copyTo(zos) }
                }
            }
        }
    }
}
