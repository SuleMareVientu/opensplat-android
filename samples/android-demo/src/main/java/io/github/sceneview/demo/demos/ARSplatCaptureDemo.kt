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
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.sqrt

@Serializable
data class NerfstudioTransform(
    val camera_model: String = "PERSPECTIVE",
    val w: Int,
    val h: Int,
    val ply_file_path: String = "points.ply",
    val frames: List<NerfstudioFrame>
)

@Serializable
data class NerfstudioFrame(
    val file_path: String,
    val fl_x: Float,
    val fl_y: Float,
    val cx: Float,
    val cy: Float,
    val transform_matrix: List<List<Float>>
)

class PointData(
    var x: Float,
    var y: Float,
    var z: Float,
    var r: Int,
    var g: Int,
    var b: Int,
    var observations: Int = 1,
    var isHighAccuracy: Boolean = true,
    var distance: Float = 0f,
    var textureVariance: Float = 0f,
    var voxelKey: Long = 0L
)

class PrimitivePointList(initialCapacity: Int = 5000) {
    var size = 0
        private set
    var xyz = FloatArray(initialCapacity * 3)
    var rgb = ByteArray(initialCapacity * 3)
    var highAccuracy = BooleanArray(initialCapacity)
    var distance = FloatArray(initialCapacity)
    var textureVariance = FloatArray(initialCapacity)

    fun add(x: Float, y: Float, z: Float, r: Int, g: Int, b: Int, highAcc: Boolean, dist: Float, variance: Float) {
        if (size >= xyz.size / 3) {
            val newCap = size * 2
            xyz = xyz.copyOf(newCap * 3)
            rgb = rgb.copyOf(newCap * 3)
            highAccuracy = highAccuracy.copyOf(newCap)
            distance = distance.copyOf(newCap)
            textureVariance = textureVariance.copyOf(newCap)
        }
        val idx3 = size * 3
        xyz[idx3] = x; xyz[idx3 + 1] = y; xyz[idx3 + 2] = z
        rgb[idx3] = r.toByte(); rgb[idx3 + 1] = g.toByte(); rgb[idx3 + 2] = b.toByte()
        highAccuracy[size] = highAcc
        distance[size] = dist
        textureVariance[size] = variance
        size++
    }

    fun addAll(other: PrimitivePointList) {
        val newSize = size + other.size
        if (newSize > xyz.size / 3) {
            val newCap = Math.max(xyz.size / 3 * 2, newSize)
            xyz = xyz.copyOf(newCap * 3)
            rgb = rgb.copyOf(newCap * 3)
            highAccuracy = highAccuracy.copyOf(newCap)
            distance = distance.copyOf(newCap)
            textureVariance = textureVariance.copyOf(newCap)
        }
        System.arraycopy(other.xyz, 0, xyz, size * 3, other.size * 3)
        System.arraycopy(other.rgb, 0, rgb, size * 3, other.size * 3)
        System.arraycopy(other.highAccuracy, 0, highAccuracy, size, other.size)
        System.arraycopy(other.distance, 0, distance, size, other.size)
        System.arraycopy(other.textureVariance, 0, textureVariance, size, other.size)
        size = newSize
    }
}

class IntHashSet(initialCapacity: Int) {
    private var keys = IntArray(initialCapacity) { Int.MIN_VALUE }
    var size = 0
        private set

    fun add(key: Int): Boolean {
        if (size >= keys.size * 0.75f) {
            resize(keys.size * 2)
        }
        val safeKey = if (key == Int.MIN_VALUE) Int.MAX_VALUE else key
        var idx = (safeKey and 0x7FFFFFFF) % keys.size
        while (keys[idx] != Int.MIN_VALUE) {
            if (keys[idx] == safeKey) return false
            idx = (idx + 1) % keys.size
        }
        keys[idx] = safeKey
        size++
        return true
    }

    private fun resize(newCapacity: Int) {
        val oldKeys = keys
        keys = IntArray(newCapacity) { Int.MIN_VALUE }
        size = 0
        for (key in oldKeys) {
            if (key != Int.MIN_VALUE) {
                add(key)
            }
        }
    }
}

class OccupancyVoxel {
    val frames = HashSet<Int>()
    var hasHighAccuracy = false
    var rSum = 0
    var gSum = 0
    var bSum = 0
    var highAccCount = 0
}

data class CapturedFrameData(
    val anchor: Anchor,
    val filePath: String,
    val fl_x: Float,
    val fl_y: Float,
    val cx: Float,
    val cy: Float,
    val localPoints: PrimitivePointList = PrimitivePointList()
)

object ByteArrayPool {
    private val pool = java.util.concurrent.ConcurrentLinkedQueue<ByteArray>()
    fun acquire(size: Int): ByteArray {
        val arr = pool.poll()
        if (arr != null && arr.size == size) return arr
        return ByteArray(size)
    }
    fun release(arr: ByteArray) {
        pool.offer(arr)
    }
}

object ShortArrayPool {
    private val pool = java.util.concurrent.ConcurrentLinkedQueue<ShortArray>()
    fun acquire(size: Int): ShortArray {
        val arr = pool.poll()
        if (arr != null && arr.size == size) return arr
        return ShortArray(size)
    }
    fun release(arr: ShortArray) {
        pool.offer(arr)
    }
}

private fun packVoxel(x: Int, y: Int, z: Int): Long {
    val mask = 0x1FFFFFL
    return ((x.toLong() and mask) shl 42) or
           ((y.toLong() and mask) shl 21) or
           (z.toLong() and mask)
}

private fun unpackX(packed: Long): Int {
    val x21 = (packed shr 42) and 0x1FFFFFL
    return if (x21 >= 0x100000L) (x21 - 0x200000L).toInt() else x21.toInt()
}

private fun unpackY(packed: Long): Int {
    val y21 = (packed shr 21) and 0x1FFFFFL
    return if (y21 >= 0x100000L) (y21 - 0x200000L).toInt() else y21.toInt()
}

private fun unpackZ(packed: Long): Int {
    val z21 = packed and 0x1FFFFFL
    return if (z21 >= 0x100000L) (z21 - 0x200000L).toInt() else z21.toInt()
}

@Suppress("NOTHING_TO_INLINE")
private inline fun fastRound(value: Float): Int {
    return if (value >= 0f) (value + 0.5f).toInt() else (value - 0.5f).toInt()
}

class CaptureContext {
    var lastDepthTime = 0L
    var lastPose: Pose? = null
}

@Composable
fun ArSplatCaptureDemo(onBack: () -> Unit) {
    val context = LocalContext.current
    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine)
    val materialLoader = rememberMaterialLoader(engine)
    val coroutineScope = rememberCoroutineScope()

    var isCapturing by remember { mutableStateOf(false) }
    var isExporting by remember { mutableStateOf(false) }
    var isAutoFocus by remember { mutableStateOf(true) }
    var arSession by remember { mutableStateOf<com.google.ar.core.Session?>(null) }
    var targetResolutionIdx by remember { mutableStateOf(1f) } // Default to 720p
    val resolutions = listOf("480p", "720p", "1080p")
    val resolutionHeights = listOf(480, 720, 1080)

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

    val internalFrameCount = remember { java.util.concurrent.atomic.AtomicInteger(0) }
    val internalPointCount = remember { java.util.concurrent.atomic.AtomicInteger(0) }
    var displayFrameCount by remember { mutableIntStateOf(0) }
    var displayPointCount by remember { mutableIntStateOf(0) }

    LaunchedEffect(isCapturing) {
        if (isCapturing) {
            while (isActive) {
                displayFrameCount = internalFrameCount.get()
                displayPointCount = internalPointCount.get()
                kotlinx.coroutines.delay(200)
            }
        } else {
            displayFrameCount = internalFrameCount.get()
            displayPointCount = internalPointCount.get()
        }
    }

    // Capture context for tracking capture states without triggering recomposition
    val captureContext = remember { CaptureContext() }

    val capturedFrames = remember { mutableListOf<CapturedFrameData>() }
    val activeJobs = remember { ConcurrentHashMap.newKeySet<kotlinx.coroutines.Job>() }

    // Session intrinsics state
    var lastW by remember { mutableIntStateOf(0) }
    var lastH by remember { mutableIntStateOf(0) }

    val tempDir = remember {
        File(context.cacheDir, "splat_capture").apply {
            deleteRecursively()
            mkdirs()
            File(this, "images").mkdirs()
        }
    }

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
                    "Frames are automatically captured when you move 10cm or rotate 10 degrees.",
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

            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text("Capture Resolution: ", style = MaterialTheme.typography.labelLarge)
                androidx.compose.material3.Slider(
                    value = targetResolutionIdx,
                    onValueChange = {
                        val newIdx = Math.round(it).toFloat()
                        if (newIdx != targetResolutionIdx) {
                            targetResolutionIdx = newIdx
                            lastW = 0
                            lastH = 0
                        }
                    },
                    valueRange = 0f..2f,
                    steps = 1,
                    enabled = !isCapturing,
                    modifier = Modifier.weight(1f).padding(horizontal = 8.dp)
                )
                Text(resolutions[targetResolutionIdx.toInt()], style = MaterialTheme.typography.labelLarge)
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (!isCapturing) {
                    Button(
                        onClick = {
                            isCapturing = true
                            internalFrameCount.set(0)
                            internalPointCount.set(0)
                            displayFrameCount = 0
                            displayPointCount = 0
                            capturedFrames.clear()
                            captureContext.lastPose = null
                            tempDir.deleteRecursively()
                            tempDir.mkdirs()
                            File(tempDir, "images").mkdirs()
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                    ) {
                        Icon(Icons.Default.FiberManualRecord, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.ar_splat_capture_start))
                    }
                } else {
                    Button(
                        onClick = { isCapturing = false },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.ar_splat_capture_stop))
                    }
                }

                Button(
                    onClick = {
                        if (isExporting || capturedFrames.isEmpty()) return@Button
                        isExporting = true
                        coroutineScope.launch(Dispatchers.IO) {
                            try {
                                // Wait for all active background accumulation jobs to finish
                                activeJobs.toList().forEach { it.join() }

                                val jsonFile = File(tempDir, "transforms.json")
                                val framesList = capturedFrames.map { data ->
                                    val pose = data.anchor.pose
                                    val matrix = FloatArray(16)
                                    pose.toMatrix(matrix, 0)
                                    val transformMatrix = listOf(
                                        listOf(matrix[0], matrix[4], matrix[8], matrix[12]),
                                        listOf(matrix[1], matrix[5], matrix[9], matrix[13]),
                                        listOf(matrix[2], matrix[6], matrix[10], matrix[14]),
                                        listOf(matrix[3], matrix[7], matrix[11], matrix[15])
                                    )
                                    NerfstudioFrame(data.filePath, data.fl_x, data.fl_y, data.cx, data.cy, transformMatrix)
                                }
                                val nerfstudioData = NerfstudioTransform(w = lastW, h = lastH, frames = framesList)
                                val json = Json { encodeDefaults = true; prettyPrint = true }
                                jsonFile.writeText(json.encodeToString(nerfstudioData))

                                val plyFile = File(tempDir, "points.ply")
                                writePlyVoxelFiltered(capturedFrames, plyFile)
                                val zipFile = File(context.cacheDir, "export.zip")
                                zip(tempDir, zipFile)
                                withContext(Dispatchers.Main) {
                                    zipLauncher.launch("splat_dataset.zip")
                                }
                            } catch (e: Exception) {
                                // Ignore or log
                            } finally {
                                withContext(Dispatchers.Main) { isExporting = false }
                            }
                        }
                    },
                    enabled = !isCapturing && displayFrameCount > 0 && !isExporting,
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
            }
        }
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            androidx.compose.runtime.key(targetResolutionIdx) {
                ARSceneView(
                    modifier = Modifier.fillMaxSize(),
                    engine = engine,
                    modelLoader = modelLoader,
                    materialLoader = materialLoader,
                    sessionCameraConfig = { session ->
                        val targetHeight = resolutionHeights[targetResolutionIdx.toInt()]

                        // Try forcing 60 FPS to slice exposure time and reduce motion blur
                        val filter60 = com.google.ar.core.CameraConfigFilter(session)
                        filter60.targetFps = java.util.EnumSet.of(com.google.ar.core.CameraConfig.TargetFps.TARGET_FPS_60)
                        val configs60 = session.getSupportedCameraConfigs(filter60)

                        val configs = if (configs60.isNotEmpty()) configs60 else {
                            // Fallback to any FPS (30) if 60 is unavailable on this device
                            session.getSupportedCameraConfigs(com.google.ar.core.CameraConfigFilter(session))
                        }

                        configs.minByOrNull {
                            Math.abs(it.imageSize.height - targetHeight) + Math.abs(it.textureSize.height - targetHeight)
                        } ?: configs.first()
                    },
                    sessionConfiguration = { session, config ->
                        config.focusMode = if (isAutoFocus) Config.FocusMode.AUTO else Config.FocusMode.FIXED
                        when {
                            session.isDepthModeSupported(Config.DepthMode.AUTOMATIC) -> config.depthMode = Config.DepthMode.AUTOMATIC
                            session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY) -> config.depthMode = Config.DepthMode.RAW_DEPTH_ONLY
                        }
                    },
                    onSessionUpdated = { session, frame ->
                        if (arSession != session) {
                            arSession = session
                        }
                        if (lastW == 0 && frame.camera.trackingState == TrackingState.TRACKING) {
                            val intrinsics = frame.camera.imageIntrinsics
                            lastW = intrinsics.imageDimensions[0]
                            lastH = intrinsics.imageDimensions[1]
                        }
                        if (isCapturing && frame.camera.trackingState == TrackingState.TRACKING) {
                            val currentPose = frame.camera.pose
                            if (captureContext.lastPose == null || shouldCapture(captureContext.lastPose!!, currentPose)) {
                                captureContext.lastPose = currentPose
                                captureFrameBackground(frame, session, tempDir, coroutineScope) { data ->
                                    capturedFrames.add(data)
                                    internalFrameCount.set(capturedFrames.size)
                                }
                            }
                            val now = System.currentTimeMillis()
                            if (capturedFrames.isNotEmpty() && now - captureContext.lastDepthTime > 130) {
                                captureContext.lastDepthTime = now
                                val job = accumulateHybridPointCloudBackground(frame, capturedFrames.last(), coroutineScope) {
                                    internalPointCount.set(capturedFrames.sumOf { it.localPoints.size })
                                }
                                if (job != null) {
                                    activeJobs.add(job)
                                    job.invokeOnCompletion { activeJobs.remove(job) }
                                }
                            }
                        }
                    }
                )
            }

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
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = stringResource(R.string.ar_splat_capture_points, displayPointCount), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    }
                    if (isCapturing) {
                        Icon(Icons.Default.FiberManualRecord, null, tint = Color.Red, modifier = Modifier.size(12.dp))
                    }
                }
            }
        }
    }
}

private fun shouldCapture(lastPose: Pose, currentPose: Pose): Boolean {
    val dist = distance(lastPose, currentPose)
    val angle = angleBetween(lastPose, currentPose)
    return dist > 0.1f || angle > 10f
}

private fun distance(p1: Pose, p2: Pose): Float {
    val dx = p1.tx() - p2.tx(); val dy = p1.ty() - p2.ty(); val dz = p1.tz() - p2.tz()
    return sqrt(dx * dx + dy * dy + dz * dz)
}

private fun angleBetween(p1: Pose, p2: Pose): Float {
    val dot = p1.qx() * p2.qx() + p1.qy() * p2.qy() + p1.qz() * p2.qz() + p1.qw() * p2.qw()
    return Math.toDegrees(2.0 * Math.acos(Math.min(1.0, Math.abs(dot).toDouble()))).toFloat()
}

private fun accumulateHybridPointCloudBackground(
    frame: Frame,
    targetFrame: CapturedFrameData,
    coroutineScope: CoroutineScope,
    onUpdated: () -> Unit
): kotlinx.coroutines.Job? {
    var rawDepthImage: Image? = null
    var confidenceImage: Image? = null
    var smoothDepthImage: Image? = null
    var colorImage: Image? = null

    var rawDepthArray: ShortArray? = null
    var confArray: ByteArray? = null
    var smoothDepthArray: ShortArray? = null
    var colorNv21: ByteArray? = null

    try {
        rawDepthImage = runCatching { frame.acquireRawDepthImage16Bits() }.getOrNull() ?: return null
        confidenceImage = runCatching { frame.acquireRawDepthConfidenceImage() }.getOrNull() ?: return null
        smoothDepthImage = runCatching { frame.acquireDepthImage16Bits() }.getOrNull()
        colorImage = frame.cameraImage() ?: return null

        val depthWidth = rawDepthImage.width
        val depthHeight = rawDepthImage.height
        val colorWidth = colorImage.width
        val colorHeight = colorImage.height

        val cameraMatrix = FloatArray(16); frame.camera.pose.toMatrix(cameraMatrix, 0)
        val anchorMatrix = FloatArray(16); targetFrame.anchor.pose.toMatrix(anchorMatrix, 0)
        val worldToAnchorMatrix = FloatArray(16); android.opengl.Matrix.invertM(worldToAnchorMatrix, 0, anchorMatrix, 0)
        val localToAnchorMatrix = FloatArray(16); android.opengl.Matrix.multiplyMM(localToAnchorMatrix, 0, worldToAnchorMatrix, 0, cameraMatrix, 0)

        val intrinsics = frame.camera.imageIntrinsics
        val scaleX = depthWidth.toFloat() / intrinsics.imageDimensions[0].toFloat()
        val scaleY = depthHeight.toFloat() / intrinsics.imageDimensions[1].toFloat()
        val fx = intrinsics.focalLength[0] * scaleX; val fy = intrinsics.focalLength[1] * scaleY
        val cx = intrinsics.principalPoint[0] * scaleX; val cy = intrinsics.principalPoint[1] * scaleY
        val scaleU = colorWidth.toFloat() / depthWidth.toFloat(); val scaleV = colorHeight.toFloat() / depthHeight.toFloat()

        val rda = copyShortPlane(rawDepthImage.planes[0], depthWidth, depthHeight)
        rawDepthArray = rda
        val ca = copyBytePlane(confidenceImage.planes[0], depthWidth, depthHeight)
        confArray = ca
        val sda = smoothDepthImage?.let { copyShortPlane(it.planes[0], depthWidth, depthHeight) }
        smoothDepthArray = sda
        val cn = colorImage.toNv21ByteArray()
        colorNv21 = cn

        return coroutineScope.launch(Dispatchers.Default) {
            try {
                val newPoints = PrimitivePointList()
                val voxelGrid = IntHashSet(16384)

                for (v in 0 until depthHeight step 2) {
                    for (u in 0 until depthWidth step 2) {
                        val depthIdx = v * depthWidth + u

                        val rawDepthMm = rda[depthIdx].toInt() and 0xFFFF
                        val conf = ca[depthIdx].toInt() and 0xFF
                        val smoothDepthMm = sda?.get(depthIdx)?.toInt()?.and(0xFFFF) ?: 0

                        var depthMm = 0; var isHighAccuracy = false
                        if (rawDepthMm in 1..4000 && conf >= 230) { depthMm = rawDepthMm; isHighAccuracy = true }
                        else if (smoothDepthMm in 1..4000) { depthMm = smoothDepthMm; isHighAccuracy = false }
                        if (depthMm == 0) continue

                        val zCam = depthMm / 1000f
                        val xCam = (u - cx) * zCam / fx
                        val yCam = -(v - cy) * zCam / fy
                        val zCamCoord = -zCam

                        val xAnchor = localToAnchorMatrix[0] * xCam + localToAnchorMatrix[4] * yCam + localToAnchorMatrix[8] * zCamCoord + localToAnchorMatrix[12]
                        val yAnchor = localToAnchorMatrix[1] * xCam + localToAnchorMatrix[5] * yCam + localToAnchorMatrix[9] * zCamCoord + localToAnchorMatrix[13]
                        val zAnchor = localToAnchorMatrix[2] * xCam + localToAnchorMatrix[6] * yCam + localToAnchorMatrix[10] * zCamCoord + localToAnchorMatrix[14]

                        val colorU = (u * scaleU).toInt().coerceIn(0, colorWidth - 1)
                        val colorV = (v * scaleV).toInt().coerceIn(0, colorHeight - 1)
                        val color = getPixelColorFromNv21(cn, colorWidth, colorHeight, colorU, colorV)
                        val r = (color shr 16) and 0xFF
                        val g = (color shr 8) and 0xFF
                        val b = color and 0xFF
                        val variance = getTextureGradientFromNv21(cn, colorWidth, colorHeight, colorU, colorV)

                        val textureMultiplier = if (variance < 10f) 4f else if (variance < 30f) 2f else 1f
                        val localVoxelSize = (zCam * 0.01f * textureMultiplier).coerceIn(0.002f, 0.05f)
                        val vx = fastRound(xAnchor / localVoxelSize)
                        val vy = fastRound(yAnchor / localVoxelSize)
                        val vz = fastRound(zAnchor / localVoxelSize)
                        val hash = (vx * 73856093) xor (vy * 19349663) xor (vz * 83492791)
                        if (!voxelGrid.add(hash)) continue
                        newPoints.add(xAnchor, yAnchor, zAnchor, r, g, b, isHighAccuracy, zCam, variance)
                    }
                }
                withContext(Dispatchers.Main) {
                    targetFrame.localPoints.addAll(newPoints)
                    onUpdated()
                }
            } catch (e: Exception) {
            } finally {
                ShortArrayPool.release(rda)
                ByteArrayPool.release(ca)
                if (sda != null) ShortArrayPool.release(sda)
                ByteArrayPool.release(cn)
            }
        }
    } catch (e: Exception) {
        rawDepthArray?.let { ShortArrayPool.release(it) }
        confArray?.let { ByteArrayPool.release(it) }
        smoothDepthArray?.let { ShortArrayPool.release(it) }
        colorNv21?.let { ByteArrayPool.release(it) }
        return null
    } finally {
        rawDepthImage?.close()
        confidenceImage?.close()
        smoothDepthImage?.close()
        colorImage?.close()
    }
}

private fun writePlyVoxelFiltered(frames: List<CapturedFrameData>, file: File) {
    val occupancyGrid = HashMap<Long, OccupancyVoxel>()
    val occupancyVoxelSize = 0.02f
    frames.forEachIndexed { frameIndex, frameData ->
        val anchorMatrix = FloatArray(16); frameData.anchor.pose.toMatrix(anchorMatrix, 0)
        val pts = frameData.localPoints
        for (i in 0 until pts.size) {
            val i3 = i * 3
            val px = pts.xyz[i3]; val py = pts.xyz[i3+1]; val pz = pts.xyz[i3+2]
            val pHighAcc = pts.highAccuracy[i]
            val pr = pts.rgb[i3].toInt() and 0xFF; val pg = pts.rgb[i3+1].toInt() and 0xFF; val pb = pts.rgb[i3+2].toInt() and 0xFF

            val xWorld = anchorMatrix[0] * px + anchorMatrix[4] * py + anchorMatrix[8] * pz + anchorMatrix[12]
            val yWorld = anchorMatrix[1] * px + anchorMatrix[5] * py + anchorMatrix[9] * pz + anchorMatrix[13]
            val zWorld = anchorMatrix[2] * px + anchorMatrix[6] * py + anchorMatrix[10] * pz + anchorMatrix[14]
            
            val vx = fastRound(xWorld / occupancyVoxelSize)
            val vy = fastRound(yWorld / occupancyVoxelSize)
            val vz = fastRound(zWorld / occupancyVoxelSize)
            val coord = packVoxel(vx, vy, vz)
            
            val voxel = occupancyGrid.getOrPut(coord) { OccupancyVoxel() }
            voxel.frames.add(frameIndex)
            if (pHighAcc) {
                voxel.hasHighAccuracy = true; voxel.rSum += pr; voxel.gSum += pg; voxel.bSum += pb; voxel.highAccCount++
            }
        }
    }
    
    val populatedVoxels = HashMap<Long, Int>() // Store packed RGB color
    occupancyGrid.forEach { (coord, voxel) ->
        if (voxel.hasHighAccuracy && voxel.highAccCount > 0) {
            val rAvg = voxel.rSum / voxel.highAccCount
            val gAvg = voxel.gSum / voxel.highAccCount
            val bAvg = voxel.bSum / voxel.highAccCount
            populatedVoxels[coord] = (rAvg shl 16) or (gAvg shl 8) or bAvg
        }
    }
    
    val finalPoints = HashMap<Int, PointData>()
    frames.forEach { frameData ->
        val anchorMatrix = FloatArray(16); frameData.anchor.pose.toMatrix(anchorMatrix, 0)
        val pts = frameData.localPoints
        for (i in 0 until pts.size) {
            val i3 = i * 3
            val px = pts.xyz[i3]; val py = pts.xyz[i3+1]; val pz = pts.xyz[i3+2]
            val pr = pts.rgb[i3].toInt() and 0xFF; val pg = pts.rgb[i3+1].toInt() and 0xFF; val pb = pts.rgb[i3+2].toInt() and 0xFF
            val pHighAcc = pts.highAccuracy[i]
            val pDist = pts.distance[i]
            val pVar = pts.textureVariance[i]

            val xWorld = anchorMatrix[0] * px + anchorMatrix[4] * py + anchorMatrix[8] * pz + anchorMatrix[12]
            val yWorld = anchorMatrix[1] * px + anchorMatrix[5] * py + anchorMatrix[9] * pz + anchorMatrix[13]
            val zWorld = anchorMatrix[2] * px + anchorMatrix[6] * py + anchorMatrix[10] * pz + anchorMatrix[14]
            
            val vx = fastRound(xWorld / occupancyVoxelSize)
            val vy = fastRound(yWorld / occupancyVoxelSize)
            val vz = fastRound(zWorld / occupancyVoxelSize)
            val coord = packVoxel(vx, vy, vz)
            
            if (!pHighAcc) {
                val refColor = populatedVoxels[coord] ?: continue
                val refR = (refColor shr 16) and 0xFF
                val refG = (refColor shr 8) and 0xFF
                val refB = refColor and 0xFF
                val rDiff = pr - refR; val gDiff = pg - refG; val bDiff = pb - refB
                if (sqrt((rDiff*rDiff + gDiff*gDiff + bDiff*bDiff).toDouble()) > 60.0) continue
                if ((occupancyGrid[coord]?.frames?.size ?: 0) < 2) continue
            }
            val textureMultiplier = if (pVar < 10f) 4f else if (pVar < 30f) 2f else 1f
            val res = (pDist * 0.01f * textureMultiplier).coerceIn(0.002f, 0.05f)
            val hash = (fastRound(xWorld / res) * 73856093) xor (fastRound(yWorld / res) * 19349663) xor (fastRound(zWorld / res) * 83492791)
            val existing = finalPoints[hash]
            if (existing == null) finalPoints[hash] = PointData(xWorld, yWorld, zWorld, pr, pg, pb, 1, pHighAcc, pDist, pVar)
            else { val n = existing.observations; existing.r = (existing.r * n + pr) / (n + 1); existing.g = (existing.g * n + pg) / (n + 1); existing.b = (existing.b * n + pb) / (n + 1); existing.observations++ }
        }
    }
    
    val exportedPoints = finalPoints.values.toList()
    val outlierGrid = HashMap<Long, Int>()
    exportedPoints.forEach { p ->
        val coord = packVoxel(fastRound(p.x / 0.02f), fastRound(p.y / 0.02f), fastRound(p.z / 0.02f))
        p.voxelKey = coord
        outlierGrid[coord] = (outlierGrid[coord] ?: 0) + 1
    }
    
    val validVoxels = HashSet<Long>()
    outlierGrid.forEach { (coord, count) ->
        val cx = unpackX(coord)
        val cy = unpackY(coord)
        val cz = unpackZ(coord)
        var neighbors = count
        for (dx in -1..1) {
            for (dy in -1..1) {
                for (dz in -1..1) {
                    if (dx == 0 && dy == 0 && dz == 0) continue
                    val neighborCoord = packVoxel(cx + dx, cy + dy, cz + dz)
                    neighbors += outlierGrid[neighborCoord] ?: 0
                }
            }
        }
        if (neighbors >= 3) {
            validVoxels.add(coord)
        }
    }
    
    val finalCleanPoints = exportedPoints.filter { p ->
        p.voxelKey in validVoxels
    }
    
    file.outputStream().buffered().use { out ->
        val header = "ply\nformat binary_little_endian 1.0\nelement vertex ${finalCleanPoints.size}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        out.write(header.toByteArray(Charsets.US_ASCII))
        
        val buffer = java.nio.ByteBuffer.allocate(15).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        finalCleanPoints.forEach { p ->
            buffer.clear()
            buffer.putFloat(p.x)
            buffer.putFloat(p.y)
            buffer.putFloat(p.z)
            buffer.put(p.r.toByte())
            buffer.put(p.g.toByte())
            buffer.put(p.b.toByte())
            out.write(buffer.array(), 0, 15)
        }
    }
}

private fun captureFrameBackground(
    frame: Frame,
    session: Session,
    tempDir: File,
    coroutineScope: CoroutineScope,
    onCaptured: (CapturedFrameData) -> Unit
) {
    val image = frame.cameraImage() ?: return
    val width = image.width; val height = image.height
    val nv21 = image.toNv21ByteArray()
    image.close()

    try {
        val intrinsics = frame.camera.imageIntrinsics
        var fx = intrinsics.focalLength[0]
        var fy = intrinsics.focalLength[1]
        try {
            val metadata = frame.imageMetadata
            val focusDiopters = metadata.getFloat(com.google.ar.core.ImageMetadata.LENS_FOCUS_DISTANCE)
            val physicalFocalLengthM = metadata.getFloat(com.google.ar.core.ImageMetadata.LENS_FOCAL_LENGTH) / 1000f
            val multiplier = 1f / (1f - (focusDiopters * physicalFocalLengthM))
            if (!multiplier.isNaN() && !multiplier.isInfinite()) {
                fx *= multiplier
                fy *= multiplier
            }
        } catch (e: Exception) { }

        val anchor = session.createAnchor(frame.camera.pose)
        val filePath = "images/frame_${System.currentTimeMillis()}.jpg"

        coroutineScope.launch(Dispatchers.IO) {
            try {
                val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
                File(tempDir, filePath).writeBytes(out.toByteArray())
                withContext(Dispatchers.Main) {
                    onCaptured(CapturedFrameData(anchor, filePath, fx, fy, intrinsics.principalPoint[0], intrinsics.principalPoint[1]))
                }
            } catch (e: Exception) {
                // Ignore if file system is cleared during rapid restart
            } finally {
                ByteArrayPool.release(nv21)
            }
        }
    } catch (e: Exception) {
        ByteArrayPool.release(nv21)
    }
}

private fun Image.toNv21ByteArray(): ByteArray {
    val yBuffer = planes[0].buffer; val uBuffer = planes[1].buffer; val vBuffer = planes[2].buffer
    val yRowStride = planes[0].rowStride; val uvRowStride = planes[1].rowStride; val uvPixelStride = planes[1].pixelStride
    val size = width * height * 3 / 2
    val nv21 = ByteArrayPool.acquire(size)
    for (i in 0 until height) {
        yBuffer.position(i * yRowStride)
        yBuffer.get(nv21, i * width, width)
    }
    var pos = width * height
    for (i in 0 until height / 2) {
        for (j in 0 until width / 2) {
            nv21[pos++] = vBuffer.get(i * uvRowStride + j * uvPixelStride)
            nv21[pos++] = uBuffer.get(i * uvRowStride + j * uvPixelStride)
        }
    }
    return nv21
}

private fun getPixelColorFromNv21(nv21: ByteArray, width: Int, height: Int, x: Int, y: Int): Int {
    val yVal = nv21[y * width + x].toInt() and 0xFF
    val uvOffset = width * height + (y / 2) * width + (x / 2) * 2
    val vVal = nv21[uvOffset].toInt() and 0xFF
    val uVal = nv21[uvOffset + 1].toInt() and 0xFF
    
    val c = yVal - 16
    val d = uVal - 128
    val e = vVal - 128
    
    val r = ((298 * c + 409 * e + 128) shr 8).coerceIn(0, 255)
    val g = ((298 * c - 100 * d - 208 * e + 128) shr 8).coerceIn(0, 255)
    val b = ((298 * c + 516 * d + 128) shr 8).coerceIn(0, 255)
    
    return (r shl 16) or (g shl 8) or b
}

private fun getTextureGradientFromNv21(nv21: ByteArray, width: Int, height: Int, x: Int, y: Int): Float {
    val xc = x.coerceIn(1, width - 2); val yc = y.coerceIn(1, height - 2)
    val i = yc * width + xc
    val l = nv21[i - 1].toInt() and 0xFF; val r = nv21[i + 1].toInt() and 0xFF
    val u = nv21[i - width].toInt() and 0xFF; val d = nv21[i + width].toInt() and 0xFF
    return (Math.abs(r - l) + Math.abs(d - u)).toFloat()
}

private fun copyShortPlane(plane: Image.Plane, width: Int, height: Int): ShortArray {
    val buffer = plane.buffer.apply { order(java.nio.ByteOrder.nativeOrder()) }
    val size = width * height
    val result = ShortArrayPool.acquire(size)
    if (plane.rowStride == width * 2 && plane.pixelStride == 2) {
        buffer.position(0)
        buffer.asShortBuffer().get(result)
    } else {
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (v in 0 until height) {
            val offset = v * rowStride
            for (u in 0 until width) {
                result[v * width + u] = buffer.getShort(offset + u * pixelStride)
            }
        }
    }
    return result
}

private fun copyBytePlane(plane: Image.Plane, width: Int, height: Int): ByteArray {
    val buffer = plane.buffer
    val size = width * height
    val result = ByteArrayPool.acquire(size)
    if (plane.rowStride == width && plane.pixelStride == 1) {
        buffer.position(0)
        buffer.get(result)
    } else {
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (v in 0 until height) {
            val offset = v * rowStride
            for (u in 0 until width) {
                result[v * width + u] = buffer.get(offset + u * pixelStride)
            }
        }
    }
    return result
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
