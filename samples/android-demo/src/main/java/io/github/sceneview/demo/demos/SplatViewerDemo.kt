package io.github.sceneview.demo.demos

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.sceneview.Aabb
import io.github.sceneview.SceneView
import io.github.sceneview.core.splat.SplatCloud
import io.github.sceneview.core.splat.SplatParser
import io.github.sceneview.demo.DemoScaffold
import io.github.sceneview.demo.LoadingScrim
import io.github.sceneview.demo.R
import io.github.sceneview.demo.common.rememberModelDemoEnvironment
import io.github.sceneview.demo.rememberFirstFrameState
import io.github.sceneview.demo.rememberHeroOrbitCameraManipulator
import io.github.sceneview.fitDistanceForBounds
import io.github.sceneview.math.Position
import io.github.sceneview.rememberCameraNode
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberEnvironmentLoader
import io.github.sceneview.rememberMaterialLoader
import io.github.sceneview.splat.SplatBuffers
import io.github.sceneview.verticalFovDegreesForFocalLength
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class SplatFraming(
    val radius: Float,
    val center: Position,
)

@Composable
fun SplatViewerDemo(onBack: () -> Unit) {
    val engine = rememberEngine()
    val materialLoader = rememberMaterialLoader(engine)
    val environmentLoader = rememberEnvironmentLoader(engine)

    var splatCloud by remember { mutableStateOf<SplatCloud?>(null) }
    var splatCount by remember { mutableIntStateOf(0) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var zoomSpeedMultiplier by remember { mutableStateOf(1f) }

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            isLoading = true
            errorMessage = null
            scope.launch {
                try {
                    val bytes = withContext(Dispatchers.IO) {
                        context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    }
                    if (bytes == null) {
                        errorMessage = "Could not read file data."
                    } else {
                        val cloud = withContext(Dispatchers.Default) {
                            SplatParser.parse(bytes)
                        }
                        if (cloud.count > 0) {
                            splatCloud = cloud
                            splatCount = cloud.count
                        } else {
                            errorMessage = "Empty splat cloud loaded."
                        }
                    }
                } catch (e: Exception) {
                    errorMessage = "Error parsing file: ${e.localizedMessage ?: e.message}"
                } finally {
                    isLoading = false
                }
            }
        }
    }

    val framing = remember(splatCloud) {
        val cloud = splatCloud ?: return@remember null
        val bounds = SplatBuffers.boundingBox(cloud)
        val aabb = Aabb(
            center = Position(bounds[0], bounds[1], bounds[2]),
            halfExtent = Position(bounds[3], bounds[4], bounds[5])
        )
        val radius = fitDistanceForBounds(
            bounds = aabb,
            verticalFovDegrees = verticalFovDegreesForFocalLength(28.0),
            aspect = 0.5,
        ).coerceIn(0.2f, 50f)
        SplatFraming(radius = radius, center = aabb.center)
    }

    val cameraManipulator = rememberHeroOrbitCameraManipulator(
        trigger = splatCloud != null,
        radius = framing?.radius ?: 2.5f,
        yHeight = 0f,
        pinchZoomSpeed = io.github.sceneview.gesture.CameraGestureDetector.DefaultCameraManipulator.DEFAULT_PINCH_ZOOM_SPEED * zoomSpeedMultiplier
    )

    val cameraNode = rememberCameraNode(engine)

    val firstFrame = rememberFirstFrameState()
    val firstFrameRendered = if (splatCloud != null) firstFrame.rendered else remember { mutableStateOf(true) }

    DemoScaffold(
        title = stringResource(R.string.demo_splat_viewer),
        onBack = onBack,
        firstFrameRendered = firstFrameRendered,
        controls = {
            if (splatCloud != null) {
                Text(
                    text = "Rendered Splats: $splatCount / ${splatCloud?.count ?: 0}",
                    style = MaterialTheme.typography.labelLarge
                )
                Slider(
                    value = splatCount.toFloat(),
                    onValueChange = { splatCount = it.toInt() },
                    valueRange = 0f..(splatCloud?.count?.toFloat() ?: 1f)
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(
                        R.string.demo_splat_viewer_zoom_speed,
                        "%.1fx".format(zoomSpeedMultiplier)
                    ),
                    style = MaterialTheme.typography.labelLarge
                )
                Slider(
                    value = zoomSpeedMultiplier,
                    onValueChange = { zoomSpeedMultiplier = it },
                    valueRange = 0.1f..10f
                )
                Spacer(modifier = Modifier.height(12.dp))
                Button(
                    onClick = { filePickerLauncher.launch("*/*") },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(text = stringResource(R.string.demo_splat_viewer_select_file))
                }
            }
        }
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            if (splatCloud != null) {
                val cloud = splatCloud!!
                SceneView(
                    modifier = Modifier.fillMaxSize(),
                    onFrame = { frameTimeNanos ->
                        firstFrame.onFrame(frameTimeNanos)
                    },
                    engine = engine,
                    cameraNode = cameraNode,
                    environmentLoader = environmentLoader,
                    environment = rememberModelDemoEnvironment(environmentLoader),
                    cameraManipulator = cameraManipulator,
                ) {
                    SplatNode(
                        splatCloud = cloud,
                        splatCount = splatCount,
                        cameraPositionProvider = { cameraNode.worldPosition },
                        position = framing?.let { -it.center } ?: Position(0f, 0f, 0f),
                        apply = { isTouchable = false }
                    )
                }
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = androidx.compose.foundation.layout.Arrangement.Center
                ) {
                    Text(
                        text = stringResource(R.string.demo_splat_viewer_placeholder),
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Normal,
                        modifier = Modifier.padding(bottom = 24.dp),
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                    Button(onClick = { filePickerLauncher.launch("*/*") }) {
                        Text(text = stringResource(R.string.demo_splat_viewer_select_file))
                    }
                    if (errorMessage != null) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = errorMessage!!,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }
                }
            }

            LoadingScrim(
                loading = isLoading,
                label = stringResource(R.string.demo_splat_viewer_loading),
            )
        }
    }
}
