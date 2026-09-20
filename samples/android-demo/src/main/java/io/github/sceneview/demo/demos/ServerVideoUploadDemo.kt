@file:OptIn(ExperimentalMaterial3Api::class)

package io.github.sceneview.demo.demos

import android.content.Context
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.sceneview.demo.R
import io.github.sceneview.demo.service.JobMetadata
import io.github.sceneview.demo.service.ServerVideoUploadService
import io.github.sceneview.demo.service.SystemStatsResponse
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun ServerVideoUploadDemo(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Server Config
    var serverUrl by remember { mutableStateOf("http://10.0.2.2:8000") }
    var isConnected by remember { mutableStateOf(false) }
    var connectionStatus by remember { mutableStateOf("Not checked") }
    var systemStats by remember { mutableStateOf<SystemStatsResponse?>(null) }

    // Video selection state
    var selectedVideoUri by remember { mutableStateOf<Uri?>(null) }
    var selectedVideoName by remember { mutableStateOf<String?>(null) }
    var isUploading by remember { mutableStateOf(false) }
    var uploadProgress by remember { mutableIntStateOf(0) }

    // Training configuration
    var numItersInput by remember { mutableStateOf("30000") }
    var downscaleFactor by remember { mutableIntStateOf(1) }
    var useCpu by remember { mutableStateOf(false) }

    // Active Job state
    var activeJobId by remember { mutableStateOf<String?>(null) }
    var activeJobMetadata by remember { mutableStateOf<JobMetadata?>(null) }
    var activeJobLogs by remember { mutableStateOf<List<String>>(emptyList()) }
    var jobList by remember { mutableStateOf<List<JobMetadata>>(emptyList()) }

    // Model Download launcher
    var downloadingJobId by remember { mutableStateOf<String?>(null) }
    val downloadLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/octet-stream")
    ) { uri ->
        val jId = downloadingJobId
        if (uri != null && jId != null) {
            scope.launch {
                Toast.makeText(context, "Downloading model file...", Toast.LENGTH_SHORT).show()
                val result = runCatching {
                    context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                        ServerVideoUploadService.downloadModel(serverUrl, jId, "ply", outputStream).getOrThrow()
                    } ?: throw Exception("Failed to open destination file")
                }
                result.fold(
                    onSuccess = {
                        Toast.makeText(context, "Model download complete!", Toast.LENGTH_LONG).show()
                    },
                    onFailure = { ex ->
                        Toast.makeText(context, "Download failed: ${ex.message}", Toast.LENGTH_LONG).show()
                    }
                )
            }
        }
    }

    // Video File Picker
    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            val name = getFileName(context, uri)
            if (name != null && !name.lowercase().endsWith(".mp4")) {
                Toast.makeText(context, "Please select an .mp4 video file", Toast.LENGTH_LONG).show()
            } else {
                selectedVideoUri = uri
                selectedVideoName = name ?: "video.mp4"
            }
        }
    }

    // Server check function
    fun testConnection() {
        scope.launch {
            connectionStatus = "Checking..."
            ServerVideoUploadService.checkStatus(serverUrl).fold(
                onSuccess = { res ->
                    isConnected = true
                    connectionStatus = "Online (${res.service})"
                    // Fetch system stats & job list
                    ServerVideoUploadService.getSystemStats(serverUrl).onSuccess { systemStats = it }
                    ServerVideoUploadService.listJobs(serverUrl).onSuccess { jobList = it }
                },
                onFailure = { ex ->
                    isConnected = false
                    connectionStatus = "Offline: ${ex.message}"
                    systemStats = null
                }
            )
        }
    }

    // Periodically poll active job status and server stats if connected
    LaunchedEffect(isConnected, activeJobId) {
        while (isConnected) {
            if (activeJobId != null) {
                ServerVideoUploadService.getJobStatus(serverUrl, activeJobId!!).onSuccess { resp ->
                    activeJobMetadata = resp.metadata
                    activeJobLogs = resp.logs
                }
            }
            ServerVideoUploadService.getSystemStats(serverUrl).onSuccess { systemStats = it }
            delay(3000L)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.demo_server_video_upload)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            // 1. SERVER CONFIGURATION & STATUS
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "OpenSplat Server",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedTextField(
                            value = serverUrl,
                            onValueChange = { serverUrl = it },
                            label = { Text(stringResource(R.string.demo_server_video_upload_server_url)) },
                            modifier = Modifier.weight(1f),
                            singleLine = true
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Button(onClick = { testConnection() }) {
                            Text(stringResource(R.string.demo_server_video_upload_test_connection))
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Status: $connectionStatus",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (isConnected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                    )

                    systemStats?.let { stats ->
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            stats.cpu?.let { Text("CPU: ${"%.1f".format(it.percent)}%", style = MaterialTheme.typography.labelSmall) }
                            stats.ram?.let { Text("RAM: ${"%.1f".format(it.percent)}%", style = MaterialTheme.typography.labelSmall) }
                            stats.gpu?.let { Text("GPU: ${if (it.available) "${"%.1f".format(it.load ?: 0f)}%" else "N/A"}", style = MaterialTheme.typography.labelSmall) }
                            Text("Active Jobs: ${stats.activeJobsCount}", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }

            // 2. VIDEO UPLOADER (.mp4)
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Upload Video (.mp4)",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = selectedVideoName ?: "No MP4 video selected",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (selectedVideoUri != null) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = { videoPickerLauncher.launch("video/mp4") },
                            modifier = Modifier.weight(1f),
                            enabled = !isUploading
                        ) {
                            Icon(Icons.Default.CloudUpload, contentDescription = null)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(stringResource(R.string.demo_server_video_upload_select_video))
                        }

                        Button(
                            onClick = {
                                if (selectedVideoUri != null) {
                                    isUploading = true
                                    uploadProgress = 0
                                    scope.launch {
                                        val result = ServerVideoUploadService.uploadVideo(
                                            baseUrl = serverUrl,
                                            context = context,
                                            videoUri = selectedVideoUri!!,
                                            fileName = selectedVideoName ?: "video.mp4",
                                            onProgress = { uploadProgress = it }
                                        )
                                        isUploading = false
                                        result.fold(
                                            onSuccess = { res ->
                                                activeJobId = res.jobId
                                                Toast.makeText(context, "Uploaded! Job ID: ${res.jobId}", Toast.LENGTH_SHORT).show()
                                                ServerVideoUploadService.listJobs(serverUrl).onSuccess { jobList = it }
                                            },
                                            onFailure = { ex ->
                                                Toast.makeText(context, "Upload failed: ${ex.message}", Toast.LENGTH_LONG).show()
                                            }
                                        )
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f),
                            enabled = selectedVideoUri != null && isConnected && !isUploading
                        ) {
                            Text(stringResource(R.string.demo_server_video_upload_upload_button))
                        }
                    }

                    if (isUploading) {
                        Spacer(modifier = Modifier.height(12.dp))
                        LinearProgressIndicator(
                            progress = { uploadProgress / 100f },
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Uploading: $uploadProgress%",
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.align(Alignment.End)
                        )
                    }
                }
            }

            // 3. ACTIVE JOB MONITOR & CONTROLS
            activeJobId?.let { jobId ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Active Job: $jobId",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))

                        activeJobMetadata?.let { meta ->
                            Text(text = "Source: ${meta.sourceFile}", style = MaterialTheme.typography.bodySmall)
                            Text(
                                text = "Status: ${meta.status}",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Bold,
                                color = when (meta.status) {
                                    "completed" -> MaterialTheme.colorScheme.primary
                                    "failed" -> MaterialTheme.colorScheme.error
                                    "training", "preprocessing" -> MaterialTheme.colorScheme.tertiary
                                    else -> MaterialTheme.colorScheme.onSecondaryContainer
                                }
                            )

                            if (meta.status == "training" || meta.status == "preprocessing") {
                                Spacer(modifier = Modifier.height(8.dp))
                                LinearProgressIndicator(
                                    progress = { (meta.progress / 100f).coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text("Step: ${meta.step} / ${meta.totalSteps}", style = MaterialTheme.typography.bodySmall)
                                    meta.loss?.let { Text("Loss: ${"%.5f".format(it)}", style = MaterialTheme.typography.bodySmall) }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // Training Options (when ready to start)
                        Text("Training Parameters", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = numItersInput,
                                onValueChange = { numItersInput = it },
                                label = { Text("Iterations") },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                modifier = Modifier.width(120.dp),
                                singleLine = true
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Checkbox(checked = useCpu, onCheckedChange = { useCpu = it })
                            Text("Use CPU", style = MaterialTheme.typography.bodySmall)
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // Actions Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = {
                                    val iters = numItersInput.toIntOrNull() ?: 30000
                                    scope.launch {
                                        ServerVideoUploadService.startJob(serverUrl, jobId, numIters = iters, downscale = downscaleFactor, useCpu = useCpu).fold(
                                            onSuccess = { Toast.makeText(context, "Job started", Toast.LENGTH_SHORT).show() },
                                            onFailure = { ex -> Toast.makeText(context, "Start failed: ${ex.message}", Toast.LENGTH_LONG).show() }
                                        )
                                    }
                                },
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = null)
                                Text("Start")
                            }

                            Button(
                                onClick = {
                                    scope.launch {
                                        ServerVideoUploadService.stopJob(serverUrl, jobId).fold(
                                            onSuccess = { Toast.makeText(context, "Job stopped", Toast.LENGTH_SHORT).show() },
                                            onFailure = { ex -> Toast.makeText(context, "Stop failed: ${ex.message}", Toast.LENGTH_LONG).show() }
                                        )
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                            ) {
                                Icon(Icons.Default.Stop, contentDescription = null)
                                Text("Stop")
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = {
                                    downloadingJobId = jobId
                                    downloadLauncher.launch("$jobId.ply")
                                },
                                modifier = Modifier.weight(1f),
                                enabled = activeJobMetadata?.status == "completed"
                            ) {
                                Icon(Icons.Default.Download, contentDescription = null)
                                Text("Model")
                            }

                            Button(
                                onClick = {
                                    scope.launch {
                                        ServerVideoUploadService.deleteJob(serverUrl, jobId).fold(
                                            onSuccess = {
                                                activeJobId = null
                                                activeJobMetadata = null
                                                Toast.makeText(context, "Job deleted", Toast.LENGTH_SHORT).show()
                                                ServerVideoUploadService.listJobs(serverUrl).onSuccess { jobList = it }
                                            },
                                            onFailure = { ex -> Toast.makeText(context, "Delete failed: ${ex.message}", Toast.LENGTH_LONG).show() }
                                        )
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                            ) {
                                Icon(Icons.Default.Delete, contentDescription = null)
                                Text("Delete")
                            }
                        }

                        // Log Console Tail
                        if (activeJobLogs.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(12.dp))
                            Text("Server Logs", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                            Spacer(modifier = Modifier.height(4.dp))
                            Surface(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 140.dp)
                                    .clip(RoundedCornerShape(6.dp)),
                                color = MaterialTheme.colorScheme.surface
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .verticalScroll(rememberScrollState())
                                        .padding(8.dp)
                                ) {
                                    Text(
                                        text = activeJobLogs.takeLast(30).joinToString("\n"),
                                        fontFamily = FontFamily.Monospace,
                                        fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 4. ALL SERVER JOBS LIST
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Server Jobs (${jobList.size})",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        IconButton(
                            onClick = {
                                scope.launch {
                                    ServerVideoUploadService.listJobs(serverUrl).onSuccess { jobList = it }
                                }
                            },
                            enabled = isConnected
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = "Refresh")
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    if (jobList.isEmpty()) {
                        Text("No jobs found on server", style = MaterialTheme.typography.bodySmall)
                    } else {
                        jobList.forEach { job ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(text = job.jobId, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
                                    Text(text = "${job.sourceFile} • ${job.status}", style = MaterialTheme.typography.bodySmall)
                                }
                                Button(
                                    onClick = { activeJobId = job.jobId },
                                    enabled = activeJobId != job.jobId
                                ) {
                                    Text("Select")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun getFileName(context: Context, uri: Uri): String? {
    var result: String? = null
    if (uri.scheme == "content") {
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        try {
            if (cursor != null && cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    result = cursor.getString(index)
                }
            }
        } finally {
            cursor?.close()
        }
    }
    if (result == null) {
        result = uri.path
        val cut = result?.lastIndexOf('/') ?: -1
        if (cut != -1) {
            result = result?.substring(cut + 1)
        }
    }
    return result
}
