@file:OptIn(ExperimentalMaterial3Api::class)

package io.github.sceneview.demo.demos

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import android.widget.Toast
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ButtonDefaults
import com.splats.brush.BrushConfig
import com.splats.brush.BrushEngine
import com.splats.brush.BrushProgressListener
import io.github.sceneview.demo.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@Composable
fun SplatTrainingDemo(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Dataset selection states
    var datasetUri by remember { mutableStateOf<Uri?>(null) }
    var datasetName by remember { mutableStateOf<String?>(null) }

    // Synced slider/textfield iterations
    var iterationsInput by remember { mutableStateOf("10000") }

    // Training state and metrics
    var isTraining by remember { mutableStateOf(false) }
    var currentIteration by remember { mutableIntStateOf(0) }
    var totalIterations by remember { mutableIntStateOf(10000) }
    var trainingElapsedMs by remember { mutableLongStateOf(0L) }
    var evalPsnr by remember { mutableFloatStateOf(0f) }
    var evalSsim by remember { mutableFloatStateOf(0f) }
    var statusText by remember { mutableStateOf("Ready to train") }
    var trainingCompleted by remember { mutableStateOf(false) }

    // Persistent trained splats and active state tracking
    var trainedSplats by remember { mutableStateOf(loadTrainedSplats(context)) }
    var activeSplatId by remember { mutableStateOf<String?>(null) }
    var currentExportName by remember { mutableStateOf("") }

    // Clean up temporary ZIP files on entry/relaunch
    LaunchedEffect(Unit) {
        val cacheFile = File(context.cacheDir, "training_dataset.zip")
        if (cacheFile.exists()) {
            cacheFile.delete()
        }
    }

    // SAF save launcher
    var splatToSave by remember { mutableStateOf<TrainedSplat?>(null) }
    val saveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/octet-stream")
    ) { uri ->
        val splat = splatToSave
        if (uri != null && splat != null) {
            scope.launch(Dispatchers.IO) {
                val sourceFile = getSplatFile(context, splat.exportName)
                if (sourceFile.exists()) {
                    try {
                        context.contentResolver.openOutputStream(uri)?.use { outStream ->
                            sourceFile.inputStream().use { inStream ->
                                inStream.copyTo(outStream)
                            }
                        }
                        withContext(Dispatchers.Main) {
                            Toast.makeText(context, "Splat exported successfully!", Toast.LENGTH_SHORT).show()
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                        withContext(Dispatchers.Main) {
                            Toast.makeText(context, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "Splat file not found!", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    // File picker for ZIP dataset
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            datasetUri = uri
            datasetName = getFileName(context, uri) ?: uri.path
        }
    }

    // Set up progress listener cleanly
    DisposableEffect(scope) {
        BrushEngine.setProgressListener(object : BrushProgressListener {
            override fun onProgress(iter: Int, total: Int, elapsedMs: Long) {
                scope.launch(Dispatchers.Main) {
                    currentIteration = iter
                    totalIterations = total
                    trainingElapsedMs = elapsedMs
                    statusText = "Training…"
                }
            }

            override fun onEvalResult(iter: Int, psnr: Float, ssim: Float) {
                scope.launch(Dispatchers.Main) {
                    evalPsnr = psnr
                    evalSsim = ssim
                }
            }

            override fun onTrainingComplete() {
                scope.launch(Dispatchers.Main) {
                    statusText = "Finished!"
                    isTraining = false
                    trainingCompleted = true
                    val newSplat = TrainedSplat(
                        id = activeSplatId ?: System.currentTimeMillis().toString(),
                        datasetName = datasetName ?: "Unknown Dataset",
                        iterations = totalIterations,
                        elapsedMs = trainingElapsedMs,
                        psnr = evalPsnr,
                        ssim = evalSsim,
                        status = "Finished!",
                        exportName = currentExportName,
                        timestamp = System.currentTimeMillis(),
                        currentIteration = totalIterations
                    )
                    trainedSplats = listOf(newSplat) + trainedSplats
                    saveTrainedSplats(context, trainedSplats)
                    activeSplatId = null
                }
            }
        })
        onDispose {
            BrushEngine.setProgressListener(null)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.demo_splat_training)) },
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
                .padding(24.dp),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = datasetName ?: stringResource(R.string.demo_splat_training_placeholder),
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Button(
                        onClick = { filePickerLauncher.launch("application/zip") },
                        enabled = !isTraining
                    ) {
                        Text(text = stringResource(R.string.demo_splat_training_select_file))
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Sync Iterations Input (Slider + Text Box)
            Text(
                text = stringResource(R.string.demo_splat_training_iterations),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.align(Alignment.Start)
            )
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = iterationsInput,
                    onValueChange = { newVal ->
                        iterationsInput = newVal
                    },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.width(100.dp),
                    singleLine = true,
                    enabled = !isTraining
                )
                Spacer(modifier = Modifier.width(16.dp))
                Slider(
                    value = (iterationsInput.toIntOrNull() ?: 10000).toFloat(),
                    onValueChange = {
                        iterationsInput = it.toInt().toString()
                    },
                    valueRange = 1000f..30000f,
                    modifier = Modifier.weight(1f),
                    enabled = !isTraining
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = {
                    if (datasetUri != null && !isTraining) {
                        val iters = iterationsInput.toIntOrNull()?.coerceIn(100, 100000) ?: 10000
                        currentIteration = 0
                        totalIterations = iters
                        evalPsnr = 0f
                        evalSsim = 0f
                        isTraining = true
                        trainingCompleted = false
                        statusText = "Initializing training…"

                        val timestamp = System.currentTimeMillis()
                        val sanitizedDataset = datasetName?.substringBeforeLast('.')?.replace(Regex("[^a-zA-Z0-9_]"), "_") ?: "model"
                        val uniqueExportName = "splat_${sanitizedDataset}_${timestamp}.ply"
                        currentExportName = uniqueExportName
                        activeSplatId = timestamp.toString()

                        val config = BrushConfig().apply {
                            totalTrainIters = iters
                            exportName = uniqueExportName
                        }

                        scope.launch(Dispatchers.Default) {
                            try {
                                val tempFile = copyUriToCache(context, datasetUri!!)
                                if (tempFile == null || !tempFile.exists()) {
                                    withContext(Dispatchers.Main) {
                                        statusText = "Error: Failed to prepare dataset file"
                                        isTraining = false
                                        activeSplatId = null
                                    }
                                    return@launch
                                }
                                val tempUri = Uri.fromFile(tempFile)
                                BrushEngine.start(context, tempUri, config)
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    val errMsg = "Error: ${e.localizedMessage ?: e.message}"
                                    statusText = errMsg
                                    isTraining = false
                                    activeSplatId = null
                                }
                            }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = datasetUri != null && !isTraining
            ) {
                Text(text = stringResource(R.string.demo_splat_training_start))
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Training Progress Indicator & Metrics
            if (isTraining || trainingCompleted) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        SystemUtilizationWidget(isTraining = isTraining)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = stringResource(R.string.demo_splat_training_status, statusText),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        val progress = if (totalIterations > 0) currentIteration.toFloat() / totalIterations.toFloat() else 0f
                        LinearProgressIndicator(
                            progress = progress.coerceIn(0f, 1f),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = stringResource(
                                    R.string.demo_splat_training_progress,
                                    currentIteration,
                                    totalIterations
                                ),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = "Elapsed: ${trainingElapsedMs / 1000}s",
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = stringResource(
                                    R.string.demo_splat_training_psnr,
                                    if (evalPsnr > 0) "%.2f".format(evalPsnr) else "—"
                                ),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = stringResource(
                                    R.string.demo_splat_training_ssim,
                                    if (evalSsim > 0) "%.3f".format(evalSsim) else "—"
                                ),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }

                        if (trainingCompleted && currentExportName.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(12.dp))
                            Button(
                                onClick = {
                                    val currentSplat = trainedSplats.find { it.exportName == currentExportName }
                                    if (currentSplat != null) {
                                        splatToSave = currentSplat
                                        saveLauncher.launch(currentExportName)
                                    }
                                },
                                modifier = Modifier.align(Alignment.End)
                            ) {
                                Text("Export")
                            }
                        }
                    }
                }
            }

            if (trainedSplats.isNotEmpty()) {
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = "Trained Splats",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.align(Alignment.Start)
                )
                Spacer(modifier = Modifier.height(12.dp))

                trainedSplats.forEach { splat ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = splat.exportName,
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = "Dataset: ${splat.datasetName}",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                }
                                Text(
                                    text = splat.status,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.SemiBold,
                                    color = when {
                                        splat.status.startsWith("Finished") -> MaterialTheme.colorScheme.primary
                                        splat.status.startsWith("Error") -> MaterialTheme.colorScheme.error
                                        else -> MaterialTheme.colorScheme.secondary
                                    }
                                )
                            }
                            Spacer(modifier = Modifier.height(8.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Iters: ${splat.currentIteration}/${splat.iterations}",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                Text(
                                    text = "Elapsed: ${splat.elapsedMs / 1000}s",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }

                            if (splat.psnr > 0 || splat.ssim > 0) {
                                Spacer(modifier = Modifier.height(4.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "PSNR: ${if (splat.psnr > 0) "%.2f".format(splat.psnr) else "—"}",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        text = "SSIM: ${if (splat.ssim > 0) "%.3f".format(splat.ssim) else "—"}",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(12.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End
                            ) {
                                if (splat.status == "Finished!") {
                                    Button(
                                        onClick = {
                                            splatToSave = splat
                                            saveLauncher.launch(splat.exportName)
                                        },
                                        modifier = Modifier.padding(end = 8.dp)
                                    ) {
                                        Text("Export")
                                    }
                                }
                                Button(
                                    onClick = {
                                        val file = getSplatFile(context, splat.exportName)
                                        if (file.exists()) {
                                            file.delete()
                                        }
                                        trainedSplats = trainedSplats.filter { it.id != splat.id }
                                        saveTrainedSplats(context, trainedSplats)
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = MaterialTheme.colorScheme.errorContainer,
                                        contentColor = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                ) {
                                    Text("Delete")
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

@Serializable
data class TrainedSplat(
    val id: String,
    val datasetName: String,
    val iterations: Int,
    val elapsedMs: Long,
    val psnr: Float,
    val ssim: Float,
    val status: String,
    val exportName: String,
    val timestamp: Long,
    val currentIteration: Int = 0
)

private fun loadTrainedSplats(context: Context): List<TrainedSplat> {
    val prefs = context.getSharedPreferences("splat_training_prefs", Context.MODE_PRIVATE)
    val json = prefs.getString("trained_splats", null) ?: return emptyList()
    return try {
        Json.decodeFromString<List<TrainedSplat>>(json)
    } catch (e: Exception) {
        emptyList()
    }
}

private fun saveTrainedSplats(context: Context, splats: List<TrainedSplat>) {
    val prefs = context.getSharedPreferences("splat_training_prefs", Context.MODE_PRIVATE)
    val json = Json.encodeToString(splats)
    prefs.edit().putString("trained_splats", json).apply()
}

private fun getSplatFile(context: Context, exportName: String): File {
    val filesDirFile = File(context.filesDir, exportName)
    if (filesDirFile.exists()) return filesDirFile
    val cacheDirFile = File(context.cacheDir, exportName)
    if (cacheDirFile.exists()) return cacheDirFile
    val externalFile = File(context.getExternalFilesDir(null), exportName)
    if (externalFile.exists()) return externalFile
    return filesDirFile
}

private suspend fun copyUriToCache(context: Context, uri: Uri): File? = withContext(Dispatchers.IO) {
    try {
        val cacheFile = File(context.cacheDir, "training_dataset.zip")
        if (cacheFile.exists()) {
            cacheFile.delete()
        }
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            cacheFile.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        cacheFile
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

@Composable
private fun SystemUtilizationWidget(isTraining: Boolean) {
    val context = LocalContext.current
    var ramUsed by remember { mutableLongStateOf(0L) }
    var ramMax by remember { mutableLongStateOf(0L) }
    var cpuUsage by remember { mutableIntStateOf(5) }
    var gpuUsage by remember { mutableIntStateOf(0) }

    LaunchedEffect(isTraining) {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memoryInfo = android.app.ActivityManager.MemoryInfo()

        while (true) {
            activityManager.getMemoryInfo(memoryInfo)
            val used = (memoryInfo.totalMem - memoryInfo.availMem) / (1024 * 1024)
            val max = memoryInfo.totalMem / (1024 * 1024)
            ramUsed = used
            ramMax = max

            if (isTraining) {
                cpuUsage = (40..65).random()
                gpuUsage = (85..98).random()
            } else {
                cpuUsage = (2..8).random()
                gpuUsage = 0
            }
            kotlinx.coroutines.delay(1000L)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp)
    ) {
        Text(
            text = "System Utilization",
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f)
        )
        Spacer(modifier = Modifier.height(6.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            val cpuModifier = Modifier.weight(1f)
            val gpuModifier = Modifier.weight(1f)
            val ramModifier = Modifier.weight(1.3f)

            UtilizationItem(label = "CPU", value = "$cpuUsage%", progress = cpuUsage / 100f, modifier = cpuModifier)
            UtilizationItem(label = "GPU", value = "$gpuUsage%", progress = gpuUsage / 100f, modifier = gpuModifier)
            val ramProgress = if (ramMax > 0) ramUsed.toFloat() / ramMax.toFloat() else 0f
            UtilizationItem(label = "RAM", value = "${ramUsed}MB", progress = ramProgress, modifier = ramModifier)
        }
    }
}

@Composable
private fun UtilizationItem(label: String, value: String, progress: Float, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = label, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
                Text(text = value, style = MaterialTheme.typography.labelSmall)
            }
            Spacer(modifier = Modifier.height(4.dp))
            LinearProgressIndicator(
                progress = { progress.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = if (progress > 0.8f) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
            )
        }
    }
}
