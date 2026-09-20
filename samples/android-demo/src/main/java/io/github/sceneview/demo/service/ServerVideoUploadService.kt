package io.github.sceneview.demo.service

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okio.BufferedSink
import okio.source
import java.io.File
import java.io.OutputStream
import java.util.concurrent.TimeUnit

@Serializable
data class ServerStatusResponse(
    val status: String = "",
    val service: String = ""
)

@Serializable
data class UploadVideoResponse(
    @SerialName("job_id") val jobId: String,
    val status: String
)

@Serializable
data class JobMetadata(
    @SerialName("job_id") val jobId: String,
    @SerialName("job_type") val jobType: String = "video",
    @SerialName("source_file") val sourceFile: String = "",
    val status: String = "",
    val progress: Float = 0f,
    val step: Int = 0,
    @SerialName("total_steps") val totalSteps: Int = 0,
    val loss: Float? = null,
    @SerialName("error_message") val errorMessage: String? = null,
    @SerialName("created_at") val createdAt: String? = null,
    @SerialName("updated_at") val updatedAt: String? = null
)

@Serializable
data class JobStatusResponse(
    val metadata: JobMetadata,
    val logs: List<String> = emptyList()
)

@Serializable
data class SimpleStatusResponse(
    val status: String = ""
)

@Serializable
data class ErrorDetail(
    val detail: String = ""
)

@Serializable
data class CpuStats(
    val percent: Float = 0f,
    val cores: Int = 0
)

@Serializable
data class RamStats(
    val total: Long = 0L,
    val available: Long = 0L,
    val used: Long = 0L,
    val percent: Float = 0f
)

@Serializable
data class DiskStats(
    val total: Long = 0L,
    val used: Long = 0L,
    val free: Long = 0L,
    val percent: Float = 0f
)

@Serializable
data class GpuStats(
    val available: Boolean = false,
    val name: String? = null,
    val load: Float? = null,
    @SerialName("memory_used") val memoryUsed: Long? = null,
    @SerialName("memory_total") val memoryTotal: Long? = null,
    @SerialName("memory_percent") val memoryPercent: Float? = null,
    val temp: Int? = null
)

@Serializable
data class SystemStatsResponse(
    val cpu: CpuStats? = null,
    val ram: RamStats? = null,
    val disk: DiskStats? = null,
    val gpu: GpuStats? = null,
    @SerialName("active_jobs_count") val activeJobsCount: Int = 0
)

object ServerVideoUploadService {

    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        isLenient = true
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(300, TimeUnit.SECONDS)
        .build()

    private fun normalizeUrl(baseUrl: String): String {
        var url = baseUrl.trim()
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = "http://$url"
        }
        return url.removeSuffix("/")
    }

    suspend fun checkStatus(baseUrl: String): Result<ServerStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<ServerStatusResponse>(bodyStr)
            }
        }
    }

    suspend fun uploadVideo(
        baseUrl: String,
        context: Context,
        videoUri: Uri,
        fileName: String,
        onProgress: ((progressPercent: Int) -> Unit)? = null
    ): Result<UploadVideoResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val uploadUrl = "$rootUrl/api/upload/video"

            val contentResolver = context.contentResolver
            val fileSize = contentResolver.openAssetFileDescriptor(videoUri, "r")?.use { it.length } ?: -1L

            val mediaType = "video/mp4".toMediaType()

            val requestBody = object : RequestBody() {
                override fun contentType() = mediaType
                override fun contentLength() = fileSize

                override fun writeTo(sink: BufferedSink) {
                    contentResolver.openInputStream(videoUri)?.use { inputStream ->
                        val buffer = ByteArray(8192)
                        var uploaded = 0L
                        var read: Int
                        var lastProgress = -1
                        val source = inputStream.source()

                        while (source.read(sink.buffer, 8192).also { read = it.toInt() } != -1L) {
                            sink.flush()
                            uploaded += read
                            if (fileSize > 0) {
                                val currentProgress = ((uploaded * 100) / fileSize).toInt()
                                if (currentProgress != lastProgress) {
                                    lastProgress = currentProgress
                                    onProgress?.invoke(currentProgress)
                                }
                            }
                        }
                    } ?: throw Exception("Could not open video file input stream")
                }
            }

            val multipartBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", fileName, requestBody)
                .build()

            val request = Request.Builder()
                .url(uploadUrl)
                .post(multipartBody)
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "Upload failed: HTTP ${response.code} ${response.message}" })
                }
                json.decodeFromString<UploadVideoResponse>(bodyStr)
            }
        }
    }

    suspend fun listJobs(baseUrl: String): Result<List<JobMetadata>> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/jobs")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<List<JobMetadata>>(bodyStr)
            }
        }
    }

    suspend fun getJobStatus(baseUrl: String, jobId: String): Result<JobStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/jobs/$jobId/status")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<JobStatusResponse>(bodyStr)
            }
        }
    }

    suspend fun startJob(
        baseUrl: String,
        jobId: String,
        numIters: Int = 30000,
        downscale: Int = 1,
        useCpu: Boolean = false
    ): Result<SimpleStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val formBody = FormBody.Builder()
                .add("num_iters", numIters.toString())
                .add("downscale", downscale.toString())
                .add("use_cpu", useCpu.toString())
                .build()

            val request = Request.Builder()
                .url("$rootUrl/api/jobs/$jobId/start")
                .post(formBody)
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<SimpleStatusResponse>(bodyStr)
            }
        }
    }

    suspend fun stopJob(baseUrl: String, jobId: String): Result<SimpleStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/jobs/$jobId/stop")
                .post(FormBody.Builder().build())
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<SimpleStatusResponse>(bodyStr)
            }
        }
    }

    suspend fun downloadModel(
        baseUrl: String,
        jobId: String,
        format: String = "ply",
        outputStream: OutputStream
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/jobs/$jobId/download?format=$format")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val bodyStr = response.body?.string() ?: ""
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "Download failed: HTTP ${response.code}" })
                }
                response.body?.byteStream()?.use { inputStream ->
                    inputStream.copyTo(outputStream)
                } ?: throw Exception("Response body is empty")
                Unit
            }
        }
    }

    suspend fun deleteJob(baseUrl: String, jobId: String): Result<SimpleStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/jobs/$jobId")
                .delete()
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<SimpleStatusResponse>(bodyStr)
            }
        }
    }

    suspend fun getSystemStats(baseUrl: String): Result<SystemStatsResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val rootUrl = normalizeUrl(baseUrl)
            val request = Request.Builder()
                .url("$rootUrl/api/system/stats")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val bodyStr = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    val errorDetail = parseError(bodyStr)
                    throw Exception(errorDetail.ifEmpty { "HTTP ${response.code}: ${response.message}" })
                }
                json.decodeFromString<SystemStatsResponse>(bodyStr)
            }
        }
    }

    private fun parseError(jsonBody: String): String {
        return try {
            json.decodeFromString<ErrorDetail>(jsonBody).detail
        } catch (e: Exception) {
            ""
        }
    }
}
