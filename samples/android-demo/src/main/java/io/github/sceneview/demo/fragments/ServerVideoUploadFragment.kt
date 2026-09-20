package io.github.sceneview.demo.fragments

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.runtime.Composable
import io.github.sceneview.demo.DemoCategory
import io.github.sceneview.demo.DemoEntry
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.ServerVideoUploadDemo

/** Append-only fragment for the `server-video-upload` demo. See [DemoFragment]. */
object ServerVideoUploadFragment : DemoFragment {
    override val entry: DemoEntry = DemoEntry(
        id = "server-video-upload",
        titleRes = R.string.demo_server_video_upload,
        subtitleRes = R.string.demo_server_video_upload_subtitle,
        category = DemoCategory.CONTENT,
        icon = Icons.Filled.CloudUpload,
    )

    @Composable
    override fun Screen(onBack: () -> Unit) {
        ServerVideoUploadDemo(onBack)
    }
}
