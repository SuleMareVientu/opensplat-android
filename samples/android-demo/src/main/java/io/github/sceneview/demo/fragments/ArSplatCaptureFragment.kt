package io.github.sceneview.demo.fragments

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.runtime.Composable
import io.github.sceneview.demo.DemoCategory
import io.github.sceneview.demo.DemoEntry
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.ArSplatCaptureDemo

/** Append-only fragment for the `ar-splat-capture` demo. See [DemoFragment]. */
object ArSplatCaptureFragment : DemoFragment {
    override val entry: DemoEntry = DemoEntry(
        id = "ar-splat-capture",
        titleRes = R.string.demo_ar_splat_capture_title,
        subtitleRes = R.string.demo_ar_splat_capture_subtitle,
        category = DemoCategory.AUGMENTED_REALITY,
        icon = Icons.Filled.Camera,
    )

    @Composable
    override fun Screen(onBack: () -> Unit) {
        ArSplatCaptureDemo(onBack)
    }
}
