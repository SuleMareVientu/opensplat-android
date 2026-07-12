package io.github.sceneview.demo.fragments

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.runtime.Composable
import io.github.sceneview.demo.DemoCategory
import io.github.sceneview.demo.DemoEntry
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.SplatViewerDemo

/** Append-only fragment for the `splat-viewer` demo. See [DemoFragment]. */
object SplatViewerFragment : DemoFragment {
    override val entry: DemoEntry = DemoEntry(
        id = "splat-viewer",
        titleRes = R.string.demo_splat_viewer,
        subtitleRes = R.string.demo_splat_viewer_subtitle,
        category = DemoCategory.BASICS_3D,
        icon = Icons.Filled.Cloud,
    )

    @Composable
    override fun Screen(onBack: () -> Unit) {
        SplatViewerDemo(onBack)
    }
}
