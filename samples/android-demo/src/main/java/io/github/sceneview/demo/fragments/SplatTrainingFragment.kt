package io.github.sceneview.demo.fragments

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Brush
import androidx.compose.runtime.Composable
import io.github.sceneview.demo.DemoCategory
import io.github.sceneview.demo.DemoEntry
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.SplatTrainingDemo

/** Append-only fragment for the `splat-training` demo. See [DemoFragment]. */
object SplatTrainingFragment : DemoFragment {
    override val entry: DemoEntry = DemoEntry(
        id = "splat-training",
        titleRes = R.string.demo_splat_training,
        subtitleRes = R.string.demo_splat_training_subtitle,
        category = DemoCategory.CONTENT,
        icon = Icons.Filled.Brush,
    )

    @Composable
    override fun Screen(onBack: () -> Unit) {
        SplatTrainingDemo(onBack)
    }
}
