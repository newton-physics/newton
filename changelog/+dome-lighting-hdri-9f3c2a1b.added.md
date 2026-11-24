Add HDRI dome lighting to `SensorCamera` via `SensorCamera.set_dome_light()`. An
equirectangular environment (a linear-radiance array or a `.hdr` file, loaded with a
dependency-free decoder) drives diffuse image-based ambient lighting using order-2
spherical harmonics, replacing the hemispheric ambient term while preserving
`SensorCamera.set_ambient_light()` and `RenderConfig.enable_ambient_lighting` for
hemispheric lighting. Directional and spot lights still contribute on top. Set
`SensorCamera.RenderConfig.dome_shadow_samples` to `N > 0` to cast `N` stochastic shadow
rays per pixel, importance-sampled toward the bright parts of the environment, for soft
shadows from concentrated sources (a sun) as well as diffuse sky without fireflies; the
estimate is unbiased and converges under temporal accumulation, at the cost of per-frame
Monte-Carlo noise that decreases as `N` grows.

Set `SensorCamera.RenderConfig.enable_dome_background` to show the HDRI environment
itself as the background for camera rays that miss all geometry, instead of the
render's clear color. Requires an environment set via `set_dome_light()`.

Add `SensorCamera.set_dome_light_color()` as a convenience for a flat, uniform-color
dome light, without needing to build a uniform-value array or HDRI file.
