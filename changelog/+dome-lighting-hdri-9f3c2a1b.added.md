Add HDRI dome lighting to `SensorCamera` with `set_dome_light()` and
`set_dome_light_color()`. Support linear NumPy/Warp environment arrays, local
Radiance RGBE files, and sRGB images, with configurable intensity and rotation,
optional importance-sampled soft shadows, and environment backgrounds. Preserve
hemispheric lighting through `set_ambient_light()`.
