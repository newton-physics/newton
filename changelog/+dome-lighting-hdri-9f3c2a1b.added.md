Add HDRI dome lighting to `SensorCamera` with `set_dome_light()` and
`set_dome_light_color()`. Support linear NumPy/Warp environment arrays, local
Radiance RGBE files, and sRGB images, with configurable intensity and rotation,
optional importance-sampled soft shadows, and environment backgrounds. Shadows
and backgrounds also work with the default sky/ground map.
The sensor camera example keeps its directional light with HDRIs and switches
to dome-only lighting when dome shadow samples are enabled.
