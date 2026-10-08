Add `SensorCamera` raytraced rendering with caller-supplied camera rays and
per-view camera transforms passed to `SensorCamera.update()`. Provide
`create_image_output_<kind>()` helpers for color, depth, forward depth, shape
index, normal, albedo, and HDR color buffers, and
`compute_camera_rays_pinhole_usd()` for USD camera rays. Support `(near, far)`
tuples and Warp arrays for `depth_range` in both per-view and tiled depth
conversion, with consistent range validation and clamping at the range endpoints.
