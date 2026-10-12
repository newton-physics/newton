Add `SensorCamera` raytraced rendering with caller-supplied camera rays and
per-view camera transforms passed to `SensorCamera.update()`. Provide
`allocate_image()` and `allocate_image_<kind>()` helpers for zero-initialized
color, HDR color, depth, forward depth, normal, albedo, and shape-index buffers
on the model device, and
`compute_camera_rays_pinhole_usd()` for USD camera rays. Support `(near, far)`
tuples and Warp arrays for `depth_range` in both per-view and tiled depth
conversion, with consistent range validation and clamping at the range endpoints.
