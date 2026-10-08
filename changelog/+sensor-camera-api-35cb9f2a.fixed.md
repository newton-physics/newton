Accept the same `(near, far)` tuples and Warp arrays for `depth_range` in
`SensorCamera.Utils.flatten_depth_image_to_rgba()` and `to_rgba_from_depth()`.
Validate tuple ranges and array shape, dtype, and device consistently before
launching depth conversion, and clamp tiled depth colors at the range endpoints
to match per-view conversion instead of wrapping out-of-range values.
