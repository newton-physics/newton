Rename the `SensorCamera` helpers introduced in 1.7.0rc1: use
`create_image_output_<kind>()` instead of `create_<kind>_image_output()` for
color, depth, forward depth, shape index, normal, albedo, and HDR color buffers,
and `compute_camera_rays_pinhole_usd()` instead of
`compute_camera_rays_usd_pinhole()`. Update RC1 callers to the new names.
