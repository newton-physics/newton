Deprecate `SensorCamera.create_<kind>_image_output()` in favor of
`create_image_output_<kind>()` for color, depth, forward depth, shape index,
normal, albedo, and HDR color buffers. Deprecate
`SensorCamera.compute_camera_rays_usd_pinhole()` in favor of
`compute_camera_rays_pinhole_usd()` to match the other lens helpers. The old
names remain functional aliases.
