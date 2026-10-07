Honor requested texture SDFs for mesh and box shapes that collide only with particles, including CUDA validation for those requests. For CPU particle-only boxes, leave `ShapeConfig.sdf_max_resolution` and `ShapeConfig.sdf_target_voxel_size` unset; box particle contacts remain analytic and do not need a texture.

Build internal deferred particle-only mesh SDFs with shape scale baked in, which can require a separate texture per distinct scale and increase setup time and live GPU memory. Requests using only `force_sdf=True` continue to share an unscaled SDF across scales of the same mesh when the other generation settings match.

Normalize full-surface contact normals for mesh SDFs with baked scale, and preserve `force_sdf` texture generation when a supplied SDF contains only volume data.
