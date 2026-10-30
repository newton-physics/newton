Add experimental extensions to `SensorCamera` sensors bound with `cameras=` or `sites=`:

  - User ray functions through the `ray_function` and `ray_params` constructor arguments, which receive a `SensorCamera.RayQuery` per pixel sample.
  - `face_index_image` and `barycentric_image` outputs of `SensorCamera.update()`, with `allocate_image_face_index()`, `allocate_image_barycentric()` and the `SensorCamera.HIT_*` shape-index constants.
  - A `view_mask` argument of `SensorCamera.update()` that skips views and keeps their previous outputs.
  - Sharing of one renderer, and with it lights, ambient light and checkerboard materials, between bound sensors of a model with the same `load_textures` and `deformable_surface_mode` settings.
