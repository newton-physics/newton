Add model lights:

  - Declare lights with `ModelBuilder.add_light()` and `newton.LightType`, stored in `Model.light_*` arrays with a `LIGHT` attribute frequency; lights belong to a world or are global, replicate per world, and follow `collapse_fixed_joints()`.
  - Import lights from USD `UsdLux.DistantLight` and `UsdLux.SphereLight` prims (`add_usd(load_lights=True)`) and MJCF `<light>` elements (`add_mjcf(parse_lights=True)`).
  - Render model lights in `SensorCamera`, posed from the state on every update; a light in a world lights only that world.
