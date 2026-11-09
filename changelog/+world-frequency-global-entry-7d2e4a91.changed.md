Give global world `-1` its own entry in every `Model.AttributeFrequency.WORLD`
array, including `WORLD`-frequency custom attributes, the `model.mujoco`
per-world solver options, and `SolverMuJoCo.Observables.overflow`, and in
`Model.gravity`, instead of sharing world `0`'s entry. These arrays now have
shape `(world_count + 1,)`, with the final entry for global world `-1`;
`*_world_start` arrays keep their own layout. Models built without
`ModelBuilder.begin_world()` place every entity in global world `-1`, so on
such models:

  - `Model.set_gravity(g, world=0)`, or `set_gravity()` with one vector per
    local world, no longer changes their gravity and warns. Writing
    `model.gravity` directly with the previous one-entry array updates only
    world `0`. Pass `world=-1` or a single vector for every world.
  - `SolverMuJoCo` reads per-world options and reports `overflow` in the final
    entry, for global world `-1`. Index it with `[-1]` instead of `[0]`.
  - Builder `WORLD` values authored or imported outside a world context are
    stored under key `-1`, the global world. Their value is also the default of
    local worlds without their own value, unless the attribute stores
    references.

Merging a builder places all its entities in one destination world, which
takes the source's global value or else its world `0` value; global
`add_builder()` merges write the global entry instead of world `0`, and
values for other source worlds raise `ValueError`. Use
`values[:model.world_count]` for the previous per-world layout.
