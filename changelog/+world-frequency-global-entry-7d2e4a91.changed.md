Separate global world `-1` from world `0` in every per-world array.
`Model.AttributeFrequency.WORLD` arrays, including `WORLD`-frequency custom
attributes, the `model.mujoco` per-world solver options, and
`SolverMuJoCo.Observables.overflow`, now have shape `(world_count + 1,)`, with
the final entry for global world `-1`. `Model.gravity` always has this shape;
models built without `ModelBuilder.begin_world()` no longer share one gravity
entry between world `0` and their global entities. Use `world=-1`, or a single
vector for every world, to change the gravity of global entities.
`WORLD` values imported or authored outside a world context, and those merged by
`add_builder()` outside one, now go to the global entry instead of world `0`.
Use `values[:model.world_count]` for the previous per-world layout.
