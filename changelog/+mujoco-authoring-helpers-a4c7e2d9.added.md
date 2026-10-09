Add `newton.solvers.mujoco` helpers for programmatically authoring MuJoCo actuators, contact pairs, tendons, and equality constraints, and `ModelBuilder.CustomAttribute.reference_value_transformer` for remapping row-dependent references in `ModelBuilder.add_builder()`.

Joint actuator targets select individual ball/free-joint axes with a single-value gear, and welds preserve the initial relative pose by default.
