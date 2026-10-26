Add `newton.solvers.mujoco` helpers for programmatically authoring MuJoCo actuators, contact pairs, tendons, and equality constraints, and `ModelBuilder.CustomAttribute.reference_value_transformer` for remapping row-dependent references in `ModelBuilder.add_builder()`.

Scalar and D6 actuator targets select individual joint DOFs. Ball/free-joint targets require explicit gear vectors in MuJoCo's native frames. Welds preserve the initial relative pose by default, including translated world copies.
