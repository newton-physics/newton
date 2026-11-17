Add PADMM options to `SolverKamino` for redundant constraints, such as those of closed kinematic loops:

  - `dual_residual_mode="delassus"` measures the dual residual as the change of constraint-space velocity, which ignores impulse drift along the Delassus null space.
  - `redundant_joint_tolerance` drops numerically redundant joint constraints from the dense Delassus matrix before each factorization.
