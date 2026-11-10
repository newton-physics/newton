Add PADMM options to `SolverKamino` for redundant constraints, such as those of closed kinematic loops:

  - `dual_residual_mode="delassus"` measures the dual residual as the change of constraint-space velocity, which ignores impulse drift along the Delassus null space.
