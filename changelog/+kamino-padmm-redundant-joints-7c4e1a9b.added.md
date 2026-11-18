Add PADMM options to `SolverKamino` for redundant constraints, such as those of closed kinematic loops:

  - `dual_residual_mode="delassus"` measures the dual residual as the change of constraint-space velocity, which ignores impulse drift along the Delassus null space.
  - `detect_redundant_kinematics` drops numerically redundant kinematic joint constraints from the Delassus matrix using a diagonal shift before each solve, with dense and sparse dynamics and every linear solver, below the relative pivot `redundant_kinematics_pivot_threshold`.
  - `reuse_redundant_kinematics_set` keeps the dropped joint constraints across steps, and detects them again only in the worlds where they fail a check.
  - `unpenalized_bilateral_rows` removes the ALM penalty from the bilateral joint constraints, so that they are solved exactly at each iteration while the penalty applies to the other constraints.
