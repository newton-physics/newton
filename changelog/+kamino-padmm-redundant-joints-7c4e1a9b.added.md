Add PADMM options to `SolverKamino` for redundant constraints, such as those of closed kinematic loops:

  - `dual_residual_mode="delassus"` measures the dual residual as the change of constraint-space velocity, which ignores impulse drift along the Delassus null space.
  - `redundant_joint_tolerance` drops numerically redundant joint constraints from the Delassus matrix before each solve, with dense and sparse dynamics and every linear solver.
  - `joint_rho_scale` scales the ALM penalty on the bilateral joint constraints, so that they can be solved exactly at each iteration while the penalty applies to the other constraints.
  - `redundant_joint_reuse` keeps the dropped joint constraints across steps, and detects them again only in the worlds where a check on the joint constraint block fails.
