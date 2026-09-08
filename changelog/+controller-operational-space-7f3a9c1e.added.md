Add `ControllerOperationalSpace`, a model-based operational-space (task-space)
controller supporting simultaneous motion and wrench (hybrid force/motion)
control per task axis, with optional inertia decoupling and null-space
posture control for redundant robots. Add
`ControllerOperationalSpaceModelFree`, taking the tool pose/twist, Jacobian,
mass matrix, and gravity term as inputs instead of computing them from a
`Model`. Add `controller_operational_space_hybrid_force_motion`, an example
demonstrating `ControllerOperationalSpace` on two heterogeneous robots (a
redundant Franka Panda and a non-redundant UR10) pressing into tables with
interactively steered position and force targets.
