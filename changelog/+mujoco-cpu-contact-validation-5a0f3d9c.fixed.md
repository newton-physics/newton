Reject `SolverMuJoCo(use_mujoco_cpu=True, use_mujoco_contacts=False)` instead of silently ignoring supplied contacts; use `use_mujoco_contacts=True`.
Planar mesh colliders are therefore unsupported on the MuJoCo CPU backend; replace them with planes, boxes, or meshes with thickness.
