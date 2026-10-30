Fix `SolverMuJoCo` with `use_mujoco_cpu=True` ignoring fixed-root pose updates from `joint_X_p` and `joint_X_c`, which left MuJoCo colliding against the original pose.
