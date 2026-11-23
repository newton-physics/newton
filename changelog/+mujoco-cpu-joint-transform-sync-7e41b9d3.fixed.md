Fix `SolverMuJoCo` with `use_mujoco_cpu=True` ignoring runtime edits of `joint_X_p`, `joint_X_c` and `joint_axis` after `notify_model_changed(ModelFlags.JOINT_PROPERTIES)`.
