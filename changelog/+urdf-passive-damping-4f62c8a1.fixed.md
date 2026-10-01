Map URDF joint damping to passive velocity damping (`joint_damping`) instead of
target-drive damping (`joint_target_kd`). Drive damping for URDF-imported joints
now comes from `ModelBuilder.default_joint_cfg.target_kd`; set it explicitly if
you relied on URDF damping as a PD gain. `SolverXPBD` and `SolverVBD` do not
currently consume `joint_damping`, so URDF-authored passive damping has no effect
with these solvers. Applications that relied on the previous drive-damping
mapping must explicitly configure `joint_target_kd` to retain that behavior.
