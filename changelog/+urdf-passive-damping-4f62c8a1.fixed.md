Map URDF joint damping to passive velocity damping (`joint_damping`) instead of
target-drive damping (`joint_target_kd`). Drive damping for URDF-imported joints
now comes from `ModelBuilder.default_joint_cfg.target_kd`; set it explicitly if
you relied on URDF damping as a PD gain.
