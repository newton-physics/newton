Import URDF joint damping as passive velocity damping (`joint_damping`) when
`ModelBuilder.add_urdf(joint_damping_mode="passive")` is selected. In this mode,
drive damping comes from `ModelBuilder.default_joint_cfg.target_kd`; configure
per-joint drive gains explicitly if needed. `SolverXPBD` and `SolverVBD` do not
currently consume `joint_damping`, so choose `joint_damping_mode="drive"` or
configure `joint_target_kd` to retain their former imported damping. Newton 1.7
keeps the drive mapping by default and warns for authored damping; the default
will switch to passive damping in Newton 1.8.
