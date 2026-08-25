Remove the `default_dof_indices` constructor argument and the per-port index
overrides (`joint_q_des_idx`, `joint_qd_des_idx`, `joint_qdd_idx`,
`gravity_force_idx`, `coriolis_force_idx`, `joint_f_idx`) from the joint
impedance controllers. Bind an indexed view to the port instead —
`inputs.joint_q_des = sim_q_des[controller.q_start]` replaces a gather override, and
`outputs.joint_f = control.joint_f[controller.qd_start]` replaces `joint_f_idx`.
`ControllerJointImpedanceModelFree` also drops its `robot_count` and `max_dofs`
arguments, both of which are now derived from `controlled_dofs_per_robot`.
