Add a `total_controlled_dofs` property to `ControllerJointImpedance` and
`ControllerJointImpedanceModelFree`, reporting the controlled-DOF count that
every compact port is sized to. `ControllerJointImpedance` also exposes
`q_start`/`qd_start`, the resolved coordinate/DOF index of each controlled
joint.
