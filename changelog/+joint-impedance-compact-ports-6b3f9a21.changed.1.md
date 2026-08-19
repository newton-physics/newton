Constrain `ControllerJointImpedance` to the joints it actually controls rather
than to the whole model. A model containing a free, ball, distance, or D6 joint
was previously rejected outright; it is now accepted, and only the addressed
joints must span a single coordinate and a single DOF — which admits a
single-axis D6 alongside revolute and prismatic joints. An articulation may be
left entirely uncontrolled, in which case it occupies no slot in any of the
controller's buffers and is masked out of the forward kinematics and dynamics
evaluations, so its `State.body_q` is left untouched by `step()`. Addressing a
joint that belongs to no articulation, or the same DOF twice, now raises
`ValueError` at construction instead of producing silently wrong torques, and
`step()` raises when a port is written whose feature is disabled — setting
`inputs.joint_qdd` without `has_qdd_feedforward`, for instance, was previously
ignored. `device` and `requires_grad` must match the model's.
