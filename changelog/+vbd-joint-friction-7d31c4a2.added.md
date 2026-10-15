Add per-DOF dry Coulomb friction to local compliant-ALM VBD for revolute, prismatic,
D6, and ball joints. Interpret `Model.joint_friction` as an absolute force or
torque bound [N or N·m], with static sticking at convergence and saturated
sliding, including on both reference and follower joints in mimic relationships.
Add the `vbd_joint_friction` and `vbd_joint_friction_pendulum` examples.

Retain regularized friction for sparse mode and legacy AVBD, allowing slow creep
rather than exact static sticking. Ball friction requires
`rigid_compliant_alm=True` and `rigid_articulation_solve="local"`.
