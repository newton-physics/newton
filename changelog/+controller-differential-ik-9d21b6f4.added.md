Add `ControllerDifferentialIK`, a model-based differential-kinematics (Jacobian-based)
controller driving joint velocity/position targets toward a desired tool
pose, with a selectable inverse-Jacobian solve
(`DifferentialIKMethod.DAMPED_LEAST_SQUARES`, `PSEUDO_INVERSE`, `TRANSPOSE`,
`ADAPTIVE_DAMPING`, or `TRUNCATED_SVD`) and optional null-space joint-limit
avoidance/posture control for redundant robots. Add
`ControllerDifferentialIKModelFree`, taking the tool pose and Jacobian as inputs
instead of computing them from a `Model`. Add `controller_differential_ik`, an
example driving four heterogeneous robots at once with a single controller
call -- a redundant 7-DOF Franka Panda, a non-redundant 6-DOF UR10, a 4-DOF
planar arm restricted to a 3D (X, Y, yaw) task via `axis_weight`, made
redundant by that restriction, and a 5-DOF elbow-type arm demonstrating
`null_space_axes` (protecting a different set of axes than `axis_weight`
tracks) -- each tracking its own draggable gizmo target, with null-space
posture control anchoring the redundant arms and
`DifferentialIKMethod.ADAPTIVE_DAMPING` ramping damping up automatically near a
singularity or the edge of reach.
