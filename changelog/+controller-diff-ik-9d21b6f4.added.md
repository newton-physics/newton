Add `ControllerDiffIK`, a model-based differential-kinematics (Jacobian-based)
controller driving joint velocity/position targets toward a desired tool
pose, with a selectable inverse-Jacobian solve
(`IkMethod.DAMPED_LEAST_SQUARES`, `PSEUDO_INVERSE`, `TRANSPOSE`,
`ADAPTIVE_DAMPING`, or `TRUNCATED_SVD`) and optional null-space joint-limit
avoidance/posture control for redundant robots. Add
`ControllerDiffIKModelFree`, taking the tool pose and Jacobian as inputs
instead of computing them from a `Model`. Add `controller_diff_ik`, an
example demonstrating `ControllerDiffIK` on a Franka Panda and a UR10 with
draggable 6-DOF gizmo targets.
