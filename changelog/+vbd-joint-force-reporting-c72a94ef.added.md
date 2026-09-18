Add experimental `SolverVBD.eval_joint_forces()` post-processing for incoming
joint wrenches in the child joint frame, including structural reactions,
drives, limits, actuation, friction, viscous damping, and mimic reactions in
compliant ALM and legacy VBD. Report final-iterate force estimates on demand
with CUDA graph support. Optionally estimate motor-side effort, including
joint-side armature inertia and mimic-ratio reflection, for the observed
motion without changing VBD dynamics.
