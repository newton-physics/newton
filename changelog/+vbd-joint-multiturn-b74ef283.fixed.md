Preserve accumulated angles for all VBD revolute and D6 joints, including
drives, limits, friction, damping, and mimic relationships. Publish continuous
`State.joint_q` and corresponding `State.joint_qd` for revolute, prismatic, and
D6 joints. Support CUDA graph replay and world-masked resets; seed initial
turns from `State.joint_q` with matching body poses. XPBD and SemiImplicit
behavior is unchanged.
