Constrain `ControllerJointImpedance` to the joints it controls rather than the
whole model: a model containing a free, ball, distance, or D6 joint is now
accepted, and only an addressed joint must span a single coordinate and a single
DOF. An articulation may be left uncontrolled, in which case it occupies no slot
and is masked out of the forward kinematics and dynamics evaluations. Mistakes
that previously produced silently wrong torques now raise — a joint belonging to
no articulation, the same DOF addressed twice, or a write to a port whose
feature is disabled. `device` and `requires_grad` are no longer constructor
arguments; both are taken from `model` directly.
