Skip non-1-coordinate/1-DOF joints (Free, Ball, Distance, a multi-axis D6,
...) in `ControllerJointImpedance`'s default `joints` selection instead of
letting them through and then rejecting the whole construction. A model
mixing a floating base or ball joint with controllable joints no longer
needs its `joints` pruned by hand; naming such a joint explicitly still
raises.
