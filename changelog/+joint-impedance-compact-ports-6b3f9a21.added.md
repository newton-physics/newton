Add `newton.controllers.select_joints` and `newton.controllers.JointSelection`,
which resolve articulations and joints into the `JointSelection`
`ControllerJointImpedance` requires for its `joint_selection` argument.
`JointSelection.q_start`/`qd_start` hold each selected joint's *starting*
coordinate/DOF index. Both `articulations` and `joints` accept indices or
label patterns — a glob, a compiled regular expression, or a list of either —
following the same label-matching rules as the rest of Newton. Duplicate
articulations and duplicate joints are collapsed, and passing no `joints`
records the starting index of every joint with at least one coordinate and
one DOF in each selected articulation (a zero-DOF Fixed joint has no starting
index to give, so it contributes no entry); the controller, not
`select_joints`, checks whether each remaining joint is otherwise
controllable.
