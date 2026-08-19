Add `newton.controllers.select_joints` and `newton.controllers.JointSelection`,
which resolve articulations and joints into the matched `joint_q_idx` /
`joint_qd_idx` pair `ControllerJointImpedance` requires. Both arguments accept
indices or label patterns — a glob, a compiled regular expression, or a list of
either — following the same label-matching rules as the rest of Newton.
Duplicate articulations and duplicate joints are collapsed, and passing no
`joints` selects every single-coordinate, single-DOF joint of each selected
articulation.
