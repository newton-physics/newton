Rework the `ControllerJointImpedance` and `ControllerJointImpedanceModelFree`
port layout. Every port except `inputs.joint_q` and `inputs.joint_qd` is now
compact — one entry per controlled DOF — including the gain arrays, which change
from 2-D `(robot_count, max_dofs)` to 1-D `(total_dofs,)`. Ports accept a
`wp.indexedarray` view, so gathering from or scattering to a simulation-sized
array is expressed at the bind site:

```python
selection = select_joints(model)
controller = ControllerJointImpedance(
    model, joint_q_idx=selection.q_idx, joint_qd_idx=selection.qd_idx, ...
)
outputs.joint_f = control.joint_f[selection.qd_idx]
```

`ControllerJointImpedance` now takes a finalized `newton.Model` as its first
positional argument instead of a `newton.ModelBuilder`, which it no longer
finalizes on the caller's behalf. It requires `joint_q_idx` (coordinate space)
and `joint_qd_idx` (DOF space) instead of `default_dof_indices`; the two spaces
differ whenever the model contains a ball, free, or distance joint. Index arrays
are `int32` rather than `uint32`.

Rename the controllers' counts so each states what it counts:
`robot_count` becomes `model_robot_count` (every robot in the model, controlled
or not), `dofs_per_robot` becomes `controlled_dofs_per_robot`, `max_dofs`
becomes `max_controlled_dofs`, and `total_dofs` becomes
`total_controlled_dofs`.

Size every buffer the controllers own to the robots actually controlled.
`inputs.mass_matrix` is now
`[controlled_robot_count, max_controlled_dofs, max_controlled_dofs]`, so an
uncontrolled robot no longer reserves a block, and
`ControllerJointImpedanceModelFree` rejects a zero entry in
`controlled_dofs_per_robot` — leave the robot out instead. Both controllers
report `controlled_robot_count`; `ControllerJointImpedance` also reports
`model_robot_count`, the number of articulations in the model.
