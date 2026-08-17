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

`ControllerJointImpedance` now requires `joint_q_idx` (coordinate space) and
`joint_qd_idx` (DOF space) instead of `default_dof_indices`; the two spaces
differ whenever the model contains a ball, free, or distance joint. Index arrays
are `int32` rather than `uint32`. An articulation may now be left entirely
uncontrolled, and `device=None` resolves to the default Warp device instead of
always raising.
