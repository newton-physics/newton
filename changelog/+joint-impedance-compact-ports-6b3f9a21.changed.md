Rework the joint impedance controllers' ports. Every port of
`ControllerJointImpedanceModelFree` is compact — one entry per controlled DOF —
as are the gains and `outputs.joint_f` of `ControllerJointImpedance`, which keeps
whole-model `inputs.joint_q` and `inputs.joint_qd` to evaluate the dynamics from.
Ports accept a `wp.indexedarray` view, so a gather or scatter is expressed at the
bind site.

`ControllerJointImpedance` takes a finalized `newton.Model` and the `int32`
`joint_q_idx` / `joint_qd_idx` pair instead of a `ModelBuilder` and
`default_dof_indices`. Buffers are sized to the robots actually controlled, so
`inputs.mass_matrix` is `[controlled_robot_count, max_controlled_dofs,
max_controlled_dofs]`. The counts are renamed to say what they count:
`robot_count`, `dofs_per_robot`, `max_dofs`, and `total_dofs` become
`model_robot_count`, `controlled_dofs_per_robot`, `max_controlled_dofs`, and
`total_controlled_dofs`.

```python
# Was: ControllerJointImpedance(builder=builder, default_dof_indices=idx, ...)
#      outputs.joint_f = control.joint_f
selection = select_joints(model)
controller = ControllerJointImpedance(
    model, joint_q_idx=selection.q_idx, joint_qd_idx=selection.qd_idx, ...
)
outputs.joint_f = control.joint_f[selection.qd_idx]
```
