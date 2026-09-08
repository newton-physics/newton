Rework the joint impedance controllers' ports. Every port of
`ControllerJointImpedanceModelFree` is compact — one entry per controlled DOF —
as are the gains and `outputs.joint_f` of `ControllerJointImpedance`, which keeps
whole-model `inputs.joint_q` and `inputs.joint_qd` to evaluate the dynamics from.
Ports accept a `wp.indexedarray` view, so a gather or scatter is expressed at the
bind site.

`ControllerJointImpedance` takes a finalized `newton.Model` and
`articulations`/`joints` arguments selecting the controlled joints, instead of
a `ModelBuilder` and `default_dof_indices`. Each accepts model indices and/or
label patterns — a glob, a compiled regular expression, or a list of either —
following the same label-matching rules as the rest of Newton; omitting
`joints` controls every eligible joint of each selected articulation. Buffers
are sized to the robots actually controlled, so `inputs.mass_matrix` is
`[controlled_robot_count, max_controlled_dofs, max_controlled_dofs]`. The
counts are renamed to say what they count: `robot_count`, `dofs_per_robot`,
`max_dofs`, and `total_dofs` become `model_robot_count`,
`controlled_dofs_per_robot`, `max_controlled_dofs`, and `total_controlled_dofs`.

```python
# Was: ControllerJointImpedance(builder=builder, default_dof_indices=idx, ...)
#      outputs.joint_f = control.joint_f
controller = ControllerJointImpedance(model, joints=["shoulder", "elbow"], ...)
outputs.joint_f = control.joint_f[controller.qd_start]
```
