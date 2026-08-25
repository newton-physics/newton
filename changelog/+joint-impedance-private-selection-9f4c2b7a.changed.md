`ControllerJointImpedance` takes `articulations`/`joints` query patterns or
model indices directly instead of a `JointSelection`. `JointSelection` and
`select_joints` are no longer public; the resolved coordinate/DOF indices are
available afterward as `controller.q_start`/`controller.qd_start`.

```python
# Was:
selection = select_joints(model, joints=["panda_joint1", ...])
controller = ControllerJointImpedance(model, joint_selection=selection, ...)
outputs.joint_f = control.joint_f[selection.qd_start]

# Now:
controller = ControllerJointImpedance(model, joints=["panda_joint1", ...], ...)
outputs.joint_f = control.joint_f[controller.qd_start]
```
