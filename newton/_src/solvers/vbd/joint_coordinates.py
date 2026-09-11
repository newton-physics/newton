# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continuous scalar joint coordinates shared by VBD drives, limits, and mimics."""

import warp as wp

from ...sim import JointType, Model, State
from ...sim.joint_coordinates import eval_joint_coordinate, eval_joint_velocity, unwrap_joint_coordinate

wp.set_module_options({"enable_backward": False})

_Vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.struct
class JointCoordinateData:
    body_com: wp.array[wp.vec3]
    joint_type: wp.array[int]
    parent: wp.array[int]
    child: wp.array[int]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    q_start: wp.array[int]
    qd_start: wp.array[int]
    dof_dim: wp.array2d[int]
    axis: wp.array[wp.vec3]
    world: wp.array[int]
    history: wp.array[float]


@wp.func
def evaluate_coordinate(data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform]):
    """Evaluate a coordinate on the current turn, together with its pose gradients."""
    angular_reference = wp.vec3()
    angular_start = data.q_start[joint] + data.dof_dim[joint, 0]
    for axis in range(data.dof_dim[joint, 1]):
        angular_reference[axis] = data.history[angular_start + axis]
    q, g_p, g_c = eval_joint_coordinate(
        joint,
        component,
        body_q,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        data.X_p,
        data.X_c,
        data.qd_start,
        data.dof_dim,
        data.axis,
        angular_reference,
        True,
    )
    q = unwrap_joint_coordinate(q, data.history[data.q_start[joint] + component], component, data.dof_dim[joint, 0])
    return q, g_p, g_c


@wp.kernel
def _begin_step(
    data: JointCoordinateData,
    joints: wp.array[int],
    body_q: wp.array[wp.transform],
    joint_q_seed: wp.array[float],
    rebaseline_mask: wp.array[bool],
    mimic_multipliers: wp.array[float],
):
    joint = joints[wp.tid()]
    start = data.q_start[joint]
    world = data.world[joint]
    slot = wp.where(world >= 0, world, rebaseline_mask.shape[0] - 1)
    count = data.dof_dim[joint, 0] + data.dof_dim[joint, 1]
    if rebaseline_mask[slot]:
        for component in range(count):
            data.history[start + component] = joint_q_seed[start + component]
    coordinates = _Vec6()
    for component in range(count):
        q, _g_p, _g_c = evaluate_coordinate(data, joint, component, body_q)
        coordinates[component] = q
        if mimic_multipliers:
            mimic_multipliers[data.qd_start[joint] + component] = 0.0
    for component in range(count):
        data.history[start + component] = coordinates[component]


@wp.kernel
def _end_step(
    data: JointCoordinateData,
    joints: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_mirror: wp.array[float],
    joint_qd_mirror: wp.array[float],
):
    joint = joints[wp.tid()]
    coordinates = _Vec6()
    for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
        q, g_p, g_c = evaluate_coordinate(data, joint, component, body_q)
        qd = eval_joint_velocity(data.parent[joint], data.child[joint], g_p, g_c, body_qd)
        qi = data.q_start[joint] + component
        vi = data.qd_start[joint] + component
        coordinates[component] = q
        if joint_q:
            joint_q[qi] = q
        if joint_qd:
            joint_qd[vi] = qd
        if joint_q_mirror:
            joint_q_mirror[qi] = q
        if joint_qd_mirror:
            joint_qd_mirror[vi] = qd
    for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
        data.history[data.q_start[joint] + component] = coordinates[component]


@wp.kernel
def _reset_coordinates(
    data: JointCoordinateData,
    joints: wp.array[int],
    model_joint_q: wp.array[float],
    model_joint_qd: wp.array[float],
    world_mask: wp.array[bool],
    reset_all: bool,
    world_count: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    joint = joints[wp.tid()]
    slot = wp.where(data.world[joint] >= 0, data.world[joint], world_count)
    selected = reset_all
    if not reset_all and slot < world_mask.shape[0]:
        selected = world_mask[slot]
    if selected:
        start = data.q_start[joint]
        for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
            if joint_q:
                joint_q[start + component] = model_joint_q[start + component]
            if joint_qd:
                dof = data.qd_start[joint] + component
                joint_qd[dof] = model_joint_qd[dof]


class JointCoordinates:
    """Own continuous coordinates independently of any particular joint feature.

    History is fixed throughout each solve and committed from its final poses.
    Initial/reset turn counts come from State.joint_q; disabled joints are still
    tracked. Each angular coordinate must move less than pi per timestep.
    """

    def __init__(self, model: Model):
        self.model = model
        self.data = JointCoordinateData()
        self.data.body_com = model.body_com
        for field, source in (
            ("joint_type", "joint_type"),
            ("parent", "joint_parent"),
            ("child", "joint_child"),
            ("X_p", "joint_X_p"),
            ("X_c", "joint_X_c"),
            ("q_start", "joint_q_start"),
            ("qd_start", "joint_qd_start"),
            ("dof_dim", "joint_dof_dim"),
            ("axis", "joint_axis"),
            ("world", "joint_world"),
        ):
            setattr(self.data, field, getattr(model, source))
        supported = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6)
        joints = [j for j, joint_type in enumerate(model.joint_type.numpy()) if joint_type in supported]
        self.joints = wp.array(joints, dtype=int, device=model.device)
        self.data.history = wp.clone(model.joint_q) if joints else wp.empty(0, dtype=float, device=model.device)

    def begin_step(self, state: State, rebaseline_mask: wp.array[bool], mimic_multipliers: wp.array[float] | None):
        if self.joints.size:
            wp.launch(
                _begin_step,
                dim=self.joints.size,
                inputs=[
                    self.data,
                    self.joints,
                    state.body_q,
                    state.joint_q if state.joint_q is not None else self.model.joint_q,
                    rebaseline_mask,
                ],
                outputs=[mimic_multipliers],
                device=self.model.device,
            )

    def end_step(self, state_in: State, state_out: State):
        if self.joints.size:
            wp.launch(
                _end_step,
                dim=self.joints.size,
                inputs=[self.data, self.joints, state_out.body_q, state_out.body_qd],
                outputs=[state_out.joint_q, state_out.joint_qd, state_in.joint_q, state_in.joint_qd],
                device=self.model.device,
            )

    def reset_coordinates(
        self, joint_q: wp.array[float] | None, joint_qd: wp.array[float] | None, world_mask: wp.array[bool] | None
    ):
        if self.joints.size:
            wp.launch(
                _reset_coordinates,
                dim=self.joints.size,
                inputs=[
                    self.data,
                    self.joints,
                    self.model.joint_q,
                    self.model.joint_qd,
                    world_mask,
                    world_mask is None,
                    self.model.world_count,
                ],
                outputs=[joint_q, joint_qd],
                device=self.model.device,
            )
