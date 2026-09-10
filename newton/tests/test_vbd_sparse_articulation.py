# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import rigid_sparse_articulation_kernels
from newton._src.solvers.vbd.rigid_sparse_articulation import build_rigid_articulation_sparse_layout
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import _joint_projectors
from newton.examples.cable import example_cable_cross_slide_table
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


@wp.kernel
def _evaluate_angular_projector(
    joint_type: int,
    joint_axis: wp.array[wp.vec3],
    parent_anchor_q: wp.quat,
    angular_projector: wp.array[wp.mat33],
):
    _, P_ang = _joint_projectors(joint_type, joint_axis, 0, 0, 1, parent_anchor_q)
    angular_projector[0] = P_ang


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / float(np.dot(q, q))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_mul(_quat_mul(q, np.array([v[0], v[1], v[2], 0.0])), _quat_inv(q))[0:3]


def _quat_angle(q: np.ndarray) -> float:
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return 2.0 * math.atan2(float(np.linalg.norm(q[0:3])), float(q[3]))


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.array(
        [axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)],
        dtype=np.float64,
    )


def _transform_point(xform: np.ndarray, local: np.ndarray) -> np.ndarray:
    return xform[0:3] + _quat_rotate(xform[3:7], local)


def _joint_residual(model: newton.Model, state: newton.State) -> float:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()

    residuals = []
    for joint_index in range(model.joint_count):
        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])
        jt = int(joint_type[joint_index])
        if jt not in (int(newton.JointType.FIXED), int(newton.JointType.REVOLUTE), int(newton.JointType.BALL)):
            continue

        x_child = _transform_point(body_q[child], joint_x_c[joint_index, 0:3])
        if parent >= 0:
            x_parent = _transform_point(body_q[parent], joint_x_p[joint_index, 0:3])
            q_parent = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
        else:
            x_parent = joint_x_p[joint_index, 0:3]
            q_parent = joint_x_p[joint_index, 3:7]

        q_child = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
        residuals.append(np.linalg.norm(x_child - x_parent))
        if jt == int(newton.JointType.FIXED):
            residuals.append(_quat_angle(_quat_mul(_quat_inv(q_parent), q_child)))

    return float(np.linalg.norm(np.asarray(residuals, dtype=np.float64)))


def _joint_split_residual(model: newton.Model, state: newton.State) -> tuple[float, float]:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()

    linear_residuals = []
    angular_residuals = []
    for joint_index in range(model.joint_count):
        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])
        jt = int(joint_type[joint_index])
        if child < 0:
            continue

        x_child = _transform_point(body_q[child], joint_x_c[joint_index, 0:3])
        q_child = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
        child_rest_q = _quat_mul(model.body_q.numpy()[child, 3:7], joint_x_c[joint_index, 3:7])
        if parent >= 0:
            x_parent = _transform_point(body_q[parent], joint_x_p[joint_index, 0:3])
            q_parent = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
            parent_rest_q = _quat_mul(model.body_q.numpy()[parent, 3:7], joint_x_p[joint_index, 3:7])
        else:
            x_parent = joint_x_p[joint_index, 0:3]
            q_parent = joint_x_p[joint_index, 3:7]
            parent_rest_q = q_parent

        P_lin = np.eye(3)
        P_ang = np.eye(3)
        qd_start = int(joint_qd_start[joint_index])
        if jt == int(newton.JointType.PRISMATIC):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            P_lin = P_lin - np.outer(axis, axis)
        elif jt == int(newton.JointType.REVOLUTE):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            P_ang = P_ang - np.outer(axis, axis)
        elif jt == int(newton.JointType.D6):
            lin_count = int(joint_dof_dim[joint_index, 0])
            ang_count = int(joint_dof_dim[joint_index, 1])
            for axis_index in range(lin_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + axis_index])
                axis /= np.linalg.norm(axis)
                P_lin = P_lin - np.outer(axis, axis)
            for axis_index in range(ang_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + lin_count + axis_index])
                axis /= np.linalg.norm(axis)
                P_ang = P_ang - np.outer(axis, axis)

        if jt in (
            int(newton.JointType.ROD),
            int(newton.JointType.BALL),
            int(newton.JointType.FIXED),
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.D6),
        ):
            linear_residuals.append(P_lin @ (x_child - x_parent))

        if jt in (
            int(newton.JointType.ROD),
            int(newton.JointType.FIXED),
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.D6),
        ):
            q_rel = _quat_mul(_quat_inv(q_parent), q_child)
            q_rel_rest = _quat_mul(_quat_inv(parent_rest_q), child_rest_q)
            q_err = _quat_mul(q_rel, _quat_inv(q_rel_rest))
            angular_world = _quat_rotate(q_parent, _quat_rotvec(q_err))
            angular_residuals.append(P_ang @ angular_world)

    linear = np.concatenate(linear_residuals) if linear_residuals else np.zeros(0)
    angular = np.concatenate(angular_residuals) if angular_residuals else np.zeros(0)
    return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[0:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return q[0:3] * (angle / vector_norm)


def _make_loop_model(device: str = "cpu") -> newton.Model:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.1
    positions = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.3, 0.0, 1.0]),
        np.array([0.3, 0.3, 1.0]),
        np.array([0.0, 0.3, 1.0]),
    ]
    bodies = []
    for i, pos in enumerate(positions):
        bodies.append(
            builder.add_link(
                xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                label=f"body_{i}",
            )
        )

    joints = [
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )
    ]
    for i in range(1, len(bodies)):
        parent_pos = positions[i - 1]
        child_pos = positions[i]
        mid = 0.5 * (parent_pos + child_pos)
        joints.append(
            builder.add_joint_fixed(
                parent=bodies[i - 1],
                child=bodies[i],
                parent_xform=wp.transform(p=wp.vec3(*(mid - parent_pos)), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(*(mid - child_pos)), q=wp.quat_identity()),
            )
        )

    mid = 0.5 * (positions[-1] + positions[0])
    joints.append(
        builder.add_joint_fixed(
            parent=bodies[-1],
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(*(mid - positions[-1])), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(*(mid - positions[0])), q=wp.quat_identity()),
        )
    )
    builder.add_articulation(joints, allow_closed_loops=True)
    builder.color()
    return builder.finalize(device=device)


def _make_single_body_model(com: wp.vec3 | None = None) -> newton.Model:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        com=com,
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32)) * 0.1,
        label="body",
    )
    builder.color()
    return builder.finalize(device="cpu")


def _make_fixed_ring_model(body_count: int) -> newton.Model:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.1
    radius = 0.05 * float(body_count)
    center = np.array([0.0, 0.0, 1.0])
    positions = []
    for i in range(body_count):
        theta = 2.0 * math.pi * float(i) / float(body_count)
        positions.append(center + np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0]))

    bodies = []
    for i, pos in enumerate(positions):
        bodies.append(
            builder.add_link(
                xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                label=f"ring_{i}",
            )
        )

    joints = [
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )
    ]

    for i in range(1, body_count):
        parent_pos = positions[i - 1]
        child_pos = positions[i]
        mid = 0.5 * (parent_pos + child_pos)
        joints.append(
            builder.add_joint_fixed(
                parent=bodies[i - 1],
                child=bodies[i],
                parent_xform=wp.transform(p=wp.vec3(*(mid - parent_pos)), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(*(mid - child_pos)), q=wp.quat_identity()),
            )
        )

    mid = 0.5 * (positions[-1] + positions[0])
    joints.append(
        builder.add_joint_fixed(
            parent=bodies[-1],
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(*(mid - positions[-1])), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(*(mid - positions[0])), q=wp.quat_identity()),
        )
    )
    builder.add_articulation(joints, allow_closed_loops=True)
    builder.color()
    return builder.finalize(device="cpu")


def _make_projected_joint_chain_model(joint_kind: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.1
    positions = [np.array([0.0, 0.0, 1.0]), np.array([0.35, 0.0, 1.0])]

    bodies = []
    for i, pos in enumerate(positions):
        bodies.append(
            builder.add_link(
                xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                label=f"{joint_kind}_{i}",
            )
        )

    joints = []
    JointDofConfig = newton.ModelBuilder.JointDofConfig
    if joint_kind == "revolute":
        joints.append(
            builder.add_joint_revolute(
                parent=-1,
                child=bodies[0],
                parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
                child_xform=wp.transform(),
                axis=newton.Axis.Y,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
            )
        )
        joints.append(
            builder.add_joint_revolute(
                parent=bodies[0],
                child=bodies[1],
                parent_xform=wp.transform(p=wp.vec3(0.175, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-0.175, 0.0, 0.0), q=wp.quat_identity()),
                axis=newton.Axis.Y,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
            )
        )
    elif joint_kind == "prismatic":
        joints.append(
            builder.add_joint_prismatic(
                parent=-1,
                child=bodies[0],
                parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
                child_xform=wp.transform(),
                axis=newton.Axis.X,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
            )
        )
        joints.append(
            builder.add_joint_prismatic(
                parent=bodies[0],
                child=bodies[1],
                parent_xform=wp.transform(p=wp.vec3(0.175, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-0.175, 0.0, 0.0), q=wp.quat_identity()),
                axis=newton.Axis.X,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
            )
        )
    elif joint_kind == "d6":
        linear_axis = JointDofConfig(axis=newton.Axis.X, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0)
        angular_axis = JointDofConfig(axis=newton.Axis.Y, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0)
        joints.append(
            builder.add_joint_d6(
                parent=-1,
                child=bodies[0],
                parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
                child_xform=wp.transform(),
                linear_axes=[linear_axis],
                angular_axes=[angular_axis],
            )
        )
        joints.append(
            builder.add_joint_d6(
                parent=bodies[0],
                child=bodies[1],
                parent_xform=wp.transform(p=wp.vec3(0.175, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-0.175, 0.0, 0.0), q=wp.quat_identity()),
                linear_axes=[linear_axis],
                angular_axes=[angular_axis],
            )
        )
    elif joint_kind == "cable":
        joints.append(
            builder.add_joint_rod(
                parent=-1,
                child=bodies[0],
                parent_xform=wp.transform(p=wp.vec3(*positions[0]), q=wp.quat_identity()),
                child_xform=wp.transform(),
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e5,
            )
        )
        joints.append(
            builder.add_joint_rod(
                parent=bodies[0],
                child=bodies[1],
                parent_xform=wp.transform(p=wp.vec3(0.175, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-0.175, 0.0, 0.0), q=wp.quat_identity()),
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e5,
            )
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")

    builder.add_articulation(joints)
    builder.color()
    return builder.finalize(device="cpu")


def _solve_coupled_revolute_armature() -> tuple[float, float, float]:
    parent_inertia = 0.2
    child_inertia = 0.3
    armature = 0.4
    drive_ke = 1000.0
    target_angle = 1.0e-3
    dt = 1.0e-2

    with mock.patch("newton.use_coord_layout_targets", True):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        parent = builder.add_link(
            xform=wp.transform(),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.25, parent_inertia, 0.35]).astype(np.float32)),
        )
        child = builder.add_link(
            xform=wp.transform(),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.4, child_inertia, 0.5]).astype(np.float32)),
        )
        root_joint = builder.add_joint_free(child=parent)
        revolute_joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=newton.Axis.Y,
            target_ke=drive_ke,
            target_kd=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            armature=armature,
        )
        builder.add_articulation([root_joint, revolute_joint])
        builder.color()
        model = builder.finalize(device="cpu")

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    target_q = control.joint_target_q.numpy()
    target_q[int(model.joint_target_q_start.numpy()[revolute_joint])] = target_angle
    control.joint_target_q.assign(target_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve="block_sparse_joints",
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=1.0e5,
        rigid_joint_angular_ke=1.0e5,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_joint_angular_k_start=drive_ke,
    )
    solver.step(state_in, state_out, control, None, dt)

    poses = state_out.body_q.numpy()
    parent_rotation = poses[parent, 3:7].astype(np.float64)
    child_rotation = poses[child, 3:7].astype(np.float64)
    parent_angle = float(_quat_rotvec(parent_rotation)[1])
    child_angle = float(_quat_rotvec(child_rotation)[1])
    relative_angle = float(_quat_rotvec(_quat_mul(_quat_inv(parent_rotation), child_rotation))[1])

    reduced_inertia = parent_inertia * child_inertia / (parent_inertia + child_inertia)
    expected_relative_angle = drive_ke * target_angle / (drive_ke + (reduced_inertia + armature) / (dt * dt))
    angular_momentum = parent_inertia * parent_angle + child_inertia * child_angle
    return relative_angle, expected_relative_angle, angular_momentum


def _make_cable_rod_model(closed: bool, bend_damping: float = 0.0) -> newton.Model:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    if closed:
        segment_count = 8
        radius = 0.35
        z = 1.0
        theta = np.linspace(0.0, 2.0 * np.pi, segment_count + 1, endpoint=True)
        points = [wp.vec3(float(radius * np.cos(t)), float(radius * np.sin(t)), z) for t in theta]
        rod = newton.Rod(points, closed=True, radius=0.02)
    else:
        segment_count = 8
        rod = newton.Rod.create_straight(
            start=wp.vec3(-0.4, 0.0, 1.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=0.8,
            segment_count=segment_count,
            radius=0.02,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="add_rod: wrap_in_articulation=False", category=UserWarning)
        _bodies, joints = builder.add_rod(
            rod=rod,
            cfg=builder.default_shape_cfg.copy(),
            stretch_stiffness=1.0e6,
            stretch_damping=0.0,
            bend_stiffness=1.0e4,
            bend_damping=bend_damping,
            wrap_in_articulation=False,
            body_frame_origin="start",
            label="sparse_cable_loop" if closed else "sparse_cable_chain",
        )
    builder.add_articulation(joints, allow_closed_loops=closed)
    builder.color()
    return builder.finalize(device="cpu")


def _perturb_body_poses(state: newton.State, translation_amplitude: float, rotation_amplitude: float) -> None:
    body_q = state.body_q.numpy().copy()
    axis = np.array([0.3, 0.5, 0.8], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    for body_id in range(body_q.shape[0]):
        phase = float(body_id + 1)
        body_q[body_id, 0] += translation_amplitude * math.sin(1.7 * phase)
        body_q[body_id, 1] += 0.5 * translation_amplitude * math.cos(2.3 * phase)
        dq = _quat_from_axis_angle(axis, rotation_amplitude * math.sin(1.1 * phase))
        body_q[body_id, 3:7] = _quat_mul(dq, body_q[body_id, 3:7])
        body_q[body_id, 3:7] /= np.linalg.norm(body_q[body_id, 3:7])
    state.body_q.assign(body_q)


def _perturb_projected_joint_chain(state: newton.State) -> None:
    body_q = state.body_q.numpy().copy()
    offsets = np.array([[0.0, 0.035, -0.025], [0.0, -0.025, 0.04]])
    axes = [np.array([0.7, 0.2, 0.1]), np.array([0.2, 0.1, 0.8])]
    angles = [0.12, -0.09]
    for body_id in range(body_q.shape[0]):
        body_q[body_id, 0:3] += offsets[body_id]
        dq = _quat_from_axis_angle(axes[body_id], angles[body_id])
        body_q[body_id, 3:7] = _quat_mul(dq, body_q[body_id, 3:7])
        body_q[body_id, 3:7] /= np.linalg.norm(body_q[body_id, 3:7])
    state.body_q.assign(body_q)


def _fixed_joint_weighted_energy(model: newton.Model, state: newton.State, solver: newton.solvers.SolverVBD) -> float:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_constraint_start = solver.joint_constraint_start.numpy()
    joint_penalty_k = solver.joint_penalty_k.numpy()

    energy = 0.0
    for joint_index in range(model.joint_count):
        if int(joint_type[joint_index]) != int(newton.JointType.FIXED):
            continue

        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])
        x_child = _transform_point(body_q[child], joint_x_c[joint_index, 0:3])
        q_child = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
        if parent >= 0:
            x_parent = _transform_point(body_q[parent], joint_x_p[joint_index, 0:3])
            q_parent = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
        else:
            x_parent = joint_x_p[joint_index, 0:3]
            q_parent = joint_x_p[joint_index, 3:7]

        c_start = int(joint_constraint_start[joint_index])
        linear = float(np.linalg.norm(x_child - x_parent))
        angular = _quat_angle(_quat_mul(_quat_inv(q_parent), q_child))
        energy += 0.5 * float(joint_penalty_k[c_start]) * linear * linear
        energy += 0.5 * float(joint_penalty_k[c_start + 1]) * angular * angular

    return energy


def _solve_stiffness_ratio_energy(mode: str) -> float:
    model = _make_fixed_ring_model(8)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    _perturb_body_poses(state_in, translation_amplitude=0.02, rotation_amplitude=0.08)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
        rigid_joint_linear_k_start=1.0e6,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_k_start=1.0,
        rigid_joint_angular_ke=1.0,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return _fixed_joint_weighted_energy(model, state_out, solver)


def _solve_projected_joint_split_residual(joint_kind: str, mode: str) -> tuple[float, float]:
    model = _make_projected_joint_chain_model(joint_kind)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    _perturb_projected_joint_chain(state_in)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e5,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return _joint_split_residual(model, state_out)


def _solve_cable_rod_split_residual(closed: bool, mode: str, bend_damping: float = 0.0) -> tuple[float, float]:
    model = _make_cable_rod_model(closed, bend_damping=bend_damping)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    _perturb_body_poses(state_in, translation_amplitude=0.012, rotation_amplitude=0.05)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e4,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return _joint_split_residual(model, state_out)


def _make_xy_table_example(mode: str):
    viewer = newton.viewer.ViewerNull(num_frames=1)
    args = SimpleNamespace(device="cpu", rigid_articulation_solve=mode)
    example = example_cable_cross_slide_table.Example(viewer, args)
    example.sim_substeps = 1
    example.solver.iterations = 1
    return example


def _solve_residual(mode: str, compliant_alm: bool = False) -> float:
    model = _make_loop_model()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    body_q = state_in.body_q.numpy().copy()
    for body_id in range(model.body_count):
        body_q[body_id, 0] += 0.02 * math.sin(1.7 * float(body_id + 1))
        body_q[body_id, 1] += 0.01 * math.cos(2.3 * float(body_id + 1))
    state_in.body_q.assign(body_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=compliant_alm,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return _joint_residual(model, state_out)


def _solve_loop_q(mode: str, device: str, compliant_alm: bool = False) -> np.ndarray:
    model = _make_loop_model(device)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    _perturb_body_poses(state_in, translation_amplitude=0.02, rotation_amplitude=0.0)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=compliant_alm,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return state_out.body_q.numpy()


def _solve_single_body_q(mode: str) -> np.ndarray:
    model = _make_single_body_model()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    body_q = state_in.body_q.numpy().copy()
    body_q[0, 0:3] += np.array([0.02, -0.01, 0.005])
    state_in.body_q.assign(body_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return state_out.body_q.numpy()


def _solve_offset_com_single_body_q(mode: str) -> np.ndarray:
    model = _make_single_body_model(com=wp.vec3(0.17, -0.08, 0.04))
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    body_q = state_in.body_q.numpy().copy()
    body_q[0, 0:3] += np.array([0.015, -0.02, 0.01])
    dq = _quat_from_axis_angle(np.array([0.2, 0.5, 0.7], dtype=np.float64), 0.12)
    body_q[0, 3:7] = _quat_mul(dq, body_q[0, 3:7])
    body_q[0, 3:7] /= np.linalg.norm(body_q[0, 3:7])
    state_in.body_q.assign(body_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=1.0,
    )
    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
    return state_out.body_q.numpy()


class TestVBDSparseArticulation(unittest.TestCase):
    def test_builder_rejects_closed_loop_articulation(self):
        """Reject closed-loop articulations unless the caller opts in."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        inertia = wp.mat33(np.eye(3, dtype=np.float32))
        bodies = [
            builder.add_link(xform=wp.transform(wp.vec3(float(i), 0.0, 0.0)), mass=1.0, inertia=inertia)
            for i in range(3)
        ]
        joints = [builder.add_joint_fixed(parent=-1, child=bodies[0])]
        joints.append(builder.add_joint_fixed(parent=bodies[0], child=bodies[1]))
        joints.append(builder.add_joint_fixed(parent=bodies[1], child=bodies[2]))
        joints.append(builder.add_joint_fixed(parent=bodies[0], child=bodies[2]))

        with self.assertRaisesRegex(ValueError, "multiple parents"):
            builder.add_articulation(joints)

    def test_builder_accepts_closed_loop_articulation_opt_in(self):
        """Include every loop joint and body when closed loops are enabled."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        inertia = wp.mat33(np.eye(3, dtype=np.float32))
        bodies = [
            builder.add_link(xform=wp.transform(wp.vec3(float(i), 0.0, 0.0)), mass=1.0, inertia=inertia)
            for i in range(3)
        ]
        joints = [builder.add_joint_fixed(parent=-1, child=bodies[0])]
        joints.append(builder.add_joint_fixed(parent=bodies[0], child=bodies[1]))
        joints.append(builder.add_joint_fixed(parent=bodies[1], child=bodies[2]))
        joints.append(builder.add_joint_fixed(parent=bodies[0], child=bodies[2]))

        builder.add_articulation(joints, allow_closed_loops=True)
        builder.color()
        model = builder.finalize(device="cpu")

        self.assertEqual(model.articulation_count, 1)
        np.testing.assert_array_equal(model.joint_articulation.numpy(), np.zeros(4, dtype=np.int32))

        # The loop closure is inside the articulation range, so the direct factorization
        # covers every joint and body of the articulation.
        layout = build_rigid_articulation_sparse_layout(model, "cpu")
        self.assertEqual(layout.articulation_count, 1)
        np.testing.assert_array_equal(np.sort(layout.articulation_joints.numpy()), np.arange(model.joint_count))
        np.testing.assert_array_equal(np.sort(layout.articulation_bodies.numpy()), np.arange(model.body_count))

    def test_sparse_revolute_projector_uses_parent_frame(self):
        """Construct the revolute angular projector in the parent frame."""
        axis = np.array([0.2, -0.4, 0.7], dtype=np.float32)
        axis /= np.linalg.norm(axis)
        parent_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.6, 0.1, -0.3)), 1.1)
        projector = wp.empty(1, dtype=wp.mat33, device="cpu")
        wp.launch(
            _evaluate_angular_projector,
            dim=1,
            inputs=[
                int(newton.JointType.REVOLUTE),
                wp.array([axis], dtype=wp.vec3, device="cpu"),
                parent_rotation,
            ],
            outputs=[projector],
            device="cpu",
        )
        expected = np.eye(3) - np.outer(axis, axis)
        np.testing.assert_allclose(projector.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_sparse_single_body_matches_local(self):
        """Match the local solve for an unconstrained body."""
        local_q = _solve_single_body_q("local")
        sparse_q = _solve_single_body_q("block_sparse_joints")
        np.testing.assert_allclose(sparse_q, local_q, rtol=1.0e-5, atol=1.0e-5)

    def test_sparse_single_body_offset_com_matches_local(self):
        """Match the local solve when the body center of mass is offset."""
        local_q = _solve_offset_com_single_body_q("local")
        sparse_q = _solve_offset_com_single_body_q("block_sparse_joints")
        np.testing.assert_allclose(sparse_q, local_q, rtol=1.0e-5, atol=1.0e-5)

    def test_sparse_articulation_reduces_loop_residual(self):
        """Reduce a stiff closed-loop residual relative to the local solve."""
        local_residual = _solve_residual("local")
        sparse_residual = _solve_residual("block_sparse_joints")
        self.assertLess(sparse_residual, 0.9 * local_residual)

    def test_sparse_articulation_reduces_compliant_alm_loop_residual(self):
        """Reduce a compliant-ALM loop residual relative to the local solve."""
        local_residual = _solve_residual("local", compliant_alm=True)
        sparse_residual = _solve_residual("block_sparse_joints", compliant_alm=True)
        self.assertLess(sparse_residual, 0.9 * local_residual)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_sparse_articulation_cuda_matches_cpu_serial(self):
        """Match the CUDA block solve to the CPU serial reference."""
        for compliant_alm in (False, True):
            with self.subTest(compliant_alm=compliant_alm):
                cpu_q = _solve_loop_q("block_sparse_joints", "cpu", compliant_alm=compliant_alm)
                cuda_q = _solve_loop_q("block_sparse_joints", "cuda:0", compliant_alm=compliant_alm)
                np.testing.assert_allclose(cuda_q, cpu_q, rtol=2.0e-4, atol=2.0e-4)

    def test_sparse_articulation_default_relaxation_is_tuned(self):
        """Retain the tuned sparse-articulation relaxation default."""
        model = _make_single_body_model()
        solver = newton.solvers.SolverVBD(
            model, iterations=1, rigid_compliant_alm=False, rigid_articulation_solve="block_sparse_joints"
        )
        self.assertEqual(solver.rigid_articulation_relaxation, 0.65)

    def test_builder_preserves_empty_joint_dof_rank(self):
        """Keep the joint DoF table two-dimensional when the model has no joints."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
        builder.color()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.joint_count, 0)
        self.assertEqual(model.joint_dof_dim.shape, (0, 2))

    def test_sparse_articulation_rejects_nonfinite_tuning(self):
        """Reject non-finite relaxation and regularization values."""
        model = _make_single_body_model()
        for name, value in (
            ("rigid_articulation_relaxation", math.nan),
            ("rigid_articulation_diagonal_regularization", math.nan),
            ("rigid_articulation_diagonal_regularization", math.inf),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    newton.solvers.SolverVBD(model, rigid_compliant_alm=False, **{name: value})

    def test_sparse_articulation_honors_deterministic_option(self):
        """Forward the requested deterministic mode to sparse kernels."""
        original = wp.get_module_options(module=rigid_sparse_articulation_kernels)
        try:
            model = _make_single_body_model()
            newton.solvers.SolverVBD(
                model,
                iterations=1,
                rigid_compliant_alm=False,
                rigid_articulation_solve="block_sparse_joints",
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            options = wp.get_module_options(module=rigid_sparse_articulation_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
        finally:
            wp.set_module_options(original, module=rigid_sparse_articulation_kernels)

    def test_sparse_articulation_couples_revolute_armature(self):
        """Couple revolute armature inertia across both incident bodies."""
        relative_angle, expected_relative_angle, angular_momentum = _solve_coupled_revolute_armature()
        self.assertAlmostEqual(relative_angle, expected_relative_angle, delta=1.0e-6)
        self.assertAlmostEqual(angular_momentum, 0.0, delta=1.0e-7)

    def test_sparse_articulation_handles_joint_stiffness_ratio(self):
        """Remain stable across a large joint-stiffness ratio."""
        local_energy = _solve_stiffness_ratio_energy("local")
        sparse_energy = _solve_stiffness_ratio_energy("block_sparse_joints")
        self.assertLess(sparse_energy, 0.05 * local_energy)

    def test_sparse_articulation_supports_projected_joint_types(self):
        """Improve constrained residuals for each projected joint type."""
        for joint_kind in ("revolute", "prismatic", "d6", "cable"):
            with self.subTest(joint_kind=joint_kind):
                local_linear, local_angular = _solve_projected_joint_split_residual(joint_kind, "local")
                sparse_linear, sparse_angular = _solve_projected_joint_split_residual(joint_kind, "block_sparse_joints")
                self.assertLess(sparse_linear, 0.9 * local_linear)
                self.assertLess(sparse_angular, 0.9 * local_angular)

    def test_sparse_articulation_improves_cable_rods(self):
        """Improve open and closed rod residuals over the local solve."""
        for closed in (False, True):
            with self.subTest(closed=closed):
                local_linear, local_angular = _solve_cable_rod_split_residual(closed, "local")
                sparse_linear, sparse_angular = _solve_cable_rod_split_residual(closed, "block_sparse_joints")
                self.assertLess(sparse_linear, 0.9 * local_linear)
                if closed:
                    self.assertLess(sparse_angular, 1.1 * local_angular)
                else:
                    self.assertLess(sparse_angular, 0.9 * local_angular)

    def test_sparse_articulation_includes_declared_xy_table_closure_joint(self):
        """Factor the cross-slide table closure inside its articulation."""
        example = _make_xy_table_example("block_sparse_joints")
        self.assertEqual(example.model.articulation_count, 1)
        self.assertEqual(int(example.model.joint_articulation.numpy()[-1]), 0)

        layout = example.solver.rigid_articulation_sparse_layout
        self.assertIsNotNone(layout)
        joint_offsets = layout.articulation_joint_offsets.numpy()
        joint_counts = np.diff(joint_offsets)
        self.assertEqual(int(np.max(joint_counts)), example.model.joint_count)
        self.assertEqual(
            len(np.unique(layout.articulation_bodies.numpy())) + layout.local_body_count,
            example.model.body_count,
        )

        example.step()
        example.test_post_step()
        self.assertTrue(np.isfinite(example.state_0.body_q.numpy()).all())


class TestVBDSparseArticulationDevices(unittest.TestCase):
    pass


def _run_unregistered_fixed_body(device, mode: str) -> np.ndarray:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01),
    )
    builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        rigid_compliant_alm=True,
        rigid_articulation_solve=mode,
    )
    for _ in range(20):
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    return state_in.body_q.numpy()


def test_sparse_falls_back_to_local_without_articulations(test, device):
    """Fall back to the local solve when no articulation is declared."""
    local_q = _run_unregistered_fixed_body(device, "local")
    sparse_q = _run_unregistered_fixed_body(device, "block_sparse_joints")
    np.testing.assert_allclose(sparse_q, local_q, rtol=1.0e-4, atol=1.0e-4)
    test.assertGreater(float(sparse_q[0, 2]), 0.99)


def test_sparse_uses_local_solve_for_standalone_body(test, device):
    """Use the local solve for bodies outside sparse articulations."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)
    articulated_body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=inertia
    )
    root_joint = builder.add_joint_fixed(
        parent=-1,
        child=articulated_body,
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([root_joint])

    standalone_body = builder.add_link(
        xform=wp.transform((0.0, 1.0, 2.0), wp.quat_identity()), mass=1.0, inertia=inertia
    )
    builder.add_joint_fixed(
        parent=-1,
        child=standalone_body,
        parent_xform=wp.transform((0.0, 1.0, 2.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=True,
        rigid_articulation_solve="block_sparse_joints",
    )
    layout = solver.rigid_articulation_sparse_layout
    test.assertIsNotNone(layout)
    test.assertEqual(layout.articulation_body_count, 1)
    test.assertEqual(layout.local_body_count, 1)

    for _ in range(20):
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    poses = state_in.body_q.numpy()
    test.assertGreater(float(poses[articulated_body, 2]), 0.99)
    test.assertGreater(float(poses[standalone_body, 2]), 1.99)


def _make_cross_articulation_closure_model(device, single_articulation: bool):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.1)
    groups = []
    joints = []
    for y in (0.0, 2.0):
        root = builder.add_link(xform=wp.transform((0.0, y, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
        tip = builder.add_link(xform=wp.transform((1.0, y, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
        root_joint = builder.add_joint_fixed(parent=-1, child=root)
        tip_joint = builder.add_joint_revolute(parent=root, child=tip, axis=newton.Axis.Z)
        joints.extend((root_joint, tip_joint))
        if not single_articulation:
            builder.add_articulation([root_joint, tip_joint])
        groups.append((root, tip))
    joints.append(builder.add_joint_fixed(parent=groups[0][1], child=groups[1][1]))
    if single_articulation:
        builder.add_articulation(joints, allow_closed_loops=True)
    builder.color()
    return builder.finalize(device=device)


def test_sparse_rejects_cross_articulation_joint(test, device):
    """Reject joints that connect two declared sparse articulations."""
    model = _make_cross_articulation_closure_model(device, single_articulation=False)
    with test.assertRaisesRegex(ValueError, "outside the declared articulation ranges"):
        newton.solvers.SolverVBD(
            model,
            iterations=8,
            rigid_compliant_alm=False,
            rigid_articulation_solve="block_sparse_joints",
        )


def test_sparse_rejects_omitted_articulation_closure(test, device):
    """Reject a closure joint omitted from its articulation range."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.1)
    root = builder.add_link(mass=1.0, inertia=inertia)
    tip = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    tip_joint = builder.add_joint_revolute(parent=root, child=tip, axis=newton.Axis.Z)
    builder.add_articulation([root_joint, tip_joint])
    builder.add_joint_fixed(
        parent=-1,
        child=tip,
        parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "outside the declared articulation ranges"):
        newton.solvers.SolverVBD(
            model,
            rigid_compliant_alm=False,
            rigid_articulation_solve="block_sparse_joints",
        )


def test_sparse_factorizes_closed_loop_articulation(test, device):
    """Factor every body and joint in an opted-in closed loop."""
    # Declaring the loop closure inside one articulation puts every joint and body of
    # that articulation into a single direct factorization.
    model = _make_cross_articulation_closure_model(device, single_articulation=True)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=False,
        rigid_articulation_solve="block_sparse_joints",
        rigid_articulation_relaxation=1.0,
    )
    layout = solver.rigid_articulation_sparse_layout
    test.assertIsNotNone(layout)
    test.assertEqual(layout.articulation_count, 1)
    np.testing.assert_array_equal(np.sort(layout.articulation_bodies.numpy()), np.arange(model.body_count))
    np.testing.assert_array_equal(np.sort(layout.articulation_joints.numpy()), np.arange(model.joint_count))

    state_in = model.state()
    state_out = model.state()
    for _ in range(5):
        solver.step(state_in, state_out, None, None, 1.0 / 240.0)
        state_in, state_out = state_out, state_in
    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())


def _run_anisotropic_rod(device, mode: str) -> np.ndarray:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    rod = newton.Rod.create_straight(
        start=wp.vec3(0.0, 0.0, 1.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=0.4,
        segment_count=4,
        radius=0.02,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="add_rod: wrap_in_articulation=False", category=UserWarning)
        bodies, joints = builder.add_rod(
            rod=rod,
            stretch_stiffness=1.0e4,
            shear_stiffness=1.0e3,
            bend_stiffness=1.0e2,
            twist_stiffness=5.0e2,
            wrap_in_articulation=False,
            body_frame_origin="start",
        )
    joints.append(
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(wp.vec3(rod.points[0]), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    )
    builder.add_articulation(joints, allow_closed_loops=True)
    builder.color()
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=100,
        rigid_compliant_alm=True,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=0.5,
    )
    for _ in range(10):
        solver.step(state_in, state_out, None, None, 1.0 / 240.0)
        state_in, state_out = state_out, state_in
    return state_in.body_q.numpy()


def test_sparse_anisotropic_rod_matches_local_trajectory(test, device):
    """Match the local trajectory for an anisotropic rod."""
    local_q = _run_anisotropic_rod(device, "local")
    sparse_q = _run_anisotropic_rod(device, "block_sparse_joints")
    test.assertTrue(np.isfinite(sparse_q).all())
    np.testing.assert_allclose(sparse_q[:, :3], local_q[:, :3], rtol=2.0e-3, atol=2.0e-3)


def test_sparse_default_relaxation_with_contact(test, device):
    """Keep the tuned default stable on the sparse contact path."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    builder.add_ground_plane()
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.12), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01),
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    root_joint = builder.add_joint_free(child=body)
    builder.add_articulation([root_joint])
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=True,
        rigid_articulation_solve="block_sparse_joints",
    )
    test.assertEqual(solver.rigid_articulation_relaxation, 0.65)
    test.assertIsNotNone(solver.rigid_articulation_sparse_layout)
    for _ in range(20):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    pose = state_in.body_q.numpy()[body]
    test.assertTrue(np.isfinite(pose).all())
    test.assertGreater(float(pose[2]), 0.08)


def test_sparse_updates_soft_contact_penalty_without_rigid_capacity(test, device):
    """Update soft-contact penalties when rigid-contact capacity is zero."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    builder.default_shape_cfg.ke = 1.0e3
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01))
    builder.add_shape_box(body, hx=0.4, hy=0.4, hz=0.1)
    root_joint = builder.add_joint_fixed(parent=-1, child=body)
    builder.add_articulation([root_joint])
    builder.add_cloth_grid(
        pos=(-0.15, -0.15, 0.25),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0,
        edge_ke=10.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=0, soft_contact_max=256)
    contacts = pipeline.contacts()
    test.assertEqual(contacts.rigid_contact_max, 0)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_compliant_alm=False,
        rigid_avbd_linear_beta=1.0e3,
        rigid_contact_k_start=1.0,
        rigid_articulation_solve="block_sparse_joints",
    )
    for _ in range(30):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(float(np.max(solver.body_particle_contact_penalty_k.numpy())), 1.0)


add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_falls_back_to_local_without_articulations",
    test_sparse_falls_back_to_local_without_articulations,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_uses_local_solve_for_standalone_body",
    test_sparse_uses_local_solve_for_standalone_body,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_rejects_cross_articulation_joint",
    test_sparse_rejects_cross_articulation_joint,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_rejects_omitted_articulation_closure",
    test_sparse_rejects_omitted_articulation_closure,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_factorizes_closed_loop_articulation",
    test_sparse_factorizes_closed_loop_articulation,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_anisotropic_rod_matches_local_trajectory",
    test_sparse_anisotropic_rod_matches_local_trajectory,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_default_relaxation_with_contact",
    test_sparse_default_relaxation_with_contact,
    devices=devices,
)
add_function_test(
    TestVBDSparseArticulationDevices,
    "test_sparse_updates_soft_contact_penalty_without_rigid_capacity",
    test_sparse_updates_soft_contact_penalty_without_rigid_capacity,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
