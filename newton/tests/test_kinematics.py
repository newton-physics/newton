# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.mujoco import kernels as mujoco_kernels
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def origin_velocity_from_body_qd(model, body_q, body_qd, body_idx):
    """Recover body-origin velocity from COM-referenced `body_qd`."""
    rot = wp.quat(
        float(body_q[body_idx, 3]),
        float(body_q[body_idx, 4]),
        float(body_q[body_idx, 5]),
        float(body_q[body_idx, 6]),
    )
    com_local = model.body_com.numpy()[body_idx]
    com_world = np.array(
        wp.quat_rotate(rot, wp.vec3(float(com_local[0]), float(com_local[1]), float(com_local[2]))),
        dtype=np.float32,
    )
    return body_qd[body_idx, :3] - np.cross(body_qd[body_idx, 3:6], com_world)


def eval_fk_mujoco_kernel(model, joint_q, joint_qd, state):
    """Evaluate the duplicated MuJoCo FK kernel directly."""
    wp.launch(
        kernel=mujoco_kernels.eval_articulation_fk,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_articulation,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
        ],
        outputs=[state.body_q, state.body_qd],
        device=model.device,
    )


def _add_free_distance_joint(builder, joint_type, parent, child, parent_xform, child_xform):
    if joint_type == newton.JointType.FREE:
        return builder.add_joint_free(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    if joint_type == newton.JointType.DISTANCE:
        return builder.add_joint_distance(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            min_distance=-1.0,
            max_distance=-1.0,
        )
    raise AssertionError(f"Unsupported joint type for test helper: {joint_type}")


def test_fk_ik(test, device):
    builder = newton.ModelBuilder()

    world_count = 1

    for i in range(world_count):
        builder.add_mjcf(newton.examples.get_asset("nv_ant.xml"), up_axis="Y")

        coord_count = 15
        dof_count = 14

        coord_start = i * coord_count
        dof_start = i * dof_count

        # base
        builder.joint_q[coord_start : coord_start + 3] = [i * 2.0, 0.70, 0.0]
        builder.joint_q[coord_start + 3 : coord_start + 7] = wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5
        )

        # joints
        builder.joint_q[coord_start + 7 : coord_start + coord_count] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_qd[dof_start + 6 : dof_start + dof_count] = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]

    # finalize model
    model = builder.finalize(device=device)

    state = model.state()

    # save a copy of joint values
    q_fk = model.joint_q.numpy()
    qd_fk = model.joint_qd.numpy()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    q_ik = wp.zeros_like(model.joint_q, device=device)
    qd_ik = wp.zeros_like(model.joint_qd, device=device)

    newton.eval_ik(model, state, q_ik, qd_ik)

    assert_np_equal(q_fk, q_ik.numpy(), tol=1e-6)
    assert_np_equal(qd_fk, qd_ik.numpy(), tol=1e-6)


def test_fk_ik_with_analytical_solution(test, device):
    # Verify FK computes correct positions for a 2-link planar arm, and IK recovers joint angles.
    # Test parameters: length of the two links
    L1, L2 = 1.0, 0.8

    # Add two dummy links with revolute joint
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    link0 = builder.add_link()
    builder.add_shape_sphere(link0, radius=0.01)
    link1 = builder.add_link()
    builder.add_shape_sphere(link1, radius=0.01)
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, L1, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, L2, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])
    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qi0 = q_start[0]
    qi1 = q_start[1]

    angle_configs = [(0.0, 0.0), (0.3, 0.0), (0.0, -0.5), (np.pi / 4, np.pi / 4), (0.3, -0.2)]
    tol = 1e-4
    for theta1, theta2 in angle_configs:
        # Set desired angles
        state = model.state()
        q_init = state.joint_q.numpy()
        q_init[qi0] = theta1
        q_init[qi1] = theta2
        state.joint_q.assign(q_init)

        # Call Fk
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        body_q = state.body_q.numpy()
        pos0 = body_q[0][:3]
        pos1 = body_q[1][:3]

        # Calculate analytical pose
        expected_pos0_x = L1 * np.sin(theta1)
        expected_pos0_y = -L1 * np.cos(theta1)
        expected_pos1_x = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        expected_pos1_y = -L1 * np.cos(theta1) - L2 * np.cos(theta1 + theta2)

        test.assertAlmostEqual(pos0[0], expected_pos0_x, delta=tol, msg=f"Link0 X @ ({theta1:.2f},{theta2:.2f})")
        test.assertAlmostEqual(pos0[1], expected_pos0_y, delta=tol, msg=f"Link0 Y @ ({theta1:.2f},{theta2:.2f})")
        test.assertAlmostEqual(pos0[2], 0.0, delta=tol, msg=f"Link0 Z @ ({theta1:.2f},{theta2:.2f})")

        test.assertAlmostEqual(pos1[0], expected_pos1_x, delta=tol, msg=f"Link1 X @ ({theta1:.2f},{theta2:.2f})")
        test.assertAlmostEqual(pos1[1], expected_pos1_y, delta=tol, msg=f"Link1 Y @ ({theta1:.2f},{theta2:.2f})")
        test.assertAlmostEqual(pos1[2], 0.0, delta=tol, msg=f"Link1 Z @ ({theta1:.2f},{theta2:.2f})")

        # Call IK to recover joint angles from body state
        q_ik = wp.zeros_like(model.joint_q, device=device)
        qd_ik = wp.zeros_like(model.joint_qd, device=device)
        newton.eval_ik(model, state, q_ik, qd_ik)

        q_recovered = q_ik.numpy()
        test.assertAlmostEqual(
            float(q_recovered[qi0]), theta1, delta=tol, msg=f"IK theta1 @ ({theta1:.2f},{theta2:.2f})"
        )
        test.assertAlmostEqual(
            float(q_recovered[qi1]), theta2, delta=tol, msg=f"IK theta2 @ ({theta1:.2f},{theta2:.2f})"
        )


def test_fk_descendant_linear_velocity_matches_finite_difference(test, device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    link0 = builder.add_link()
    link1 = builder.add_link()

    builder.body_com[link0] = wp.vec3(0.35, 0.0, 0.0)
    builder.body_com[link1] = wp.vec3(0.0, 0.0, 0.0)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        # Nonzero child offset exercises transport from the joint anchor to the
        # descendant body origin even for a purely revolute chain.
        child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    state = model.state()
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()

    q[q_start[0]] = 0.7
    q[q_start[1]] = -0.35
    qd[qd_start[0]] = 1.1
    qd[qd_start[1]] = -0.45

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    dt = 1.0e-4
    q_next = q.copy()
    q_next[q_start[0]] += qd[qd_start[0]] * dt
    q_next[q_start[1]] += qd[qd_start[1]] * dt

    state_next = model.state()
    state_next.joint_q.assign(q_next)
    state_next.joint_qd.assign(qd)
    newton.eval_fk(model, state_next.joint_q, state_next.joint_qd, state_next)

    body_q = state.body_q.numpy().reshape(-1, 7)
    body_q_next = state_next.body_q.numpy().reshape(-1, 7)
    body_qd = state.body_qd.numpy().reshape(-1, 6)

    tip_idx = link1
    origin_vel_fd = (body_q_next[tip_idx, :3] - body_q[tip_idx, :3]) / dt
    origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, tip_idx)

    # body_q is float32, so forward finite differences at small dt carry ~1e-3
    # quantization error. This tolerance is still tight enough to catch the
    # descendant velocity-propagation bug fixed for articulated chains.
    assert_np_equal(origin_vel_fd, origin_vel_from_body_qd, tol=5.0e-3)


def test_fk_prismatic_descendant_linear_velocity_matches_finite_difference(test, device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link()
    slider = builder.add_link()

    builder.body_com[base] = wp.vec3(0.2, 0.0, 0.0)
    builder.body_com[slider] = wp.vec3(0.35, 0.0, -0.1)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_prismatic(
        parent=base,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.4), wp.quat_identity()),
        # Nonzero child offset exercises transport from the child joint anchor
        # to the child body origin in addition to the translated parent path.
        child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    state = model.state()
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()

    q[q_start[0]] = 0.55
    q[q_start[1]] = 0.8
    qd[qd_start[0]] = 1.1
    qd[qd_start[1]] = -0.35

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    dt = 1.0e-4
    q_next = q.copy()
    q_next[q_start[0]] += qd[qd_start[0]] * dt
    q_next[q_start[1]] += qd[qd_start[1]] * dt

    state_next = model.state()
    state_next.joint_q.assign(q_next)
    state_next.joint_qd.assign(qd)
    newton.eval_fk(model, state_next.joint_q, state_next.joint_qd, state_next)

    body_q = state.body_q.numpy().reshape(-1, 7)
    body_q_next = state_next.body_q.numpy().reshape(-1, 7)
    body_qd = state.body_qd.numpy().reshape(-1, 6)

    tip_idx = slider
    origin_vel_fd = (body_q_next[tip_idx, :3] - body_q[tip_idx, :3]) / dt
    origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, tip_idx)

    assert_np_equal(origin_vel_fd, origin_vel_from_body_qd, tol=5.0e-3)


def test_ik_prismatic_descendant_recovers_joint_state(test, device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link()
    slider = builder.add_link()

    builder.body_com[base] = wp.vec3(0.2, 0.0, 0.0)
    builder.body_com[slider] = wp.vec3(0.35, 0.0, -0.1)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_prismatic(
        parent=base,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.4), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    state = model.state()
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()

    q[q_start[0]] = 0.55
    q[q_start[1]] = 0.8
    qd[qd_start[0]] = 1.1
    qd[qd_start[1]] = -0.35

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    recovered_q = wp.zeros_like(state.joint_q)
    recovered_qd = wp.zeros_like(state.joint_qd)
    newton.eval_ik(model, state, recovered_q, recovered_qd)

    assert_np_equal(recovered_q.numpy(), q, tol=1.0e-6)
    assert_np_equal(recovered_qd.numpy(), qd, tol=1.0e-6)


def test_fk_free_distance_root_descendant_linear_velocity_matches_finite_difference(test, device, joint_type):
    """FK finite-difference regression for a FREE/DISTANCE-rooted chain with an offset-COM descendant.

    This pins the public ``newton.eval_fk`` (and the duplicated MuJoCo
    ``eval_articulation_fk`` kernel) against a forward finite difference for
    a FREE/DISTANCE-rooted articulation whose root carries a non-zero
    ``body_com`` and whose descendant body has a non-zero ``body_com`` and a
    non-trivial child joint anchor -- the exact boundary the PR reworked.

    A regression in the FREE/DISTANCE velocity transport (e.g. dropping the
    ``omega x com_offset`` term, or mishandling the public
    ``(v_com_world, omega_world)`` convention) shows up as a mismatch
    between the origin velocity recovered from ``body_qd`` and the
    forward-difference of ``body_q`` for either the root or the descendant.
    Looping over both FK entry points guards both code paths against
    divergence on the FREE/DISTANCE root.
    """
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link()
    child = builder.add_link()

    # Non-zero body COMs for both links so COM-aware velocity transport is
    # exercised at the FK boundary for the root and the descendant.
    builder.body_com[base] = wp.vec3(0.15, -0.1, 0.05)
    builder.body_com[child] = wp.vec3(0.3, 0.1, -0.15)

    # Identity parent/child anchors on the FREE/DISTANCE root so integrating
    # the seven root-joint coordinates corresponds directly to integrating
    # the base body pose; this keeps the finite-difference reference math
    # self-contained while still exercising the FK boundary.
    j0 = _add_free_distance_joint(
        builder=builder,
        joint_type=joint_type,
        parent=-1,
        child=base,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    # Non-trivial anchors on the revolute descendant exercise the transport
    # from the child joint anchor to the descendant body origin under a
    # moving parent.
    j1 = builder.add_joint_revolute(
        parent=base,
        child=child,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.75, 0.25, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.2, -0.05, 0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    q = model.joint_q.numpy()
    qd = model.joint_qd.numpy()

    # Root pose: translated origin plus a non-axis-aligned rotation so the
    # COM offset is rotated into all three world axes.
    q[q_start[0] : q_start[0] + 3] = np.array([0.2, -0.1, 0.15], dtype=np.float32)
    root_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, -0.2)), 0.55)
    q[q_start[0] + 3 : q_start[0] + 7] = np.array(
        [float(root_rot[0]), float(root_rot[1]), float(root_rot[2]), float(root_rot[3])],
        dtype=np.float32,
    )

    # Root joint_qd uses Newton's public (v_com_world, omega_world) convention.
    v_com_world = np.array([0.6, -0.3, 0.2], dtype=np.float32)
    omega_world = np.array([0.4, 0.25, -0.5], dtype=np.float32)
    qd[qd_start[0] : qd_start[0] + 3] = v_com_world
    qd[qd_start[0] + 3 : qd_start[0] + 6] = omega_world

    # Descendant revolute with non-zero angle and angular velocity.
    q[q_start[1]] = 0.35
    qd[qd_start[1]] = -0.7

    dt = 1.0e-4

    # Integrate the FREE/DISTANCE root coordinates under the public
    # (v_com_world, omega_world) convention:
    #   - advance the root orientation by omega * dt in world frame;
    #   - advance the root COM by v_com * dt;
    #   - back out the new root origin from the new orientation and COM.
    base_com_local = model.body_com.numpy()[base]
    base_com_local_wp = wp.vec3(
        float(base_com_local[0]),
        float(base_com_local[1]),
        float(base_com_local[2]),
    )
    omega_mag = float(np.linalg.norm(omega_world))
    if omega_mag > 0.0:
        omega_axis = wp.vec3(
            float(omega_world[0]) / omega_mag,
            float(omega_world[1]) / omega_mag,
            float(omega_world[2]) / omega_mag,
        )
        delta_r = wp.quat_from_axis_angle(omega_axis, float(omega_mag * dt))
    else:
        delta_r = wp.quat_identity()
    root_rot_next = delta_r * root_rot

    p = q[q_start[0] : q_start[0] + 3].astype(np.float32)
    com_world_initial = p + np.array(wp.quat_rotate(root_rot, base_com_local_wp), dtype=np.float32)
    com_world_next = com_world_initial + v_com_world * dt
    p_next = com_world_next - np.array(wp.quat_rotate(root_rot_next, base_com_local_wp), dtype=np.float32)

    q_next = q.copy()
    q_next[q_start[0] : q_start[0] + 3] = p_next
    q_next[q_start[0] + 3 : q_start[0] + 7] = np.array(
        [float(root_rot_next[0]), float(root_rot_next[1]), float(root_rot_next[2]), float(root_rot_next[3])],
        dtype=np.float32,
    )
    q_next[q_start[1]] = q[q_start[1]] + qd[qd_start[1]] * dt

    for eval_fk_fn in (newton.eval_fk, eval_fk_mujoco_kernel):
        state = model.state()
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        eval_fk_fn(model, state.joint_q, state.joint_qd, state)

        state_next = model.state()
        state_next.joint_q.assign(q_next)
        state_next.joint_qd.assign(qd)
        eval_fk_fn(model, state_next.joint_q, state_next.joint_qd, state_next)

        body_q = state.body_q.numpy().reshape(-1, 7)
        body_q_next = state_next.body_q.numpy().reshape(-1, 7)
        body_qd = state.body_qd.numpy().reshape(-1, 6)

        # Root body origin FD check -- pins the FREE/DISTANCE COM-velocity contract.
        root_origin_vel_fd = (body_q_next[base, :3] - body_q[base, :3]) / dt
        root_origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, base)
        assert_np_equal(root_origin_vel_fd, root_origin_vel_from_body_qd, tol=5.0e-3)

        # Descendant body origin FD check -- covers the FREE/DISTANCE-rooted
        # chain coverage that the removed Featherstone-specific oracle used to pin.
        child_origin_vel_fd = (body_q_next[child, :3] - body_q[child, :3]) / dt
        child_origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, child)
        assert_np_equal(child_origin_vel_fd, child_origin_vel_from_body_qd, tol=5.0e-3)


def test_ik_free_distance_descendant_recovers_joint_state(test, device, joint_type):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link()
    child = builder.add_link()

    builder.body_com[base] = wp.vec3(0.25, -0.1, 0.0)
    builder.body_com[child] = wp.vec3(0.3, 0.15, -0.2)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = _add_free_distance_joint(
        builder=builder,
        joint_type=joint_type,
        parent=base,
        child=child,
        parent_xform=wp.transform(wp.vec3(1.0, 0.2, 0.3), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.1, -0.05, 0.2), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    state = model.state()
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()

    q[q_start[0]] = 0.35
    q[q_start[1] : q_start[1] + 3] = np.array([0.4, -0.2, 0.3], dtype=np.float32)
    q_free_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.45)
    q[q_start[1] + 3 : q_start[1] + 7] = np.array(
        [q_free_rot[0], q_free_rot[1], q_free_rot[2], q_free_rot[3]],
        dtype=np.float32,
    )

    qd[qd_start[0]] = 0.9
    qd[qd_start[1] : qd_start[1] + 6] = np.array([0.2, -0.15, 0.1, 0.4, -0.3, 0.25], dtype=np.float32)

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    recovered_q = wp.zeros_like(state.joint_q)
    recovered_qd = wp.zeros_like(state.joint_qd)
    newton.eval_ik(model, state, recovered_q, recovered_qd)

    assert_np_equal(recovered_q.numpy(), q, tol=1.0e-5)
    assert_np_equal(recovered_qd.numpy(), qd, tol=1.0e-5)


def test_solver_fk_prismatic_descendant_linear_velocity_matches_finite_difference(test, device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link()
    slider = builder.add_link()

    builder.body_com[base] = wp.vec3(0.2, 0.0, 0.0)
    builder.body_com[slider] = wp.vec3(0.35, 0.0, -0.1)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j1 = builder.add_joint_prismatic(
        parent=base,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.4), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    q = model.joint_q.numpy()
    qd = model.joint_qd.numpy()
    q[q_start[0]] = 0.55
    q[q_start[1]] = 0.8
    qd[qd_start[0]] = 1.1
    qd[qd_start[1]] = -0.35

    dt = 1.0e-4
    q_next = q.copy()
    q_next[q_start[0]] += qd[qd_start[0]] * dt
    q_next[q_start[1]] += qd[qd_start[1]] * dt

    for eval_fk_fn in (newton.eval_fk, eval_fk_mujoco_kernel):
        state = model.state()
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        eval_fk_fn(model, state.joint_q, state.joint_qd, state)

        state_next = model.state()
        state_next.joint_q.assign(q_next)
        state_next.joint_qd.assign(qd)
        eval_fk_fn(model, state_next.joint_q, state_next.joint_qd, state_next)

        body_q = state.body_q.numpy().reshape(-1, 7)
        body_q_next = state_next.body_q.numpy().reshape(-1, 7)
        body_qd = state.body_qd.numpy().reshape(-1, 6)

        origin_vel_fd = (body_q_next[slider, :3] - body_q[slider, :3]) / dt
        origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, slider)

        assert_np_equal(origin_vel_fd, origin_vel_from_body_qd, tol=5.0e-3)


def test_fk_with_indices(test, device):
    """Test eval_fk with articulation indices parameter"""
    builder = newton.ModelBuilder()

    # Create 3 simple pendulums (articulations)
    for i in range(3):
        b1 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()))
        b2 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0 + 1.0, 0.0, 0.0), wp.quat_identity()))
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1, j2], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    # First, establish initial positions with zero angles
    joint_q_initial = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    newton.eval_fk(model, joint_q_initial, joint_qd, state)

    # Now set different joint angles for articulation 1 only
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_q_np = joint_q.numpy()
    joint_q_np[2:4] = [0.3, 0.4]  # Only set angles for articulation 1
    joint_q = wp.array(joint_q_np, dtype=float, device=device)

    # Update only articulation 1 using indices
    indices = wp.array([1], dtype=int, device=device)
    newton.eval_fk(model, joint_q, joint_qd, state, indices=indices)

    # Check the body positions
    body_q = state.body_q.numpy()

    # Verify max_joints_per_articulation was computed correctly
    test.assertEqual(model.max_joints_per_articulation, 2)

    # Check articulation mapping
    test.assertEqual(model.articulation_count, 3)

    # Check the body positions and rotations
    body_q = state.body_q.numpy()

    # Bodies 0,1 (articulation 0) should still be at their initial positions
    test.assertAlmostEqual(body_q[0, 0], 0.0, places=6)  # body 0 x position
    test.assertAlmostEqual(body_q[1, 0], 1.0, places=6)  # body 1 x position
    test.assertAlmostEqual(body_q[0, 1], 0.0, places=6)  # body 0 y position
    test.assertAlmostEqual(body_q[1, 1], 0.0, places=6)  # body 1 y position

    # For articulation 1:
    # Body 2 is the base link connected to world - it rotates around its anchor at (2,0,0)
    # Since the anchor is at the body center, position doesn't change but orientation does
    test.assertAlmostEqual(body_q[2, 0], 2.0, places=6)  # body 2 x position stays the same
    test.assertAlmostEqual(body_q[2, 1], 0.0, places=6)  # body 2 y position stays the same

    # Body 3 is connected to body 2 and should have moved due to both joint rotations
    # With joint angles [0.3, 0.4], body 3 should be displaced
    test.assertNotAlmostEqual(body_q[3, 0], 3.0, places=2)  # body 3 x should have changed
    test.assertNotAlmostEqual(body_q[3, 1], 0.0, places=2)  # body 3 y should have changed

    # Bodies 4,5 (articulation 2) should still be at their initial positions
    test.assertAlmostEqual(body_q[4, 0], 4.0, places=6)  # body 4 x position
    test.assertAlmostEqual(body_q[5, 0], 5.0, places=6)  # body 5 x position
    test.assertAlmostEqual(body_q[4, 1], 0.0, places=6)  # body 4 y position
    test.assertAlmostEqual(body_q[5, 1], 0.0, places=6)  # body 5 y position


def test_ik_with_indices(test, device):
    """Test eval_ik with articulation indices parameter"""
    builder = newton.ModelBuilder()

    # Create 2 simple pendulums
    for i in range(2):
        b1 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()))
        b2 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0 + 1.0, 0.0, 0.0), wp.quat_identity()))
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1, j2], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    # Set joint angles and compute FK
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)

    joint_q_np = joint_q.numpy()
    joint_q_np[0:2] = [0.1, 0.2]  # Articulation 0
    joint_q_np[2:4] = [0.3, 0.4]  # Articulation 1
    joint_q = wp.array(joint_q_np, dtype=float, device=device)

    newton.eval_fk(model, joint_q, joint_qd, state)

    # Test IK with indices - only recover articulation 0
    joint_q_ik = wp.zeros_like(joint_q)
    joint_qd_ik = wp.zeros_like(joint_qd)
    indices = wp.array([0], dtype=int, device=device)

    newton.eval_ik(model, state, joint_q_ik, joint_qd_ik, indices=indices)

    joint_q_ik_np = joint_q_ik.numpy()

    # Articulation 0 should be recovered
    assert_np_equal(joint_q_np[0:2], joint_q_ik_np[0:2], tol=2e-6)

    # Articulation 1 should remain zero
    assert_np_equal(np.array([0.0, 0.0]), joint_q_ik_np[2:4], tol=1e-6)


def test_fk_error_mask_and_indices(test, device):
    """Test that eval_fk raises error when both mask and indices are provided"""
    builder = newton.ModelBuilder()

    # Create a simple model
    b1 = builder.add_link()
    j1 = builder.add_joint_revolute(parent=-1, child=b1, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([j1])

    model = builder.finalize(device=device)
    state = model.state()

    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)

    mask = wp.array([True], dtype=bool, device=device)
    indices = wp.array([0], dtype=int, device=device)

    # Should raise ValueError
    with test.assertRaises(ValueError) as context:
        newton.eval_fk(model, joint_q, joint_qd, state, mask=mask, indices=indices)

    test.assertIn("Cannot specify both mask and indices", str(context.exception))


def test_isaac_lab_use_case(test, device):
    """Test the Isaac Lab pattern of updating specific world articulations"""
    builder = newton.ModelBuilder()

    # Create 8 identical robots (worlds)
    world_count = 8
    for i in range(world_count):
        b1 = builder.add_link(xform=wp.transform(wp.vec3(i * 3.0, 0.0, 0.0), wp.quat_identity()))
        b2 = builder.add_link(xform=wp.transform(wp.vec3(i * 3.0 + 1.0, 0.0, 0.0), wp.quat_identity()))
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 3.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1, j2], label=f"env_{i}")

    model = builder.finalize(device=device)

    # Test pattern: reset specific worlds
    world_indices_to_reset = wp.array([1, 3, 5], dtype=int, device=device)

    # Set all joints to some non-zero value
    joint_q = wp.full(model.joint_coord_count, 0.5, dtype=float, device=device)
    joint_qd = wp.full(model.joint_dof_count, 0.1, dtype=float, device=device)

    # Create reset values (zeros)
    reset_q = wp.zeros_like(joint_q)
    reset_qd = wp.zeros_like(joint_qd)

    # Update state with non-zero values for all
    state = model.state()
    newton.eval_fk(model, joint_q, joint_qd, state)

    # Reset only specific worlds
    newton.eval_fk(model, reset_q, reset_qd, state, indices=world_indices_to_reset)

    # Verify with IK
    recovered_q = wp.zeros_like(joint_q)
    recovered_qd = wp.zeros_like(joint_qd)
    newton.eval_ik(model, state, recovered_q, recovered_qd)

    recovered_q_np = recovered_q.numpy()

    # Check that reset worlds have zero values
    for world_idx in [1, 3, 5]:
        joint_start = world_idx * 2
        assert_np_equal(np.array([0.0, 0.0]), recovered_q_np[joint_start : joint_start + 2], tol=1e-6)

    # Check that non-reset worlds still have original values
    for world_idx in [0, 2, 4, 6, 7]:
        joint_start = world_idx * 2
        assert_np_equal(np.array([0.5, 0.5]), recovered_q_np[joint_start : joint_start + 2], tol=1e-6)


def test_bounds_checking(test, device):
    """Test that invalid articulation indices are handled gracefully"""
    builder = newton.ModelBuilder()

    # Create 2 articulations
    for _ in range(2):
        b1 = builder.add_link()
        j1 = builder.add_joint_revolute(parent=-1, child=b1, axis=wp.vec3(0.0, 0.0, 1.0))
        builder.add_articulation([j1])

    model = builder.finalize(device=device)
    state = model.state()

    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)

    # Test with invalid indices (negative and out of range)
    invalid_indices = wp.array([-1, 0, 5, 1, 100], dtype=int, device=device)

    # Should not crash - invalid indices are skipped
    newton.eval_fk(model, joint_q, joint_qd, state, indices=invalid_indices)
    newton.eval_ik(model, state, joint_q, joint_qd, indices=invalid_indices)

    # The test passes if no exception is raised


def test_ik_with_mask(test, device):
    """Test eval_ik with mask parameter"""
    builder = newton.ModelBuilder()

    # Create 3 simple pendulums
    for i in range(3):
        b1 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()))
        b2 = builder.add_link(xform=wp.transform(wp.vec3(i * 2.0 + 1.0, 0.0, 0.0), wp.quat_identity()))
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1, j2])

    model = builder.finalize(device=device)
    state = model.state()

    # Set joint angles for all articulations
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    joint_q_np = joint_q.numpy()
    # Each articulation has 2 joints
    joint_q_np[0:2] = [0.1, 0.2]  # articulation 0
    joint_q_np[2:4] = [0.3, 0.4]  # articulation 1
    joint_q_np[4:6] = [0.5, 0.6]  # articulation 2
    joint_q = wp.array(joint_q_np, dtype=float, device=device)

    # Run FK to update body transforms
    newton.eval_fk(model, joint_q, joint_qd, state)

    # Now run IK with mask to recover joint values for only articulations 0 and 2
    recovered_q = wp.zeros_like(joint_q)
    recovered_qd = wp.zeros_like(joint_qd)
    mask = wp.array([True, False, True], dtype=bool, device=device)
    newton.eval_ik(model, state, recovered_q, recovered_qd, mask=mask)

    recovered_q_np = recovered_q.numpy()

    # Check articulation 0 recovered correctly
    assert_np_equal(np.array([0.1, 0.2]), recovered_q_np[0:2], tol=2e-6)

    # Check articulation 1 still has zero values (masked out)
    assert_np_equal(np.array([0.0, 0.0]), recovered_q_np[2:4], tol=1e-6)

    # Check articulation 2 recovered correctly
    assert_np_equal(np.array([0.5, 0.6]), recovered_q_np[4:6], tol=2e-6)


def test_ik_error_mask_and_indices(test, device):
    """Test that eval_ik raises error when both mask and indices are provided"""
    builder = newton.ModelBuilder()
    parent = builder.add_link(xform=wp.transform((0, 0, 0), wp.quat_identity()))
    child = builder.add_link(xform=wp.transform((1, 0, 0), wp.quat_identity()))
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint])

    model = builder.finalize(device=device)
    state = model.state()

    mask = wp.array([True], dtype=bool, device=device)
    indices = wp.array([0], dtype=int, device=device)

    # Should raise ValueError
    with test.assertRaises(ValueError) as cm:
        newton.eval_ik(model, state, state.joint_q, state.joint_qd, mask=mask, indices=indices)

    test.assertIn("mutually exclusive", str(cm.exception))


def _build_dynamic_and_kinematic_single_joint_model(device):
    builder = newton.ModelBuilder()

    dynamic_body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    dynamic_joint = builder.add_joint_revolute(
        parent=-1,
        child=dynamic_body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([dynamic_joint], label="dynamic_articulation")

    kinematic_body = builder.add_link(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    kinematic_joint = builder.add_joint_revolute(
        parent=-1,
        child=kinematic_body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([kinematic_joint], label="kinematic_articulation")

    return builder.finalize(device=device)


def test_fk_body_flag_filter_dynamic_only(test, device):
    model = _build_dynamic_and_kinematic_single_joint_model(device)
    state = model.state()

    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    zero_q = wp.zeros(model.joint_coord_count, dtype=float, device=device)
    newton.eval_fk(model, zero_q, joint_qd, state)
    initial_body_q = state.body_q.numpy().copy()

    joint_q = wp.array(np.array([0.35, -0.45]), dtype=float, device=device)
    newton.eval_fk(
        model,
        joint_q,
        joint_qd,
        state,
        body_flag_filter=int(newton.BodyFlags.DYNAMIC),
    )
    body_q = state.body_q.numpy()

    # dynamic body should be updated by FK
    dynamic_quat_dot = abs(float(np.dot(initial_body_q[0, 3:7], body_q[0, 3:7])))
    test.assertLess(dynamic_quat_dot, 0.999)

    # kinematic body should be restored to the previous state
    assert_np_equal(initial_body_q[1], body_q[1], tol=1e-8)


def test_ik_body_flag_filter_dynamic_only(test, device):
    model = _build_dynamic_and_kinematic_single_joint_model(device)
    state = model.state()

    joint_q = wp.array(np.array([0.2, -0.4]), dtype=float, device=device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    newton.eval_fk(model, joint_q, joint_qd, state)

    recovered_q = wp.full(model.joint_coord_count, 7.0, dtype=float, device=device)
    recovered_qd = wp.full(model.joint_dof_count, 3.0, dtype=float, device=device)
    newton.eval_ik(
        model,
        state,
        recovered_q,
        recovered_qd,
        body_flag_filter=int(newton.BodyFlags.DYNAMIC),
    )

    recovered_q_np = recovered_q.numpy()
    recovered_qd_np = recovered_qd.numpy()

    # dynamic articulation is updated
    test.assertAlmostEqual(float(recovered_q_np[0]), 0.2, places=5)
    test.assertAlmostEqual(float(recovered_qd_np[0]), 0.0, places=6)

    # kinematic articulation is preserved from the previous values
    test.assertAlmostEqual(float(recovered_q_np[1]), 7.0, places=6)
    test.assertAlmostEqual(float(recovered_qd_np[1]), 3.0, places=6)


devices = get_test_devices()


class TestSimKinematics(unittest.TestCase):
    pass


add_function_test(TestSimKinematics, "test_fk_ik", test_fk_ik, devices=devices)
add_function_test(
    TestSimKinematics, "test_fk_ik_with_analytical_solution", test_fk_ik_with_analytical_solution, devices=devices
)
add_function_test(
    TestSimKinematics,
    "test_fk_descendant_linear_velocity_matches_finite_difference",
    test_fk_descendant_linear_velocity_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestSimKinematics,
    "test_fk_prismatic_descendant_linear_velocity_matches_finite_difference",
    test_fk_prismatic_descendant_linear_velocity_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestSimKinematics,
    "test_fk_free_root_descendant_linear_velocity_matches_finite_difference",
    test_fk_free_distance_root_descendant_linear_velocity_matches_finite_difference,
    devices=devices,
    joint_type=newton.JointType.FREE,
)
add_function_test(
    TestSimKinematics,
    "test_fk_distance_root_descendant_linear_velocity_matches_finite_difference",
    test_fk_free_distance_root_descendant_linear_velocity_matches_finite_difference,
    devices=devices,
    joint_type=newton.JointType.DISTANCE,
)
add_function_test(
    TestSimKinematics,
    "test_ik_prismatic_descendant_recovers_joint_state",
    test_ik_prismatic_descendant_recovers_joint_state,
    devices=devices,
)
add_function_test(
    TestSimKinematics,
    "test_ik_free_descendant_recovers_joint_state",
    test_ik_free_distance_descendant_recovers_joint_state,
    devices=devices,
    joint_type=newton.JointType.FREE,
)
add_function_test(
    TestSimKinematics,
    "test_ik_distance_descendant_recovers_joint_state",
    test_ik_free_distance_descendant_recovers_joint_state,
    devices=devices,
    joint_type=newton.JointType.DISTANCE,
)
add_function_test(
    TestSimKinematics,
    "test_solver_fk_prismatic_descendant_linear_velocity_matches_finite_difference",
    test_solver_fk_prismatic_descendant_linear_velocity_matches_finite_difference,
    devices=devices,
)
add_function_test(TestSimKinematics, "test_fk_with_indices", test_fk_with_indices, devices=devices)
add_function_test(TestSimKinematics, "test_ik_with_indices", test_ik_with_indices, devices=devices)
add_function_test(TestSimKinematics, "test_fk_error_mask_and_indices", test_fk_error_mask_and_indices, devices=devices)
add_function_test(TestSimKinematics, "test_isaac_lab_use_case", test_isaac_lab_use_case, devices=devices)
add_function_test(TestSimKinematics, "test_bounds_checking", test_bounds_checking, devices=devices)
add_function_test(TestSimKinematics, "test_ik_with_mask", test_ik_with_mask, devices=devices)
add_function_test(TestSimKinematics, "test_ik_error_mask_and_indices", test_ik_error_mask_and_indices, devices=devices)
add_function_test(
    TestSimKinematics,
    "test_fk_body_flag_filter_dynamic_only",
    test_fk_body_flag_filter_dynamic_only,
    devices=devices,
)
add_function_test(
    TestSimKinematics,
    "test_ik_body_flag_filter_dynamic_only",
    test_ik_body_flag_filter_dynamic_only,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
