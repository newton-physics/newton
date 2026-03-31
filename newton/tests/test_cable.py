# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


# -----------------------------------------------------------------------------
# Assert helpers
# -----------------------------------------------------------------------------


def _assert_bodies_above_ground(
    test: unittest.TestCase,
    body_q: np.ndarray,
    body_ids: list[int],
    context: str,
    margin: float = 1.0e-4,
) -> None:
    """Assert a set of bodies are not below the z=0 ground plane (within margin)."""
    z_pos = body_q[body_ids, 2]
    z_min = z_pos.min()
    test.assertGreaterEqual(
        z_min,
        -margin,
        msg=f"{context}: body below ground: z_min={z_min:.6f} < {-margin:.6f}",
    )


def _assert_capsule_attachments(
    test: unittest.TestCase,
    body_q: np.ndarray,
    body_ids: list[int],
    context: str,
    segment_length: float,
    tol_ratio: float = 0.05,
) -> None:
    """Assert that adjacent capsules remain attached within tolerance.

    Approximates the parent capsule end and child capsule start in world space and
    checks that their separation is small relative to the rest capsule length.
    """
    tol = tol_ratio * segment_length
    for i in range(len(body_ids) - 1):
        idx_p = body_ids[i]
        idx_c = body_ids[i + 1]

        p_pos = body_q[idx_p, :3]
        c_pos = body_q[idx_c, :3]

        dir_vec = c_pos - p_pos
        seg_len = np.linalg.norm(dir_vec)
        if seg_len > 1.0e-6:
            dir_hat = dir_vec / seg_len
        else:
            dir_hat = np.array([1.0, 0.0, 0.0], dtype=float)

        parent_end = p_pos + dir_hat * segment_length
        child_start = c_pos
        gap = np.linalg.norm(parent_end - child_start)

        test.assertLessEqual(
            gap,
            tol,
            msg=f"{context}: capsule attachment gap too large at segment {i} (gap={gap:.6g}, tol={tol:.6g})",
        )


def _assert_surface_attachment(
    test: unittest.TestCase,
    body_q: np.ndarray,
    anchor_body: int,
    child_body: int,
    context: str,
    parent_anchor_local: wp.vec3,
    tol: float = 1.0e-3,
) -> None:
    """Assert that the child body origin lies on the anchor-frame attachment point.

    Intended attach point (world):
        x_expected = x_anchor + R_anchor * parent_anchor_local
    """
    with wp.ScopedDevice("cpu"):
        x_anchor = wp.vec3(body_q[anchor_body][0], body_q[anchor_body][1], body_q[anchor_body][2])
        q_anchor = wp.quat(
            body_q[anchor_body][3], body_q[anchor_body][4], body_q[anchor_body][5], body_q[anchor_body][6]
        )
        x_expected = x_anchor + wp.quat_rotate(q_anchor, parent_anchor_local)

        x_child = wp.vec3(body_q[child_body][0], body_q[child_body][1], body_q[child_body][2])
        err = float(wp.length(x_child - x_expected))
        test.assertLess(
            err,
            tol,
            msg=f"{context}: surface-attachment error is {err:.6e} (tol={tol:.1e})",
        )


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------


@wp.kernel
def _set_kinematic_body_pose(
    body_id: wp.int32,
    pose: wp.transform,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_q[body_id] = pose
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _drive_gripper_boxes_kernel(
    ramp_time: float,
    t: float,
    body_ids: wp.array[wp.int32],
    signs: wp.array[wp.float32],
    anchor_p: wp.vec3,
    anchor_q: wp.quat,
    seg_half_len: float,
    target_offset_mag: float,
    initial_offset_mag: float,
    pull_start_time: float,
    pull_ramp_time: float,
    pull_distance: float,
    body_q: wp.array[wp.transform],
):
    """Kinematically move two gripper boxes toward an anchor frame, then pull along anchor +Z.

    Used by `test_cable_kinematic_gripper_picks_capsule` to validate that **friction with kinematic
    bodies** transfers motion to a dynamic payload (i.e., the payload can be lifted without gravity).

    Notes:
        - This kernel is purely a scripted pose driver (no joints/constraints involved).
        - It writes only `body_q` (poses).
    """
    tid = wp.tid()
    b = body_ids[tid]
    sgn = signs[tid]

    rot = anchor_q
    center = anchor_p + wp.quat_rotate(rot, wp.vec3(0.0, 0.0, seg_half_len))

    t = wp.float32(t)
    pull_end_time = wp.float32(pull_start_time + pull_ramp_time)
    t_eff = wp.min(t, pull_end_time)

    # Linear close-in: ramp from initial_offset_mag -> target_offset_mag over ramp_time.
    u = wp.clamp(t_eff / wp.float32(ramp_time), 0.0, 1.0)
    offset_mag = (1.0 - u) * initial_offset_mag + u * target_offset_mag

    # Linear lift: ramp from 0 -> pull_distance over pull_ramp_time starting at pull_start_time.
    tp = wp.clamp((t_eff - wp.float32(pull_start_time)) / wp.float32(pull_ramp_time), 0.0, 1.0)
    pull = wp.float32(pull_distance) * tp

    pull_dir = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    local_off = wp.vec3(0.0, sgn * offset_mag, 0.0)
    pos = center + pull_dir * pull + wp.quat_rotate(rot, local_off)

    body_q[b] = wp.transform(pos, rot)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _make_straight_cable_along_x(num_elements: int, segment_length: float, z_height: float):
    """Create points/quats for `ModelBuilder.add_rod()` with a straight cable along +X.

    Notes:
        - Points are centered about x=0 (first point is at x=-0.5*cable_length).
        - Capsules have local +Z as their axis; quaternions rotate local +Z to world +X.
    """
    length = float(num_elements * segment_length)
    start = wp.vec3(-0.5 * length, 0.0, float(z_height))
    return newton.utils.create_straight_cable_points_and_quaternions(
        start=start,
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=length,
        num_segments=int(num_elements),
    )


def _make_straight_cable_along_y(num_elements: int, segment_length: float, z_height: float):
    """Create points/quats for `ModelBuilder.add_rod()` with a straight cable along +Y.

    Notes:
        - Points are centered about y=0 (first point is at y=-0.5*cable_length).
        - Capsules have local +Z as their axis; quaternions rotate local +Z to world +Y.
    """
    length = float(num_elements * segment_length)
    start = wp.vec3(0.0, -0.5 * length, float(z_height))
    return newton.utils.create_straight_cable_points_and_quaternions(
        start=start,
        direction=wp.vec3(0.0, 1.0, 0.0),
        length=length,
        num_segments=int(num_elements),
    )


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------


def _build_cable_chain(
    device,
    num_links: int = 6,
    pin_first: bool = True,
    bend_stiffness: float = 1.0e1,
    bend_damping: float = 1.0e-2,
    segment_length: float = 0.2,
):
    """Build a simple cable.

    Args:
        device: Warp device to build the model on.
        num_links: Number of rod elements (segments) in the cable.
        pin_first: If True, make the first rod body kinematic (anchor); if False, leave all dynamic.
        bend_stiffness: Cable bend stiffness passed to :func:`add_rod`.
        bend_damping: Cable bend damping passed to :func:`add_rod`.
        segment_length: Rest length of each capsule segment.
    """
    builder = newton.ModelBuilder()

    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Geometry: straight cable along +X, centered around the origin
    num_elements = num_links
    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=3.0)

    # Create a rod-based cable
    rod_bodies, _rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=0.05,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        label="test_cable_chain",
    )

    if pin_first and len(rod_bodies) > 0:
        first_body = rod_bodies[0]
        builder.body_flags[first_body] = int(newton.BodyFlags.KINEMATIC)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    return model, state0, state1, control, rod_bodies


def _build_cable_loop(device, num_links: int = 6):
    """Build a closed (circular) cable loop using the rod API.

    This uses the same material style as the open chain, but with ``closed=True``
    so the last segment connects back to the first.
    """
    builder = newton.ModelBuilder()

    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Geometry: points on a circle in the X-Y plane at fixed height
    num_elements = num_links
    radius = 1.0
    z_height = 3.0

    points = []
    for i in range(num_elements + 1):
        # For a closed loop we wrap the last point back to the first
        angle = 2.0 * wp.pi * (i / num_elements)
        x = radius * wp.cos(angle)
        y = radius * wp.sin(angle)
        points.append(wp.vec3(x, y, z_height))

    edge_q = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=0.0)

    _rod_bodies, _rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=0.05,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        closed=True,
        label="test_cable_loop",
    )

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    return model, state0, state1, control


# -----------------------------------------------------------------------------
# Compute helpers
# -----------------------------------------------------------------------------


def _numpy_to_transform(arr):
    """Convert a [p(3), q(4)] numpy row (xyzw quaternion) to wp.transform."""
    return wp.transform(wp.vec3(arr[0], arr[1], arr[2]), wp.quat(arr[3], arr[4], arr[5], arr[6]))


def _get_joint_rest_relative_rotation(model: newton.Model, joint_id: int) -> wp.quat:
    """Return q_rel_rest = quat_inverse(q_wp_rest) * q_wc_rest for the joint's rest configuration."""
    with wp.ScopedDevice("cpu"):
        jp = model.joint_parent.numpy()[joint_id].item()
        jc = model.joint_child.numpy()[joint_id].item()
        X_pj = _numpy_to_transform(model.joint_X_p.numpy()[joint_id])
        X_cj = _numpy_to_transform(model.joint_X_c.numpy()[joint_id])
        bq_rest = model.body_q.numpy()
        if jp >= 0:
            X_wp_rest = _numpy_to_transform(bq_rest[jp]) * X_pj
        else:
            X_wp_rest = X_pj
        X_wc_rest = _numpy_to_transform(bq_rest[jc]) * X_cj
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        return wp.normalize(wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest))


def _get_joint_world_frames(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[wp.transform, wp.transform]:
    """Compute world-space joint frames (X_wp, X_wc) for a given joint."""
    with wp.ScopedDevice("cpu"):
        jp = model.joint_parent.numpy()[joint_id].item()
        jc = model.joint_child.numpy()[joint_id].item()
        # joint_X_p / joint_X_c are stored as [p(3), q(4)] with q in xyzw order
        X_p = model.joint_X_p.numpy()[joint_id]
        X_c = model.joint_X_c.numpy()[joint_id]
        X_pj = wp.transform(wp.vec3(X_p[0], X_p[1], X_p[2]), wp.quat(X_p[3], X_p[4], X_p[5], X_p[6]))
        X_cj = wp.transform(wp.vec3(X_c[0], X_c[1], X_c[2]), wp.quat(X_c[3], X_c[4], X_c[5], X_c[6]))

        bq = body_q.to("cpu").numpy()

        # World joint (parent=-1): parent frame is identity.
        if jp >= 0:
            q_p = bq[jp]
            T_p = wp.transform(wp.vec3(q_p[0], q_p[1], q_p[2]), wp.quat(q_p[3], q_p[4], q_p[5], q_p[6]))
        else:
            T_p = wp.transform(wp.vec3(0.0), wp.quat_identity())

        q_c = bq[jc]
        T_c = wp.transform(wp.vec3(q_c[0], q_c[1], q_c[2]), wp.quat(q_c[3], q_c[4], q_c[5], q_c[6]))
        return T_p * X_pj, T_c * X_cj


def _get_joint_axis_local(model: newton.Model, joint_id: int) -> wp.vec3:
    """Return the normalized joint axis in the parent joint frame."""
    with wp.ScopedDevice("cpu"):
        qd_start = model.joint_qd_start.numpy()[joint_id].item()
        axis_np = model.joint_axis.numpy()[qd_start]
        return wp.normalize(wp.vec3(axis_np[0], axis_np[1], axis_np[2]))


def _compute_ball_joint_anchor_error(model: newton.Model, body_q: wp.array, joint_id: int) -> float:
    """Compute BALL joint anchor coincidence error |x_c - x_p| [m]."""
    X_wp, X_wc = _get_joint_world_frames(model, body_q, joint_id)
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    return float(wp.length(x_c - x_p))


def _compute_fixed_joint_frame_error(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[float, float]:
    """Compute FIXED joint world-space frame error (CPU floats).

    Returns:
        (pos_err, ang_err)

        - pos_err: anchor coincidence error |x_c - x_p| [m].
        - ang_err: rotation angle relative to the rest configuration [rad].
          Measures how much the joint has deviated from its initial
          rest-relative orientation, not the absolute angle between frames.
    """
    X_wp, X_wc = _get_joint_world_frames(model, body_q, joint_id)

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    pos_err = float(wp.length(x_c - x_p))

    # Current relative rotation
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))

    # Measure deviation from rest: q_err = q_rel * q_rel_rest^{-1}
    q_rel_rest = _get_joint_rest_relative_rotation(model, joint_id)
    q_err = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rel_rest)))
    ang_err = float(2.0 * wp.acos(wp.clamp(wp.abs(q_err[3]), 0.0, 1.0)))

    return pos_err, ang_err


def _compute_revolute_joint_error(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[float, float, float]:
    """Compute REVOLUTE joint world-space error (CPU floats).

    Returns:
        (pos_err, ang_perp_err, rot_along_free)

        - pos_err: anchor coincidence error |x_c - x_p| [m].
        - ang_perp_err: rotation angle perpendicular to the joint axis relative to rest [rad].
        - rot_along_free: rotation about the free (joint) axis relative to rest [rad].
    """
    X_wp, X_wc = _get_joint_world_frames(model, body_q, joint_id)
    a = _get_joint_axis_local(model, joint_id)

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    pos_err = float(wp.length(x_c - x_p))

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))

    # Measure relative to rest configuration
    q_rel_rest = _get_joint_rest_relative_rotation(model, joint_id)
    q_err = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rel_rest)))
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])

    axis_angle, angle = wp.quat_to_axis_angle(q_err)
    rot_vec = axis_angle * angle
    rot_perp = rot_vec - wp.dot(rot_vec, a) * a
    ang_perp_err = float(wp.length(rot_perp))
    rot_along_free = abs(float(wp.dot(rot_vec, a)))

    return pos_err, ang_perp_err, rot_along_free


def _compute_prismatic_joint_error(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[float, float, float]:
    """Compute PRISMATIC joint world-space error (CPU floats).

    Returns:
        (pos_perp_err, ang_err, c_along)

        - pos_perp_err: position error perpendicular to the joint axis [m].
        - ang_err: rotation angle relative to the rest configuration [rad].
        - c_along: signed displacement along the free (joint) axis [m].
    """
    X_wp, X_wc = _get_joint_world_frames(model, body_q, joint_id)
    a_local = _get_joint_axis_local(model, joint_id)

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    C = x_c - x_p

    q_wp = wp.transform_get_rotation(X_wp)
    a_world = wp.normalize(wp.quat_rotate(q_wp, a_local))
    C_perp = C - wp.dot(C, a_world) * a_world
    pos_perp_err = float(wp.length(C_perp))
    c_along = float(wp.dot(C, a_world))

    q_wc = wp.transform_get_rotation(X_wc)
    q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))

    # Rest relative rotation
    q_rel_rest = _get_joint_rest_relative_rotation(model, joint_id)
    q_err = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rel_rest)))
    ang_err = float(2.0 * wp.acos(wp.clamp(wp.abs(q_err[3]), 0.0, 1.0)))

    return pos_perp_err, ang_err, c_along


def _compute_d6_joint_error(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[float, float, float, float]:
    """Compute D6 joint world-space error (CPU floats).

    Assumes the D6 joint has exactly 1 linear axis (first DOF) and 1 angular axis (second DOF).

    Returns:
        (pos_perp_err, ang_perp_err, d_along, rot_along_free)

        - pos_perp_err: position error perpendicular to the free linear axis [m].
        - ang_perp_err: rotation angle perpendicular to the free angular axis relative to rest [rad].
        - d_along: signed displacement along the free linear axis [m].
        - rot_along_free: rotation about the free angular axis relative to rest [rad].
    """
    X_wp, X_wc = _get_joint_world_frames(model, body_q, joint_id)

    with wp.ScopedDevice("cpu"):
        qd_start = model.joint_qd_start.numpy()[joint_id].item()
        lin_axis_np = model.joint_axis.numpy()[qd_start]
        ang_axis_np = model.joint_axis.numpy()[qd_start + 1]
        lin_a = wp.normalize(wp.vec3(lin_axis_np[0], lin_axis_np[1], lin_axis_np[2]))
        ang_a = wp.normalize(wp.vec3(ang_axis_np[0], ang_axis_np[1], ang_axis_np[2]))

    # --- Linear: perpendicular position error + along-axis displacement ---
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    C = x_c - x_p

    q_wp = wp.transform_get_rotation(X_wp)
    lin_a_world = wp.normalize(wp.quat_rotate(q_wp, lin_a))
    d_along = float(wp.dot(C, lin_a_world))
    C_perp = C - d_along * lin_a_world
    pos_perp_err = float(wp.length(C_perp))

    # --- Angular: perpendicular angle error + free-axis rotation ---
    q_wc = wp.transform_get_rotation(X_wc)
    q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))

    # Measure relative to rest configuration
    q_rel_rest = _get_joint_rest_relative_rotation(model, joint_id)
    q_err = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rel_rest)))
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])

    axis_angle, angle = wp.quat_to_axis_angle(q_err)
    rot_vec = axis_angle * angle
    rot_perp = rot_vec - wp.dot(rot_vec, ang_a) * ang_a
    ang_perp_err = float(wp.length(rot_perp))
    rot_along_free = abs(float(wp.dot(rot_vec, ang_a)))

    return pos_perp_err, ang_perp_err, d_along, rot_along_free


# -----------------------------------------------------------------------------
# Test implementations
# -----------------------------------------------------------------------------


def _cable_chain_connectivity_impl(test: unittest.TestCase, device):
    """Cable VBD: verify that cable joints form a connected chain with expected types."""
    model, _state0, _state1, _control, _rod_bodies = _build_cable_chain(device, num_links=4)

    jt = model.joint_type.numpy()
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()

    # Ensure we have at least one cable joint and that the chain is contiguous
    cable_indices = np.where(jt == newton.JointType.CABLE)[0]
    test.assertGreater(len(cable_indices), 0)

    # Extract parent/child arrays for cable joints only
    cable_parents = parent[cable_indices]
    cable_children = child[cable_indices]

    # Each cable joint should connect valid, in-range bodies
    test.assertTrue(np.all(cable_parents >= 0))
    test.assertTrue(np.all(cable_children >= 0))
    test.assertTrue(np.all(cable_parents < model.body_count))
    test.assertTrue(np.all(cable_children < model.body_count))

    # No duplicate (parent, child) pairs
    pairs_list = list(zip(cable_parents.tolist(), cable_children.tolist(), strict=True))
    cable_pairs = set(pairs_list)
    test.assertEqual(len(cable_pairs), len(pairs_list))

    # Simple sequential connectivity check: in the current joint order,
    # the child of joint i should be the parent of joint i+1.
    if len(cable_indices) > 1:
        for i in range(len(cable_indices) - 1):
            idx0 = cable_indices[i]
            idx1 = cable_indices[i + 1]
            test.assertEqual(
                child[idx0],
                parent[idx1],
                msg=f"Expected child of joint {idx0} to match parent of joint {idx1}",
            )


def _cable_loop_connectivity_impl(test: unittest.TestCase, device):
    """Cable VBD: verify connectivity for a closed (circular) cable loop."""
    model, _state0, _state1, _control = _build_cable_loop(device, num_links=4)

    jt = model.joint_type.numpy()
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()

    cable_indices = np.where(jt == newton.JointType.CABLE)[0]
    test.assertGreater(len(cable_indices), 0)

    cable_parents = parent[cable_indices]
    cable_children = child[cable_indices]

    # Valid indices
    test.assertTrue(np.all(cable_parents >= 0))
    test.assertTrue(np.all(cable_children >= 0))
    test.assertTrue(np.all(cable_parents < model.body_count))
    test.assertTrue(np.all(cable_children < model.body_count))

    # No duplicate (parent, child) pairs
    cable_pairs = list(zip(cable_parents.tolist(), cable_children.tolist(), strict=True))
    test.assertEqual(len(set(cable_pairs)), len(cable_pairs))

    # Sequential loop connectivity: child[i] == parent[i+1], and last child == first parent
    n = len(cable_indices)
    if n > 1:
        for i in range(n):
            idx0 = cable_indices[i]
            idx1 = cable_indices[(i + 1) % n]
            test.assertEqual(
                child[idx0],
                parent[idx1],
                msg=f"Expected child of joint {idx0} to match parent of joint {idx1} in closed loop",
            )


def _cable_bend_stiffness_impl(test: unittest.TestCase, device):
    """Cable VBD: bend stiffness sweep should have a noticeable effect on tip position."""
    # From soft to stiff. Build multiple cables in one model.
    bend_values = [1.0e1, 1.0e2, 1.0e3]
    segment_length = 0.2
    num_links = 10

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Place cables far apart along Y so they don't interact.
    y_offsets = [-5.0, 0.0, 5.0]
    tip_bodies: list[int] = []
    all_rod_bodies: list[list[int]] = []

    for k, y0 in zip(bend_values, y_offsets, strict=True):
        points, edge_q = _make_straight_cable_along_x(num_links, segment_length, z_height=3.0)
        points = [wp.vec3(p[0], p[1] + y0, p[2]) for p in points]

        rod_bodies, _rod_joints = builder.add_rod(
            positions=points,
            quaternions=edge_q,
            radius=0.05,
            bend_stiffness=k,
            bend_damping=1.0e1,
            stretch_stiffness=1.0e6,
            stretch_damping=1.0e-2,
            label=f"bend_stiffness_{k:.0e}",
        )

        # Pin the first body of each cable.
        first_body = rod_bodies[0]
        builder.body_flags[first_body] = int(newton.BodyFlags.KINEMATIC)

        all_rod_bodies.append(rod_bodies)
        tip_bodies.append(rod_bodies[-1])

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    # Run for a short duration to let bending respond to gravity
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    final_q = state0.body_q.numpy()
    tip_heights = np.array([final_q[tip_body, 2] for tip_body in tip_bodies], dtype=float)

    # Check capsule attachments for each dynamic configuration
    for k, rod_bodies in zip(bend_values, all_rod_bodies, strict=True):
        _assert_capsule_attachments(
            test,
            body_q=final_q,
            body_ids=rod_bodies,
            segment_length=segment_length,
            context=f"Bend stiffness {k}",
        )

    # Check that stiffer cables have higher tip positions (less sagging under gravity)
    # Expect monotonic increase: tip_heights[0] < tip_heights[1] < tip_heights[2]
    for i in range(len(tip_heights) - 1):
        test.assertLess(
            tip_heights[i],
            tip_heights[i + 1],
            msg=(
                f"Stiffer cable should have higher tip (less sag): "
                f"k={bend_values[i]:.1e} -> z={tip_heights[i]:.4f}, "
                f"k={bend_values[i + 1]:.1e} -> z={tip_heights[i + 1]:.4f}"
            ),
        )

    # Additionally check that the variation is noticeable (not just numerical noise)
    test.assertGreater(
        tip_heights[-1] - tip_heights[0],
        1.0e-3,
        msg=f"Tip height variation too small across stiffness sweep: {tip_heights}",
    )


def _cable_sagging_and_stability_impl(test: unittest.TestCase, device):
    """Cable VBD: pinned chain should sag under gravity while remaining numerically stable."""
    segment_length = 0.2
    model, state0, state1, control, _rod_bodies = _build_cable_chain(device, num_links=6, segment_length=segment_length)
    contacts = model.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=10)
    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    # Record initial positions
    initial_q = state0.body_q.numpy().copy()
    z_initial = initial_q[:, 2]

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    final_q = state0.body_q.numpy()
    z_final = final_q[:, 2]

    # At least one cable body should move downward
    test.assertTrue((z_final < z_initial).any())

    # Positions should remain within a band relative to initial height and cable length
    z0 = z_initial.min()
    x_initial = initial_q[:, 0]
    cable_length = x_initial.max() - x_initial.min()
    lower_bound = z0 - 2.0 * cable_length
    upper_bound = z0 + 2.0 * cable_length

    test.assertTrue(np.all(z_final > lower_bound))
    test.assertTrue(np.all(z_final < upper_bound))


def _cable_twist_response_impl(test: unittest.TestCase, device):
    """Cable VBD: twisting the anchored capsule should induce rotation in the child while preserving attachment."""
    segment_length = 0.2

    # Two-link cable in an orthogonal "L" configuration:
    #  - First segment along +X
    #  - Second segment along +Y from the end of the first
    # This isolates twist response when rotating the first (anchored) capsule about its local axis.
    builder = newton.ModelBuilder()

    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    z_height = 3.0
    p0 = wp.vec3(0.0, 0.0, z_height)
    p1 = wp.vec3(segment_length, 0.0, z_height)
    p2 = wp.vec3(segment_length, segment_length, z_height)

    positions = [p0, p1, p2]

    local_z = wp.vec3(0.0, 0.0, 1.0)
    dir0 = wp.normalize(p1 - p0)  # +X
    dir1 = wp.normalize(p2 - p1)  # +Y

    q0 = wp.quat_between_vectors(local_z, dir0)
    q1 = wp.quat_between_vectors(local_z, dir1)
    quats = [q0, q1]

    rod_bodies, _rod_joints = builder.add_rod(
        positions=positions,
        quaternions=quats,
        radius=0.05,
        bend_stiffness=1.0e4,
        bend_damping=0.0,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        label="twist_chain_orthogonal",
    )

    # Pin the first body (anchored capsule)
    first_body = rod_bodies[0]
    builder.body_flags[first_body] = int(newton.BodyFlags.KINEMATIC)

    builder.color()
    model = builder.finalize(device=device)
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10)

    # Disable gravity to isolate twist response
    model.set_gravity((0.0, 0.0, 0.0))

    # Record initial orientation of the free (child) body
    child_body = rod_bodies[-1]
    q_initial = state0.body_q.numpy().copy()
    # Quaternion components in the transform are stored as [qx, qy, qz, qw]
    q_child_initial = q_initial[child_body, 3:].copy()

    # Apply a 180-degree twist about the local cable axis to the parent body by composing
    # the twist with the existing parent rotation.
    parent_body = rod_bodies[0]
    q_parent_initial = q_initial[parent_body, 3:].copy()
    q_parent_twist = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    # Compose world-space twist with initial orientation
    q_parent_new = wp.mul(q_parent_twist, wp.quat(*q_parent_initial))
    q_parent_new_arr = np.array([q_parent_new[0], q_parent_new[1], q_parent_new[2], q_parent_new[3]])
    q_initial[parent_body, 3:] = q_parent_new_arr

    # Write back to the device array (CPU or CUDA) explicitly
    state0.body_q = wp.array(q_initial, dtype=wp.transform, device=device)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    # Run a short simulation to let twist propagate
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    final_q = state0.body_q.numpy()

    # Check capsule attachments remain good
    _assert_capsule_attachments(
        test, body_q=final_q, body_ids=rod_bodies, segment_length=segment_length, context="Twist"
    )

    # Check that the child orientation has changed significantly due to twist
    q_child_final = final_q[child_body, 3:]

    # Quaternion dot product magnitude indicates orientation similarity (1 => identical up to sign)
    dot = abs(np.dot(q_child_initial, q_child_final))
    test.assertLess(
        dot,
        0.999,
        msg=f"Twist: child orientation changed too little (|dot|={dot}); expected noticeable rotation from twist.",
    )

    # Also check a specific geometric response: in the orthogonal "L" configuration,
    # twisting 180 degrees about the +X axis should reflect the free capsule across the X-Z plane:
    # its Y coordinate should change sign while X and Z remain approximately the same.

    # We check the tip of the capsule, because the body origin is at the pivot (which doesn't move).
    def get_tip_pos(body_idx, q_all):
        p = q_all[body_idx, :3]
        q = q_all[body_idx, 3:]  # x, y, z, w
        rot = wp.quat(q[0], q[1], q[2], q[3])
        v = wp.vec3(0.0, 0.0, segment_length)
        v_rot = wp.quat_rotate(rot, v)
        return np.array([p[0] + v_rot[0], p[1] + v_rot[1], p[2] + v_rot[2]])

    tip_initial = get_tip_pos(child_body, q_initial)
    tip_final = get_tip_pos(child_body, final_q)

    tol = 0.1 * segment_length

    # X and Z should stay close to their original values
    tip_x0, tip_y0, tip_z0 = tip_initial
    tip_x1, tip_y1, tip_z1 = tip_final
    test.assertAlmostEqual(
        tip_x1,
        tip_x0,
        delta=tol,
        msg=f"Twist: expected child tip x to stay near {tip_x0}, got {tip_x1}",
    )
    test.assertAlmostEqual(
        tip_z1,
        tip_z0,
        delta=tol,
        msg=f"Twist: expected child tip z to stay near {tip_z0}, got {tip_z1}",
    )

    # Y should approximately flip sign (reflect across the X-Z plane)
    # Initial tip Y should be approx segment_length (0.2)
    # We check if the sign is flipped, but allow for some deviation in magnitude
    # because the twist might not be perfectly 180 degrees or there might be some energy loss/damping
    test.assertTrue(
        tip_y1 * tip_y0 < 0,
        msg=f"Twist: expected child tip y to flip sign from {tip_y0}, got {tip_y1}",
    )
    test.assertAlmostEqual(
        tip_y1,
        -tip_y0,
        delta=tol,
        msg=(f"Twist: expected child tip y magnitude to be similar from {abs(tip_y0)}, got {abs(tip_y1)}"),
    )


def _two_layer_cable_pile_collision_impl(test: unittest.TestCase, device):
    """Cable VBD: two-layer straight cable pile should form two vertical layers.

    Creates a 2x2 cable pile (2 cables per layer, 2 layers) forming a sharp/cross
    pattern from top view:
      - Bottom layer: 2 cables along +X axis
      - Top layer: 2 cables along +Y axis
      - All cables are straight (no waviness)
      - High bend stiffness (1.0e3) to maintain straightness

    After settling under gravity and contact, bodies should cluster into two
    vertical bands:
      - bottom layer: between ground (z=0) and one cable width,
      - top layer: between one and two cable widths,
    up to a small margin for numerical tolerance and contact compliance.
    """
    builder = newton.ModelBuilder()

    # Contact material (stiff contacts, noticeable friction)
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e-1
    builder.default_shape_cfg.mu = 1.0

    # Cable geometric parameters
    num_elements = 30
    segment_length = 0.05
    cable_length = num_elements * segment_length
    cable_radius = 0.012
    cable_width = 2.0 * cable_radius

    # Vertical spacing between the two layers (start positions; they will fall)
    layer_gap = 2.0 * cable_radius  # Increased gap for clearer separation
    base_height = 0.08  # Lower starting height to stack from ground

    # Horizontal spacing of cables within each layer
    # Cables are centered at origin (0, 0) with symmetric offset
    lane_spacing = 10.0 * cable_radius  # Increased spacing for clearer separation

    # High bend stiffness to keep cables nearly straight
    bend_stiffness = 1.0e3

    # Ground plane at z=0 (Z-up)
    builder.add_ground_plane()

    # Build two layers: bottom layer along X, top layer along Y
    # Both layers centered at origin (0, 0) in horizontal plane
    cable_bodies: list[int] = []
    for layer in range(2):
        orient = "x" if layer == 0 else "y"
        z0 = base_height + layer * layer_gap

        for lane in range(2):
            # Symmetric offset: lane 0 -> -0.5*spacing, lane 1 -> +0.5*spacing
            # This centers both layers at the same (x,y) = (0,0) position
            offset = (lane - 0.5) * lane_spacing

            # Build straight cable geometry manually
            points = []
            start_coord = -0.5 * cable_length

            for i in range(num_elements + 1):
                coord = start_coord + i * segment_length
                if orient == "x":
                    # Cable along X axis, offset in Y
                    points.append(wp.vec3(coord, offset, z0))
                else:
                    # Cable along Y axis, offset in X
                    points.append(wp.vec3(offset, coord, z0))

            # Create quaternions for capsule orientation using quat_between_vectors
            # Capsule internal axis is +Z; rotate to align with cable direction
            local_axis = wp.vec3(0.0, 0.0, 1.0)
            if orient == "x":
                cable_direction = wp.vec3(1.0, 0.0, 0.0)
            else:
                cable_direction = wp.vec3(0.0, 1.0, 0.0)

            rot = wp.quat_between_vectors(local_axis, cable_direction)
            edge_q = [rot] * num_elements

            rod_bodies, _rod_joints = builder.add_rod(
                positions=points,
                quaternions=edge_q,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-1,
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                label=f"pile_l{layer}_{lane}",
            )
            cable_bodies.extend(rod_bodies)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10, friction_epsilon=0.1)
    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Let the pile settle under gravity and contact
    num_steps = 20
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    body_q = state0.body_q.numpy()
    positions = body_q[:, :3]
    z_positions = positions[:, 2]

    # Basic sanity checks
    test.assertTrue(np.isfinite(positions).all(), "Non-finite positions detected in cable pile")
    _assert_bodies_above_ground(
        test,
        body_q=body_q,
        body_ids=cable_bodies,
        margin=0.25 * cable_width,
        context="Cable pile",
    )

    # Define vertical bands with a small margin for soft contact tolerance
    # Increased margin to account for contact compression and stiff cable deformation
    margin = 0.1 * cable_width

    # Bottom layer should live between ground and one cable width (+/- margin)
    bottom_band = (z_positions >= -margin) & (z_positions <= cable_width + margin)

    # Second layer between one and two cable widths (+/- margin)
    top_band = (z_positions >= cable_width - margin) & (z_positions <= 2.0 * cable_width + margin)

    # All bodies should fall within one of the two bands
    in_bands = bottom_band | top_band
    test.assertTrue(
        np.all(in_bands),
        msg=(
            "Some cable bodies lie outside the expected two-layer vertical bands: "
            f"min_z={z_positions.min():.4f}, max_z={z_positions.max():.4f}, "
            f"cable_width={cable_width:.4f}, expected in [0, {2.0 * cable_width + margin:.4f}] "
            f"with band margin {margin:.4f}."
        ),
    )

    # Ensure we actually formed two distinct layers
    num_bottom = int(np.sum(bottom_band))
    num_top = int(np.sum(top_band))

    test.assertGreater(
        num_bottom,
        0,
        msg=f"No bodies found in the bottom cable layer band [0, {cable_width:.4f}].",
    )
    test.assertGreater(
        num_top,
        0,
        msg=f"No bodies found in the top cable layer band [{cable_width:.4f}, {2.0 * cable_width:.4f}].",
    )

    # Verify the layers are reasonably balanced (not all bodies in one layer)
    total_bodies = len(z_positions)
    test.assertGreater(
        num_bottom,
        0.1 * total_bodies,
        msg=f"Bottom layer has too few bodies: {num_bottom}/{total_bodies} (< 10%)",
    )
    test.assertGreater(
        num_top,
        0.1 * total_bodies,
        msg=f"Top layer has too few bodies: {num_top}/{total_bodies} (< 10%)",
    )


def _cable_ball_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: BALL joint should keep rod start endpoint attached to a kinematic anchor."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Kinematic anchor body at the rod start point.
    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    # Anchor marker sphere.
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    # Build a straight cable (rod) and attach its start endpoint to the anchor with a BALL joint.
    num_elements = 20
    segment_length = 0.05
    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius
    # Attach the cable endpoint to the sphere surface (not the center), accounting for cable radius so the
    # capsule endcap surface and the sphere surface are coincident along the rod axis (+X).
    attach_offset = wp.float32(anchor_radius + rod_radius)
    parent_anchor_local = wp.vec3(attach_offset, 0.0, 0.0)  # parent local == world (identity rotation)
    anchor_world_attach = anchor_pos + wp.vec3(attach_offset, 0.0, 0.0)

    # Reposition the generated cable so its first point coincides with the sphere-surface attach point.
    # (The helper builds a cable centered about x=0.)
    p0 = points[0]
    offset = anchor_world_attach - p0
    points = [p + offset for p in points]

    rod_bodies, rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_ball_joint_attach",
    )

    # `add_rod()` convention: rod body origin is at `positions[i]` (segment start), so the start endpoint is at z=0 local.
    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
    j_ball = builder.add_joint_ball(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=wp.transform(parent_anchor_local, wp.quat_identity()),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints, j_ball])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
    )

    # Smoothly move the anchor with substeps (mirrors cable example pattern).
    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            err = _compute_ball_joint_anchor_error(model, state0.body_q, j_ball)
            test.assertLess(err, 1.0e-3)

    # Also verify the rod joints remained well-attached along the chain.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms detected in BALL joint test")
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies[0],
        context="Cable BALL joint attachment",
        parent_anchor_local=parent_anchor_local,
    )

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * cable_width,
        context="Cable BALL joint attachment",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable BALL joint attachment",
    )

    # Angular freedom check: gravity sags the horizontal cable, so the child body's
    # orientation should differ from the parent's without any explicit rotation driving.
    q_parent = final_q[anchor][3:7]  # xyzw quaternion
    q_child = final_q[rod_bodies[0]][3:7]
    dot_val = float(np.clip(np.abs(np.dot(q_parent, q_child)), 0.0, 1.0))
    rel_angle = 2.0 * float(np.arccos(dot_val))
    test.assertGreater(
        rel_angle,
        0.1,
        f"BALL joint angular freedom not exercised: relative rotation {rel_angle:.4f} rad < 0.1 rad",
    )


def _cable_fixed_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: FIXED joint should keep rod start frame welded to a kinematic anchor.

    Two cables (+X and +Y) are attached to the same anchor sphere with fixed joints.
    This tests that both translation and rotation are locked in all directions -- a single
    cable along one axis can't fully demonstrate this because gravity only tests one orientation.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)

    # Kinematic anchor body with identity rotation (can't match rotation to two different cables).
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius
    attach_offset = wp.float32(anchor_radius + rod_radius)
    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)

    # --- Cable X (+X direction) ---
    points_x, edge_q_x = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    parent_anchor_local_x = wp.vec3(attach_offset, 0.0, 0.0)
    anchor_world_attach_x = anchor_pos + wp.vec3(attach_offset, 0.0, 0.0)
    p0_x = points_x[0]
    offset_x = anchor_world_attach_x - p0_x
    points_x = [p + offset_x for p in points_x]

    rod_bodies_x, rod_joints_x = builder.add_rod(
        positions=points_x,
        quaternions=edge_q_x,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_fixed_joint_attach_x",
    )

    j_fixed_x = builder.add_joint_fixed(
        parent=anchor,
        child=rod_bodies_x[0],
        parent_xform=wp.transform(parent_anchor_local_x, edge_q_x[0]),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints_x, j_fixed_x])

    # --- Cable Y (+Y direction) ---
    points_y, edge_q_y = _make_straight_cable_along_y(num_elements, segment_length, z_height=anchor_pos[2])
    parent_anchor_local_y = wp.vec3(0.0, attach_offset, 0.0)
    anchor_world_attach_y = anchor_pos + wp.vec3(0.0, attach_offset, 0.0)
    p0_y = points_y[0]
    offset_y = anchor_world_attach_y - p0_y
    points_y = [p + offset_y for p in points_y]

    rod_bodies_y, rod_joints_y = builder.add_rod(
        positions=points_y,
        quaternions=edge_q_y,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_fixed_joint_attach_y",
    )

    j_fixed_y = builder.add_joint_fixed(
        parent=anchor,
        child=rod_bodies_y[0],
        parent_xform=wp.transform(parent_anchor_local_y, edge_q_y[0]),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints_y, j_fixed_y])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
    )

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_err_x, ang_err_x = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed_x)
            test.assertLess(pos_err_x, 1.0e-3)
            test.assertLess(ang_err_x, 2.0e-2)

            pos_err_y, ang_err_y = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed_y)
            test.assertLess(pos_err_y, 1.0e-3)
            test.assertLess(ang_err_y, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms detected in FIXED joint test")
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies_x[0],
        context="Cable FIXED joint attachment (X cable)",
        parent_anchor_local=parent_anchor_local_x,
    )
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies_y[0],
        context="Cable FIXED joint attachment (Y cable)",
        parent_anchor_local=parent_anchor_local_y,
    )

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies_x,
        margin=0.25 * cable_width,
        context="Cable FIXED joint attachment (X cable)",
    )
    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies_y,
        margin=0.25 * cable_width,
        context="Cable FIXED joint attachment (Y cable)",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies_x,
        segment_length=segment_length,
        context="Cable FIXED joint attachment (X cable)",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies_y,
        segment_length=segment_length,
        context="Cable FIXED joint attachment (Y cable)",
    )


def _cable_revolute_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: REVOLUTE joint should keep rod start endpoint attached and perpendicular axes aligned.

    Two cables (+X and +Y) are attached to the same anchor sphere with revolute joints.
    Both have world Y as the free axis. For cable X, gravity creates torque about Y (free),
    so it sags. For cable Y, gravity creates torque about X (constrained), so it stays rigid.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)

    # Kinematic anchor body with identity rotation.
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius
    attach_offset = wp.float32(anchor_radius + rod_radius)
    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)

    # --- Cable X (+X direction) ---
    points_x, edge_q_x = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    parent_anchor_local_x = wp.vec3(attach_offset, 0.0, 0.0)
    anchor_world_attach_x = anchor_pos + wp.vec3(attach_offset, 0.0, 0.0)
    p0_x = points_x[0]
    offset_x = anchor_world_attach_x - p0_x
    points_x = [p + offset_x for p in points_x]

    rod_bodies_x, rod_joints_x = builder.add_rod(
        positions=points_x,
        quaternions=edge_q_x,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_revolute_joint_attach_x",
    )

    # Revolute axis: Y in joint frame -> world Y free (edge_q_x[0] maps local Z->world X,
    # so local Y stays world Y).
    j_revolute_x = builder.add_joint_revolute(
        parent=anchor,
        child=rod_bodies_x[0],
        parent_xform=wp.transform(parent_anchor_local_x, edge_q_x[0]),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
    )
    builder.add_articulation([*rod_joints_x, j_revolute_x])

    # --- Cable Y (+Y direction) ---
    points_y, edge_q_y = _make_straight_cable_along_y(num_elements, segment_length, z_height=anchor_pos[2])
    parent_anchor_local_y = wp.vec3(0.0, attach_offset, 0.0)
    anchor_world_attach_y = anchor_pos + wp.vec3(0.0, attach_offset, 0.0)
    p0_y = points_y[0]
    offset_y = anchor_world_attach_y - p0_y
    points_y = [p + offset_y for p in points_y]

    rod_bodies_y, rod_joints_y = builder.add_rod(
        positions=points_y,
        quaternions=edge_q_y,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_revolute_joint_attach_y",
    )

    # Revolute axis: Z in joint frame -> world Y free.
    # Derivation: edge_q_y[0] maps local +Z->world +Y (rotation about X by 90 deg).
    # Its inverse maps world Y->local Z. So to get world Y as free axis, use local (0,0,1).
    j_revolute_y = builder.add_joint_revolute(
        parent=anchor,
        child=rod_bodies_y[0],
        parent_xform=wp.transform(parent_anchor_local_y, edge_q_y[0]),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_articulation([*rod_joints_y, j_revolute_y])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
    )

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_err_x, ang_perp_err_x, _rot_free_x = _compute_revolute_joint_error(model, state0.body_q, j_revolute_x)
            test.assertLess(pos_err_x, 1.0e-3)
            test.assertLess(ang_perp_err_x, 2.0e-2)

            pos_err_y, ang_perp_err_y, _rot_free_y = _compute_revolute_joint_error(model, state0.body_q, j_revolute_y)
            test.assertLess(pos_err_y, 1.0e-3)
            test.assertLess(ang_perp_err_y, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms detected in REVOLUTE joint test")
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies_x[0],
        context="Cable REVOLUTE joint attachment (X cable)",
        parent_anchor_local=parent_anchor_local_x,
    )
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies_y[0],
        context="Cable REVOLUTE joint attachment (Y cable)",
        parent_anchor_local=parent_anchor_local_y,
    )

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies_x,
        margin=0.25 * cable_width,
        context="Cable REVOLUTE joint attachment (X cable)",
    )
    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies_y,
        margin=0.25 * cable_width,
        context="Cable REVOLUTE joint attachment (Y cable)",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies_x,
        segment_length=segment_length,
        context="Cable REVOLUTE joint attachment (X cable)",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies_y,
        segment_length=segment_length,
        context="Cable REVOLUTE joint attachment (Y cable)",
    )

    # Angular freedom check: gravity sags cable X (torque about Y = free axis).
    _pos_err, _ang_perp_err, rot_free_x = _compute_revolute_joint_error(model, state0.body_q, j_revolute_x)
    test.assertGreater(
        rot_free_x,
        0.1,
        f"REVOLUTE joint angular freedom not exercised: X cable free-axis rotation {rot_free_x:.4f} rad < 0.1 rad",
    )


def _cable_revolute_drive_tracks_target_impl(test: unittest.TestCase, device):
    """Cable VBD: revolute drive (target_ke/kd + target_pos) should track target angle on a cable.

    A single cable hangs from a kinematic sphere anchor via a revolute joint (Y-axis)
    with drive parameters. A static target angle is set, and the cable should converge
    toward it despite gravity.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    # Kinematic anchor body.
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_revolute_drive",
    )

    target_angle = 0.4  # rad
    drive_ke = 2000.0
    drive_kd = 0.05

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_revolute = builder.add_joint_revolute(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        axis=(0.0, 1.0, 0.0),
        target_ke=drive_ke,
        target_kd=drive_kd,
    )
    builder.add_articulation([*rod_joints, j_revolute])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    # Find the revolute joint and its DOF index after finalize().
    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    rev_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.REVOLUTE))
    dof_idx = int(joint_qd_start[rev_idx])

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Set drive target position.
    tp = control.joint_target_pos.numpy()
    tp[dof_idx] = target_angle
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Joint constraint checks.
    pos_err, ang_perp_err, rot_free = _compute_revolute_joint_error(model, state0.body_q, rev_idx)
    test.assertLess(pos_err, 1.0e-3, f"Revolute drive: position error {pos_err:.6f}")
    test.assertLess(ang_perp_err, 2.0e-2, f"Revolute drive: perpendicular angle error {ang_perp_err:.4f}")

    # Drive convergence: free-axis rotation should be near target.
    test.assertAlmostEqual(
        rot_free,
        target_angle,
        delta=0.15,
        msg=f"Revolute drive not tracking: rot_free={rot_free:.4f}, target={target_angle}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in revolute drive test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable revolute drive",
    )


def _cable_revolute_drive_limit_impl(test: unittest.TestCase, device):
    """Cable VBD: revolute drive with limits should clamp rotation within bounds.

    Vertical cable hanging -Z from a static kinematic anchor. Revolute joint (Y-axis)
    with drive and limits. Drive target is set beyond the limit bounds. After convergence
    the cable should reach the limit bound, not the drive target.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_revolute_drive_limit",
    )

    target_angle = 1.5  # rad -- beyond limits
    ang_limit = 0.3
    drive_ke = 2000.0
    drive_kd = 0.05

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_revolute = builder.add_joint_revolute(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        axis=(0.0, 1.0, 0.0),
        target_ke=drive_ke,
        target_kd=drive_kd,
        limit_lower=-ang_limit,
        limit_upper=ang_limit,
        limit_ke=1.0e5,
        limit_kd=1.0e-4,
    )
    builder.add_articulation([*rod_joints, j_revolute])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    rev_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.REVOLUTE))
    dof_idx = int(joint_qd_start[rev_idx])

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    tp = control.joint_target_pos.numpy()
    tp[dof_idx] = target_angle
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Joint constraint checks.
    pos_err, ang_perp_err, rot_free = _compute_revolute_joint_error(model, state0.body_q, rev_idx)
    test.assertLess(pos_err, 2.0e-3, f"Revolute drive limit: position error {pos_err:.6f}")
    test.assertLess(ang_perp_err, 5.0e-2, f"Revolute drive limit: perpendicular angle error {ang_perp_err:.4f}")

    # Limit enforcement: rotation should be near the limit bound.
    ang_tolerance = 0.05
    test.assertLessEqual(
        rot_free,
        ang_limit + ang_tolerance,
        msg=f"Revolute limit violated: rot={rot_free:.4f} > bound {ang_limit} + tol {ang_tolerance}",
    )
    test.assertLess(
        rot_free,
        target_angle * 0.5,
        msg=f"Revolute limit not effective: rot={rot_free:.4f} too close to target {target_angle}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in revolute drive limit test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable revolute drive limit",
    )


def _cable_prismatic_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: PRISMATIC joint should keep perpendicular position constrained and rotation locked.

    Single cable along world +X. The prismatic free axis is world Y (perpendicular to
    both the cable and gravity). The anchor oscillates in both X and Y. The locked DOFs
    (X, Z, rotation) are tested under load, while the free Y axis allows sliding.

    Note: unlike fixed/revolute, prismatic uses a single cable because a second cable
    along the free axis is a degenerate configuration (it can slide away from the anchor).
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)

    # Kinematic anchor body with identity rotation (matching ball/fixed/revolute).
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius
    attach_offset = wp.float32(anchor_radius + rod_radius)
    parent_anchor_local = wp.vec3(attach_offset, 0.0, 0.0)
    anchor_world_attach = anchor_pos + wp.vec3(attach_offset, 0.0, 0.0)

    p0 = points[0]
    offset = anchor_world_attach - p0
    points = [p + offset for p in points]

    rod_bodies, rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_prismatic_joint_attach",
    )

    # Prismatic axis: Y in joint frame. edge_q[0] maps local Z->world X and
    # preserves local Y->world Y. So axis (0,1,0) gives free sliding along world Y
    # -- perpendicular to both the cable (+X) and gravity (-Z).
    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
    j_prismatic = builder.add_joint_prismatic(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=wp.transform(parent_anchor_local, edge_q[0]),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
    )
    builder.add_articulation([*rod_joints, j_prismatic])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
    )

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))
            dy = wp.float32(0.04 * np.sin(2.0 * t))

            pose = wp.transform(wp.vec3(dx, dy, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_perp_err, ang_err, _c_along = _compute_prismatic_joint_error(model, state0.body_q, j_prismatic)
            test.assertLess(pos_perp_err, 1.0e-3)
            test.assertLess(ang_err, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms detected in PRISMATIC joint test")

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * cable_width,
        context="Cable PRISMATIC joint attachment",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable PRISMATIC joint attachment",
    )

    # Free DOF freedom: anchor oscillates in Y (free axis), cable should slide freely.
    _pos_perp_err, _ang_err, c_along = _compute_prismatic_joint_error(model, state0.body_q, j_prismatic)
    test.assertGreater(
        abs(c_along),
        0.005,
        f"PRISMATIC joint linear freedom not exercised: |c_along|={abs(c_along):.4f} m < 0.005 m",
    )


def _cable_prismatic_drive_tracks_target_impl(test: unittest.TestCase, device):
    """Cable VBD: prismatic drive (target_ke/kd + target_pos) should track target displacement on a cable.

    A single cable hangs from a kinematic sphere anchor via a prismatic joint (X-axis)
    with drive parameters. A static target displacement is set, and the cable should
    converge toward it despite gravity.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    # Kinematic anchor body.
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_prismatic_drive",
    )

    target_displacement = 0.1  # m
    drive_ke = 5000.0
    drive_kd = 0.04

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_prismatic = builder.add_joint_prismatic(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        axis=(1.0, 0.0, 0.0),
        target_ke=drive_ke,
        target_kd=drive_kd,
    )
    builder.add_articulation([*rod_joints, j_prismatic])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    # Find the prismatic joint and its DOF index after finalize().
    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    prismatic_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.PRISMATIC))
    dof_idx = int(joint_qd_start[prismatic_idx])

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Set drive target position.
    tp = control.joint_target_pos.numpy()
    tp[dof_idx] = target_displacement
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Joint constraint checks.
    pos_perp_err, ang_err, c_along = _compute_prismatic_joint_error(model, state0.body_q, prismatic_idx)
    test.assertLess(pos_perp_err, 2.0e-3, f"Prismatic drive: perpendicular position error {pos_perp_err:.6f}")
    test.assertLess(ang_err, 5.0e-2, f"Prismatic drive: angular error {ang_err:.4f}")

    # Drive convergence: signed free-axis displacement should be near target.
    test.assertAlmostEqual(
        c_along,
        target_displacement,
        delta=0.03,
        msg=f"Prismatic drive not tracking: d={c_along:.4f}, target={target_displacement}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in prismatic drive test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable prismatic drive",
    )


def _cable_prismatic_drive_limit_impl(test: unittest.TestCase, device):
    """Cable VBD: prismatic drive with limits should clamp displacement within bounds.

    Vertical cable hanging -Z from a static kinematic anchor. Prismatic joint (X-axis)
    with drive and limits. Drive target is set beyond the limit bounds. After convergence
    the cable should reach the limit bound, not the drive target.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_prismatic_drive_limit",
    )

    target_displacement = 0.5  # m -- beyond limits
    lin_limit = 0.05
    drive_ke = 5000.0
    drive_kd = 0.04

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_prismatic = builder.add_joint_prismatic(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        axis=(1.0, 0.0, 0.0),
        target_ke=drive_ke,
        target_kd=drive_kd,
        limit_lower=-lin_limit,
        limit_upper=lin_limit,
        limit_ke=1.0e5,
        limit_kd=1.0e-3,
    )
    builder.add_articulation([*rod_joints, j_prismatic])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    prismatic_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.PRISMATIC))
    dof_idx = int(joint_qd_start[prismatic_idx])

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    tp = control.joint_target_pos.numpy()
    tp[dof_idx] = target_displacement
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Joint constraint checks.
    pos_perp_err, ang_err, c_along = _compute_prismatic_joint_error(model, state0.body_q, prismatic_idx)
    test.assertLess(pos_perp_err, 2.0e-3, f"Prismatic drive limit: perp pos error {pos_perp_err:.6f}")
    test.assertLess(ang_err, 5.0e-2, f"Prismatic drive limit: angular error {ang_err:.4f}")

    # Limit enforcement: displacement should be near the limit bound.
    lin_tolerance = 0.02
    test.assertLessEqual(
        abs(c_along),
        lin_limit + lin_tolerance,
        msg=f"Prismatic limit violated: |d|={abs(c_along):.4f} > bound {lin_limit} + tol {lin_tolerance}",
    )
    test.assertLess(
        abs(c_along),
        target_displacement * 0.5,
        msg=f"Prismatic limit not effective: |d|={abs(c_along):.4f} too close to target {target_displacement}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in prismatic drive limit test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable prismatic drive limit",
    )


def _cable_d6_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 joint (free linear X + free angular Y) should keep constrained DOFs locked.

    Vertical cable hanging -Z from a kinematic anchor. D6 joint with 1 free linear
    axis (X) and 1 free angular axis (Y) in the joint parent anchor frame.
    For -Z cables the parent frame rotates +Z to -Z (180 deg about Y), so
    joint-frame X maps to world -X and Y stays world Y.

    Anchor oscillates in X and rotates around Y. The free linear axis
    allows the cable to slide (not follow the X motion). Gravity in -Z
    stresses the locked Z linear constraint. The anchor rotation around Y
    directly exercises the free angular Y axis — the cable should not
    follow the rotation.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius

    JointDofConfig = newton.ModelBuilder.JointDofConfig

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_d6_joint_attach",
    )

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_d6 = builder.add_joint_d6(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        linear_axes=[JointDofConfig(axis=(1, 0, 0))],
        angular_axes=[JointDofConfig(axis=(0, 1, 0))],
    )
    builder.add_articulation([*rod_joints, j_d6])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))
            # Rotate anchor around Y to exercise the free angular Y DOF
            ang_y = wp.float32(0.2 * np.sin(2.0 * t))
            q_anchor = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang_y)

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), q_anchor)
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_perp_err, ang_perp_err, _d_along, _rot_free = _compute_d6_joint_error(model, state0.body_q, j_d6)
            test.assertLess(pos_perp_err, 1.0e-3)
            test.assertLess(ang_perp_err, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in D6 joint test")

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * cable_width,
        context="Cable D6 joint attachment",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable D6 joint attachment",
    )

    # Free DOF freedom: a D6 that secretly locks all DOFs must not pass.
    _, _, d_along, rot_free = _compute_d6_joint_error(model, state0.body_q, j_d6)
    test.assertGreater(
        abs(d_along),
        0.005,
        msg=f"D6 free linear X not exercised: |d_along|={abs(d_along):.4f} m",
    )
    test.assertGreater(
        rot_free,
        0.005,
        msg=f"D6 free angular Y not exercised: rot_free={rot_free:.4f} rad",
    )


def _cable_d6_joint_all_locked_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 joint with all DOFs locked should behave like a fixed joint.

    Vertical cable hanging -Z from a kinematic anchor. D6 joint with no free axes
    (lin_axes=[], ang_axes=[]). Anchor oscillates in X; the cable endpoint must
    follow exactly, matching the lock_xyz config in example_cable_d6_joints.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_d6_all_locked",
    )

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_d6 = builder.add_joint_d6(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        linear_axes=[],
        angular_axes=[],
    )
    builder.add_articulation([*rod_joints, j_d6])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_err, ang_err = _compute_fixed_joint_frame_error(model, state0.body_q, j_d6)
            test.assertLess(pos_err, 1.0e-3)
            test.assertLess(ang_err, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in D6 all-locked test")

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * cable_width,
        context="Cable D6 all-locked",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable D6 all-locked",
    )


def _cable_d6_joint_locked_x_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 joint with X linear locked, Y/Z free, all angular locked.

    Vertical cable hanging -Z from a kinematic anchor. D6 joint with
    lin_axes=[(0,1,0), (0,0,1)] (free Y and Z in joint frame; X locked).
    All angular DOFs locked. Anchor oscillates in X. The cable must follow the
    X motion (locked) while being free to sag under gravity (Z free).
    Matches the lock_x config in example_cable_d6_joints.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01
    cable_width = 2.0 * rod_radius

    JointDofConfig = newton.ModelBuilder.JointDofConfig

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e-1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_d6_locked_x",
    )

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    # Free Y and Z linear (in joint frame), locked X. All angular locked.
    # For -Z cable: joint-frame X = world -X, Y = world Y, Z = world -Z.
    j_d6 = builder.add_joint_d6(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        linear_axes=[JointDofConfig(axis=(0, 1, 0)), JointDofConfig(axis=(0, 0, 1))],
        angular_axes=[],
    )
    builder.add_articulation([*rod_joints, j_d6])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    locked_axis_local = wp.vec3(1.0, 0.0, 0.0)

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            dx = wp.float32(0.05 * np.sin(1.5 * t))

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), wp.quat_identity())
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            # Position error along the locked X axis.
            X_wp, X_wc = _get_joint_world_frames(model, state0.body_q, j_d6)
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            q_wp = wp.transform_get_rotation(X_wp)
            axis_world = wp.normalize(wp.quat_rotate(q_wp, locked_axis_local))
            d_locked = abs(float(wp.dot(x_c - x_p, axis_world)))
            test.assertLess(d_locked, 1.0e-3, "D6 locked X: position error along locked axis")

            # Angular error (all angular locked).
            q_wc = wp.transform_get_rotation(X_wc)
            q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))
            q_rest = _get_joint_rest_relative_rotation(model, j_d6)
            q_err = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rest)))
            ang_err = float(2.0 * wp.acos(wp.clamp(wp.abs(q_err[3]), 0.0, 1.0)))
            test.assertLess(ang_err, 2.0e-2, "D6 locked X: angular error")

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in D6 locked-X test")

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * cable_width,
        context="Cable D6 locked-X",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable D6 locked-X",
    )

    # Freedom: displacement in free directions should be non-zero (gravity sag in Z).
    X_wp, X_wc = _get_joint_world_frames(model, state0.body_q, j_d6)
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    C = x_c - x_p
    q_wp = wp.transform_get_rotation(X_wp)
    axis_world = wp.normalize(wp.quat_rotate(q_wp, locked_axis_local))
    d_locked_val = float(wp.dot(C, axis_world))
    C_free = C - d_locked_val * axis_world
    free_displacement = float(wp.length(C_free))
    test.assertGreater(
        free_displacement,
        0.005,
        msg=f"D6 free Y/Z displacement not exercised: {free_displacement:.4f} m",
    )


def _cable_d6_drive_tracks_target_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 drive (target_ke/kd + target_pos) should track targets on a cable.

    A single cable hangs from a kinematic sphere anchor via a D6 joint (1 linear X + 1 angular Y)
    with drive parameters. Static targets are set for both DOFs, and the cable should converge
    toward them despite gravity.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    # Kinematic anchor body.
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_d6_drive",
    )

    target_displacement = 0.1  # m
    target_angle = 0.4  # rad
    lin_drive_ke = 5000.0
    lin_drive_kd = 0.04
    ang_drive_ke = 2000.0
    ang_drive_kd = 0.05

    JointDofConfig = newton.ModelBuilder.JointDofConfig

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_d6 = builder.add_joint_d6(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        linear_axes=[JointDofConfig(axis=(1, 0, 0), target_ke=lin_drive_ke, target_kd=lin_drive_kd)],
        angular_axes=[JointDofConfig(axis=(0, 1, 0), target_ke=ang_drive_ke, target_kd=ang_drive_kd)],
    )
    builder.add_articulation([*rod_joints, j_d6])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    # Find the D6 joint and its DOF indices after finalize().
    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    d6_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.D6))
    qd_s = int(joint_qd_start[d6_idx])
    lin_dof_idx = qd_s
    ang_dof_idx = qd_s + 1

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Set drive target positions.
    tp = control.joint_target_pos.numpy()
    tp[lin_dof_idx] = target_displacement
    tp[ang_dof_idx] = target_angle
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Joint constraint checks.
    pos_perp_err, ang_perp_err, c_along, rot_free = _compute_d6_joint_error(model, state0.body_q, d6_idx)
    test.assertLess(pos_perp_err, 2.0e-3, f"D6 drive: perpendicular position error {pos_perp_err:.6f}")
    test.assertLess(ang_perp_err, 5.0e-2, f"D6 drive: perpendicular angle error {ang_perp_err:.4f}")

    # Drive convergence: linear displacement should be near target.
    test.assertAlmostEqual(
        c_along,
        target_displacement,
        delta=0.03,
        msg=f"D6 linear drive not tracking: d={c_along:.4f}, target={target_displacement}",
    )

    # Drive convergence: angular rotation should be near target.
    test.assertAlmostEqual(
        rot_free,
        target_angle,
        delta=0.15,
        msg=f"D6 angular drive not tracking: rot_free={rot_free:.4f}, target={target_angle}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in D6 drive test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable D6 drive",
    )


def _cable_d6_drive_limit_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 drive with limits should clamp DOFs within bounds.

    Vertical cable hanging -Z from a static kinematic anchor. D6 joint with
    1 free linear axis (X) and 1 free angular axis (Y), both with drives and limits.
    Drive targets are set well beyond the limit bounds. After convergence the
    cable should reach the limit bounds, not the drive targets.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor_radius = 0.1

    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 20
    segment_length = 0.05
    rod_radius = 0.01

    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -anchor_radius)
    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements * segment_length),
        num_segments=num_elements,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        wrap_in_articulation=False,
        label="test_cable_d6_drive_limit",
    )

    # Drive targets are intentionally beyond the limit bounds.
    target_displacement = 0.5
    target_angle = 1.5
    lin_limit = 0.05
    ang_limit = 0.3

    JointDofConfig = newton.ModelBuilder.JointDofConfig

    parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), rod_quats[0])
    child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    j_d6 = builder.add_joint_d6(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=parent_xform,
        child_xform=child_xform,
        linear_axes=[
            JointDofConfig(
                axis=(1, 0, 0),
                target_ke=5000.0,
                target_kd=0.04,
                limit_lower=-lin_limit,
                limit_upper=lin_limit,
                limit_ke=1.0e5,
                limit_kd=1.0e-3,
            )
        ],
        angular_axes=[
            JointDofConfig(
                axis=(0, 1, 0),
                target_ke=2000.0,
                target_kd=0.05,
                limit_lower=-ang_limit,
                limit_upper=ang_limit,
                limit_ke=1.0e5,
                limit_kd=1.0e-4,
            )
        ],
    )
    builder.add_articulation([*rod_joints, j_d6])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    joint_types = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    d6_idx = next(i for i in range(model.joint_count) if int(joint_types[i]) == int(newton.JointType.D6))
    qd_s = int(joint_qd_start[d6_idx])

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    tp = control.joint_target_pos.numpy()
    tp[qd_s] = target_displacement
    tp[qd_s + 1] = target_angle
    control.joint_target_pos = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Locked DOF checks.
    pos_perp_err, ang_perp_err, c_along, rot_free = _compute_d6_joint_error(model, state0.body_q, d6_idx)
    test.assertLess(pos_perp_err, 2.0e-3, f"D6 drive limit: perp pos error {pos_perp_err:.6f}")
    test.assertLess(ang_perp_err, 5.0e-2, f"D6 drive limit: perp ang error {ang_perp_err:.4f}")

    # Linear limit: displacement should be near the limit bound, not the drive target.
    lin_tolerance = 0.02
    test.assertLessEqual(
        abs(c_along),
        lin_limit + lin_tolerance,
        msg=f"D6 linear limit violated: |d|={abs(c_along):.4f} > bound {lin_limit} + tol {lin_tolerance}",
    )
    test.assertLess(
        abs(c_along),
        target_displacement * 0.5,
        msg=f"D6 linear limit not effective: |d|={abs(c_along):.4f} too close to target {target_displacement}",
    )

    # Angular limit: rotation should be near the limit bound, not the drive target.
    ang_tolerance = 0.05
    test.assertLessEqual(
        rot_free,
        ang_limit + ang_tolerance,
        msg=f"D6 angular limit violated: rot={rot_free:.4f} > bound {ang_limit} + tol {ang_tolerance}",
    )
    test.assertLess(
        rot_free,
        target_angle * 0.5,
        msg=f"D6 angular limit not effective: rot={rot_free:.4f} too close to target {target_angle}",
    )

    # Cable integrity.
    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in D6 drive limit test")
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable D6 drive limit",
    )


def _cable_kinematic_gripper_picks_capsule_impl(test: unittest.TestCase, device):
    """Kinematic friction regression: moving kinematic grippers should lift a dynamic capsule.

    - two kinematic box "fingers" close on a capsule and then lift upward
    - gravity is disabled, so any lift must come from kinematic contact/friction transfer

    Assertions:
    - the capsule must be lifted upward by a non-trivial amount
    - the capsule final z should roughly track the grippers' final z (within tolerance)
    """
    builder = newton.ModelBuilder()

    # Contact/friction: large mu to encourage sticking if kinematic friction is working.
    builder.default_shape_cfg.mu = 1.0e3

    # Payload: capsule sized to match old box AABB (0.20, 0.10, 0.10) in (X,Y,Z)
    box_hx = 0.10
    box_hy = 0.05
    capsule_radius = float(box_hy)
    capsule_half_height = float(box_hx - capsule_radius)
    capsule_rot_z_to_x = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))

    capsule_center = wp.vec3(0.0, 0.015, capsule_radius)
    capsule_body = builder.add_body(
        xform=wp.transform(p=capsule_center, q=wp.quat_identity()),
        mass=1.0,
        label="ut_gripper_capsule",
    )
    payload_cfg = builder.default_shape_cfg.copy()
    payload_cfg.mu = 1.0e3
    builder.add_shape_capsule(
        body=capsule_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=capsule_rot_z_to_x),
        radius=capsule_radius,
        half_height=capsule_half_height,
        cfg=payload_cfg,
        label="ut_gripper_capsule_shape",
    )

    # Kinematic box grippers
    grip_hx = 0.52
    grip_hy = 0.02
    grip_hz = 0.56

    anchor_p = wp.vec3(0.0, 0.0, float(capsule_center[2]))
    anchor_q = wp.quat_identity()

    target_offset_mag = float(capsule_radius) + 0.95 * float(grip_hy)
    initial_offset_mag = target_offset_mag + 3.0 * (2.0 * float(grip_hy))

    g_neg = builder.add_body(
        xform=wp.transform(p=anchor_p + wp.vec3(0.0, -initial_offset_mag, 0.0), q=anchor_q),
        mass=0.0,
        label="ut_gripper_neg",
    )
    g_pos = builder.add_body(
        xform=wp.transform(p=anchor_p + wp.vec3(0.0, initial_offset_mag, 0.0), q=anchor_q),
        mass=0.0,
        label="ut_gripper_pos",
    )

    builder.body_mass[g_neg] = 0.0
    builder.body_inv_mass[g_neg] = 0.0
    builder.body_inv_inertia[g_neg] = wp.mat33(0.0)
    builder.body_mass[g_pos] = 0.0
    builder.body_inv_mass[g_pos] = 0.0
    builder.body_inv_inertia[g_pos] = wp.mat33(0.0)

    grip_cfg = builder.default_shape_cfg.copy()
    grip_cfg.mu = 1.0e3

    # Keep grippers kinematic (no mass contribution from density)
    grip_cfg.density = 0.0
    builder.add_shape_box(body=g_neg, hx=float(grip_hx), hy=float(grip_hy), hz=float(grip_hz), cfg=grip_cfg)
    builder.add_shape_box(body=g_pos, hx=float(grip_hx), hy=float(grip_hy), hz=float(grip_hz), cfg=grip_cfg)

    builder.color()
    model = builder.finalize(device=device)
    # Disable gravity: any upward motion must be due to kinematic friction/contact transfer.
    model.set_gravity((0.0, 0.0, 0.0))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
    )

    # Drive arrays
    gripper_body_ids = wp.array([g_neg, g_pos], dtype=wp.int32, device=device)
    gripper_signs = wp.array([-1.0, 1.0], dtype=wp.float32, device=device)

    # Timeline
    ramp_time = 0.25
    pull_start_time = 0.25
    pull_ramp_time = 1.0
    pull_distance = 0.75

    fps = 60.0
    frame_dt = 1.0 / fps
    sim_substeps = 1
    sim_dt = frame_dt / sim_substeps

    # Record initial pose
    q0 = state0.body_q.numpy()
    capsule_z0 = float(q0[capsule_body, 2])

    # Run a fixed number of frames for a lightweight regression test.
    num_frames = 100
    sim_time = 0.0
    num_steps = num_frames * sim_substeps
    for _step in range(num_steps):
        state0.clear_forces()

        wp.launch(
            kernel=_drive_gripper_boxes_kernel,
            dim=2,
            inputs=[
                float(ramp_time),
                float(sim_time),
                gripper_body_ids,
                gripper_signs,
                anchor_p,
                anchor_q,
                0.0,  # seg_half_len
                float(target_offset_mag),
                float(initial_offset_mag),
                float(pull_start_time),
                float(pull_ramp_time),
                float(pull_distance),
                state0.body_q,
            ],
            device=device,
        )

        model.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, sim_dt)
        state0, state1 = state1, state0

        sim_time += sim_dt

    qf = state0.body_q.numpy()
    test.assertTrue(np.isfinite(qf).all(), "Non-finite body transforms detected in gripper friction test")

    capsule_zf = float(qf[capsule_body, 2])
    z_lift = capsule_zf - capsule_z0

    # 1) Must lift upward significantly.
    test.assertGreater(
        z_lift,
        0.25,
        msg=f"Capsule was not lifted enough by kinematic friction: dz={z_lift:.4f} (z0={capsule_z0:.4f}, zf={capsule_zf:.4f})",
    )

    # 2) Capsule should roughly track the grippers' final lift height.
    gripper_z = 0.5 * (float(qf[g_neg, 2]) + float(qf[g_pos, 2]))
    test.assertLess(
        abs(capsule_zf - gripper_z),
        0.01,
        msg=f"Capsule Z does not track grippers: capsule_z={capsule_zf:.4f}, gripper_z={gripper_z:.4f}",
    )


def _cable_graph_y_junction_spanning_tree_impl(test: unittest.TestCase, device):
    """Cable graph: Y-junction should build (and simulate) with wrap_in_articulation=True."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Simple Y: 0-1-2 and 1-3
    node_positions = [
        wp.vec3(0.0, 0.0, 0.5),
        wp.vec3(0.2, 0.0, 0.5),
        wp.vec3(0.4, 0.0, 0.5),
        wp.vec3(0.2, 0.2, 0.5),
    ]
    edges = [(0, 1), (1, 2), (1, 3)]

    node_positions_any: list[Any] = node_positions

    cable_radius = 0.05
    cable_width = 2.0 * cable_radius
    rod_bodies, rod_joints = builder.add_rod_graph(
        node_positions=node_positions_any,
        edges=edges,
        radius=cable_radius,
        cfg=builder.default_shape_cfg.copy(),
        bend_stiffness=1.0e2,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        label="ut_cable_graph_y",
        wrap_in_articulation=True,
    )

    test.assertEqual(len(rod_bodies), len(edges))

    # Spanning forest on a tree with E=3 => E-1 joints.
    test.assertEqual(len(rod_joints), 2)

    # Also verify the produced joints connect edge-bodies consistently:
    # - every rod joint connects two rod_bodies
    # - each such pair corresponds to two input edges sharing a node (sanity check)
    # - the resulting rod-body graph is connected (so E-1 joints => spanning tree)
    body_to_edge = {int(body): edges[i] for i, body in enumerate(rod_bodies)}
    rod_body_set = set(body_to_edge.keys())

    # Union-find over rod bodies (treat joints as undirected edges for connectivity).
    parent_uf = {b: b for b in rod_body_set}

    def find_uf(x: int) -> int:
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union_uf(a: int, b: int):
        ra, rb = find_uf(a), find_uf(b)
        if ra != rb:
            parent_uf[rb] = ra

    for j in rod_joints:
        jb = int(j)
        a = int(builder.joint_parent[jb])
        b = int(builder.joint_child[jb])

        test.assertIn(a, rod_body_set, msg=f"Y-junction rod joint {jb} parent body {a} not in rod_bodies")
        test.assertIn(b, rod_body_set, msg=f"Y-junction rod joint {jb} child body {b} not in rod_bodies")
        test.assertNotEqual(a, b, msg=f"Y-junction rod joint {jb} connects body {a} to itself")

        eu0, ev0 = body_to_edge[a]
        eu1, ev1 = body_to_edge[b]
        shared = {eu0, ev0}.intersection({eu1, ev1})
        test.assertTrue(
            shared,
            msg=(
                f"Y-junction rod joint {jb} connects edges {body_to_edge[a]} and {body_to_edge[b]} "
                f"that do not share a graph node"
            ),
        )

        union_uf(a, b)

    roots = {find_uf(b) for b in rod_body_set}
    test.assertEqual(len(roots), 1, msg=f"Y-junction rod bodies not connected by rod joints, roots={sorted(roots)}")

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0, state1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=10)

    q_init = state0.body_q.numpy()
    z_init_min = float(np.min(q_init[rod_bodies, 2]))

    frame_dt = 1.0 / 60.0
    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    contacts = model.contacts()
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            state0.clear_forces()
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    qf = state0.body_q.numpy()
    test.assertTrue(np.isfinite(qf).all(), "Non-finite body transforms detected in Y-junction graph simulation")

    z_min = float(np.min(qf[rod_bodies, 2]))
    test.assertLess(z_min, z_init_min - 0.01, msg="Y-junction did not fall noticeably under gravity")
    _assert_bodies_above_ground(test, qf, rod_bodies, context="y-junction", margin=0.25 * cable_width)


def _cable_rod_ring_closed_in_articulation_impl(test: unittest.TestCase, device):
    """Closed ring via add_rod(closed=True) should build and simulate."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Build a planar ring polyline (duplicate last point so the last segment returns to the start).
    num_segments = 16
    radius = 0.35
    z0 = 0.5
    theta = np.linspace(0.0, 2.0 * np.pi, num_segments + 1, endpoint=True)
    points = [wp.vec3(float(radius * np.cos(t)), float(radius * np.sin(t)), float(z0)) for t in theta]
    quats = newton.utils.create_parallel_transport_cable_quaternions(points)

    # Avoid list[Vec3] invariance issues in static checking.
    points_any: list[Any] = points
    quats_any: list[Any] = quats

    cable_radius = 0.05
    cable_width = 2.0 * cable_radius
    rod_bodies, rod_joints = builder.add_rod(
        positions=points_any,
        quaternions=quats_any,
        radius=cable_radius,
        cfg=builder.default_shape_cfg.copy(),
        bend_stiffness=1.0e2,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        closed=True,
        label="ut_cable_rod_ring_closed",
        wrap_in_articulation=True,
    )

    test.assertEqual(len(rod_bodies), num_segments)
    test.assertEqual(len(rod_joints), num_segments)

    # Ensure the loop-closing joint exists (added after articulation wrapping).
    first_body = int(rod_bodies[0])
    last_body = int(rod_bodies[-1])
    loop_pairs = {(int(builder.joint_parent[int(j)]), int(builder.joint_child[int(j)])) for j in rod_joints}
    test.assertIn(
        (last_body, first_body),
        loop_pairs,
        msg="Closed ring is missing the loop-closing joint between the last and first segment bodies",
    )

    builder.add_ground_plane()
    builder.color()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    # Drop the ring onto the ground and ensure stable simulation.
    state0, state1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    q_init = state0.body_q.numpy()
    z_init_min = float(np.min(q_init[rod_bodies, 2]))

    contacts = model.contacts()
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    qf = state0.body_q.numpy()
    test.assertTrue(np.isfinite(qf).all(), "Non-finite body transforms detected in closed-ring simulation")

    z_min = float(np.min(qf[rod_bodies, 2]))
    test.assertLess(z_min, z_init_min - 0.1, msg="Ring did not fall noticeably under gravity")
    _assert_bodies_above_ground(test, qf, rod_bodies, context="closed-ring", margin=0.25 * cable_width)


def _cable_graph_default_quat_aligns_z_impl(test: unittest.TestCase, device):
    """Cable graph: when quaternions are not provided, local +Z should align to the edge direction."""
    builder = newton.ModelBuilder()

    p0 = wp.vec3(0.0, 0.0, 3.0)
    p1 = wp.vec3(0.4, 0.1, 3.0)
    node_positions = [p0, p1]
    edges = [(0, 1)]

    # Use wp.vec3 at runtime but avoid list[Vec3] invariance issues in static checking.
    node_positions_any: list[Any] = node_positions

    rod_bodies, rod_joints = builder.add_rod_graph(
        node_positions=node_positions_any,
        edges=edges,
        radius=0.05,
        cfg=builder.default_shape_cfg.copy(),
        bend_stiffness=0.0,
        bend_damping=0.0,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        label="ut_cable_graph_quat",
        wrap_in_articulation=True,
        quaternions=None,
    )
    test.assertEqual(len(rod_bodies), 1)
    test.assertEqual(len(rod_joints), 0)

    builder.color()
    model = builder.finalize(device=device)

    # Read initial pose and verify axis alignment.
    q0 = model.state().body_q.numpy()
    body_id = int(rod_bodies[0])
    rot = wp.quat(q0[body_id][3], q0[body_id][4], q0[body_id][5], q0[body_id][6])
    z_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    d_hat = wp.normalize(p1 - p0)
    dot = float(wp.dot(z_world, d_hat))
    test.assertGreater(dot, 0.999, msg=f"Default quaternion does not align +Z with edge direction (dot={dot:.6f})")


def _cable_graph_collision_filter_pairs_impl(test: unittest.TestCase, device):
    """Cable graph: collision filtering should be applied at junctions.

    For a Y-junction (degree-3 node): two pairs are jointed; the remaining sibling pair should also be
    collision-filtered automatically by `add_rod_graph()`'s junction filtering.
    """

    def assert_body_pair_filtered(builder: newton.ModelBuilder, model: newton.Model, body_a: int, body_b: int):
        test.assertIn(body_a, builder.body_shapes)
        test.assertIn(body_b, builder.body_shapes)
        test.assertEqual(len(builder.body_shapes[body_a]), 1, msg="expected one shape per rod body in this test")
        test.assertEqual(len(builder.body_shapes[body_b]), 1, msg="expected one shape per rod body in this test")
        sa = int(builder.body_shapes[body_a][0])
        sb = int(builder.body_shapes[body_b][0])
        pair = (min(sa, sb), max(sa, sb))
        test.assertIn(
            pair, model.shape_collision_filter_pairs, msg=f"missing collision filter pair for bodies {body_a}-{body_b}"
        )

    # Y-junction (3 edges).
    builder = newton.ModelBuilder()
    node_positions = [
        wp.vec3(0.0, 0.0, 0.5),
        wp.vec3(0.25, 0.0, 0.5),
        wp.vec3(-0.125, 0.21650635, 0.5),
        wp.vec3(-0.125, -0.21650635, 0.5),
    ]
    edges = [(0, 1), (0, 2), (0, 3)]
    node_positions_any = node_positions

    rod_bodies, rod_joints = builder.add_rod_graph(
        node_positions=node_positions_any,
        edges=edges,
        radius=0.05,
        cfg=builder.default_shape_cfg.copy(),
        bend_stiffness=0.0,
        bend_damping=0.0,
        stretch_stiffness=1.0e6,
        stretch_damping=0.0,
        label="ut_cable_graph_y_filter",
        wrap_in_articulation=True,
    )
    test.assertEqual(len(rod_bodies), 3)
    test.assertEqual(len(rod_joints), 2)

    builder.color()
    model = builder.finalize(device=device)

    bodies = [int(b) for b in rod_bodies]
    all_pairs = {(min(a, b), max(a, b)) for i, a in enumerate(bodies) for b in bodies[i + 1 :]}

    jointed_pairs: set[tuple[int, int]] = set()
    for j in rod_joints:
        jb = int(j)
        a = int(builder.joint_parent[jb])
        b = int(builder.joint_child[jb])
        jointed_pairs.add((min(a, b), max(a, b)))

    # 1) Jointed pairs must be filtered (from collision_filter_parent=True on the joint).
    for a, b in sorted(jointed_pairs):
        assert_body_pair_filtered(builder, model, a, b)

    # 2) The remaining sibling pair(s) at the junction should also be filtered (from junction filtering).
    sibling_pairs = all_pairs - jointed_pairs
    test.assertEqual(
        len(sibling_pairs), 1, msg=f"expected exactly one non-jointed sibling pair, got {sorted(sibling_pairs)}"
    )
    for a, b in sibling_pairs:
        assert_body_pair_filtered(builder, model, a, b)


def _collect_rigid_body_contact_forces_impl(test: unittest.TestCase, device):
    """VBD rigid contact-force query returns valid per-contact buffers."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 0.5

    # Two overlapping dynamic boxes to guarantee rigid-rigid contact generation.
    b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, label="box0")
    b1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()), mass=1.0, label="box1")
    builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    state0 = model.state()
    contacts = model.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=1)

    dt = 1.0 / 60.0

    # Build contacts on the current state and query them directly.
    # This keeps the test focused on contact-force extraction, not on integration dynamics.
    model.collide(state0, contacts)

    c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = solver.collect_rigid_contact_forces(state0, contacts, dt)
    count = int(c_count.numpy()[0])

    # Buffer lengths must match rigid contact capacity.
    expected_len = int(contacts.rigid_contact_shape0.shape[0])
    test.assertEqual(int(c_b0.shape[0]), expected_len)
    test.assertEqual(int(c_b1.shape[0]), expected_len)
    test.assertEqual(int(c_p0w.shape[0]), expected_len)
    test.assertEqual(int(c_p1w.shape[0]), expected_len)
    test.assertEqual(int(c_f_b1.shape[0]), expected_len)

    # We set up overlapping boxes, so at least one rigid contact should be queryable.
    test.assertGreater(count, 0, msg="Expected at least one rigid-rigid contact")

    b0_np = c_b0.numpy()
    b1_np = c_b1.numpy()
    f_np = c_f_b1.numpy()
    test.assertTrue(np.all(b0_np[:count] >= 0), msg="Invalid body0 ids in active contact range")
    test.assertTrue(np.all(b1_np[:count] >= 0), msg="Invalid body1 ids in active contact range")
    test.assertTrue(np.isfinite(f_np[:count]).all(), msg="Non-finite contact force values in active contact range")

    force_norms = np.linalg.norm(f_np[:count], axis=1)
    test.assertTrue(np.any(force_norms > 1.0e-8), msg="Expected at least one non-zero rigid contact force")


def _cable_world_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: joints with parent=-1 (world) should anchor rod start to a fixed world frame.

    Builds a short cable rod for each joint type (BALL, FIXED, REVOLUTE, PRISMATIC), attaches
    the first capsule to the world with parent=-1, and verifies that the joint constraint
    error stays small under gravity.
    """
    num_elements = 10
    segment_length = 0.05
    rod_radius = 0.01
    z_height = 3.0

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 15

    # World-space attach point for the rod start.
    attach_pos = wp.vec3(0.0, 0.0, z_height)

    joint_configs = [
        ("BALL", "ball"),
        ("FIXED", "fixed"),
        ("REVOLUTE", "revolute"),
        ("PRISMATIC", "prismatic"),
        ("D6", "d6"),
    ]

    for joint_label, joint_kind in joint_configs:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 1.0

        points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=z_height)
        # Shift cable so its first point starts at the attach position.
        offset = attach_pos - points[0]
        points = [p + offset for p in points]

        rod_bodies, rod_joints = builder.add_rod(
            positions=points,
            quaternions=edge_q,
            radius=rod_radius,
            bend_stiffness=1.0e-1,
            bend_damping=1.0e-2,
            stretch_stiffness=1.0e9,
            stretch_damping=0.0,
            wrap_in_articulation=False,
            label=f"test_cable_world_{joint_kind}",
        )

        child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
        parent_xform = wp.transform(attach_pos, wp.quat_identity())
        child_xform = wp.transform(child_anchor_local, wp.quat_identity())

        if joint_kind == "ball":
            j = builder.add_joint_ball(
                parent=-1,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
            )
        elif joint_kind == "fixed":
            j = builder.add_joint_fixed(
                parent=-1,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
            )
        elif joint_kind == "revolute":
            j = builder.add_joint_revolute(
                parent=-1,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
                axis=wp.vec3(0.0, 1.0, 0.0),
            )
        elif joint_kind == "prismatic":
            j = builder.add_joint_prismatic(
                parent=-1,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
                axis=wp.vec3(1.0, 0.0, 0.0),
            )
        elif joint_kind == "d6":
            JointDofConfig = newton.ModelBuilder.JointDofConfig
            j = builder.add_joint_d6(
                parent=-1,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
                linear_axes=[JointDofConfig(axis=(1, 0, 0))],
                angular_axes=[JointDofConfig(axis=(0, 1, 0))],
            )

        builder.add_articulation([*rod_joints, j])
        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))

        state0 = model.state()
        state1 = model.state()
        control = model.control()
        contacts = model.contacts()

        solver = newton.solvers.SolverVBD(model, iterations=10)

        for _step in range(num_steps):
            for _substep in range(sim_substeps):
                model.collide(state0, contacts)
                solver.step(state0, state1, control, contacts, dt=sim_dt)
                state0, state1 = state1, state0

        final_q = state0.body_q.numpy()
        test.assertTrue(
            np.isfinite(final_q).all(),
            msg=f"{joint_label} world joint: non-finite body transforms",
        )

        if joint_kind == "ball":
            err = _compute_ball_joint_anchor_error(model, state0.body_q, j)
            test.assertLess(
                err,
                1.0e-3,
                msg=f"{joint_label} world joint: anchor error {err:.6f} m > 1e-3 m",
            )
        elif joint_kind == "fixed":
            pos_err, ang_err = _compute_fixed_joint_frame_error(model, state0.body_q, j)
            test.assertLess(
                pos_err,
                1.0e-3,
                msg=f"{joint_label} world joint: pos error {pos_err:.6f} m > 1e-3 m",
            )
            test.assertLess(
                ang_err,
                2.0e-2,
                msg=f"{joint_label} world joint: ang error {ang_err:.4f} rad > 2e-2 rad",
            )
        elif joint_kind == "revolute":
            pos_err, ang_perp_err, _ = _compute_revolute_joint_error(model, state0.body_q, j)
            test.assertLess(
                pos_err,
                1.0e-3,
                msg=f"{joint_label} world joint: pos error {pos_err:.6f} m > 1e-3 m",
            )
            test.assertLess(
                ang_perp_err,
                2.0e-2,
                msg=f"{joint_label} world joint: perp ang error {ang_perp_err:.4f} rad > 2e-2 rad",
            )
        elif joint_kind == "prismatic":
            pos_perp_err, ang_err, _ = _compute_prismatic_joint_error(model, state0.body_q, j)
            test.assertLess(
                pos_perp_err,
                1.0e-3,
                msg=f"{joint_label} world joint: perp pos error {pos_perp_err:.6f} m > 1e-3 m",
            )
            test.assertLess(
                ang_err,
                2.0e-2,
                msg=f"{joint_label} world joint: ang error {ang_err:.4f} rad > 2e-2 rad",
            )
        elif joint_kind == "d6":
            pos_perp_err, ang_perp_err, _, _ = _compute_d6_joint_error(model, state0.body_q, j)
            test.assertLess(
                pos_perp_err,
                1.0e-3,
                msg=f"{joint_label} world joint: perp pos error {pos_perp_err:.6f} m > 1e-3 m",
            )
            test.assertLess(
                ang_perp_err,
                2.0e-2,
                msg=f"{joint_label} world joint: perp ang error {ang_perp_err:.4f} rad > 2e-2 rad",
            )


def _joint_enabled_toggle_impl(test: unittest.TestCase, device):
    """VBD: disabling a joint lets the cable detach; re-enabling pulls it back.

    Uses a BALL joint between a kinematic anchor sphere and a short cable (rod).
    The early-return guard in evaluate_joint_force_hessian / update_duals_joint
    is joint-type-agnostic, so one joint type covers all.
    """
    builder = newton.ModelBuilder()

    # Kinematic anchor sphere at height.
    anchor_pos = wp.vec3(0.0, 0.0, 3.0)
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    # Short cable (rod) hanging below the anchor.
    num_elements = 6
    segment_length = 0.05
    rod_radius = 0.01
    attach_offset = anchor_radius + rod_radius
    cable_start = anchor_pos + wp.vec3(0.0, 0.0, -attach_offset)

    rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
        start=cable_start,
        direction=wp.vec3(0.0, 0.0, -1.0),
        length=float(num_elements) * segment_length,
        num_segments=num_elements,
        twist_total=0.0,
    )

    rod_bodies, rod_joints = builder.add_rod(
        positions=rod_points,
        quaternions=rod_quats,
        radius=rod_radius,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        wrap_in_articulation=False,
        label="test_joint_enabled_cable",
    )

    # BALL joint: anchor sphere → first rod body.
    parent_anchor_local = wp.vec3(0.0, 0.0, -attach_offset)
    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
    j = builder.add_joint_ball(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=wp.transform(parent_anchor_local, wp.quat_identity()),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints, j])

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=10)

    sim_dt = 1.0 / 60.0 / 4

    def step_n(n):
        nonlocal state0, state1
        for _ in range(n):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    # Phase 1: joint enabled (default) — cable stays attached to anchor.
    step_n(10)
    err_connected = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertLess(err_connected, 1.0e-3, f"Phase 1 (enabled): pos error {err_connected:.6f} m > 1e-3")

    # Phase 2: disable joint — cable detaches and falls under gravity.
    enabled_np = model.joint_enabled.numpy()
    enabled_np[j] = False
    model.joint_enabled.assign(wp.array(enabled_np, dtype=bool, device=device))
    step_n(10)
    err_disabled = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertGreater(
        err_disabled, 5.0e-3, f"Phase 2 (disabled): pos error {err_disabled:.6f} m — cable did not separate"
    )

    # Phase 3: re-enable joint — solver pulls cable back toward anchor.
    enabled_np[j] = True
    model.joint_enabled.assign(wp.array(enabled_np, dtype=bool, device=device))
    step_n(10)
    err_reenabled = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertLess(
        err_reenabled,
        err_disabled,
        f"Phase 3 (re-enabled): pos error {err_reenabled:.6f} m did not decrease from {err_disabled:.6f} m",
    )


class TestCable(unittest.TestCase):
    pass


add_function_test(
    TestCable,
    "test_joint_enabled_toggle",
    _joint_enabled_toggle_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_chain_connectivity",
    _cable_chain_connectivity_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_loop_connectivity",
    _cable_loop_connectivity_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_sagging_and_stability",
    _cable_sagging_and_stability_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_bend_stiffness",
    _cable_bend_stiffness_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_twist_response",
    _cable_twist_response_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_two_layer_cable_pile_collision",
    _two_layer_cable_pile_collision_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_ball_joint_attaches_rod_endpoint",
    _cable_ball_joint_attaches_rod_endpoint_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_fixed_joint_attaches_rod_endpoint",
    _cable_fixed_joint_attaches_rod_endpoint_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_revolute_joint_attaches_rod_endpoint",
    _cable_revolute_joint_attaches_rod_endpoint_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_revolute_drive_tracks_target",
    _cable_revolute_drive_tracks_target_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_revolute_drive_limit",
    _cable_revolute_drive_limit_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_prismatic_joint_attaches_rod_endpoint",
    _cable_prismatic_joint_attaches_rod_endpoint_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_prismatic_drive_tracks_target",
    _cable_prismatic_drive_tracks_target_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_prismatic_drive_limit",
    _cable_prismatic_drive_limit_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_d6_joint_attaches_rod_endpoint",
    _cable_d6_joint_attaches_rod_endpoint_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_d6_joint_all_locked",
    _cable_d6_joint_all_locked_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_d6_joint_locked_x",
    _cable_d6_joint_locked_x_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_d6_drive_tracks_target",
    _cable_d6_drive_tracks_target_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_d6_drive_limit",
    _cable_d6_drive_limit_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_kinematic_gripper_picks_capsule",
    _cable_kinematic_gripper_picks_capsule_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_graph_y_junction_spanning_tree",
    _cable_graph_y_junction_spanning_tree_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_rod_ring_closed_in_articulation",
    _cable_rod_ring_closed_in_articulation_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_graph_default_quat_aligns_z",
    _cable_graph_default_quat_aligns_z_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_graph_collision_filter_pairs",
    _cable_graph_collision_filter_pairs_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_collect_rigid_body_contact_forces",
    _collect_rigid_body_contact_forces_impl,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_world_joint_attaches_rod_endpoint",
    _cable_world_joint_attaches_rod_endpoint_impl,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
