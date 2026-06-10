# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    CableBendTwistMeasure,
    _bishop_transport_quat,
    _cable_bend_twist_directional_derivatives_from_measure,
    _finite_curvature_binormal,
    _finite_curvature_binormal_derivative,
    _normalize_with_fallback,
    _project_perp,
    _transported_twist_angle_derivative_from_measure,
    _transported_twist_angle_from_material_axes,
    compute_cable_dahl_parameters,
    evaluate_angular_constraint_force_hessian,
    evaluate_cable_bend_twist_force_hessian_z,
)
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


# -----------------------------------------------------------------------------
# Device-side time kernels (for CUDA graph capture with kinematic bodies)
# -----------------------------------------------------------------------------


@wp.kernel
def _advance_time(sim_time: wp.array[float], dt: float):
    sim_time[0] = sim_time[0] + dt


@wp.kernel
def _set_kinematic_sinusoidal_pose(
    body_id: wp.int32,
    sim_time: wp.array[float],
    anchor_z: float,
    x_amp: float,
    x_freq: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    t = wp.float32(sim_time[0])
    dx = x_amp * wp.sin(x_freq * t)
    body_q[body_id] = wp.transform(wp.vec3(dx, 0.0, anchor_z), wp.quat_identity())
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _set_kinematic_sinusoidal_xy_pose(
    body_id: wp.int32,
    sim_time: wp.array[float],
    anchor_z: float,
    x_amp: float,
    x_freq: float,
    y_amp: float,
    y_freq: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    t = wp.float32(sim_time[0])
    dx = x_amp * wp.sin(x_freq * t)
    dy = y_amp * wp.sin(y_freq * t)
    body_q[body_id] = wp.transform(wp.vec3(dx, dy, anchor_z), wp.quat_identity())
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _set_kinematic_d6_pose(
    body_id: wp.int32,
    sim_time: wp.array[float],
    anchor_z: float,
    x_amp: float,
    x_freq: float,
    ang_amp: float,
    ang_freq: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    t = wp.float32(sim_time[0])
    dx = x_amp * wp.sin(x_freq * t)
    ang_y = ang_amp * wp.sin(ang_freq * t)
    half = ang_y * 0.5
    q_anchor = wp.quat(0.0, wp.sin(half), 0.0, wp.cos(half))
    body_q[body_id] = wp.transform(wp.vec3(dx, 0.0, anchor_z), q_anchor)
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _apply_y_axis_torque(
    body_id: wp.int32,
    sim_time: wp.array[float],
    tau_amp: float,
    tau_freq: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Write an oscillating world-Y torque to body_f[body_id] (no linear force)."""
    t = wp.float32(sim_time[0])
    tau_y = tau_amp * wp.sin(tau_freq * t)
    body_f[body_id] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, tau_y, 0.0),
    )


@wp.kernel
def _set_kinematic_linear_rotating_pose(
    body_id: wp.int32,
    sim_time: wp.array[float],
    anchor_z: float,
    velocity_x: float,
    angular_velocity_z: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    t = wp.float32(sim_time[0])
    x_kin = velocity_x * t
    angle_z = angular_velocity_z * t
    half = angle_z * 0.5
    q_kin = wp.quat(0.0, 0.0, wp.sin(half), wp.cos(half))
    body_q[body_id] = wp.transform(wp.vec3(x_kin, 0.0, anchor_z), q_kin)
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _drive_gripper_boxes_graph_kernel(
    ramp_time: float,
    sim_time: wp.array[float],
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
    """Kinematically move two gripper boxes using device-side time for graph capture."""
    tid = wp.tid()
    b = body_ids[tid]
    sgn = signs[tid]
    rot = anchor_q
    center = anchor_p + wp.quat_rotate(rot, wp.vec3(0.0, 0.0, seg_half_len))
    t = wp.float32(sim_time[0])
    pull_end_time = wp.float32(pull_start_time + pull_ramp_time)
    t_eff = wp.min(t, pull_end_time)
    u = wp.clamp(t_eff / wp.float32(ramp_time), 0.0, 1.0)
    offset_mag = (1.0 - u) * initial_offset_mag + u * target_offset_mag
    tp = wp.clamp((t_eff - wp.float32(pull_start_time)) / wp.float32(pull_ramp_time), 0.0, 1.0)
    pull = wp.float32(pull_distance) * tp
    pull_dir = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    local_off = wp.vec3(0.0, sgn * offset_mag, 0.0)
    pos = center + pull_dir * pull + wp.quat_rotate(rot, local_off)
    body_q[b] = wp.transform(pos, rot)


# -----------------------------------------------------------------------------
# Graph-capture helper
# -----------------------------------------------------------------------------


def _run_sim_loop(simulate_fn, num_steps, device):
    """Run a simulation loop with optional CUDA graph capture.

    ``simulate_fn()`` must be graph-capturable: no host-side branching, no
    scalar time arguments — use device-side ``sim_time`` arrays and the
    ``_advance_time`` kernel instead.
    If it swaps ping-pong state buffers, each call must leave those buffers in
    the same orientation it received them, e.g. by performing an even number of
    ``state0, state1 = state1, state0`` swaps.
    """
    use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    graph = None
    if use_cuda_graph:
        with wp.ScopedCapture(device) as capture:
            simulate_fn()
        graph = capture.graph

    for _ in range(num_steps):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            simulate_fn()


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
    bend_stiffness: float = 5.0e1,
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

    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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

    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
    bend_values = [5.0e1, 5.0e2, 5.0e3]
    segment_length = 0.2
    num_links = 10

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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

    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=5.0e4,
        bend_damping=0.0,
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
    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
      - High bend stiffness (2.0e4) to maintain straightness

    After settling under gravity and contact, bodies should cluster into two
    vertical bands:
      - bottom layer: between ground (z=0) and one cable width,
      - top layer: between one and two cable widths,
    up to a small margin for numerical tolerance and contact compliance.
    """
    builder = newton.ModelBuilder()

    # Contact material (stiff contacts, noticeable friction)
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
    bend_stiffness = 2.0e4

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

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    err = _compute_ball_joint_anchor_error(model, state0.body_q, j_ball)
    test.assertLess(err, 1.0e-3, f"BALL joint: final anchor error {err:.6f} m > 1e-3 m")

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    pos_err_x, ang_err_x = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed_x)
    test.assertLess(pos_err_x, 1.0e-3, f"FIXED joint (X): pos error {pos_err_x:.6f}")
    test.assertLess(ang_err_x, 2.0e-2, f"FIXED joint (X): ang error {ang_err_x:.4f}")

    pos_err_y, ang_err_y = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed_y)
    test.assertLess(pos_err_y, 1.0e-3, f"FIXED joint (Y): pos error {pos_err_y:.6f}")
    test.assertLess(ang_err_y, 2.0e-2, f"FIXED joint (Y): ang error {ang_err_y:.4f}")

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    pos_err_x, ang_perp_err_x, _ = _compute_revolute_joint_error(model, state0.body_q, j_revolute_x)
    test.assertLess(pos_err_x, 1.0e-3, f"REVOLUTE joint (X): pos error {pos_err_x:.6f}")
    test.assertLess(ang_perp_err_x, 2.0e-2, f"REVOLUTE joint (X): ang perp error {ang_perp_err_x:.4f}")

    pos_err_y, ang_perp_err_y, _ = _compute_revolute_joint_error(model, state0.body_q, j_revolute_y)
    test.assertLess(pos_err_y, 1.0e-3, f"REVOLUTE joint (Y): pos error {pos_err_y:.6f}")
    test.assertLess(ang_perp_err_y, 2.0e-2, f"REVOLUTE joint (Y): ang perp error {ang_perp_err_y:.4f}")

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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
    tp = control.joint_target_q.numpy()
    tp[dof_idx] = target_angle
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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

    tp = control.joint_target_q.numpy()
    tp[dof_idx] = target_angle
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_xy_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    0.04,
                    2.0,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    pos_perp_err, ang_err, _c_along = _compute_prismatic_joint_error(model, state0.body_q, j_prismatic)
    test.assertLess(pos_perp_err, 1.0e-3, "PRISMATIC joint: perpendicular position error too large")
    test.assertLess(ang_err, 2.0e-2, "PRISMATIC joint: angular error too large")

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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
    tp = control.joint_target_q.numpy()
    tp[dof_idx] = target_displacement
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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

    tp = control.joint_target_q.numpy()
    tp[dof_idx] = target_displacement
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    """Cable VBD: D6 joint (free linear X + free angular Y), locked DOFs stay locked
    and free DOFs respond to their drivers.

    Vertical cable hanging -Z from a kinematic anchor. D6 joint with 1 free linear
    axis (X) and 1 free angular axis (Y) in the joint parent anchor frame.
    For -Z cables the parent frame rotates +Z to -Z (180 deg about Y), so
    joint-frame X maps to world -X and Y stays world Y.

    Free linear X is driven by the anchor X oscillation, coupled through the
    locked angular X/Z constraints. Free angular Y is driven by an external
    oscillating world-Y torque applied to rod[0]; the locked angular X/Z resist
    anything other than world-Y rotation, leaving the torque to spin the free Y.
    Gravity in -Z stresses the locked Z linear constraint.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    rod0_id = wp.int32(rod_bodies[0])
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_d6_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    0.2,
                    2.0,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            wp.launch(
                _apply_y_axis_torque,
                dim=1,
                inputs=[rod0_id, sim_time_arr, 1.0e-2, 2.0, state0.body_f],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    pos_perp_err, ang_perp_err, _d_along, _rot_free = _compute_d6_joint_error(model, state0.body_q, j_d6)
    test.assertLess(pos_perp_err, 1.0e-3, "D6 joint: perpendicular position error too large")
    test.assertLess(ang_perp_err, 2.0e-2, "D6 joint: perpendicular angular error too large")

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

    # Free linear X freedom: anchor X oscillation couples through the locked
    # angular constraints to slide the rod along the free X axis.
    # Free angular Y freedom: an oscillating world-Y torque on rod[0] rotates
    # the rod about the free Y axis (locked X/Z angular axes resist anything else).
    _, _, d_along, rot_free = _compute_d6_joint_error(model, state0.body_q, j_d6)
    test.assertGreater(
        abs(d_along),
        0.005,
        msg=f"D6 free linear X not exercised: |d_along|={abs(d_along):.4f} m",
    )
    test.assertGreater(
        rot_free,
        0.01,
        msg=f"D6 free angular Y not exercised: rot_free={rot_free:.4f} rad",
    )


def _cable_d6_joint_all_locked_impl(test: unittest.TestCase, device):
    """Cable VBD: D6 joint with all DOFs locked should behave like a fixed joint.

    Vertical cable hanging -Z from a kinematic anchor. D6 joint with no free axes
    (lin_axes=[], ang_axes=[]). Anchor oscillates in X; the cable endpoint must
    follow exactly, matching the lock_xyz config in example_cable_d6_joints.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

    pos_err, ang_err = _compute_fixed_joint_frame_error(model, state0.body_q, j_d6)
    test.assertLess(pos_err, 1.0e-3, "D6 all-locked: position error too large")
    test.assertLess(ang_err, 2.0e-2, "D6 all-locked: angular error too large")

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
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

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_sinusoidal_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    0.05,
                    1.5,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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
    tp = control.joint_target_q.numpy()
    tp[lin_dof_idx] = target_displacement
    tp[ang_dof_idx] = target_angle
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
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

    tp = control.joint_target_q.numpy()
    tp[qd_s] = target_displacement
    tp[qd_s + 1] = target_angle
    control.joint_target_q = wp.array(tp, dtype=float, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 30

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4

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
    # AVBD friction tracking under the surface-anchor moment arm needs either
    # dt ≲ 4 ms (substeps ≥ 4) or rigid_avbd_contact_alpha ≲ 0.5; both stay
    # well inside the 1 cm tolerance below.
    sim_substeps = 4
    sim_dt = frame_dt / sim_substeps

    # Record initial pose
    q0 = state0.body_q.numpy()
    capsule_z0 = float(q0[capsule_body, 2])

    # Run a fixed number of frames for a lightweight regression test.
    num_frames = 100

    sim_time_arr = wp.zeros(1, dtype=float, device=device)

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()

            wp.launch(
                kernel=_drive_gripper_boxes_graph_kernel,
                dim=2,
                inputs=[
                    float(ramp_time),
                    sim_time_arr,
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
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_frames, device)

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
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=5.0e2,
        bend_damping=1.0e-2,
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
    sim_substeps = 6
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    contacts = model.contacts()

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            model.collide(state0, contacts)
            state0.clear_forces()
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

    qf = state0.body_q.numpy()
    test.assertTrue(np.isfinite(qf).all(), "Non-finite body transforms detected in Y-junction graph simulation")

    z_min = float(np.min(qf[rod_bodies, 2]))
    test.assertLess(z_min, z_init_min - 0.01, msg="Y-junction did not fall noticeably under gravity")
    _assert_bodies_above_ground(test, qf, rod_bodies, context="y-junction", margin=0.25 * cable_width)


def _cable_eval_fk_preserves_body_state_impl(test: unittest.TestCase, device):
    """eval_fk should not reconstruct CABLE child poses from unsupported joint coordinates."""
    builder = newton.ModelBuilder()
    rod_bodies, rod_joints = builder.add_rod_graph(
        node_positions=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.5, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
        ],
        edges=[(0, 1), (1, 2)],
        radius=0.01,
        wrap_in_articulation=True,
        label="ut_cable_eval_fk",
    )
    test.assertEqual(len(rod_bodies), 2)
    test.assertEqual(len(rod_joints), 1)

    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    joint_types = model.joint_type.numpy()
    test.assertTrue(np.all(joint_types == int(newton.JointType.CABLE)), msg="expected only CABLE joints")

    child_body = int(rod_bodies[1])

    body_q = state.body_q.numpy().copy()
    body_q[child_body, 0] += 1.0
    body_q[child_body, 2] -= 0.7
    state.body_q.assign(body_q)

    body_qd = state.body_qd.numpy().copy()
    body_qd[child_body] = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.6], dtype=body_qd.dtype)
    state.body_qd.assign(body_qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    np.testing.assert_allclose(
        state.body_q.numpy()[child_body],
        body_q[child_body],
        rtol=0.0,
        atol=1.0e-6,
        err_msg="eval_fk should preserve VBD-owned CABLE body transform",
    )
    np.testing.assert_allclose(
        state.body_qd.numpy()[child_body],
        body_qd[child_body],
        rtol=0.0,
        atol=1.0e-6,
        err_msg="eval_fk should preserve VBD-owned CABLE body velocity",
    )


def _cable_rod_ring_closed_in_articulation_impl(test: unittest.TestCase, device):
    """Closed ring via add_rod(closed=True) should build and simulate."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
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
        bend_stiffness=7.0e2,
        bend_damping=1.0e-2,
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
    sim_substeps = 6
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    q_init = state0.body_q.numpy()
    z_init_min = float(np.min(q_init[rod_bodies, 2]))

    contacts = model.contacts()

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            state0.clear_forces()
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    _run_sim_loop(simulate, num_steps, device)

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
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.5

    # Two overlapping dynamic boxes - initial overlap guarantees contact.
    b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, label="box0")
    b1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()), mass=1.0, label="box1")
    builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    state0 = model.state()
    state1 = model.state()
    contacts = model.contacts()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=2)

    dt = 1.0 / 60.0

    # Collide + step so ALM state (penalty_k, lambda) gets populated.
    model.collide(state0, contacts)
    body_q_prev_snapshot = wp.clone(solver.body_q_prev)
    solver.step(state0, state1, control, contacts, dt)

    c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = solver.collect_rigid_contact_forces(
        state1.body_q, body_q_prev_snapshot, contacts, dt
    )
    count = int(c_count.numpy()[0])

    # Buffer lengths must match rigid contact capacity.
    expected_len = int(contacts.rigid_contact_shape0.shape[0])
    test.assertEqual(int(c_b0.shape[0]), expected_len)
    test.assertEqual(int(c_b1.shape[0]), expected_len)
    test.assertEqual(int(c_p0w.shape[0]), expected_len)
    test.assertEqual(int(c_p1w.shape[0]), expected_len)
    test.assertEqual(int(c_f_b1.shape[0]), expected_len)

    # Two overlapping boxes, so at least one rigid contact should be queryable.
    test.assertGreater(count, 0, msg="Expected at least one rigid contact")

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
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.0

        points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=z_height)
        # Shift cable so its first point starts at the attach position.
        offset = attach_pos - points[0]
        points = [p + offset for p in points]

        rod_bodies, rod_joints = builder.add_rod(
            positions=points,
            quaternions=edge_q,
            radius=rod_radius,
            bend_stiffness=2.0e0,
            bend_damping=1.0e-2,
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

        def simulate(_model=model, _solver=solver, _control=control, _contacts=contacts):
            nonlocal state0, state1
            for _substep in range(sim_substeps):
                _model.collide(state0, _contacts)
                _solver.step(state0, state1, _control, _contacts, dt=sim_dt)
                state0, state1 = state1, state0

        _run_sim_loop(simulate, num_steps, device)

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
        bend_stiffness=2.0e2,
        bend_damping=1.0e-2,
        wrap_in_articulation=False,
        label="test_joint_enabled_cable",
    )

    # BALL joint: anchor sphere -> first rod body.
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

    # Phase 1: joint enabled (default) - cable stays attached to anchor.
    step_n(10)
    err_connected = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertLess(err_connected, 1.0e-3, f"Phase 1 (enabled): pos error {err_connected:.6f} m > 1e-3")

    # Phase 2: disable joint - cable detaches and falls under gravity.
    enabled_np = model.joint_enabled.numpy()
    enabled_np[j] = False
    model.joint_enabled.assign(wp.array(enabled_np, dtype=bool, device=device))
    step_n(10)
    err_disabled = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertGreater(
        err_disabled, 5.0e-3, f"Phase 2 (disabled): pos error {err_disabled:.6f} m - cable did not separate"
    )

    # Phase 3: re-enable joint - solver pulls cable back toward anchor.
    enabled_np[j] = True
    model.joint_enabled.assign(wp.array(enabled_np, dtype=bool, device=device))
    step_n(10)
    err_reenabled = _compute_ball_joint_anchor_error(model, state0.body_q, j)
    test.assertLess(
        err_reenabled,
        err_disabled,
        f"Phase 3 (re-enabled): pos error {err_reenabled:.6f} m did not decrease from {err_disabled:.6f} m",
    )


def _cable_fixed_joint_tracks_moving_kinematic_impl(test: unittest.TestCase, device):
    """Cable VBD: fixed joint tracks a translating-and-rotating kinematic body.

    A short cable is attached via a hard FIXED joint to a kinematic body that
    translates along +X and rotates about Z.  Verifies that both positional and
    angular joint errors stay bounded every substep, exercising the linear and
    angular C0 snapshot paths against a moving kinematic parent.
    """
    builder = newton.ModelBuilder()

    anchor_pos = wp.vec3(0.0, 0.0, 1.0)
    anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()))
    builder.add_shape_sphere(anchor, radius=0.05)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    num_elements = 3
    segment_length = 0.05
    rod_radius = 0.01
    attach_offset = wp.float32(0.05 + rod_radius)

    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=float(anchor_pos[2]))
    parent_anchor_local = wp.vec3(attach_offset, 0.0, 0.0)
    anchor_world_attach = anchor_pos + wp.vec3(float(attach_offset), 0.0, 0.0)
    offset = anchor_world_attach - points[0]
    points = [p + offset for p in points]

    rod_bodies, rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=rod_radius,
        bend_stiffness=2.0e0,
        bend_damping=1.0e-2,
        wrap_in_articulation=False,
        label="test_kinematic_track",
    )

    j_fixed = builder.add_joint_fixed(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=wp.transform(parent_anchor_local, edge_q[0]),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints, j_fixed])

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverVBD(model, iterations=20)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_frames = 10
    velocity_x = 0.3  # m/s
    angular_velocity_z = 1.0  # rad/s

    pos_tol = 1.5e-2
    ang_tol = 5.0e-2

    sim_time_arr = wp.zeros(1, dtype=float, device=device)
    anchor_id = wp.int32(anchor)
    anchor_z = float(anchor_pos[2])

    def simulate():
        nonlocal state0, state1
        for _substep in range(sim_substeps):
            wp.launch(
                _set_kinematic_linear_rotating_pose,
                dim=1,
                inputs=[
                    anchor_id,
                    sim_time_arr,
                    anchor_z,
                    velocity_x,
                    angular_velocity_z,
                    state0.body_q,
                    state0.body_qd,
                ],
                device=device,
            )
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0
            wp.launch(_advance_time, dim=1, inputs=[sim_time_arr, sim_dt], device=device)

    _run_sim_loop(simulate, num_frames, device)

    pos_err, ang_err = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed)
    test.assertLess(
        pos_err,
        pos_tol,
        f"Fixed joint kinematic tracking: pos error {pos_err:.6f} m against moving kinematic body",
    )
    test.assertLess(
        ang_err,
        ang_tol,
        f"Fixed joint kinematic tracking: ang error {ang_err:.4f} rad against rotating kinematic body",
    )

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms in kinematic tracking test")


# -----------------------------------------------------------------------------
# Split cable bend/twist verification helpers
# -----------------------------------------------------------------------------


@wp.func
def _build_cable_material_basis(tangent_axis: wp.vec3) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Test reference basis for arbitrary tangent axes."""
    twist_axis = tangent_axis
    t_len = wp.length(twist_axis)
    if t_len <= 1.0e-8:
        twist_axis = wp.vec3(0.0, 0.0, 1.0)
    else:
        twist_axis = twist_axis / t_len

    if twist_axis[2] < -0.9999999:
        bend_axis0 = wp.vec3(1.0, 0.0, 0.0)
        bend_axis1 = wp.vec3(0.0, -1.0, 0.0)
    else:
        inv = 1.0 / (1.0 + twist_axis[2])
        xy = -twist_axis[0] * twist_axis[1] * inv
        bend_axis0 = wp.vec3(1.0 - twist_axis[0] * twist_axis[0] * inv, xy, -twist_axis[0])
        bend_axis1 = wp.vec3(xy, 1.0 - twist_axis[1] * twist_axis[1] * inv, -twist_axis[1])
    return bend_axis0, bend_axis1, twist_axis


@wp.func
def _build_cable_bend_twist_projectors(tangent_axis: wp.vec3) -> tuple[wp.mat33, wp.mat33]:
    """Test reference bend/twist projectors for arbitrary tangent axes."""
    _bend0, _bend1, twist_axis = _build_cable_material_basis(tangent_axis)
    P_twist = wp.outer(twist_axis, twist_axis)
    return wp.identity(3, float) - P_twist, P_twist


@wp.func
def _build_cable_stretch_shear_projectors(q_wp: wp.quat, tangent_axis: wp.vec3) -> tuple[wp.mat33, wp.mat33]:
    """Test reference stretch/shear projectors for arbitrary tangent axes."""
    _bend0, _bend1, twist_axis = _build_cable_material_basis(tangent_axis)
    t_world = wp.quat_rotate(q_wp, twist_axis)
    t_world = _normalize_with_fallback(t_world, wp.vec3(0.0, 0.0, 1.0))
    P_stretch = wp.outer(t_world, t_world)
    return P_stretch, wp.identity(3, float) - P_stretch


@wp.func
def _cable_parent_to_material_basis(
    v: wp.vec3, bend_axis0: wp.vec3, bend_axis1: wp.vec3, twist_axis: wp.vec3
) -> wp.vec3:
    """Resolve a parent-frame vector into the test reference cable basis."""
    return wp.vec3(wp.dot(v, bend_axis0), wp.dot(v, bend_axis1), wp.dot(v, twist_axis))


@wp.func
def _cable_material_basis_to_parent(
    v: wp.vec3, bend_axis0: wp.vec3, bend_axis1: wp.vec3, twist_axis: wp.vec3
) -> wp.vec3:
    """Resolve a test reference cable-basis vector into the parent frame."""
    return v[0] * bend_axis0 + v[1] * bend_axis1 + v[2] * twist_axis


@wp.func
def _cable_material_basis_hessian_to_parent(
    c: wp.vec3, bend_axis0: wp.vec3, bend_axis1: wp.vec3, twist_axis: wp.vec3
) -> wp.mat33:
    """Resolve diagonal test reference cable-basis tangents into parent frame."""
    return (
        c[0] * wp.outer(bend_axis0, bend_axis0)
        + c[1] * wp.outer(bend_axis1, bend_axis1)
        + c[2] * wp.outer(twist_axis, twist_axis)
    )


@wp.func
def _measure_cable_bend_twist_with_basis(
    q_wp: wp.quat,
    q_wc: wp.quat,
    bend_axis0: wp.vec3,
    bend_axis1: wp.vec3,
    twist_axis: wp.vec3,
) -> CableBendTwistMeasure:
    """Test reference arbitrary-axis bend/twist measurement."""
    t0 = _normalize_with_fallback(wp.quat_rotate(q_wp, twist_axis), wp.vec3(0.0, 0.0, 1.0))
    t1 = _normalize_with_fallback(wp.quat_rotate(q_wc, twist_axis), t0)
    m0 = _normalize_with_fallback(
        _project_perp(wp.quat_rotate(q_wp, bend_axis0), t0),
        wp.quat_rotate(q_wp, bend_axis1),
    )
    m1 = _normalize_with_fallback(
        _project_perp(wp.quat_rotate(q_wc, bend_axis0), t1),
        wp.quat_rotate(q_wc, bend_axis1),
    )

    measure = CableBendTwistMeasure()
    measure.t0 = t0
    measure.t1 = t1
    measure.m0 = m0
    measure.m1 = m1
    measure.twist = _transported_twist_angle_from_material_axes(t0, t1, m0, m1, m0)
    measure.kb_world = _finite_curvature_binormal(t0, t1, m0)
    return measure


@wp.func
def _measure_cable_bend_twist(q_wp: wp.quat, q_wc: wp.quat, tangent_axis: wp.vec3) -> CableBendTwistMeasure:
    """Test reference arbitrary-axis bend/twist measurement."""
    bend_axis0, bend_axis1, twist_axis = _build_cable_material_basis(tangent_axis)
    return _measure_cable_bend_twist_with_basis(q_wp, q_wc, bend_axis0, bend_axis1, twist_axis)


@wp.func
def _geometric_cable_strain_directional_derivative_from_measure(
    q_wp: wp.quat,
    P_bend: wp.mat33,
    twist_axis: wp.vec3,
    measure: CableBendTwistMeasure,
    omega_world: wp.vec3,
    is_parent: bool,
) -> wp.vec3:
    """Test reference derivative of [bend0, bend1, twist]."""
    d_bend_local, d_twist = _cable_bend_twist_directional_derivatives_from_measure(
        q_wp, measure, omega_world, is_parent
    )
    return P_bend * d_bend_local + d_twist * twist_axis


@wp.func
def _cable_bend_twist_jacobian_from_measure(
    q_wp: wp.quat,
    P_bend: wp.mat33,
    twist_axis: wp.vec3,
    measure: CableBendTwistMeasure,
    is_parent: bool,
) -> wp.mat33:
    """Test reference Jacobian of arbitrary-axis bend/twist strain."""
    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)
    j0 = _geometric_cable_strain_directional_derivative_from_measure(q_wp, P_bend, twist_axis, measure, e0, is_parent)
    j1 = _geometric_cable_strain_directional_derivative_from_measure(q_wp, P_bend, twist_axis, measure, e1, is_parent)
    j2 = _geometric_cable_strain_directional_derivative_from_measure(q_wp, P_bend, twist_axis, measure, e2, is_parent)
    return wp.outer(j0, e0) + wp.outer(j1, e1) + wp.outer(j2, e2)


@wp.func
def _compute_geometric_cable_kb_twist(q_wp: wp.quat, q_wc: wp.quat, tangent_axis: wp.vec3) -> tuple[wp.vec3, float]:
    measure = _measure_cable_bend_twist(q_wp, q_wc, tangent_axis)
    return measure.kb_world, measure.twist


@wp.func
def _assemble_geometric_cable_kappa(
    q_wp: wp.quat,
    tangent_axis: wp.vec3,
    kb_now_world: wp.vec3,
    twist_now: float,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Test reference arbitrary-axis [bend0, bend1, twist] residual."""
    bend_now_local = wp.quat_rotate(wp.quat_inverse(q_wp), kb_now_world)
    _bend0, _bend1, twist_axis = _build_cable_material_basis(tangent_axis)
    P_bend, _P_twist = _build_cable_bend_twist_projectors(tangent_axis)
    return P_bend * (bend_now_local - kb_rest_local) + (twist_now - twist_rest) * twist_axis


@wp.func
def _compute_geometric_cable_kappa(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    tangent_axis: wp.vec3,
) -> wp.vec3:
    """Test reference arbitrary-axis geometric cable strain residual."""
    kb_now_world, twist_now = _compute_geometric_cable_kb_twist(q_wp, q_wc, tangent_axis)
    kb_rest_world, twist_rest = _compute_geometric_cable_kb_twist(q_wp_rest, q_wc_rest, tangent_axis)
    kb_rest_local = wp.quat_rotate(wp.quat_inverse(q_wp_rest), kb_rest_world)
    return _assemble_geometric_cable_kappa(q_wp, tangent_axis, kb_now_world, twist_now, kb_rest_local, twist_rest)


@wp.func
def _reference_cable_bend_twist_force_hessian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    tangent_axis: wp.vec3,
    penalty_k: float,
    P: wp.mat33,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    """Test reference arbitrary-axis bend/twist force and Hessian."""
    inv_dt = 1.0 / dt
    kb_rest_world, twist_rest = _compute_geometric_cable_kb_twist(q_wp_rest, q_wc_rest, tangent_axis)
    kb_rest_local = wp.quat_rotate(wp.quat_inverse(q_wp_rest), kb_rest_world)
    measure = _measure_cable_bend_twist(q_wp, q_wc, tangent_axis)
    kappa_now_vec = _assemble_geometric_cable_kappa(
        q_wp, tangent_axis, measure.kb_world, measure.twist, kb_rest_local, twist_rest
    )

    f_local = penalty_k * (P * kappa_now_vec) - (penalty_k * alpha) * (P * C0_ang) + P * lambda_ang
    H_local = penalty_k * P

    if damping > 0.0:
        kb_prev_world, twist_prev = _compute_geometric_cable_kb_twist(q_wp_prev, q_wc_prev, tangent_axis)
        kappa_prev_vec = _assemble_geometric_cable_kappa(
            q_wp_prev, tangent_axis, kb_prev_world, twist_prev, kb_rest_local, twist_rest
        )
        dkappa_dt = (kappa_now_vec - kappa_prev_vec) * inv_dt
        f_local = f_local + (damping * penalty_k) * (P * dkappa_dt)
        H_local = H_local + (damping * penalty_k * inv_dt) * P

    _bend0, _bend1, twist_axis = _build_cable_material_basis(tangent_axis)
    P_bend = wp.identity(3, float) - wp.outer(twist_axis, twist_axis)
    J_body = _cable_bend_twist_jacobian_from_measure(q_wp, P_bend, twist_axis, measure, is_parent)
    H_aa = wp.transpose(J_body) * (H_local * J_body)
    tau_world = -(wp.transpose(J_body) * f_local)
    return tau_world, H_aa, kappa_now_vec, J_body


@wp.kernel
def _eval_cable_material_basis_invariants_kernel(
    tangents: wp.array[wp.vec3],
    basis_lengths: wp.array[wp.vec3],
    basis_dots: wp.array[wp.vec3],
    roundtrip_errors: wp.array[wp.vec3],
    hessian_errors: wp.array[wp.vec3],
    projector_errors: wp.array[wp.vec3],
):
    tid = wp.tid()
    bend0, bend1, twist = _build_cable_material_basis(tangents[tid])

    basis_lengths[tid] = wp.vec3(wp.length(bend0), wp.length(bend1), wp.length(twist))
    basis_dots[tid] = wp.vec3(wp.dot(bend0, bend1), wp.dot(bend0, twist), wp.dot(bend1, twist))

    v_parent = wp.vec3(0.37, -0.29, 0.61)
    v_basis = _cable_parent_to_material_basis(v_parent, bend0, bend1, twist)
    v_parent_rt = _cable_material_basis_to_parent(v_basis, bend0, bend1, twist)

    c_basis = wp.vec3(1.25, -0.5, 0.75)
    c_parent = _cable_material_basis_to_parent(c_basis, bend0, bend1, twist)
    c_basis_rt = _cable_parent_to_material_basis(c_parent, bend0, bend1, twist)

    tangent = tangents[tid]
    tangent_len = wp.length(tangent)
    expected_twist = wp.vec3(0.0, 0.0, 1.0)
    if tangent_len > 0.0:
        expected_twist = tangent / tangent_len
    roundtrip_errors[tid] = wp.vec3(
        wp.length(v_parent_rt - v_parent),
        wp.length(c_basis_rt - c_basis),
        wp.length(twist - expected_twist),
    )

    h_diag = wp.vec3(2.0, 3.0, 5.0)
    H_parent = _cable_material_basis_hessian_to_parent(h_diag, bend0, bend1, twist)
    hessian_errors[tid] = wp.vec3(
        wp.length(H_parent * bend0 - h_diag[0] * bend0),
        wp.length(H_parent * bend1 - h_diag[1] * bend1),
        wp.length(H_parent * twist - h_diag[2] * twist),
    )

    P_bend, P_twist = _build_cable_bend_twist_projectors(tangents[tid])
    bend_v = P_bend * v_parent
    twist_v = P_twist * v_parent
    projector_errors[tid] = wp.vec3(
        wp.length(bend_v + twist_v - v_parent),
        wp.abs(wp.dot(bend_v, twist_v)),
        wp.length(twist_v - wp.dot(v_parent, twist) * twist),
    )


@wp.kernel
def _eval_cable_material_basis_continuity_kernel(
    tangents: wp.array[wp.vec3],
    basis_dots: wp.array[wp.vec3],
):
    tid = wp.tid()
    bend0_a, bend1_a, twist_a = _build_cable_material_basis(tangents[tid])
    bend0_b, bend1_b, twist_b = _build_cable_material_basis(tangents[tid + 1])
    basis_dots[tid] = wp.vec3(
        wp.dot(bend0_a, bend0_b),
        wp.dot(bend1_a, bend1_b),
        wp.dot(twist_a, twist_b),
    )


@wp.kernel
def _eval_split_cable_bend_twist_mode_leakage_kernel(
    tangents: wp.array[wp.vec3],
    signals: wp.array[wp.vec3],
    leaks: wp.array[wp.vec3],
):
    tid = wp.tid()
    bend0, _bend1, twist = _build_cable_material_basis(tangents[tid])
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangents[tid])

    q_id = wp.quat_identity()
    angle = 0.2
    q_bend = wp.quat_from_axis_angle(bend0, angle)
    q_twist = wp.quat_from_axis_angle(twist, angle)

    _torque_bend, _H_bend, kappa_bend, _J_bend = evaluate_angular_constraint_force_hessian(
        q_id,
        q_bend,
        q_id,
        q_id,
        q_id,
        q_id,
        True,
        1.0,
        P_bend,
        wp.vec3(0.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0.01,
    )
    _torque_twist, _H_twist, kappa_twist, _J_twist = evaluate_angular_constraint_force_hessian(
        q_id,
        q_twist,
        q_id,
        q_id,
        q_id,
        q_id,
        True,
        1.0,
        P_twist,
        wp.vec3(0.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0.01,
    )

    bend_signal = wp.length(P_bend * kappa_bend)
    twist_signal = wp.length(P_twist * kappa_twist)
    bend_leak = wp.length(P_twist * kappa_bend)
    twist_leak = wp.length(P_bend * kappa_twist)

    signals[tid] = wp.vec3(bend_signal, twist_signal, 0.0)
    leaks[tid] = wp.vec3(bend_leak, twist_leak, 0.0)


@wp.kernel
def _eval_split_cable_stretch_shear_mode_leakage_kernel(
    tangents: wp.array[wp.vec3],
    signals: wp.array[wp.vec3],
    leaks: wp.array[wp.vec3],
):
    tid = wp.tid()
    t = wp.normalize(tangents[tid])
    P_stretch, P_shear = _build_cable_stretch_shear_projectors(wp.quat_identity(), t)

    # Axial probe lies along the tangent (pure stretch); transverse probe is the
    # part of an arbitrary vector perpendicular to the tangent (pure shear).
    axial = t * 0.37
    probe = wp.vec3(0.19, -0.83, 0.51)
    transverse = probe - wp.dot(probe, t) * t

    stretch_signal = wp.length(P_stretch * axial)
    shear_signal = wp.length(P_shear * transverse)
    stretch_leak = wp.length(P_shear * axial)
    shear_leak = wp.length(P_stretch * transverse)

    # Completeness: the two projectors must sum to the identity.
    recon_residual = wp.length((P_stretch + P_shear) * probe - probe)

    signals[tid] = wp.vec3(stretch_signal, shear_signal, 0.0)
    leaks[tid] = wp.vec3(stretch_leak, shear_leak, recon_residual)


@wp.func
def _eval_projected_angular_torque_and_residual(
    q_child: wp.quat,
    stiffness: float,
    projector: wp.mat33,
) -> tuple[wp.vec3, wp.vec3]:
    zero = wp.vec3(0.0)
    tau, _H, kappa, _J = evaluate_angular_constraint_force_hessian(
        wp.quat_identity(),
        q_child,
        wp.quat_identity(),
        wp.quat_identity(),
        wp.quat_identity(),
        wp.quat_identity(),
        True,
        stiffness,
        projector,
        zero,
        zero,
        0.0,
        0.0,
        0.01,
    )
    return tau, kappa


@wp.kernel
def _eval_split_cable_projector_stiffness_isolation_kernel(
    tangents: wp.array[wp.vec3],
    isolation_errors: wp.array[wp.vec3],
    scaling_errors: wp.array[wp.vec3],
    residual_errors: wp.array[wp.vec3],
):
    tid = wp.tid()
    bend0, _bend1, twist = _build_cable_material_basis(tangents[tid])
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangents[tid])

    angle = 0.17
    q_bend = wp.quat_from_axis_angle(bend0, angle)
    q_twist = wp.quat_from_axis_angle(twist, angle)

    tau_bend_2, kappa_bend = _eval_projected_angular_torque_and_residual(q_bend, 2.0, P_bend)
    tau_bend_11, _kappa_b11 = _eval_projected_angular_torque_and_residual(q_bend, 11.0, P_bend)
    tau_bend_twist_3, _kappa_bt3 = _eval_projected_angular_torque_and_residual(q_bend, 3.0, P_twist)
    tau_bend_twist_29, _kappa_bt29 = _eval_projected_angular_torque_and_residual(q_bend, 29.0, P_twist)

    tau_twist_3, kappa_twist = _eval_projected_angular_torque_and_residual(q_twist, 3.0, P_twist)
    tau_twist_29, _kappa_t29 = _eval_projected_angular_torque_and_residual(q_twist, 29.0, P_twist)
    tau_twist_bend_2, _kappa_tb2 = _eval_projected_angular_torque_and_residual(q_twist, 2.0, P_bend)
    tau_twist_bend_11, _kappa_tb11 = _eval_projected_angular_torque_and_residual(q_twist, 11.0, P_bend)

    bend_norm = wp.max(wp.length(tau_bend_2), 1.0e-8)
    twist_norm = wp.max(wp.length(tau_twist_3), 1.0e-8)

    bend_isolation = wp.length(tau_bend_twist_29 - tau_bend_twist_3) / bend_norm
    twist_isolation = wp.length(tau_twist_bend_11 - tau_twist_bend_2) / twist_norm
    leak_rel = wp.max(wp.length(tau_bend_twist_29) / bend_norm, wp.length(tau_twist_bend_11) / twist_norm)

    bend_scale = wp.length(tau_bend_11 - 5.5 * tau_bend_2) / wp.max(wp.length(tau_bend_11), 1.0e-8)
    twist_scale = wp.length(tau_twist_29 - (29.0 / 3.0) * tau_twist_3) / wp.max(wp.length(tau_twist_29), 1.0e-8)

    bend_angle_err = wp.abs(wp.length(P_bend * kappa_bend) - angle) / angle
    twist_angle_err = wp.abs(wp.length(P_twist * kappa_twist) - angle) / angle

    isolation_errors[tid] = wp.vec3(bend_isolation, twist_isolation, leak_rel)
    scaling_errors[tid] = wp.vec3(bend_scale, twist_scale, 0.0)
    residual_errors[tid] = wp.vec3(bend_angle_err, twist_angle_err, 0.0)


@wp.kernel
def _eval_split_cable_material_force_law_kernel(
    bend_stiffness: float,
    twist_stiffness: float,
    angle: float,
    torque_magnitudes: wp.array[float],
    leakage_errors: wp.array[float],
):
    q_id = wp.quat_identity()
    tangent = wp.vec3(0.0, 0.0, 1.0)
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangent)
    zero = wp.vec3(0.0)

    q_bend = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
    q_twist = wp.quat_from_axis_angle(tangent, angle)

    tau_bend, _H_bend, kappa_bend, _J_bend = evaluate_cable_bend_twist_force_hessian_z(
        q_id,
        q_bend,
        zero,
        0.0,
        q_id,
        q_id,
        True,
        wp.vec3(bend_stiffness, bend_stiffness, 0.0),
        zero,
        zero,
        zero,
        zero,
        zero,
        False,
        0.01,
    )
    tau_twist, _H_twist, kappa_twist, _J_twist = evaluate_cable_bend_twist_force_hessian_z(
        q_id,
        q_twist,
        zero,
        0.0,
        q_id,
        q_id,
        True,
        wp.vec3(0.0, 0.0, twist_stiffness),
        zero,
        zero,
        zero,
        zero,
        zero,
        False,
        0.01,
    )

    torque_magnitudes[0] = wp.length(tau_bend)
    torque_magnitudes[1] = wp.length(tau_twist)
    leakage_errors[0] = wp.length(P_twist * tau_bend)
    leakage_errors[1] = wp.length(P_bend * tau_twist)
    leakage_errors[2] = wp.abs(wp.length(P_bend * kappa_bend) - 2.0 * wp.tan(0.5 * angle))
    leakage_errors[3] = wp.abs(wp.length(P_twist * kappa_twist) - angle)


@wp.kernel
def _eval_split_cable_cantilever_moment_law_kernel(
    lever_arms: wp.array[float],
    bend_stiffness: float,
    tip_force: float,
    errors: wp.array[wp.vec3],
):
    tid = wp.tid()

    q_id = wp.quat_identity()
    tangent = wp.vec3(0.0, 0.0, 1.0)
    bend_axis = wp.vec3(1.0, 0.0, 0.0)
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangent)
    zero = wp.vec3(0.0)

    expected_moment = tip_force * lever_arms[tid]
    expected_strain = expected_moment / bend_stiffness
    theta = 2.0 * wp.atan(0.5 * expected_strain)
    q_bend = wp.quat_from_axis_angle(bend_axis, theta)

    tau_bend, _H_bend, kappa_bend, _J_bend = evaluate_cable_bend_twist_force_hessian_z(
        q_id,
        q_bend,
        zero,
        0.0,
        q_id,
        q_id,
        True,
        wp.vec3(bend_stiffness, bend_stiffness, 0.0),
        zero,
        zero,
        zero,
        zero,
        zero,
        False,
        0.01,
    )

    measured_moment = wp.length(tau_bend)
    measured_strain = wp.length(P_bend * kappa_bend)

    errors[tid] = wp.vec3(
        wp.abs(measured_moment - expected_moment),
        wp.length(P_twist * tau_bend),
        wp.abs(measured_strain - expected_strain),
    )


@wp.func
def _eval_split_cable_fd_torque(
    q_wc: wp.quat,
    stiffness: float,
    P: wp.mat33,
) -> tuple[wp.vec3, wp.mat33]:
    q_id = wp.quat_identity()
    zero = wp.vec3(0.0)
    tau, H, _kappa, _J = evaluate_angular_constraint_force_hessian(
        q_id,
        q_wc,
        q_id,
        q_id,
        q_id,
        q_id,
        True,
        stiffness,
        P,
        zero,
        zero,
        0.0,
        0.0,
        0.01,
    )
    return tau, H


@wp.kernel
def _eval_split_cable_angular_hessian_finite_difference_kernel(errors: wp.array[wp.vec3]):
    tid = wp.tid()

    tangent = wp.normalize(wp.vec3(0.3, -0.4, 0.8660254))
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangent)
    bend0, _bend1, _twist = _build_cable_material_basis(tangent)
    P = P_bend
    free_axis = tangent
    stiffness = 23.0
    if tid == 1:
        P = P_twist
        free_axis = bend0

    eps = 1.0e-3
    _tau0, H = _eval_split_cable_fd_torque(wp.quat_identity(), stiffness, P)

    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)

    tau_p0, _H_p0 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e0, eps), stiffness, P)
    tau_m0, _H_m0 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e0, -eps), stiffness, P)
    tau_p1, _H_p1 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e1, eps), stiffness, P)
    tau_m1, _H_m1 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e1, -eps), stiffness, P)
    tau_p2, _H_p2 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e2, eps), stiffness, P)
    tau_m2, _H_m2 = _eval_split_cable_fd_torque(wp.quat_from_axis_angle(e2, -eps), stiffness, P)

    fd0 = (tau_p0 - tau_m0) / (2.0 * eps)
    fd1 = (tau_p1 - tau_m1) / (2.0 * eps)
    fd2 = (tau_p2 - tau_m2) / (2.0 * eps)

    err0 = wp.length(fd0 - H * e0)
    err1 = wp.length(fd1 - H * e1)
    err2 = wp.length(fd2 - H * e2)
    max_fd_error = wp.max(err0, wp.max(err1, err2)) / stiffness

    sym_error = (
        wp.max(
            wp.abs(wp.dot(e0, H * e1) - wp.dot(e1, H * e0)),
            wp.max(
                wp.abs(wp.dot(e0, H * e2) - wp.dot(e2, H * e0)),
                wp.abs(wp.dot(e1, H * e2) - wp.dot(e2, H * e1)),
            ),
        )
        / stiffness
    )

    null_error = wp.length(H * free_axis) / stiffness
    errors[tid] = wp.vec3(max_fd_error, sym_error, null_error)


@wp.func
def _quat_perturb_world(q: wp.quat, axis: wp.vec3, angle: float) -> wp.quat:
    return wp.normalize(wp.quat_from_axis_angle(axis, angle) * q)


@wp.func
def _geometric_cable_test_energy(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    stiffness: float,
    P: wp.mat33,
    tangent_axis: wp.vec3,
) -> float:
    residual = _compute_geometric_cable_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest, tangent_axis)
    projected = P * residual
    return 0.5 * stiffness * wp.dot(projected, projected)


@wp.func
def _eval_geometric_cable_test_force_hessian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    is_parent: bool,
    stiffness: float,
    P: wp.mat33,
    tangent_axis: wp.vec3,
) -> tuple[wp.vec3, wp.mat33]:
    zero = wp.vec3(0.0)
    tau, H, _kappa, _J = _reference_cable_bend_twist_force_hessian(
        q_wp,
        q_wc,
        q_wp_rest,
        q_wc_rest,
        q_wp,
        q_wc,
        is_parent,
        tangent_axis,
        stiffness,
        P,
        zero,
        zero,
        0.0,
        0.0,
        0.01,
    )
    return tau, H


@wp.kernel
def _eval_geometric_cable_force_hessian_finite_difference_kernel(errors: wp.array[wp.vec3]):
    tid = wp.tid()

    tangent = wp.normalize(wp.vec3(0.2, -0.35, 0.915))
    _bend0, _bend1, _twist = _build_cable_material_basis(tangent)
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangent)

    P = P_bend
    stiffness = 17.0
    if tid >= 2:
        P = P_twist
        stiffness = 29.0

    is_parent = tid == 0 or tid == 2

    q_wp = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.7, -0.2)), 0.31)
    q_wc = wp.quat_from_axis_angle(wp.normalize(wp.vec3(-0.6, 0.2, 0.4)), 0.58) * q_wp
    q_wp_rest = wp.quat_identity()
    q_wc_rest = wp.quat_identity()

    tau, _H = _eval_geometric_cable_test_force_hessian(
        q_wp, q_wc, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )

    eps = 1.0e-3
    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e0, eps)
        q_wp_m = _quat_perturb_world(q_wp, e0, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e0, eps)
        q_wc_m = _quat_perturb_world(q_wc, e0, -eps)
    fd0 = -(
        _geometric_cable_test_energy(q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, stiffness, P, tangent)
        - _geometric_cable_test_energy(q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, stiffness, P, tangent)
    ) / (2.0 * eps)

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e1, eps)
        q_wp_m = _quat_perturb_world(q_wp, e1, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e1, eps)
        q_wc_m = _quat_perturb_world(q_wc, e1, -eps)
    fd1 = -(
        _geometric_cable_test_energy(q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, stiffness, P, tangent)
        - _geometric_cable_test_energy(q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, stiffness, P, tangent)
    ) / (2.0 * eps)

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e2, eps)
        q_wp_m = _quat_perturb_world(q_wp, e2, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e2, eps)
        q_wc_m = _quat_perturb_world(q_wc, e2, -eps)
    fd2 = -(
        _geometric_cable_test_energy(q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, stiffness, P, tangent)
        - _geometric_cable_test_energy(q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, stiffness, P, tangent)
    ) / (2.0 * eps)

    force_fd_error = wp.length(tau - wp.vec3(fd0, fd1, fd2)) / stiffness

    # Validate the local Gauss-Newton Hessian at finite geometry but zero residual.
    q_wp_rest = q_wp
    q_wc_rest = q_wc
    tau0, H0 = _eval_geometric_cable_test_force_hessian(
        q_wp, q_wc, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e0, eps)
        q_wp_m = _quat_perturb_world(q_wp, e0, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e0, eps)
        q_wc_m = _quat_perturb_world(q_wc, e0, -eps)
    tau_p, _Hp = _eval_geometric_cable_test_force_hessian(
        q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    tau_m, _Hm = _eval_geometric_cable_test_force_hessian(
        q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    h_fd0 = (tau_p - tau_m) / (2.0 * eps)

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e1, eps)
        q_wp_m = _quat_perturb_world(q_wp, e1, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e1, eps)
        q_wc_m = _quat_perturb_world(q_wc, e1, -eps)
    tau_p, _Hp = _eval_geometric_cable_test_force_hessian(
        q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    tau_m, _Hm = _eval_geometric_cable_test_force_hessian(
        q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    h_fd1 = (tau_p - tau_m) / (2.0 * eps)

    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, e2, eps)
        q_wp_m = _quat_perturb_world(q_wp, e2, -eps)
    else:
        q_wc_p = _quat_perturb_world(q_wc, e2, eps)
        q_wc_m = _quat_perturb_world(q_wc, e2, -eps)
    tau_p, _Hp = _eval_geometric_cable_test_force_hessian(
        q_wp_p, q_wc_p, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    tau_m, _Hm = _eval_geometric_cable_test_force_hessian(
        q_wp_m, q_wc_m, q_wp_rest, q_wc_rest, is_parent, stiffness, P, tangent
    )
    h_fd2 = (tau_p - tau_m) / (2.0 * eps)

    h_err0 = wp.length(h_fd0 + H0 * e0)
    h_err1 = wp.length(h_fd1 + H0 * e1)
    h_err2 = wp.length(h_fd2 + H0 * e2)
    hessian_fd_error = wp.max(h_err0, wp.max(h_err1, h_err2)) / stiffness

    sym_error = (
        wp.max(
            wp.abs(wp.dot(e0, H0 * e1) - wp.dot(e1, H0 * e0)),
            wp.max(
                wp.abs(wp.dot(e0, H0 * e2) - wp.dot(e2, H0 * e0)),
                wp.abs(wp.dot(e1, H0 * e2) - wp.dot(e2, H0 * e1)),
            ),
        )
        / stiffness
    )

    # tau0 should be zero at the rest pose.
    rest_force_error = wp.length(tau0) / stiffness
    errors[tid] = wp.vec3(
        wp.max(force_fd_error, rest_force_error),
        hessian_fd_error,
        sym_error,
    )


@wp.kernel
def _eval_bishop_transport_antiparallel_fallback_kernel(errors: wp.array[wp.vec3]):
    t0 = wp.vec3(1.0, 0.0, 0.0)
    t1 = wp.vec3(-1.0, 0.0, 0.0)
    fallback_parallel_to_t0 = wp.vec3(1.0, 0.0, 0.0)

    q = _bishop_transport_quat(t0, t1, fallback_parallel_to_t0)
    mapped = wp.quat_rotate(q, t0)
    errors[0] = wp.vec3(wp.length(mapped - t1), wp.abs(wp.dot(mapped, t0) + 1.0), wp.length(mapped))


@wp.func
def _rotate_tangent_for_fd(t: wp.vec3, omega: wp.vec3, h: float) -> wp.vec3:
    omega_len = wp.length(omega)
    if omega_len <= 1.0e-12:
        return t
    q = wp.quat_from_axis_angle(omega / omega_len, h * omega_len)
    return wp.quat_rotate(q, t)


@wp.func
def _curvature_binormal_derivative_fd_error(t0: wp.vec3, t1: wp.vec3, fallback: wp.vec3, h: float) -> wp.vec3:
    omega0 = wp.vec3(0.31, -0.27, 0.19)
    omega1 = wp.vec3(-0.17, 0.23, 0.29)
    dt0 = wp.cross(omega0, t0)
    dt1 = wp.cross(omega1, t1)

    analytic = _finite_curvature_binormal_derivative(t0, t1, dt0, dt1)
    t0_p = _rotate_tangent_for_fd(t0, omega0, h)
    t1_p = _rotate_tangent_for_fd(t1, omega1, h)
    t0_m = _rotate_tangent_for_fd(t0, omega0, -h)
    t1_m = _rotate_tangent_for_fd(t1, omega1, -h)
    fd = (_finite_curvature_binormal(t0_p, t1_p, fallback) - _finite_curvature_binormal(t0_m, t1_m, fallback)) / (
        2.0 * h
    )

    kb = _finite_curvature_binormal(t0, t1, fallback)
    cap_tangent_error = wp.abs(wp.dot(kb, analytic))
    if wp.length(kb) > 1.0e-8:
        cap_tangent_error = cap_tangent_error / wp.length(kb)
    return wp.vec3(wp.length(analytic - fd), wp.length(analytic), cap_tangent_error)


@wp.kernel
def _eval_geometric_curvature_binormal_derivative_kernel(errors: wp.array[wp.vec3]):
    t0 = wp.vec3(0.0, 0.0, 1.0)
    fallback = wp.vec3(0.0, 1.0, 0.0)

    t1_regular = wp.normalize(wp.vec3(0.25, -0.12, 0.96))
    regular = _curvature_binormal_derivative_fd_error(t0, t1_regular, fallback, 1.0e-3)

    q_hairpin = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 3.05)
    t1_capped = wp.quat_rotate(q_hairpin, t0)
    capped = _curvature_binormal_derivative_fd_error(t0, t1_capped, fallback, 1.0e-3)

    errors[0] = wp.vec3(regular[0], capped[0], capped[2])


@wp.func
def _transported_twist_angle_derivative_fd_error(
    q_wp: wp.quat,
    q_wc: wp.quat,
    tangent: wp.vec3,
    omega: wp.vec3,
    is_parent: bool,
    h: float,
) -> wp.vec3:
    measure = _measure_cable_bend_twist(q_wp, q_wc, tangent)
    analytic = _transported_twist_angle_derivative_from_measure(measure, omega, is_parent)
    q_wp_p = q_wp
    q_wp_m = q_wp
    q_wc_p = q_wc
    q_wc_m = q_wc
    omega_len = wp.length(omega)
    axis = omega / omega_len
    if is_parent:
        q_wp_p = _quat_perturb_world(q_wp, axis, h * omega_len)
        q_wp_m = _quat_perturb_world(q_wp, axis, -h * omega_len)
    else:
        q_wc_p = _quat_perturb_world(q_wc, axis, h * omega_len)
        q_wc_m = _quat_perturb_world(q_wc, axis, -h * omega_len)

    twist_p = _measure_cable_bend_twist(q_wp_p, q_wc_p, tangent).twist
    twist_m = _measure_cable_bend_twist(q_wp_m, q_wc_m, tangent).twist
    fd = (twist_p - twist_m) / (2.0 * h)
    return wp.vec3(wp.abs(analytic - fd), wp.abs(analytic), wp.abs(fd))


@wp.kernel
def _eval_transported_twist_angle_derivative_kernel(errors: wp.array[wp.vec3]):
    tid = wp.tid()
    is_parent = tid == 0 or tid == 2
    omega = wp.vec3(0.19, -0.31, 0.23)
    if not is_parent:
        omega = wp.vec3(-0.17, 0.29, 0.21)

    tangent = wp.normalize(wp.vec3(0.2, -0.35, 0.915))
    q_wp = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.7, -0.2)), 0.31)
    q_wc = wp.quat_from_axis_angle(wp.normalize(wp.vec3(-0.6, 0.2, 0.4)), 0.42) * q_wp

    if tid >= 2:
        tangent = wp.vec3(0.0, 0.0, 1.0)
        q_wp = wp.quat_identity()
        q_bend = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.55)
        child_tangent = wp.quat_rotate(q_bend, tangent)
        q_wc = wp.quat_from_axis_angle(child_tangent, 0.41) * q_bend

    errors[tid] = _transported_twist_angle_derivative_fd_error(q_wp, q_wc, tangent, omega, is_parent, 1.0e-3)


@wp.kernel
def _eval_geometric_precurved_twist_is_pure_twist_kernel(errors: wp.array[wp.vec3]):
    tangent = wp.vec3(0.0, 0.0, 1.0)
    P_bend, P_twist = _build_cable_bend_twist_projectors(tangent)

    bend_angle = 0.55
    twist_angle = 0.41
    q_wp_rest = wp.quat_identity()
    q_wc_rest = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), bend_angle)

    child_tangent_world = wp.quat_rotate(q_wc_rest, tangent)
    q_wp = q_wp_rest
    q_wc = wp.quat_from_axis_angle(child_tangent_world, twist_angle) * q_wc_rest

    kappa = _compute_geometric_cable_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest, tangent)
    bend_leak = wp.length(P_bend * kappa)
    twist_err = wp.abs(wp.length(P_twist * kappa) - twist_angle)
    errors[0] = wp.vec3(bend_leak, twist_err, wp.length(kappa))


@wp.kernel
def _eval_geometric_global_rotation_preserves_rest_strain_kernel(errors: wp.array[wp.vec3]):
    tangent = wp.vec3(0.0, 0.0, 1.0)

    q_wp_rest = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.2, 0.5)), 0.37)
    bend_axis_world = wp.quat_rotate(q_wp_rest, wp.vec3(0.0, 1.0, 0.0))
    q_bend = wp.quat_from_axis_angle(bend_axis_world, 0.61)
    q_wc_rest = q_bend * q_wp_rest

    child_tangent_world = wp.quat_rotate(q_wc_rest, tangent)
    q_wc_rest = wp.quat_from_axis_angle(child_tangent_world, 0.43) * q_wc_rest

    q_global = wp.quat_from_axis_angle(wp.normalize(wp.vec3(-0.4, 0.6, 0.2)), 0.79)
    q_wp = q_global * q_wp_rest
    q_wc = q_global * q_wc_rest

    kappa = _compute_geometric_cable_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest, tangent)
    errors[0] = wp.vec3(wp.length(kappa), wp.length(kappa), wp.length(kappa))


@wp.kernel
def _eval_geometric_curvature_binormal_cap_kernel(errors: wp.array[wp.vec3]):
    tangent = wp.vec3(0.0, 0.0, 1.0)
    q_wp = wp.quat_identity()
    q_wc = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 3.05)
    q_rest = wp.quat_identity()

    kappa = _compute_geometric_cable_kappa(q_wp, q_wc, q_rest, q_rest, tangent)
    cap_error = wp.abs(wp.length(kappa) - 20.0)
    errors[0] = wp.vec3(cap_error, 0.0, wp.length(kappa))


def _split_cable_angular_slot_layout(test, device):
    """Twist stiffness/damping is routed or defaulted into the split angular slots, and negative stiffness is rejected."""
    # (extra add_joint_cable kwargs, expected penalty_k_max, expected penalty_kd) for the four-slot layout.
    cases = [
        # Explicit twist stiffness + damping is routed straight to the twist slot.
        ({"twist_stiffness": 3.0, "twist_damping": 0.25}, [100.0, 100.0, 10.0, 3.0], [0.0, 0.0, 0.0, 0.25]),
        # Omitting both twist params defaults twist to bend (isotropic angular energy).
        ({"bend_damping": 0.5}, [100.0, 100.0, 10.0, 10.0], [0.0, 0.0, 0.5, 0.5]),
        # Explicit twist stiffness with omitted twist damping keeps twist damping at zero.
        ({"bend_damping": 0.5, "twist_stiffness": 3.0}, [100.0, 100.0, 10.0, 3.0], [0.0, 0.0, 0.5, 0.0]),
    ]
    for kwargs, expected_k, expected_kd in cases:
        with test.subTest(kwargs=kwargs):
            builder = newton.ModelBuilder(gravity=0.0)
            body = builder.add_link()
            joint = builder.add_joint_cable(-1, body, stretch_stiffness=100.0, bend_stiffness=10.0, **kwargs)
            builder.add_articulation([joint])
            builder.color()
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverVBD(model)

            np.testing.assert_array_equal(model.joint_dof_dim.numpy()[joint], [2, 2])
            test.assertEqual(int(solver.joint_constraint_dim.numpy()[joint]), 4)
            start = int(solver.joint_constraint_start.numpy()[joint])
            np.testing.assert_allclose(solver.joint_penalty_k_max.numpy()[start : start + 4], expected_k)
            np.testing.assert_allclose(solver.joint_penalty_kd.numpy()[start : start + 4], expected_kd)

    # Negative stiffness must be rejected before reaching the solver.
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link()
    with test.assertRaisesRegex(ValueError, "stretch_stiffness, shear_stiffness, bend_stiffness, and twist_stiffness"):
        builder.add_joint_cable(-1, body, bend_stiffness=10.0, twist_stiffness=-1.0)


def _cable_stiffness_helper_returns_physical_twist(test, device):
    """Elastic-moduli helper should return GJ/L when a shear modulus source is provided."""
    E = 200.0
    radius = 0.5
    length = 2.0
    stretch, bend, twist = newton.utils.create_cable_stiffness_from_elastic_moduli(
        E, radius, length, poissons_ratio=0.25
    )

    area = np.pi * radius * radius
    inertia = 0.25 * np.pi * radius**4
    polar_inertia = 0.5 * np.pi * radius**4
    G = E / (2.0 * (1.0 + 0.25))
    np.testing.assert_allclose(
        [stretch, bend, twist], [E * area / length, E * inertia / length, G * polar_inertia / length]
    )

    with test.assertRaisesRegex(ValueError, "mutually exclusive"):
        newton.utils.create_cable_stiffness_from_elastic_moduli(E, radius, length, poissons_ratio=0.25, shear_modulus=G)
    with test.assertRaisesRegex(ValueError, "poissons_ratio"):
        newton.utils.create_cable_stiffness_from_elastic_moduli(E, radius, length, poissons_ratio=0.5)


def _split_cable_dahl_uses_bend_and_twist_envelopes(test, device):
    """Shared Dahl eps/tau is split across bend and twist with slot-specific stiffness."""
    with wp.ScopedDevice(device):
        joint_type = wp.array(
            [int(newton.JointType.CABLE), int(newton.JointType.CABLE)],
            dtype=wp.int32,
            device=device,
        )
        joint_enabled = wp.array([True, True], dtype=bool, device=device)
        joint_parent = wp.array([-1, -1], dtype=wp.int32, device=device)
        joint_child = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_x = wp.array(
            [wp.transform_identity(), wp.transform_identity()],
            dtype=wp.transform,
            device=device,
        )
        joint_constraint_start = wp.array([0, 4], dtype=wp.int32, device=device)
        joint_penalty_k = wp.array([0.0, 0.0, 10.0, 2.0, 0.0, 0.0, 10.0, 2.0], dtype=float, device=device)
        joint_is_hard = wp.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=wp.int32, device=device)
        joint_cable_kb_rest_local = wp.zeros(2, dtype=wp.vec3, device=device)
        joint_cable_twist_rest = wp.zeros(2, dtype=float, device=device)

        q_bend = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)
        q_twist = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.1)
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0), q_bend),
                wp.transform(wp.vec3(0.0), q_twist),
            ],
            dtype=wp.transform,
            device=device,
        )
        zero_vec3 = wp.zeros(2, dtype=wp.vec3, device=device)
        eps_max = wp.array([0.2, 0.2], dtype=float, device=device)
        tau = wp.array([0.2, 0.2], dtype=float, device=device)
        sigma_start = wp.zeros(2, dtype=wp.vec3, device=device)
        C_fric = wp.zeros(2, dtype=wp.vec3, device=device)

        wp.launch(
            compute_cable_dahl_parameters,
            dim=2,
            inputs=[
                joint_type,
                joint_enabled,
                joint_parent,
                joint_child,
                joint_x,
                joint_x,
                joint_constraint_start,
                joint_penalty_k,
                joint_is_hard,
                joint_cable_kb_rest_local,
                joint_cable_twist_rest,
                body_q,
                zero_vec3,
                zero_vec3,
                zero_vec3,
                eps_max,
                tau,
            ],
            outputs=[sigma_start, C_fric],
            device=device,
        )

        sigma = np.abs(sigma_start.numpy())
        c_fric = C_fric.numpy()
        # Geometric cable bend uses the finite curvature binormal
        # |kb| = 2*tan(theta/2), not the small-angle log-map theta.
        expected_bend_strain = 2.0 * np.tan(0.5 * 0.1)
        expected_bend_sigma = 10.0 * 0.2 * (1.0 - np.exp(-expected_bend_strain / 0.2))
        expected_twist_sigma = 2.0 * 0.2 * (1.0 - np.exp(-0.1 / 0.2))

        np.testing.assert_allclose(sigma[0, 0], expected_bend_sigma, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(sigma[1, 2], expected_twist_sigma, rtol=1.0e-5, atol=1.0e-6)
        test.assertGreater(c_fric[0, 0], 0.0)
        test.assertGreater(c_fric[1, 2], 0.0)
        test.assertLessEqual(c_fric[0, 0], 10.0 + 1.0e-6)
        test.assertLessEqual(c_fric[1, 2], 2.0 + 1.0e-6)
        test.assertGreater(c_fric[0, 0], c_fric[1, 2])
        np.testing.assert_allclose(sigma[0, 1:], [0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(sigma[1, :2], [0.0, 0.0], atol=1.0e-6)
        test.assertGreater(sigma[0, 0], sigma[1, 2])


def _split_cable_material_basis_invariants(test, device):
    """cable material basis must be orthonormal and invert parent/basis transforms."""
    with wp.ScopedDevice(device):
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        tangents = wp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [inv_sqrt2, inv_sqrt2, 0.0],
                [0.2, -0.4, 0.9],
                [0.0, 0.0, 0.0],
            ],
            dtype=wp.vec3,
            device=device,
        )
        basis_lengths = wp.zeros(5, dtype=wp.vec3, device=device)
        basis_dots = wp.zeros(5, dtype=wp.vec3, device=device)
        roundtrip_errors = wp.zeros(5, dtype=wp.vec3, device=device)
        hessian_errors = wp.zeros(5, dtype=wp.vec3, device=device)
        projector_errors = wp.zeros(5, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_cable_material_basis_invariants_kernel,
            dim=5,
            inputs=[tangents],
            outputs=[basis_lengths, basis_dots, roundtrip_errors, hessian_errors, projector_errors],
            device=device,
        )

        np.testing.assert_allclose(basis_lengths.numpy(), np.ones((5, 3)), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(basis_dots.numpy(), np.zeros((5, 3)), atol=1.0e-6)
        np.testing.assert_allclose(roundtrip_errors.numpy(), np.zeros((5, 3)), atol=1.0e-6)
        np.testing.assert_allclose(hessian_errors.numpy(), np.zeros((5, 3)), atol=1.0e-6)
        np.testing.assert_allclose(projector_errors.numpy(), np.zeros((5, 3)), atol=1.0e-6)


def _split_cable_material_basis_is_continuous_near_axis_cutoff(test, device):
    """Cable material bend axes should not jump near axis-selection cutoffs."""
    with wp.ScopedDevice(device):
        xs = np.asarray([0.895, 0.899, 0.900, 0.901, 0.905], dtype=np.float64)
        tangents_np = np.stack([xs, np.zeros_like(xs), np.sqrt(1.0 - xs * xs)], axis=1)
        tangents = wp.array(tangents_np, dtype=wp.vec3, device=device)
        basis_dots = wp.zeros(len(xs) - 1, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_cable_material_basis_continuity_kernel,
            dim=len(xs) - 1,
            inputs=[tangents],
            outputs=[basis_dots],
            device=device,
        )

        dots = basis_dots.numpy()
        test.assertGreater(
            float(np.min(dots)),
            0.999,
            f"cable material basis jumped near an axis-selection cutoff: dots={dots}",
        )


def _split_cable_bend_twist_projectors_have_no_pure_mode_leakage(test, device):
    """Pure bend/twist rotations should project cleanly for +Z and tilted tangent axes."""
    with wp.ScopedDevice(device):
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        tangents = wp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [inv_sqrt2, inv_sqrt2, 0.0],
                [0.2, -0.4, 0.9],
            ],
            dtype=wp.vec3,
            device=device,
        )
        signals = wp.zeros(4, dtype=wp.vec3, device=device)
        leaks = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_split_cable_bend_twist_mode_leakage_kernel,
            dim=4,
            inputs=[tangents],
            outputs=[signals, leaks],
            device=device,
        )

        signal = signals.numpy()[:, :2]
        leak = leaks.numpy()[:, :2]
        test.assertTrue(np.all(signal > 1.0e-3), f"pure-mode signals unexpectedly small: {signal}")
        np.testing.assert_allclose(leak, np.zeros_like(leak), atol=1.0e-6)
        test.assertLess(float(np.max(leak / np.maximum(signal, 1.0e-12))), 1.0e-6)


def _split_cable_stretch_shear_projectors_have_no_pure_mode_leakage(test, device):
    """Axial/transverse displacements should project cleanly and P_stretch+P_shear=I."""
    with wp.ScopedDevice(device):
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        tangents = wp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [inv_sqrt2, inv_sqrt2, 0.0],
                [0.2, -0.4, 0.9],
            ],
            dtype=wp.vec3,
            device=device,
        )
        signals = wp.zeros(4, dtype=wp.vec3, device=device)
        leaks = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_split_cable_stretch_shear_mode_leakage_kernel,
            dim=4,
            inputs=[tangents],
            outputs=[signals, leaks],
            device=device,
        )

        signal = signals.numpy()[:, :2]
        leak = leaks.numpy()
        test.assertTrue(np.all(signal > 1.0e-3), f"pure-mode signals unexpectedly small: {signal}")
        # leak columns: [stretch->shear, shear->stretch, completeness residual]
        np.testing.assert_allclose(leak, np.zeros_like(leak), atol=1.0e-6)


def _split_cable_routes_explicit_shear_to_second_slot(test, device):
    """Explicit shear stiffness/damping must land in the split shear slot."""
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link()
    joint = builder.add_joint_cable(
        -1,
        body,
        stretch_stiffness=100.0,
        stretch_damping=0.2,
        shear_stiffness=40.0,
        shear_damping=0.7,
        bend_stiffness=10.0,
        bend_damping=0.5,
        twist_stiffness=3.0,
        twist_damping=0.25,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model)

    np.testing.assert_array_equal(model.joint_dof_dim.numpy()[joint], [2, 2])
    test.assertEqual(int(solver.joint_constraint_dim.numpy()[joint]), 4)
    start = int(solver.joint_constraint_start.numpy()[joint])
    np.testing.assert_allclose(solver.joint_penalty_k_max.numpy()[start : start + 4], [100.0, 40.0, 10.0, 3.0])
    np.testing.assert_allclose(solver.joint_penalty_kd.numpy()[start : start + 4], [0.2, 0.7, 0.5, 0.25])


def _split_cable_angular_stiffness_isolated_by_projector(test, device):
    """Bend/twist stiffness should scale only its projected angular mode."""
    with wp.ScopedDevice(device):
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        tangents = wp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [inv_sqrt2, inv_sqrt2, 0.0],
                [0.2, -0.4, 0.9],
            ],
            dtype=wp.vec3,
            device=device,
        )
        isolation_errors = wp.zeros(4, dtype=wp.vec3, device=device)
        scaling_errors = wp.zeros(4, dtype=wp.vec3, device=device)
        residual_errors = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_split_cable_projector_stiffness_isolation_kernel,
            dim=4,
            inputs=[tangents],
            outputs=[isolation_errors, scaling_errors, residual_errors],
            device=device,
        )

        max_isolation = float(np.max(isolation_errors.numpy()))
        max_scaling = float(np.max(scaling_errors.numpy()))
        max_residual = float(np.max(residual_errors.numpy()))
        test.assertLess(max_isolation, 1.0e-5, f"bend/twist stiffness isolation failed: {isolation_errors.numpy()}")
        test.assertLess(max_scaling, 1.0e-5, f"bend/twist force scaling failed: {scaling_errors.numpy()}")
        test.assertLess(max_residual, 1.0e-5, f"projected residual angle accuracy failed: {residual_errors.numpy()}")


def _split_cable_material_force_law_matches_ei_gj(test, device):
    """Per-joint bend/twist torques should match EI/h and GJ/h stiffness inputs."""
    segment_length = 0.08
    radius = 0.012
    youngs_modulus = 2.0e6
    poissons_ratio = 0.25
    angle = 0.031
    joint_count = 7

    _stretch, bend_stiffness, twist_stiffness = newton.utils.create_cable_stiffness_from_elastic_moduli(
        youngs_modulus,
        radius,
        segment_length,
        poissons_ratio=poissons_ratio,
    )

    torque_magnitudes = wp.zeros(2, dtype=float, device=device)
    leakage_errors = wp.zeros(4, dtype=float, device=device)
    wp.launch(
        _eval_split_cable_material_force_law_kernel,
        dim=1,
        inputs=[bend_stiffness, twist_stiffness, angle, torque_magnitudes, leakage_errors],
        device=device,
    )

    measured_bend, measured_twist = torque_magnitudes.numpy()
    leak_bend_to_twist, leak_twist_to_bend, bend_strain_error, twist_strain_error = leakage_errors.numpy()
    bend_strain = 2.0 * np.tan(0.5 * angle)
    bend_jacobian = 1.0 + (0.5 * bend_strain) ** 2
    expected_bend = bend_stiffness * bend_strain * bend_jacobian
    expected_twist = twist_stiffness * angle

    np.testing.assert_allclose(measured_bend, expected_bend, rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(measured_twist, expected_twist, rtol=1.0e-5, atol=1.0e-8)
    test.assertLess(leak_bend_to_twist, max(1.0e-8, 1.0e-6 * expected_bend))
    test.assertLess(leak_twist_to_bend, max(1.0e-8, 1.0e-6 * expected_twist))
    test.assertLess(bend_strain_error, 1.0e-8)
    test.assertLess(twist_strain_error, 1.0e-8)

    total_twist = angle * joint_count
    effective_length = segment_length * joint_count
    polar_inertia = 0.5 * np.pi * radius**4
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    expected_chain_torque = shear_modulus * polar_inertia * total_twist / effective_length
    np.testing.assert_allclose(measured_twist, expected_chain_torque, rtol=1.0e-5, atol=1.0e-8)


def _split_cable_discrete_cantilever_moment_law_matches_beam_limit(test, device):
    """Small-angle cantilever moments should match the Euler-Bernoulli discrete limit."""
    segment_length = 0.08
    joint_count = 14
    tip_force = 0.2
    bend_stiffness = 37.0

    lever_arms_np = segment_length * np.arange(joint_count, 0, -1, dtype=np.float32)
    lever_arms = wp.array(lever_arms_np, dtype=float, device=device)
    errors = wp.zeros(joint_count, dtype=wp.vec3, device=device)

    wp.launch(
        _eval_split_cable_cantilever_moment_law_kernel,
        dim=joint_count,
        inputs=[lever_arms, bend_stiffness, tip_force, errors],
        device=device,
    )

    errors_np = errors.numpy()
    max_moment_error, max_twist_leakage, max_strain_error = np.max(errors_np, axis=0)
    max_moment = tip_force * float(lever_arms_np[0])
    max_strain = max_moment / bend_stiffness

    test.assertLess(max_moment_error, max(1.0e-7, 1.0e-5 * max_moment))
    test.assertLess(max_twist_leakage, max(1.0e-8, 1.0e-6 * max_moment))
    test.assertLess(max_strain_error, max(1.0e-8, 1.0e-5 * max_strain))

    expected_moments = tip_force * lever_arms_np.astype(np.float64)
    expected_angles = expected_moments / bend_stiffness
    discrete_tip_deflection = float(np.dot(lever_arms_np, expected_angles))
    closed_form_deflection = (
        tip_force * segment_length**2 * joint_count * (joint_count + 1) * (2 * joint_count + 1) / (6.0 * bend_stiffness)
    )
    np.testing.assert_allclose(discrete_tip_deflection, closed_form_deflection, rtol=1.0e-7, atol=1.0e-10)

    # With k_bend = EI/h, the discrete spring-chain compliance converges to
    # the Euler-Bernoulli cantilever tip compliance F L^3 / (3 EI).
    def discrete_to_eb_ratio(n: int) -> float:
        length = n * segment_length
        ei = bend_stiffness * segment_length
        discrete = tip_force * segment_length**2 * n * (n + 1) * (2 * n + 1) / (6.0 * bend_stiffness)
        eb = tip_force * length**3 / (3.0 * ei)
        return discrete / eb

    coarse_ratio = discrete_to_eb_ratio(joint_count)
    fine_ratio = discrete_to_eb_ratio(4 * joint_count)
    test.assertLess(abs(fine_ratio - 1.0), abs(coarse_ratio - 1.0))
    test.assertLess(abs(fine_ratio - 1.0), 0.03)


def _split_cable_kinematic_arc_yields_uniform_curvature(test, device):
    """Kinematically-driven cantilever should settle to a uniform-curvature arc.

    Clamp the root and place the tip on an analytic discrete arc. The
    minimum-energy equilibrium is uniform per-joint bend and zero twist.
    """
    segment_length = 0.10
    num_segments = 14
    num_joints = num_segments - 1
    cable_length = num_segments * segment_length
    bend_stiffness = 400.0
    target_tip_angle = np.deg2rad(60.0)
    delta_theta = target_tip_angle / num_joints

    # Twist >> bend amplifies any bend energy that leaks into the twist subspace.
    twist_stiffness = 1000.0

    def _quat_mul(a, b):
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

    def _quat_distance(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a /= max(float(np.linalg.norm(a)), 1.0e-12)
        b /= max(float(np.linalg.norm(b)), 1.0e-12)
        return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))

    builder = newton.ModelBuilder(gravity=0.0)
    newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    points = newton.utils.create_straight_cable_points(
        start=wp.vec3(0.0, 0.0, 0.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=cable_length,
        num_segments=num_segments,
    )
    quats = newton.utils.create_parallel_transport_cable_quaternions(points)
    rod_bodies, _rod_joints = builder.add_rod(
        positions=points,
        quaternions=quats,
        radius=0.010,
        stretch_stiffness=1.0e6,
        bend_stiffness=bend_stiffness,
        bend_damping=1.0,
        twist_stiffness=twist_stiffness,
        twist_damping=1.0,
        label="kinematic_arc",
    )

    for body_idx in (int(rod_bodies[0]), int(rod_bodies[-1])):
        builder.body_mass[body_idx] = 0.0
        builder.body_inv_mass[body_idx] = 0.0
        builder.body_inertia[body_idx] = wp.mat33(0.0)
        builder.body_inv_inertia[body_idx] = wp.mat33(0.0)

    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=30)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    tip_body = int(rod_bodies[-1])
    body_indices = np.asarray(rod_bodies, dtype=np.int64)
    dynamic_body_indices = body_indices[1:-1]

    analytic_points = np.zeros((num_segments + 1, 3), dtype=np.float64)
    for i in range(1, num_segments + 1):
        theta = (i - 1) * delta_theta
        analytic_points[i] = analytic_points[i - 1] + segment_length * np.array(
            [np.cos(theta), 0.0, -np.sin(theta)], dtype=np.float64
        )

    tip_target_pos = analytic_points[num_segments - 1]
    tip_angle = (num_segments - 1) * delta_theta
    half = 0.5 * tip_angle
    tip_target_quat = np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=np.float64)

    rest_body_q = state_0.body_q.numpy().astype(np.float64)
    rest_tip_quat = rest_body_q[tip_body, 3:7].copy()
    rest_tip_pos = rest_body_q[tip_body, :3].copy()
    tip_final_quat = _quat_mul(tip_target_quat, rest_tip_quat)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    ramp_frames = 120
    min_hold_frames = 30
    max_hold_frames = 240
    settle_speed = 5.0e-5

    def _set_tip(scale):
        tip_pos_now = (1.0 - scale) * rest_tip_pos + scale * tip_target_pos
        half_now = 0.5 * tip_angle * scale
        delta_quat = np.array([0.0, np.sin(half_now), 0.0, np.cos(half_now)], dtype=np.float64)
        tip_quat_now = _quat_mul(delta_quat, rest_tip_quat)

        body_q = state_0.body_q.numpy()
        body_q[tip_body, :3] = tip_pos_now.astype(np.float32)
        body_q[tip_body, 3:7] = tip_quat_now.astype(np.float32)
        state_0.body_q.assign(body_q)
        state_1.body_q.assign(body_q)

    def _step_frame(scale):
        nonlocal state_0, state_1
        _set_tip(scale)
        for _ in range(sim_substeps):
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

    def _max_dynamic_speed():
        body_qd = state_0.body_qd.numpy().astype(np.float64)
        linear = float(np.max(np.linalg.norm(body_qd[dynamic_body_indices, :3], axis=1)))
        angular = float(np.max(np.linalg.norm(body_qd[dynamic_body_indices, 3:6], axis=1)))
        return linear, angular

    for frame in range(ramp_frames):
        _step_frame((frame + 1) / ramp_frames)

    for frame in range(max_hold_frames):
        _step_frame(1.0)
        max_lin_speed, max_ang_speed = _max_dynamic_speed()
        if frame + 1 >= min_hold_frames and max(max_lin_speed, max_ang_speed) < settle_speed:
            break

    body_q = state_0.body_q.numpy().astype(np.float64)
    centerline = body_q[body_indices, :3]
    edges = np.diff(centerline, axis=0)
    edge_norms = np.linalg.norm(edges, axis=1)
    edges_unit = edges / edge_norms[:, None]
    cos_per_joint = np.clip(np.einsum("ij,ij->i", edges_unit[:-1], edges_unit[1:]), -1.0, 1.0)
    angles_per_joint = np.arccos(cos_per_joint)

    angle_rms_deg = float(np.rad2deg(np.sqrt(np.mean((angles_per_joint - delta_theta) ** 2))))
    max_angle_err_deg = float(np.rad2deg(np.max(np.abs(angles_per_joint - delta_theta))))
    shape_err = np.linalg.norm(centerline - analytic_points[:num_segments], axis=1)
    shape_rms_rel = float(np.sqrt(np.mean(shape_err**2)) / cable_length)
    y_drift_rel = float(np.max(np.abs(centerline[:, 1] - centerline[0, 1])) / cable_length)
    max_stretch_rel = float(np.max(np.abs(edge_norms - segment_length) / segment_length))
    max_lin_speed, max_ang_speed = _max_dynamic_speed()
    tip_pos_err = float(np.linalg.norm(body_q[tip_body, :3] - tip_target_pos))
    tip_quat_err = _quat_distance(body_q[tip_body, 3:7], tip_final_quat)

    max_measured_bend = float(np.max(np.abs(angles_per_joint)))

    diag = {
        "angle_rms_deg": angle_rms_deg,
        "max_angle_err_deg": max_angle_err_deg,
        "shape_rms_rel": shape_rms_rel,
        "y_drift_rel": y_drift_rel,
        "max_stretch_rel": max_stretch_rel,
        "max_lin_speed": max_lin_speed,
        "max_ang_speed": max_ang_speed,
        "max_measured_bend": max_measured_bend,
    }

    test.assertTrue(np.isfinite(centerline).all(), f"non-finite cable state after settle: {diag}")
    test.assertGreater(float(np.min(edge_norms)), 0.5 * segment_length, f"segment collapsed: {diag}")
    test.assertLess(tip_pos_err, 1.0e-5, f"kinematic tip position drifted: {diag}")
    test.assertLess(tip_quat_err, 1.0e-5, f"kinematic tip orientation drifted: {diag}")
    test.assertLess(max_lin_speed, 1.0e-4, f"arc did not settle translationally: {diag}")
    test.assertLess(max_ang_speed, 2.5e-4, f"arc did not settle rotationally: {diag}")
    test.assertLess(y_drift_rel, 1.0e-4, f"pure bend produced out-of-plane drift: {diag}")
    test.assertLess(max_stretch_rel, 1.0e-3, f"segment lengths changed under pure bend: {diag}")
    test.assertLess(angle_rms_deg, 0.12, f"non-uniform per-joint bend: {diag}")
    test.assertLess(max_angle_err_deg, 0.22, f"localized bend angle error too high: {diag}")
    test.assertLess(shape_rms_rel, 1.0e-3, f"centerline drifted from analytic arc: {diag}")
    test.assertGreater(max_measured_bend, 0.5 * delta_theta, f"bend motion was not active: {diag}")


def _split_cable_angular_hessian_matches_finite_difference(test, device):
    """Projected bend/twist angular Hessians should match centered finite differences."""
    errors = wp.zeros(2, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_split_cable_angular_hessian_finite_difference_kernel,
        dim=2,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    max_fd = float(np.max(errors_np[:, 0]))
    max_sym = float(np.max(errors_np[:, 1]))
    max_null = float(np.max(errors_np[:, 2]))

    test.assertLess(max_fd, 2.0e-4, f"angular Hessian finite-difference mismatch: {errors_np}")
    test.assertLess(max_sym, 1.0e-6, f"angular Hessian symmetry mismatch: {errors_np}")
    test.assertLess(max_null, 1.0e-6, f"angular Hessian projected null-space mismatch: {errors_np}")


def _split_cable_geometric_force_hessian_matches_finite_difference(test, device):
    """Geometric cable force should be the gradient of its geometric strain energy."""
    errors = wp.zeros(4, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_geometric_cable_force_hessian_finite_difference_kernel,
        dim=4,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    max_force = float(np.max(errors_np[:, 0]))
    max_hessian = float(np.max(errors_np[:, 1]))
    max_sym = float(np.max(errors_np[:, 2]))

    test.assertLess(max_force, 5.0e-4, f"geometric force finite-difference mismatch: {errors_np}")
    test.assertLess(max_hessian, 5.0e-3, f"geometric Hessian finite-difference mismatch: {errors_np}")
    test.assertLess(max_sym, 1.0e-5, f"geometric Hessian symmetry mismatch: {errors_np}")


def _split_cable_geometric_precurved_twist_does_not_leak_to_bend(test, device):
    """Pure material twist on a pre-curved rest joint should not create bend strain."""
    errors = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_geometric_precurved_twist_is_pure_twist_kernel,
        dim=1,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    bend_leak, twist_err, twist_mag = errors_np[0]
    test.assertLess(bend_leak, 1.0e-6, f"pre-curved pure twist leaked into bend: {errors_np}")
    test.assertLess(twist_err, 1.0e-6, f"pre-curved pure twist magnitude changed: {errors_np}")
    test.assertGreater(twist_mag, 0.1, f"twist regression test is vacuous: {errors_np}")


def _split_cable_geometric_rest_strain_is_global_rotation_invariant(test, device):
    """A rigid global rotation of the authored rest shape should not create cable strain."""
    errors = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_geometric_global_rotation_preserves_rest_strain_kernel,
        dim=1,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    strain_mag = errors_np[0, 0]
    test.assertLess(strain_mag, 1.0e-6, f"global rotation changed geometric rest strain: {errors_np}")


def _split_cable_bishop_transport_handles_antiparallel_fallback(test, device):
    """Antiparallel tangents still need a transport axis perpendicular to the tangent."""
    errors = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_bishop_transport_antiparallel_fallback_kernel,
        dim=1,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    map_error, antiparallel_error, mapped_len = errors_np[0]
    test.assertLess(map_error, 1.0e-6, f"Bishop fallback did not map t0 to -t0: {errors_np}")
    test.assertLess(antiparallel_error, 1.0e-6, f"Bishop fallback result was not antiparallel: {errors_np}")
    test.assertGreater(mapped_len, 0.9, f"Bishop fallback produced a degenerate vector: {errors_np}")


def _split_cable_curvature_binormal_derivative_matches_finite_difference(test, device):
    """Curvature-binormal derivative should match the exact capped value function."""
    errors = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_geometric_curvature_binormal_derivative_kernel,
        dim=1,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    regular_error, capped_error, capped_tangent_error = errors_np[0]
    test.assertLess(regular_error, 5.0e-4, f"regular curvature derivative finite-difference mismatch: {errors_np}")
    test.assertLess(capped_error, 5.0e-3, f"capped curvature derivative finite-difference mismatch: {errors_np}")
    test.assertLess(capped_tangent_error, 1.0e-5, f"capped derivative changed capped magnitude: {errors_np}")


def _split_cable_transported_twist_derivative_matches_finite_difference(test, device):
    """Transported twist derivative should match finite differences for parent and child rotations."""
    errors = wp.zeros(4, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_transported_twist_angle_derivative_kernel,
        dim=4,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    max_error = float(np.max(errors_np[:, 0]))
    max_signal = float(np.max(errors_np[:, 1:]))
    test.assertLess(max_error, 5.0e-4, f"transported twist derivative finite-difference mismatch: {errors_np}")
    test.assertGreater(max_signal, 0.05, f"transported twist derivative test is vacuous: {errors_np}")


def _split_cable_geometric_curvature_binormal_is_capped(test, device):
    """Near-fold bend should stay finite under the VBD curvature-binormal cap."""
    errors = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_geometric_curvature_binormal_cap_kernel,
        dim=1,
        outputs=[errors],
        device=device,
    )

    errors_np = errors.numpy()
    cap_error, _finite_placeholder, kappa_mag = errors_np[0]
    test.assertLess(cap_error, 1.0e-5, f"near-fold curvature was not capped at 20: {errors_np}")
    test.assertGreater(kappa_mag, 19.9, f"cap regression test is vacuous: {errors_np}")


def _split_cable_dahl_full_step_state_stays_in_active_subspace(test, device):
    """A solver step with Dahl enabled should not leak pure bend history into twist, or vice versa."""
    builder = newton.ModelBuilder(gravity=0.0)
    newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    bend_body = builder.add_link(xform=wp.transform_identity())
    twist_body = builder.add_link(xform=wp.transform_identity())
    for body in (bend_body, twist_body):
        builder.body_mass[body] = 0.0
        builder.body_inv_mass[body] = 0.0
        builder.body_inertia[body] = wp.mat33(0.0)
        builder.body_inv_inertia[body] = wp.mat33(0.0)

    bend_joint = builder.add_joint_cable(-1, bend_body, bend_stiffness=10.0, twist_stiffness=2.0)
    twist_joint = builder.add_joint_cable(-1, twist_body, bend_stiffness=10.0, twist_stiffness=2.0)
    builder.add_articulation([bend_joint])
    builder.add_articulation([twist_joint])
    builder.color()
    model = builder.finalize(device=device)
    model.vbd.dahl_eps_max.fill_(0.2)
    model.vbd.dahl_tau.fill_(0.2)

    solver = newton.solvers.SolverVBD(model, iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    body_q = state_0.body_q.numpy()
    body_q[bend_body] = [0.0, 0.0, 0.0, *wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)]
    body_q[twist_body] = [0.0, 0.0, 0.0, *wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.1)]
    state_0.body_q.assign(body_q)
    state_1.body_q.assign(body_q)

    solver.step(state_0, state_1, control, None, 1.0 / 60.0)

    sigma = solver.joint_sigma_prev.numpy()
    kappa = solver.joint_kappa_prev.numpy()
    d_kappa = solver.joint_dkappa_prev.numpy()

    test.assertGreater(np.linalg.norm(sigma[bend_joint, :2]), 1.0e-4)
    test.assertGreater(abs(float(sigma[twist_joint, 2])), 1.0e-4)
    np.testing.assert_allclose(sigma[bend_joint, 2], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(sigma[twist_joint, :2], [0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(kappa[bend_joint, 2], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(kappa[twist_joint, :2], [0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(d_kappa[bend_joint, 2], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(d_kappa[twist_joint, :2], [0.0, 0.0], atol=1.0e-6)


class TestCable(unittest.TestCase):
    pass


add_function_test(
    TestCable,
    "test_cable_fixed_joint_tracks_moving_kinematic",
    _cable_fixed_joint_tracks_moving_kinematic_impl,
    devices=devices,
)
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
    "test_cable_eval_fk_preserves_body_state",
    _cable_eval_fk_preserves_body_state_impl,
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
add_function_test(
    TestCable,
    "test_split_cable_angular_slot_layout",
    _split_cable_angular_slot_layout,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_cable_stiffness_helper_returns_physical_twist",
    _cable_stiffness_helper_returns_physical_twist,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_dahl_uses_bend_and_twist_envelopes",
    _split_cable_dahl_uses_bend_and_twist_envelopes,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_material_basis_invariants",
    _split_cable_material_basis_invariants,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_material_basis_is_continuous_near_axis_cutoff",
    _split_cable_material_basis_is_continuous_near_axis_cutoff,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_bend_twist_projectors_have_no_pure_mode_leakage",
    _split_cable_bend_twist_projectors_have_no_pure_mode_leakage,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_stretch_shear_projectors_have_no_pure_mode_leakage",
    _split_cable_stretch_shear_projectors_have_no_pure_mode_leakage,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_routes_explicit_shear_to_second_slot",
    _split_cable_routes_explicit_shear_to_second_slot,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_angular_stiffness_isolated_by_projector",
    _split_cable_angular_stiffness_isolated_by_projector,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_material_force_law_matches_ei_gj",
    _split_cable_material_force_law_matches_ei_gj,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_discrete_cantilever_moment_law_matches_beam_limit",
    _split_cable_discrete_cantilever_moment_law_matches_beam_limit,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_kinematic_arc_yields_uniform_curvature",
    _split_cable_kinematic_arc_yields_uniform_curvature,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_angular_hessian_matches_finite_difference",
    _split_cable_angular_hessian_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_geometric_force_hessian_matches_finite_difference",
    _split_cable_geometric_force_hessian_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_geometric_precurved_twist_does_not_leak_to_bend",
    _split_cable_geometric_precurved_twist_does_not_leak_to_bend,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_geometric_rest_strain_is_global_rotation_invariant",
    _split_cable_geometric_rest_strain_is_global_rotation_invariant,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_bishop_transport_handles_antiparallel_fallback",
    _split_cable_bishop_transport_handles_antiparallel_fallback,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_curvature_binormal_derivative_matches_finite_difference",
    _split_cable_curvature_binormal_derivative_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_transported_twist_derivative_matches_finite_difference",
    _split_cable_transported_twist_derivative_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_geometric_curvature_binormal_is_capped",
    _split_cable_geometric_curvature_binormal_is_capped,
    devices=devices,
)
add_function_test(
    TestCable,
    "test_split_cable_dahl_full_step_state_stays_in_active_subspace",
    _split_cable_dahl_full_step_state_stays_in_active_subspace,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
