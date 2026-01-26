# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

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
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    body_q[body_id] = pose
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _drive_gripper_boxes_kernel(
    ramp_time: float,
    t: float,
    body_ids: wp.array(dtype=wp.int32),
    signs: wp.array(dtype=wp.float32),
    anchor_p: wp.vec3,
    anchor_q: wp.quat,
    seg_half_len: float,
    target_offset_mag: float,
    initial_offset_mag: float,
    pull_start_time: float,
    pull_ramp_time: float,
    pull_distance: float,
    body_q: wp.array(dtype=wp.transform),
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
    cable_length = num_elements * segment_length
    start_x = -0.5 * cable_length
    points = [wp.vec3(start_x + i * segment_length, 0.0, z_height) for i in range(num_elements + 1)]

    # Capsule internal axis is +Z; rotate so rod axis aligns with +X
    rot_z_to_x = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
    edge_q = [rot_z_to_x] * num_elements
    return points, edge_q


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
        key="test_cable_chain",
    )

    if pin_first and len(rod_bodies) > 0:
        first_body = rod_bodies[0]
        builder.body_mass[first_body] = 0.0
        builder.body_inv_mass[first_body] = 0.0

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

    # Orient capsules tangentially along the circle
    edge_q = []
    for i in range(num_elements):
        p0 = points[i]
        p1 = points[i + 1]
        dir_vec = wp.normalize(p1 - p0)

        # Capsule internal axis is +Z; rotate +Z into dir_vec
        q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), dir_vec)
        edge_q.append(q)

    _rod_bodies, _rod_joints = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=0.05,
        bend_stiffness=1.0e1,
        bend_damping=1.0e-2,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-2,
        closed=True,
        key="test_cable_loop",
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


def _compute_ball_joint_anchor_error(model: newton.Model, body_q: wp.array, joint_id: int) -> float:
    """Compute BALL joint world-space anchor error (CPU float).

    Returns:
        |x_c - x_p|, where x_p and x_c are the parent/child anchor positions in world space.
    """
    with wp.ScopedDevice("cpu"):
        jp = model.joint_parent.numpy()[joint_id].item()
        jc = model.joint_child.numpy()[joint_id].item()
        X_p = model.joint_X_p.numpy()[joint_id]
        X_c = model.joint_X_c.numpy()[joint_id]

        # wp.transform is [p(3), q(4)] in xyzw order
        X_pj = wp.transform(wp.vec3(X_p[0], X_p[1], X_p[2]), wp.quat(X_p[3], X_p[4], X_p[5], X_p[6]))
        X_cj = wp.transform(wp.vec3(X_c[0], X_c[1], X_c[2]), wp.quat(X_c[3], X_c[4], X_c[5], X_c[6]))

        bq = body_q.to("cpu").numpy()
        q_p = bq[jp]
        q_c = bq[jc]
        T_p = wp.transform(wp.vec3(q_p[0], q_p[1], q_p[2]), wp.quat(q_p[3], q_p[4], q_p[5], q_p[6]))
        T_c = wp.transform(wp.vec3(q_c[0], q_c[1], q_c[2]), wp.quat(q_c[3], q_c[4], q_c[5], q_c[6]))

        X_wp = T_p * X_pj
        X_wc = T_c * X_cj
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        return float(wp.length(x_c - x_p))


def _compute_fixed_joint_frame_error(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[float, float]:
    """Compute FIXED joint world-space frame error (CPU floats).

    Returns:
        (pos_err, ang_err)

        - pos_err: |x_c - x_p| where x_p/x_c are the parent/child joint-frame translations in world space.
        - ang_err: relative rotation angle between joint-frame orientations in world space [rad].
    """
    with wp.ScopedDevice("cpu"):
        jp = model.joint_parent.numpy()[joint_id].item()
        jc = model.joint_child.numpy()[joint_id].item()
        X_p = model.joint_X_p.numpy()[joint_id]
        X_c = model.joint_X_c.numpy()[joint_id]

        # wp.transform is [p(3), q(4)] in xyzw order
        X_pj = wp.transform(wp.vec3(X_p[0], X_p[1], X_p[2]), wp.quat(X_p[3], X_p[4], X_p[5], X_p[6]))
        X_cj = wp.transform(wp.vec3(X_c[0], X_c[1], X_c[2]), wp.quat(X_c[3], X_c[4], X_c[5], X_c[6]))

        bq = body_q.to("cpu").numpy()
        q_p = bq[jp]
        q_c = bq[jc]
        T_p = wp.transform(wp.vec3(q_p[0], q_p[1], q_p[2]), wp.quat(q_p[3], q_p[4], q_p[5], q_p[6]))
        T_c = wp.transform(wp.vec3(q_c[0], q_c[1], q_c[2]), wp.quat(q_c[3], q_c[4], q_c[5], q_c[6]))

        X_wp = T_p * X_pj
        X_wc = T_c * X_cj

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        pos_err = float(wp.length(x_c - x_p))

        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_rel = wp.mul(wp.quat_inverse(q_wp), q_wc)
        q_rel = wp.normalize(q_rel)
        # Quaternion sign is arbitrary; enforce shortest-path angle for robustness.
        w = wp.clamp(wp.abs(q_rel[3]), 0.0, 1.0)
        ang_err = float(2.0 * wp.acos(w))

        return pos_err, ang_err


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
            key=f"bend_stiffness_{k:.0e}",
        )

        # Pin the first body of each cable.
        first_body = rod_bodies[0]
        builder.body_mass[first_body] = 0.0
        builder.body_inv_mass[first_body] = 0.0

        all_rod_bodies.append(rod_bodies)
        tip_bodies.append(rod_bodies[-1])

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0, state1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=10)

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    # Run for a short duration to let bending respond to gravity
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            contacts = model.collide(state0)
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
            contacts = model.collide(state0)
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
        key="twist_chain_orthogonal",
    )

    # Pin the first body (anchored capsule)
    first_body = rod_bodies[0]
    builder.body_mass[first_body] = 0.0
    builder.body_inv_mass[first_body] = 0.0

    builder.color()
    model = builder.finalize(device=device)
    state0 = model.state()
    state1 = model.state()
    control = model.control()

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
            contacts = model.collide(state0)
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
                key=f"pile_l{layer}_{lane}",
            )
            cable_bodies.extend(rod_bodies)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()

    solver = newton.solvers.SolverVBD(model, iterations=10, friction_epsilon=0.1)
    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Let the pile settle under gravity and contact
    num_steps = 20
    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            state0.clear_forces()
            contacts = model.collide(state0)
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
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    # Anchor marker sphere.
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)

    # Build a straight cable (rod) and attach its start endpoint to the anchor with a BALL joint.
    num_elements = 20
    segment_length = 0.05
    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    rod_radius = 0.01
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
        key="test_cable_ball_joint_attach",
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

            contacts = model.collide(state0)
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
        margin=0.25 * segment_length,
        context="Cable BALL joint attachment",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable BALL joint attachment",
    )


def _cable_fixed_joint_attaches_rod_endpoint_impl(test: unittest.TestCase, device):
    """Cable VBD: FIXED joint should keep rod start frame welded to a kinematic anchor."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    anchor_pos = wp.vec3(0.0, 0.0, 3.0)

    # Build a straight cable along +X and use its segment orientation for the kinematic anchor.
    num_elements = 20
    segment_length = 0.05
    points, edge_q = _make_straight_cable_along_x(num_elements, segment_length, z_height=anchor_pos[2])
    anchor_rot = edge_q[0]

    # Kinematic anchor body at the rod start point (match rod orientation).
    anchor = builder.add_body(xform=wp.transform(anchor_pos, anchor_rot))
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    # Anchor marker sphere.
    anchor_radius = 0.1
    builder.add_shape_sphere(anchor, radius=anchor_radius)

    rod_radius = 0.01
    # Attach the cable endpoint to the sphere surface (not the center), accounting for cable radius so the
    # capsule endcap surface and the sphere surface are coincident along the rod axis (+X).
    # The rod axis is world +X, and anchor_rot maps parent-local +Z -> world +X, so use +Z in parent local.
    attach_offset = wp.float32(anchor_radius + rod_radius)
    parent_anchor_local = wp.vec3(0.0, 0.0, attach_offset)
    anchor_world_attach = anchor_pos + wp.quat_rotate(anchor_rot, parent_anchor_local)

    # Reposition the generated cable so its first point coincides with the sphere-surface attach point.
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
        key="test_cable_fixed_joint_attach",
    )

    child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
    j_fixed = builder.add_joint_fixed(
        parent=anchor,
        child=rod_bodies[0],
        parent_xform=wp.transform(parent_anchor_local, wp.quat_identity()),
        child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
    )
    builder.add_articulation([*rod_joints, j_fixed])

    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    state0 = model.state()
    state1 = model.state()
    control = model.control()

    # Stiffen both linear and angular caps for non-cable joints so FIXED behaves near-hard.
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        rigid_joint_linear_ke=1.0e9,
        rigid_joint_angular_ke=1.0e9,
        rigid_joint_linear_k_start=1.0e7,
        rigid_joint_angular_k_start=1.0e7,
    )

    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_steps = 20

    for _step in range(num_steps):
        for _substep in range(sim_substeps):
            t = (_step * sim_substeps + _substep) * sim_dt
            # Use wp.float32 so Warp builtins match expected scalar types (avoid numpy.float64).
            dx = wp.float32(0.05 * np.sin(1.5 * t))
            ang = wp.float32(0.4 * np.sin(1.5 * t + 0.7))
            q_drive = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang)
            q = wp.mul(q_drive, anchor_rot)

            pose = wp.transform(wp.vec3(dx, 0.0, anchor_pos[2]), q)
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[wp.int32(anchor), pose, state0.body_q, state0.body_qd],
                device=device,
            )

            contacts = model.collide(state0)
            solver.step(state0, state1, control, contacts, dt=sim_dt)
            state0, state1 = state1, state0

            pos_err, ang_err = _compute_fixed_joint_frame_error(model, state0.body_q, j_fixed)
            test.assertLess(pos_err, 1.0e-3)
            test.assertLess(ang_err, 2.0e-2)

    final_q = state0.body_q.numpy()
    test.assertTrue(np.isfinite(final_q).all(), "Non-finite body transforms detected in FIXED joint test")
    _assert_surface_attachment(
        test,
        body_q=final_q,
        anchor_body=anchor,
        child_body=rod_bodies[0],
        context="Cable FIXED joint attachment",
        parent_anchor_local=parent_anchor_local,
    )

    _assert_bodies_above_ground(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        margin=0.25 * segment_length,
        context="Cable FIXED joint attachment",
    )
    _assert_capsule_attachments(
        test,
        body_q=final_q,
        body_ids=rod_bodies,
        segment_length=segment_length,
        context="Cable FIXED joint attachment",
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
        key="ut_gripper_capsule",
    )
    payload_cfg = builder.default_shape_cfg.copy()
    payload_cfg.mu = 1.0e3
    builder.add_shape_capsule(
        body=capsule_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=capsule_rot_z_to_x),
        radius=capsule_radius,
        half_height=capsule_half_height,
        cfg=payload_cfg,
        key="ut_gripper_capsule_shape",
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
        key="ut_gripper_neg",
    )
    g_pos = builder.add_body(
        xform=wp.transform(p=anchor_p + wp.vec3(0.0, initial_offset_mag, 0.0), q=anchor_q),
        mass=0.0,
        key="ut_gripper_pos",
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

        contacts = model.collide(state0)
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


class TestCable(unittest.TestCase):
    pass


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
    "test_cable_kinematic_gripper_picks_capsule",
    _cable_kinematic_gripper_picks_capsule_impl,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
