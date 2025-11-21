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
        bend_stiffness: Cable bend stiffness passed to :func:`add_rod_mesh`.
        bend_damping: Cable bend damping passed to :func:`add_rod_mesh`.
        segment_length: Rest length of each capsule segment.
    """
    builder = newton.ModelBuilder()

    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    # Geometry: straight cable along +X, centered around the origin
    num_elements = num_links
    cable_length = num_elements * segment_length

    # Points: segment endpoints along X at some height
    z_height = 3.0
    start_x = -0.5 * cable_length
    points = []
    for i in range(num_elements + 1):
        x = start_x + i * segment_length
        points.append(wp.vec3(x, 0.0, z_height))

    # Capsule internal axis is +Z; rotate so rod axis aligns with +X
    # Use a single orientation for all segments since the cable is straight.
    # quat_between_vectors(local +Z, world +X) gives the minimal rotation that aligns the axes.
    rot_z_to_x = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
    edge_q = [rot_z_to_x] * num_elements

    # Create a rod-based cable
    rod_bodies, _rod_joints = builder.add_rod_mesh(
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

    # Geometry: points on a circle in the X–Y plane at fixed height
    num_elements = num_links
    radius = 1.0
    z_height = 3.0

    points = []
    for i in range(num_elements + 1):
        # For a closed loop we wrap the last point back to the first
        angle = 2.0 * wp.pi * float(i) / float(num_elements)
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

    _rod_bodies, _rod_joints = builder.add_rod_mesh(
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


def _assert_capsule_attachments(
    test: unittest.TestCase,
    body_q: np.ndarray,
    rod_bodies: list[int],
    segment_length: float,
    context: str,
) -> None:
    """Assert that adjacent capsules remain attached within 5% of segment length.

    Approximates the parent capsule end and child capsule start in world space and
    checks that their separation is small relative to the rest capsule length.
    """
    tol = 0.05 * segment_length
    for i in range(len(rod_bodies) - 1):
        idx_p = rod_bodies[i]
        idx_c = rod_bodies[i + 1]

        p = body_q[idx_p, :3]
        c = body_q[idx_c, :3]

        dir_vec = c - p
        seg_len_now = np.linalg.norm(dir_vec)
        if seg_len_now > 1.0e-6:
            dir = dir_vec / seg_len_now
        else:
            dir = np.array([1.0, 0.0, 0.0], dtype=float)

        parent_end = p + dir * segment_length
        child_start = c
        dist = np.linalg.norm(parent_end - child_start)

        test.assertLessEqual(
            dist,
            tol,
            msg=f"{context}: capsule attachment gap too large at segment {i} (dist={dist}, tol={tol})",
        )


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
    cable_pairs = set(zip(cable_parents.tolist(), cable_children.tolist()))
    test.assertEqual(len(cable_pairs), len(cable_indices))

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
    cable_pairs = list(zip(cable_parents.tolist(), cable_children.tolist()))
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
    # From soft to stiff
    bend_values = [1.0e1, 1.0e2, 1.0e3]
    tip_heights = []
    segment_length = 0.2

    for k in bend_values:
        model, state0, state1, control, rod_bodies = _build_cable_chain(
            device, num_links=10, bend_stiffness=k, bend_damping=1.0e1, segment_length=segment_length
        )
        solver = newton.solvers.SolverVBD(model, iterations=10)
        frame_dt = 1.0 / 60.0
        sim_substeps = 10
        sim_dt = frame_dt / sim_substeps

        # Run for a short duration to let bending respond to gravity
        for _ in range(20):
            for _ in range(sim_substeps):
                state0.clear_forces()
                contacts = model.collide(state0)
                solver.step(state0, state1, control, contacts, sim_dt)
                state0, state1 = state1, state0

        final_q = state0.body_q.numpy()
        tip_body = rod_bodies[-1]
        tip_heights.append(float(final_q[tip_body, 2]))

        # Check capsule attachments for this dynamic configuration
        _assert_capsule_attachments(
            test,
            body_q=final_q,
            rod_bodies=rod_bodies,
            segment_length=segment_length,
            context=f"Bend stiffness {k}",
        )

    tip_heights = np.array(tip_heights, dtype=float)

    # Check that stiffer cables have higher tip positions (less sagging under gravity)
    # Expect monotonic increase: tip_heights[0] < tip_heights[1] < tip_heights[2]
    for i in range(len(tip_heights) - 1):
        test.assertLess(
            tip_heights[i],
            tip_heights[i + 1],
            msg=(
                f"Stiffer cable should have higher tip (less sag): "
                f"k={bend_values[i]:.1e} → z={tip_heights[i]:.4f}, "
                f"k={bend_values[i + 1]:.1e} → z={tip_heights[i + 1]:.4f}"
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
    model, state0, state1, control, rod_bodies = _build_cable_chain(
        device, num_links=6, segment_length=segment_length
    )
    solver = newton.solvers.SolverVBD(model, iterations=10)
    frame_dt = 1.0 / 60.0
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Record initial positions
    initial_q = state0.body_q.numpy().copy()
    z_initial = initial_q[:, 2]

    for _ in range(20):
        for _ in range(sim_substeps):
            state0.clear_forces()
            contacts = model.collide(state0)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    final_q = state0.body_q.numpy()
    z_final = final_q[:, 2]

    # At least one cable body should move downward
    test.assertTrue((z_final < z_initial).any())

    # Positions should remain within a band relative to initial height and cable length
    z0 = float(z_initial.min())
    x_initial = initial_q[:, 0]
    cable_length = float(x_initial.max() - x_initial.min())
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

    rod_bodies, _rod_joints = builder.add_rod_mesh(
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

    # Record initial orientation and position of the free (child) body
    child_body = rod_bodies[-1]
    q_initial = state0.body_q.numpy().copy()
    q_child_initial = q_initial[child_body, 3:].copy()  # [qw, qx, qy, qz]
    p_child_initial = q_initial[child_body, :3].copy()

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

    # Run a short simulation to let twist propagate
    for _ in range(20):
        for _ in range(sim_substeps):
            state0.clear_forces()
            contacts = model.collide(state0)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    final_q = state0.body_q.numpy()

    # Check capsule attachments remain good
    _assert_capsule_attachments(test, body_q=final_q, rod_bodies=rod_bodies, segment_length=segment_length, context="Twist")

    # Check that the child orientation has changed significantly due to twist
    q_child_final = final_q[child_body, 3:]

    # Quaternion dot product magnitude indicates orientation similarity (1 => identical up to sign)
    dot = float(abs(np.dot(q_child_initial, q_child_final)))
    test.assertLess(
        dot,
        0.999,
        msg=f"Twist: child orientation changed too little (|dot|={dot}); expected noticeable rotation from twist.",
    )

    # Also check a specific geometric response: in the orthogonal "L" configuration,
    # twisting 180 degrees about the +X axis should reflect the free capsule across the X–Z plane:
    # its Y coordinate should change sign while X and Z remain approximately the same.
    
    # We check the tip of the capsule, because the body origin is at the pivot (which doesn't move).
    def get_tip_pos(body_idx, q_all):
        p = q_all[body_idx, :3]
        q = q_all[body_idx, 3:] # x, y, z, w
        rot = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        v = wp.vec3(0.0, 0.0, segment_length)
        v_rot = wp.quat_rotate(rot, v)
        return np.array([p[0] + v_rot[0], p[1] + v_rot[1], p[2] + v_rot[2]])

    tip_initial = get_tip_pos(child_body, q_initial)
    tip_final = get_tip_pos(child_body, final_q)
    
    tol = 0.1 * segment_length

    # X and Z should stay close to their original values
    test.assertAlmostEqual(
        float(tip_final[0]),
        float(tip_initial[0]),
        delta=tol,
        msg=f"Twist: expected child tip x to stay near {float(tip_initial[0])}, got {float(tip_final[0])}",
    )
    test.assertAlmostEqual(
        float(tip_final[2]),
        float(tip_initial[2]),
        delta=tol,
        msg=f"Twist: expected child tip z to stay near {float(tip_initial[2])}, got {float(tip_final[2])}",
    )

    # Y should approximately flip sign (reflect across the X–Z plane)
    # Initial tip Y should be approx segment_length (0.2)
    # We check if the sign is flipped, but allow for some deviation in magnitude
    # because the twist might not be perfectly 180 degrees or there might be some energy loss/damping
    test.assertTrue(
        float(tip_final[1]) * float(tip_initial[1]) < 0,
        msg=f"Twist: expected child tip y to flip sign from {float(tip_initial[1])}, got {float(tip_final[1])}"
    )
    test.assertAlmostEqual(
        float(tip_final[1]),
        float(-tip_initial[1]),
        delta=tol,
        msg=(
            "Twist: expected child tip y magnitude to be similar "
            f"from {abs(float(tip_initial[1]))}, "
            f"got {abs(float(tip_final[1]))}"
        ),
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
    for layer in range(2):
        orient = "x" if layer == 0 else "y"
        z0 = base_height + layer * layer_gap

        for lane in range(2):
            # Symmetric offset: lane 0 → -0.5*spacing, lane 1 → +0.5*spacing
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

            builder.add_rod_mesh(
                positions=points,
                quaternions=edge_q,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-1,
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                key=f"pile_l{layer}_{lane}",
            )

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
    for _ in range(num_steps):
        for _ in range(sim_substeps):
            state0.clear_forces()
            contacts = model.collide(state0)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    body_q = state0.body_q.numpy()
    positions = body_q[:, :3]
    z_positions = positions[:, 2]

    # Basic sanity checks
    test.assertTrue(np.isfinite(positions).all(), "Non-finite positions detected in cable pile")

    # Define vertical bands with a small margin for soft contact tolerance
    # Increased margin to account for contact compression and stiff cable deformation
    margin = 0.1 * cable_width

    # Bottom layer should live between ground and one cable width (±margin)
    bottom_band = (z_positions >= -margin) & (z_positions <= cable_width + margin)

    # Second layer between one and two cable widths (±margin)
    top_band = (z_positions >= cable_width - margin) & (z_positions <= 2.0 * cable_width + margin)

    # All bodies should fall within one of the two bands
    in_bands = bottom_band | top_band
    test.assertTrue(
        np.all(in_bands),
        msg=(
            "Some cable bodies lie outside the expected two-layer vertical bands: "
            f"min_z={float(z_positions.min()):.4f}, max_z={float(z_positions.max()):.4f}, "
            f"cable_width={cable_width:.4f}, expected in [0, {2.0 * cable_width + margin:.4f}] "
            f"with band margin {margin:.4f}."
        ),
    )

    # Ensure we actually formed two distinct layers
    num_bottom = np.sum(bottom_band)
    num_top = np.sum(top_band)
    
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
        total_bodies * 0.1,
        msg=f"Bottom layer has too few bodies: {num_bottom}/{total_bodies} (< 10%)",
    )
    test.assertGreater(
        num_top,
        total_bodies * 0.1,
        msg=f"Top layer has too few bodies: {num_top}/{total_bodies} (< 10%)",
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

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)


