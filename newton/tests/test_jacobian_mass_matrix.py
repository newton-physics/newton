# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval_jacobian() and eval_mass_matrix() functions."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_translated_prismatic_chain(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link(mass=1.5)
    slider = builder.add_link(mass=0.9)

    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
    builder.add_shape_box(slider, hx=0.15, hy=0.1, hz=0.08)

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
    builder.add_articulation([j0, j1], label="translated_slider")

    return builder.finalize(device=device), base, slider


def _build_free_body_with_com(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    body = builder.add_body(
        xform=wp.transform(
            wp.vec3(1.0, -0.5, 2.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35),
        ),
        mass=1.7,
    )
    builder.add_shape_box(body, hx=0.2, hy=0.1, hz=0.15)
    builder.body_com[body] = wp.vec3(0.4, -0.15, 0.2)

    return builder.finalize(device=device), body


def _build_descendant_free_with_rotated_parent(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    base = builder.add_link(is_kinematic=True, mass=1.0)
    child = builder.add_link(mass=1.8)

    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(child, hx=0.18, hy=0.12, hz=0.09)
    builder.body_com[child] = wp.vec3(0.25, -0.1, 0.2)

    j0 = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transform(
            wp.vec3(),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        child_xform=wp.transform_identity(),
    )
    j1 = builder.add_joint_free(
        parent=base,
        child=child,
        child_xform=wp.transform(
            wp.vec3(0.15, 0.0, -0.1),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.35),
        ),
    )
    builder.add_articulation([j0, j1], label="descendant_free")

    return builder.finalize(device=device), base, child


def _kinetic_energy_from_body_twists(model, state, bodies):
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    body_inertia = model.body_inertia.numpy()
    body_mass = model.body_mass.numpy()

    kinetic = 0.0
    for body in bodies:
        quat = wp.quat(
            float(body_q[body, 3]),
            float(body_q[body, 4]),
            float(body_q[body, 5]),
            float(body_q[body, 6]),
        )
        R = np.array(wp.quat_to_matrix(quat), dtype=np.float64).reshape(3, 3)
        I_world = R @ body_inertia[body].astype(np.float64) @ R.T
        v_com = body_qd[body, :3].astype(np.float64)
        omega = body_qd[body, 3:6].astype(np.float64)
        kinetic += 0.5 * float(body_mass[body]) * float(v_com @ v_com)
        kinetic += 0.5 * float(omega @ (I_world @ omega))

    return kinetic


def test_jacobian_simple_pendulum(test, device):
    """Test Jacobian computation for a simple 2-link pendulum."""
    builder = newton.ModelBuilder()

    # Create a 2-link pendulum
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Compute FK first
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern - let function allocate)
    J = newton.eval_jacobian(model, state)

    test.assertIsNotNone(J)
    test.assertEqual(J.shape[0], model.articulation_count)
    test.assertEqual(J.shape[1], model.max_joints_per_articulation * 6)
    test.assertEqual(J.shape[2], model.max_dofs_per_articulation)

    J_np = J.numpy()

    # For a revolute joint about Z-axis at identity:
    # Motion subspace should be [0, 0, 0, 0, 0, 1] (linear velocity from angular motion)
    # At identity configuration, first joint affects both links
    # Check that Jacobian has non-zero entries for angular velocity (index 5)
    test.assertNotEqual(J_np[0, 5, 0], 0.0)  # First link, angular z, first dof
    test.assertNotEqual(J_np[0, 11, 0], 0.0)  # Second link, angular z, first dof
    test.assertNotEqual(J_np[0, 11, 1], 0.0)  # Second link, angular z, second dof


def test_jacobian_numerical_verification(test, device):
    """Verify Jacobian shape and basic properties."""
    builder = newton.ModelBuilder()

    # Create a simple pendulum
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Set a non-zero joint angle
    joint_q = state.joint_q.numpy()
    joint_q[0] = 0.5
    state.joint_q.assign(joint_q)

    # Compute FK
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)
    J_np = J.numpy()

    # Verify shape
    test.assertEqual(J_np.shape[0], 1)  # One articulation
    test.assertEqual(J_np.shape[1], 6)  # One link * 6
    test.assertEqual(J_np.shape[2], 1)  # One DOF

    # For revolute joint about z-axis, the angular z component (index 5) should be 1.0
    test.assertAlmostEqual(J_np[0, 5, 0], 1.0, places=5)


def test_mass_matrix_symmetry(test, device):
    """Test that mass matrix is symmetric."""
    builder = newton.ModelBuilder()

    # Create a 2-link pendulum with different masses
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0,
    )

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    # Set some joint angles
    joint_q = state.joint_q.numpy()
    joint_q[0] = 0.3
    joint_q[1] = 0.5
    state.joint_q.assign(joint_q)

    # Compute FK first
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute mass matrix (convenience pattern)
    H = newton.eval_mass_matrix(model, state)

    test.assertIsNotNone(H)
    test.assertEqual(H.shape[0], model.articulation_count)
    test.assertEqual(H.shape[1], model.max_dofs_per_articulation)
    test.assertEqual(H.shape[2], model.max_dofs_per_articulation)

    H_np = H.numpy()

    # Check symmetry for the valid portion of the matrix
    num_dofs = 2
    H_valid = H_np[0, :num_dofs, :num_dofs]

    np.testing.assert_allclose(H_valid, H_valid.T, rtol=1e-5, atol=1e-6)


def test_mass_matrix_positive_definite(test, device):
    """Test that mass matrix is positive definite."""
    builder = newton.ModelBuilder()

    # Create a pendulum with non-trivial inertia
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="pendulum")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute mass matrix (convenience pattern)
    H = newton.eval_mass_matrix(model, state)
    H_np = H.numpy()

    # For a single DOF, the mass matrix should be a positive scalar
    test.assertGreater(H_np[0, 0, 0], 0.0)


def test_jacobian_multiple_articulations(test, device):
    """Test Jacobian computation with multiple articulations."""
    builder = newton.ModelBuilder()

    # Create 3 independent pendulums
    for i in range(3):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)

    test.assertEqual(J.shape[0], 3)  # 3 articulations
    test.assertEqual(model.articulation_count, 3)


def test_jacobian_with_mask(test, device):
    """Test Jacobian computation with articulation mask."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums
    for i in range(2):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute with mask - only first articulation (performance pattern - pre-allocate)
    J = wp.zeros(
        (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    mask = wp.array([True, False], dtype=bool, device=device)
    J_returned = newton.eval_jacobian(model, state, J, mask=mask)

    # Verify same array is returned
    test.assertIs(J_returned, J)

    J_np = J.numpy()

    # First articulation should have non-zero Jacobian
    test.assertNotEqual(np.abs(J_np[0]).max(), 0.0)

    # Second articulation should be zero (masked out)
    test.assertEqual(np.abs(J_np[1]).max(), 0.0)


def test_mass_matrix_with_mask(test, device):
    """Test mass matrix computation with articulation mask."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums
    for i in range(2):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0 + i,  # Different masses
        )
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=f"pendulum_{i}")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute with mask - only second articulation (performance pattern - pre-allocate)
    H = wp.zeros(
        (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    mask = wp.array([False, True], dtype=bool, device=device)
    H_returned = newton.eval_mass_matrix(model, state, H, mask=mask)

    # Verify same array is returned
    test.assertIs(H_returned, H)

    H_np = H.numpy()

    # First articulation should be zero (masked out)
    test.assertEqual(H_np[0, 0, 0], 0.0)

    # Second articulation should have non-zero mass matrix
    test.assertNotEqual(H_np[1, 0, 0], 0.0)


def test_prismatic_joint_jacobian(test, device):
    """Test Jacobian for prismatic joint."""
    builder = newton.ModelBuilder()

    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    j1 = builder.add_joint_prismatic(
        parent=-1,
        child=b1,
        axis=wp.vec3(1.0, 0.0, 0.0),  # Slide along X
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="slider")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Compute Jacobian (convenience pattern)
    J = newton.eval_jacobian(model, state)
    J_np = J.numpy()

    # For prismatic joint along X, the Jacobian should have:
    # Linear velocity in X direction (index 0)
    test.assertNotEqual(J_np[0, 0, 0], 0.0)
    # Angular velocity should be zero
    test.assertEqual(J_np[0, 3, 0], 0.0)
    test.assertEqual(J_np[0, 4, 0], 0.0)
    test.assertEqual(J_np[0, 5, 0], 0.0)


def test_empty_model(test, device):
    """Test that functions handle empty model gracefully."""
    builder = newton.ModelBuilder()
    model = builder.finalize(device=device)
    state = model.state()

    J = newton.eval_jacobian(model, state)
    H = newton.eval_mass_matrix(model, state)

    test.assertIsNone(J)
    test.assertIsNone(H)


def test_articulation_view_api(test, device):
    """Test Jacobian and mass matrix via ArticulationView API."""
    builder = newton.ModelBuilder()

    # Create 2 pendulums with different keys
    for i, key in enumerate(["robot_a", "robot_b"]):
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(i * 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j1], label=key)

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Create ArticulationView for just robot_a
    view_a = newton.selection.ArticulationView(model, pattern="robot_a")

    # Test eval_jacobian via ArticulationView (convenience pattern)
    J = view_a.eval_jacobian(state)
    test.assertIsNotNone(J)
    test.assertEqual(J.shape[0], model.articulation_count)

    J_np = J.numpy()
    # robot_a (index 0) should have non-zero Jacobian
    test.assertNotEqual(np.abs(J_np[0]).max(), 0.0)
    # robot_b (index 1) should be zero (not in view)
    test.assertEqual(np.abs(J_np[1]).max(), 0.0)

    # Test eval_mass_matrix via ArticulationView (convenience pattern)
    H = view_a.eval_mass_matrix(state)
    test.assertIsNotNone(H)
    test.assertEqual(H.shape[0], model.articulation_count)

    H_np = H.numpy()
    # robot_a should have non-zero mass matrix
    test.assertNotEqual(H_np[0, 0, 0], 0.0)
    # robot_b should be zero
    test.assertEqual(H_np[1, 0, 0], 0.0)

    # Test with pre-allocated buffers (performance pattern)
    J2 = wp.zeros(
        (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )
    H2 = wp.zeros(
        (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation),
        dtype=float,
        device=device,
    )

    # Create view for robot_b
    view_b = newton.selection.ArticulationView(model, pattern="robot_b")

    J2_returned = view_b.eval_jacobian(state, J2)
    H2_returned = view_b.eval_mass_matrix(state, H2)

    test.assertIs(J2_returned, J2)
    test.assertIs(H2_returned, H2)

    J2_np = J2.numpy()
    H2_np = H2.numpy()

    # robot_a should be zero (not in view_b)
    test.assertEqual(np.abs(J2_np[0]).max(), 0.0)
    test.assertEqual(H2_np[0, 0, 0], 0.0)
    # robot_b should have values
    test.assertNotEqual(np.abs(J2_np[1]).max(), 0.0)
    test.assertNotEqual(H2_np[1, 0, 0], 0.0)


def test_floating_base_jacobian(test, device):
    """Test Jacobian for a floating base articulation (FREE joint at root)."""
    builder = newton.ModelBuilder()

    # Base link with FREE joint (6 DOFs)
    b_base = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=2.0,
    )
    builder.add_shape_box(body=b_base, hx=0.2, hy=0.2, hz=0.2)

    j_free = builder.add_joint_free(
        child=b_base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )

    # Child link with revolute joint (1 DOF)
    b_child = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body=b_child, hx=0.1, hy=0.1, hz=0.1)

    j_rev = builder.add_joint_revolute(
        parent=b_base,
        child=b_child,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_free, j_rev], label="floating_robot")

    model = builder.finalize(device=device)
    state = model.state()

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # FREE joint has 6 DOFs, revolute has 1 -> total 7 DOFs, 2 links
    test.assertEqual(model.max_dofs_per_articulation, 7)
    test.assertEqual(model.max_joints_per_articulation, 2)

    J = newton.eval_jacobian(model, state)
    test.assertEqual(J.shape, (1, 12, 7))  # 1 articulation, 2*6 rows, 7 DOFs

    J_np = J.numpy()

    # Base link (rows 0-5): should be affected by the 6 FREE DOFs (columns 0-5)
    base_block = J_np[0, 0:6, 0:6]
    test.assertNotEqual(np.abs(base_block).max(), 0.0)

    # Base link should NOT be affected by the revolute DOF (column 6)
    test.assertEqual(J_np[0, 0, 6], 0.0)
    test.assertEqual(J_np[0, 1, 6], 0.0)
    test.assertEqual(J_np[0, 2, 6], 0.0)
    test.assertEqual(J_np[0, 3, 6], 0.0)
    test.assertEqual(J_np[0, 4, 6], 0.0)
    test.assertEqual(J_np[0, 5, 6], 0.0)

    # Child link (rows 6-11): should be affected by all 7 DOFs
    child_free_block = J_np[0, 6:12, 0:6]
    test.assertNotEqual(np.abs(child_free_block).max(), 0.0)
    # Revolute DOF should give angular z velocity on the child
    test.assertNotEqual(J_np[0, 11, 6], 0.0)

    # Mass matrix should be 7x7, symmetric, and positive definite
    H = newton.eval_mass_matrix(model, state)
    test.assertEqual(H.shape, (1, 7, 7))

    H_np = H.numpy()
    H_valid = H_np[0, :7, :7]
    np.testing.assert_allclose(H_valid, H_valid.T, rtol=1e-5, atol=1e-6)

    # Check positive definiteness via Cholesky
    np.linalg.cholesky(H_valid)


def test_jacobian_matches_body_qd_com(test, device):
    """`J @ joint_qd` should match COM-referenced `state.body_qd` per link."""
    model, base, slider = _build_translated_prismatic_chain(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[0] = 0.55
    q[1] = 0.8
    qd[0] = 1.1
    qd[1] = -0.35

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()
    body_qd = state.body_qd.numpy()

    base_block = J[0, 0:6, : model.joint_dof_count]
    slider_block = J[0, 6:12, : model.joint_dof_count]
    base_twist = base_block @ qd
    slider_twist = slider_block @ qd

    np.testing.assert_allclose(base_twist, body_qd[base], atol=1.0e-5, rtol=1.0e-6)
    np.testing.assert_allclose(slider_twist, body_qd[slider], atol=1.0e-5, rtol=1.0e-6)


def test_mass_matrix_matches_com_kinetic_energy(test, device):
    """`0.5 * qd^T H qd` should equal kinetic energy from COM twists."""
    model, base, slider = _build_translated_prismatic_chain(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[0] = 0.4
    q[1] = 0.6
    qd[0] = 0.9
    qd[1] = -0.25

    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)
    kinetic_from_bodies = _kinetic_energy_from_body_twists(model, state, (base, slider))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_bodies, atol=1.0e-5, rtol=1.0e-6)


def test_floating_free_jacobian_matches_body_qd_com(test, device):
    """Floating-base Jacobian should match COM-referenced body twists."""
    model, body = _build_free_body_with_com(device)
    state = model.state()

    qd = state.joint_qd.numpy()
    qd[:] = np.array([0.35, -0.2, 0.15, 0.25, -0.4, 0.7], dtype=np.float32)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()[0, :6, : model.joint_dof_count]
    body_qd = state.body_qd.numpy()[body]

    np.testing.assert_allclose(J @ qd, body_qd, atol=1.0e-5, rtol=1.0e-6)


def test_floating_free_mass_matrix_matches_com_kinetic_energy(test, device):
    """Floating-base mass matrix should match kinetic energy from COM twists."""
    model, body = _build_free_body_with_com(device)
    state = model.state()

    qd = state.joint_qd.numpy()
    qd[:] = np.array([0.3, -0.1, 0.2, 0.45, -0.25, 0.6], dtype=np.float32)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)
    kinetic_from_body = _kinetic_energy_from_body_twists(model, state, (body,))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_body, atol=1.0e-5, rtol=1.0e-6)


def test_descendant_free_jacobian_matches_body_qd_com(test, device):
    """A rotated-parent descendant FREE chain should satisfy `J @ joint_qd == body_qd[child]` at the child COM."""
    model, _base, child = _build_descendant_free_with_rotated_parent(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[:] = np.array([0.35, -0.25, 0.45, *wp.quat_rpy(0.3, -0.2, 0.4)], dtype=np.float32)
    qd[:] = np.array([0.7, -0.15, 0.25, 0.35, -0.4, 0.5], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    J = newton.eval_jacobian(model, state).numpy()[0, 6:12, : model.joint_dof_count]
    body_qd = state.body_qd.numpy()[child]

    np.testing.assert_allclose(J @ qd, body_qd, atol=1.0e-5, rtol=1.0e-6)


def test_descendant_free_mass_matrix_matches_com_kinetic_energy(test, device):
    """A rotated-parent descendant FREE chain should match COM-based kinetic energy under the public body_qd contract."""
    model, _base, child = _build_descendant_free_with_rotated_parent(device)
    state = model.state()

    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    q[:] = np.array([-0.15, 0.4, 0.3, *wp.quat_rpy(-0.25, 0.15, 0.5)], dtype=np.float32)
    qd[:] = np.array([0.25, -0.45, 0.3, 0.55, -0.2, 0.35], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    kinetic_from_h = 0.5 * float(qd @ H @ qd)

    kinetic_from_body = _kinetic_energy_from_body_twists(model, state, (child,))
    np.testing.assert_allclose(kinetic_from_h, kinetic_from_body, atol=1.0e-5, rtol=1.0e-6)


class TestJacobianMassMatrix(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestJacobianMassMatrix, "test_jacobian_simple_pendulum", test_jacobian_simple_pendulum, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_numerical_verification",
    test_jacobian_numerical_verification,
    devices=devices,
)
add_function_test(TestJacobianMassMatrix, "test_mass_matrix_symmetry", test_mass_matrix_symmetry, devices=devices)
add_function_test(
    TestJacobianMassMatrix, "test_mass_matrix_positive_definite", test_mass_matrix_positive_definite, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_jacobian_multiple_articulations",
    test_jacobian_multiple_articulations,
    devices=devices,
)
add_function_test(TestJacobianMassMatrix, "test_jacobian_with_mask", test_jacobian_with_mask, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_mass_matrix_with_mask", test_mass_matrix_with_mask, devices=devices)
add_function_test(
    TestJacobianMassMatrix, "test_prismatic_joint_jacobian", test_prismatic_joint_jacobian, devices=devices
)
add_function_test(TestJacobianMassMatrix, "test_empty_model", test_empty_model, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_articulation_view_api", test_articulation_view_api, devices=devices)
add_function_test(TestJacobianMassMatrix, "test_floating_base_jacobian", test_floating_base_jacobian, devices=devices)
add_function_test(
    TestJacobianMassMatrix, "test_jacobian_matches_body_qd_com", test_jacobian_matches_body_qd_com, devices=devices
)
add_function_test(
    TestJacobianMassMatrix,
    "test_mass_matrix_matches_com_kinetic_energy",
    test_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_floating_free_jacobian_matches_body_qd_com",
    test_floating_free_jacobian_matches_body_qd_com,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_descendant_free_jacobian_matches_body_qd_com",
    test_descendant_free_jacobian_matches_body_qd_com,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_floating_free_mass_matrix_matches_com_kinetic_energy",
    test_floating_free_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)
add_function_test(
    TestJacobianMassMatrix,
    "test_descendant_free_mass_matrix_matches_com_kinetic_energy",
    test_descendant_free_mass_matrix_matches_com_kinetic_energy,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
