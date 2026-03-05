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

"""Tests for eval_jacobian() and eval_mass_matrix() functions."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
