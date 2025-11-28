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

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def test_fk_ik(test, device):
    builder = newton.ModelBuilder()

    num_worlds = 1

    for i in range(num_worlds):
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
        builder.add_articulation([j1, j2], key=f"pendulum_{i}")

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
        builder.add_articulation([j1, j2], key=f"pendulum_{i}")

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
    num_worlds = 8
    for i in range(num_worlds):
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
        builder.add_articulation([j1, j2], key=f"env_{i}")

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


devices = get_test_devices()


class TestSimKinematics(unittest.TestCase):
    pass


add_function_test(TestSimKinematics, "test_fk_ik", test_fk_ik, devices=devices)
add_function_test(TestSimKinematics, "test_fk_with_indices", test_fk_with_indices, devices=devices)
add_function_test(TestSimKinematics, "test_ik_with_indices", test_ik_with_indices, devices=devices)
add_function_test(TestSimKinematics, "test_fk_error_mask_and_indices", test_fk_error_mask_and_indices, devices=devices)
add_function_test(TestSimKinematics, "test_isaac_lab_use_case", test_isaac_lab_use_case, devices=devices)
add_function_test(TestSimKinematics, "test_bounds_checking", test_bounds_checking, devices=devices)
add_function_test(TestSimKinematics, "test_ik_with_mask", test_ik_with_mask, devices=devices)
add_function_test(TestSimKinematics, "test_ik_error_mask_and_indices", test_ik_error_mask_and_indices, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
