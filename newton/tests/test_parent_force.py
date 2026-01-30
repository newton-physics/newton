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

"""
Tests for parent forces (body_parent_f) extended state attribute.

This module tests the `body_parent_f` attribute which stores incoming joint
wrenches (forces from the parent body through the joint) in world frame,
referenced to the body's center of mass.

Note: Only the MuJoCo solver computes this attribute.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestParentForce(unittest.TestCase):
    pass


def setup_pendulum(device, solver_fn, xform: wp.transform):
    """Setup a simple pendulum with everything transformed by xform.

    Args:
        device: Compute device
        solver_fn: Function that creates a solver given a model
        xform: Transform to apply to the entire setup (joint anchor, body position)

    Returns:
        Tuple of (model, solver, state_0, state_1, expected_force_dir_world)
    """
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    # Transform the pendulum anchor position by xform
    anchor_pos = wp.transform_point(xform, wp.vec3(0, 0, 0))
    body_rot = xform[1]

    # Use add_link() to create the pendulum body (this is recommended for articulations)
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)

    # Add revolute joint: child_xform defines where the joint attaches to the link
    # The link's COM will be 1 unit below the anchor (in -Z direction in local frame)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(anchor_pos, body_rot),
        child_xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),  # Joint attaches 1 unit above link COM
        axis=wp.vec3(0, 1, 0),
    )

    # Register as articulation (required for MuJoCo solver)
    builder.add_articulation([joint])

    model = builder.finalize(device=device)
    solver = solver_fn(model)
    state_0, state_1 = model.state(), model.state()

    # Expected force direction in world frame: opposite to gravity (+Z)
    expected_dir_world = wp.vec3(0, 0, 1)

    return model, solver, state_0, state_1, expected_dir_world


def test_parent_force_static_pendulum(test, device, solver_fn):
    """Test that parent force equals weight for a static pendulum with various transforms."""

    xforms = [
        wp.transform_identity(),  # No transform
        wp.transform(wp.vec3(5, 3, -2), wp.quat_identity()),  # Translation only
        wp.transform(
            wp.vec3(1, 2, 3), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.pi * 0.5)
        ),  # Rotation + translation
    ]

    for i, xform in enumerate(xforms):
        with test.subTest(xform_index=i):
            model, solver, state_0, state_1, expected_dir = setup_pendulum(device, solver_fn, xform)

            # Verify body_parent_f is allocated
            test.assertIsNotNone(state_0.body_parent_f)

            # Evaluate FK and step once
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            solver.step(state_0, state_1, None, None, 1.0 / 60.0)

            # Verify parent force (in world frame)
            parent_f = state_1.body_parent_f.numpy()[0]
            expected_magnitude = model.body_mass.numpy()[0] * 9.81  # m*g

            # Linear force should point opposite to gravity (+Z) with magnitude = weight
            linear_force = parent_f[:3]
            np.testing.assert_allclose(
                linear_force,
                np.array([expected_dir[0], expected_dir[1], expected_dir[2]]) * expected_magnitude,
                rtol=1e-4,
            )

            # Angular component should be ~0 (no torque at equilibrium)
            np.testing.assert_allclose(parent_f[3:6], [0, 0, 0], atol=1e-2)


# Register tests for MuJoCo warp solver only (mujoco_cpu does not support body_parent_f)
devices = get_test_devices()

for device in devices:
    add_function_test(
        TestParentForce,
        "test_parent_force_static_pendulum_mujoco_warp",
        test_parent_force_static_pendulum,
        devices=[device],
        solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
