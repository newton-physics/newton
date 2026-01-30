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
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestParentForce(unittest.TestCase):
    pass


def _setup_pendulum(
    device,
    joint_axis: wp.vec3,
    child_offset: wp.vec3,
    parent_xform: wp.transform = None,
):
    if parent_xform is None:
        parent_xform = wp.transform_identity()

    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)

    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=parent_xform,
        child_xform=wp.transform(child_offset, wp.quat_identity()),
        axis=joint_axis,
    )
    builder.add_articulation([joint])

    return builder.finalize(device=device)


def test_parent_force_static_pendulum(test, device, solver_fn):
    """Test that parent force equals weight for a static pendulum with various transforms."""

    xforms = [
        wp.transform_identity(),
        wp.transform(wp.vec3(5, 3, -2), wp.quat_identity()),
        wp.transform(wp.vec3(1, 2, 3), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.pi * 0.5)),
    ]

    dt = 5e-3

    for i, xform in enumerate(xforms):
        with test.subTest(xform_index=i):
            # Pendulum hanging down: joint 1 unit above COM, rotating about Y
            model = _setup_pendulum(
                device,
                joint_axis=wp.vec3(0, 1, 0),
                child_offset=wp.vec3(0, 0, 1),
                parent_xform=xform,
            )
            solver = solver_fn(model)
            state_0, state_1 = model.state(), model.state()

            test.assertIsNotNone(state_0.body_parent_f)

            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            solver.step(state_0, state_1, None, None, dt)

            parent_f = state_1.body_parent_f.numpy()[0]
            weight = model.body_mass.numpy()[0] * 9.81

            np.testing.assert_allclose(parent_f[:3], [0, 0, weight], rtol=1e-4)
            np.testing.assert_allclose(parent_f[3:6], [0, 0, 0], atol=1e-2)


def test_parent_force_centrifugal(test, device, solver_fn):
    """Test centrifugal force contribution when pendulum is spinning about Z axis."""
    # Horizontal pendulum: joint at origin, COM at +X, rotating about Z
    dt = 5e-3
    r = 1.0
    model = _setup_pendulum(
        device,
        joint_axis=wp.vec3(0, 0, 1),
        child_offset=wp.vec3(-r, 0, 0),
    )
    solver = solver_fn(model)
    state_0, state_1 = model.state(), model.state()

    omega = 5.0
    state_0.joint_qd[:1].assign([omega])

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver.step(state_0, state_1, None, None, dt)

    parent_f = state_1.body_parent_f.numpy()[0]
    mass = model.body_mass.numpy()[0]

    # Weight (m*g in +Z) + Centripetal (m*omega^2*r toward -X)
    expected_fx = -mass * omega**2 * r
    expected_fz = mass * 9.81

    np.testing.assert_allclose(parent_f[:3], [expected_fx, 0.0, expected_fz], rtol=1e-4)
    np.testing.assert_allclose(parent_f[3:6], [0, 0, 0], atol=1e-2)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestParentForce,
        "test_parent_force_static_pendulum_mjwarp",
        test_parent_force_static_pendulum,
        devices=[device],
        solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
    )
    add_function_test(
        TestParentForce,
        "test_parent_force_centrifugal_mjwarp",
        test_parent_force_centrifugal,
        devices=[device],
        solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
