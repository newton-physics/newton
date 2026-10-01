# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact forces of SolverFeatherPGS: Coulomb point friction and the linear-only force report."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 120.0
BOX_MASS = 2.0


def _box_on_ground(device, gravity=None, mu=None):
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    if mu is not None:
        shape_cfg.mu = mu
    builder.request_contact_attributes("force")
    # Solid-cube inertia for a 0.2 m box of mass BOX_MASS.
    inertia = wp.mat33(np.eye(3) * BOX_MASS * 0.04 / 6.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=BOX_MASS, inertia=inertia
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=shape_cfg.mu))
    return builder.finalize(device=device)


def test_resting_box_reports_weight_as_linear_force(test, device):
    """Report the resting box's weight through the linear contact force and leave the torque zero."""
    model = _box_on_ground(device)
    solver = SolverFeatherPGS(model, pgs_iterations=32)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    test.assertIsNotNone(contacts.force)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    solver.update_contacts(contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    # Force on shape0's body by shape1's; orient every record as the force on the box.
    box_shape = int(np.flatnonzero(model.shape_body.numpy() >= 0)[0])
    sign = np.where(shape0 == box_shape, 1.0, -1.0)[:, None]
    linear = contacts.rigid_contact_force.numpy()[:count] * sign
    weight = BOX_MASS * float(np.linalg.norm(model.gravity.numpy()[0]))
    np.testing.assert_allclose(linear.sum(axis=0), [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)

    wrench = contacts.force.numpy()[:count]
    np.testing.assert_allclose(wrench[:, :3] * sign, linear, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_array_equal(wrench[:, 3:], np.zeros((count, 3)))


def _tilted_gravity_box_velocity(device, slope_angle, mu, steps):
    """Box on the ground under gravity tilted by ``slope_angle``, as on an incline; return its x velocity."""
    g = 9.81
    gravity = (g * np.sin(slope_angle), 0.0, -g * np.cos(slope_angle))
    model = _box_on_ground(device, gravity=gravity, mu=mu)
    solver = SolverFeatherPGS(model, pgs_iterations=32)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return float(state_0.body_qd.numpy()[0, 0])


def test_friction_holds_inside_the_cone(test, device):
    """Hold the box at rest when the tangential load is inside the friction cone."""
    # tan(0.3) = 0.31 < mu = 0.5.
    velocity = _tilted_gravity_box_velocity(device, slope_angle=0.3, mu=0.5, steps=120)
    test.assertLess(abs(velocity), 1.0e-3)


def test_friction_slides_at_the_coulomb_bound(test, device):
    """Accelerate the box at ``g (sin a - mu cos a)`` when the load exceeds the friction cone."""
    slope_angle, mu, steps = 0.6, 0.3, 120
    velocity = _tilted_gravity_box_velocity(device, slope_angle=slope_angle, mu=mu, steps=steps)
    expected = 9.81 * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    test.assertAlmostEqual(velocity, expected, delta=0.03 * expected)


class TestFeatherPGSContactForce(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_resting_box_reports_weight_as_linear_force",
    "test_friction_holds_inside_the_cone",
    "test_friction_slides_at_the_coulomb_bound",
):
    add_function_test(TestFeatherPGSContactForce, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
