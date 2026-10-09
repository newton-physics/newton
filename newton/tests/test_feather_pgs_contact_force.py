# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact forces of SolverFeatherPGS: Coulomb point friction and the linear-only force report."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 120.0
BOX_MASS = 2.0


def _box_on_ground(device, gravity=None, mu=None, *, legacy_force_test=None):
    """Box resting on the ground; ``legacy_force_test`` requests the deprecated ``Contacts.force``."""
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    if mu is not None:
        shape_cfg.mu = mu
    if legacy_force_test is not None:
        with legacy_force_test.assertWarns(DeprecationWarning):
            builder.request_contact_attributes("force")
    # Solid-cube inertia for a 0.2 m box of mass BOX_MASS.
    inertia = wp.mat33(np.eye(3) * BOX_MASS * 0.04 / 6.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=BOX_MASS, inertia=inertia
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=shape_cfg.mu))
    return builder.finalize(device=device)


def test_resting_box_reports_weight_as_linear_force(test, device, pgs_mode="matrix_free", response="immediate"):
    """Report the resting box's weight through the linear contact force and leave the torque zero."""
    model = _box_on_ground(device, legacy_force_test=test)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_iterations=32, articulated_contact_response=response)
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


def _tilted_gravity_box_velocity(device, slope_angle, mu, steps, pgs_mode="matrix_free", response="immediate"):
    """Box on the ground under gravity tilted by ``slope_angle``, as on an incline; return its x velocity."""
    g = 9.81
    gravity = (g * np.sin(slope_angle), 0.0, -g * np.cos(slope_angle))
    model = _box_on_ground(device, gravity=gravity, mu=mu)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_iterations=32, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return float(state_0.body_qd.numpy()[0, 0])


def test_friction_holds_inside_the_cone(test, device, pgs_mode="matrix_free", response="immediate"):
    """Hold the box at rest when the tangential load is inside the friction cone."""
    # tan(0.3) = 0.31 < mu = 0.5.
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=0.3, mu=0.5, steps=120, pgs_mode=pgs_mode, response=response
    )
    test.assertLess(abs(velocity), 1.0e-3)


def test_friction_slides_at_the_coulomb_bound(test, device, pgs_mode="matrix_free", response="immediate"):
    """Accelerate the box at ``g (sin a - mu cos a)`` when the load exceeds the friction cone."""
    slope_angle, mu, steps = 0.6, 0.3, 120
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=slope_angle, mu=mu, steps=steps, pgs_mode=pgs_mode, response=response
    )
    expected = 9.81 * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    test.assertAlmostEqual(velocity, expected, delta=0.03 * expected)


ARMATURE = (9.0, 4.0, 2.0, 0.5, 1.5, 3.0)


def _sphere_on_ground(device, armature, dense, com=(0.0, 0.0, 0.0), mu=0.0):
    """A 1 kg sphere on a free joint, optionally with a massless fixed child that routes it to dense rows."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(com),
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    joints = [builder.add_joint_free(body)]
    if dense:
        child = builder.add_link(mass=0.0, inertia=wp.mat33(0.0))
        joints.append(builder.add_joint_fixed(body, child))
    builder.add_articulation(joints)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = armature
    model.joint_armature.assign(joint_armature)
    return model


def _impact(model, solver, joint_qd):
    """Step once from the given free-joint velocity; return the new velocity and summed contact force."""
    state_in, state_out = model.state(), model.state()
    qd = state_in.joint_qd.numpy()
    qd[:6] = joint_qd
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    solver.update_contacts(contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    return state_out.joint_qd.numpy()[:6], contacts.rigid_contact_force.numpy()[:count].sum(axis=0)


def test_free_body_contact_response_includes_armature(test, device, pgs_mode="matrix_free", response="immediate"):
    """Respond on free-body rows with the same armature-augmented inertia as on articulated rows."""
    stop = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    for armature, expected in ((0.0, 100.0), (9.0, 1000.0)):
        with test.subTest(armature=armature):
            model = _sphere_on_ground(device, armature, dense=False)
            solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_cfm=0.0, articulated_contact_response=response)
            qd, force = _impact(model, solver, stop)
            # The sphere starts 0.1 mm deep, so the depenetration bias adds 0.2 %.
            test.assertAlmostEqual(abs(float(force[2])), expected, delta=5.0e-3 * expected)
            test.assertAlmostEqual(float(qd[2]), 0.0, delta=5.0e-3)

    # An oblique frictional impact of an offset-COM body with per-axis armature.
    oblique = (0.3, -0.2, -1.0, 0.4, 0.5, -0.3)
    results = {}
    for dense in (False, True):
        model = _sphere_on_ground(device, ARMATURE, dense=dense, com=(0.03, -0.02, 0.01), mu=0.5)
        solver = SolverFeatherPGS(
            model, pgs_mode=pgs_mode, pgs_cfm=0.0, pgs_iterations=64, articulated_contact_response=response
        )
        results[dense] = _impact(model, solver, oblique)
    np.testing.assert_allclose(results[False][0], results[True][0], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(results[False][1], results[True][1], rtol=1.0e-3, atol=1.0e-2)

    # A notified armature change reaches the free-body response.
    model = _sphere_on_ground(device, 0.0, dense=False)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_cfm=0.0, articulated_contact_response=response)
    _impact(model, solver, stop)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    _, force = _impact(model, solver, stop)
    test.assertAlmostEqual(abs(float(force[2])), 1000.0, delta=5.0)


def test_free_body_impact_shares_armature_momentum(test, device, pgs_mode="matrix_free", response="immediate"):
    """Conserve the armature-augmented momentum in a frictionless plastic impact between free bodies."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
    for x in (0.0, 0.1999):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01)
        )
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver = SolverFeatherPGS(
        model, pgs_mode=pgs_mode, pgs_cfm=0.0, pgs_iterations=64, articulated_contact_response=response
    )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0] = 1.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    qd = state_out.joint_qd.numpy()
    # Effective masses 10 and 1: both move at 10 / 11 m/s (plus a small depenetration bias).
    test.assertAlmostEqual(float(qd[0]), 10.0 / 11.0, delta=2.0e-3)
    test.assertAlmostEqual(float(qd[6]), 10.0 / 11.0, delta=1.5e-2)
    test.assertAlmostEqual(float(10.0 * qd[0] + qd[6]), 10.0, delta=1.0e-3)


class TestFeatherPGSContactForce(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
split_devices = get_test_devices()
for _name in (
    "test_resting_box_reports_weight_as_linear_force",
    "test_friction_holds_inside_the_cone",
    "test_friction_slides_at_the_coulomb_bound",
    "test_free_body_contact_response_includes_armature",
    "test_free_body_impact_shares_armature_momentum",
):
    add_function_test(TestFeatherPGSContactForce, _name, globals()[_name], devices=devices)
    add_function_test(
        TestFeatherPGSContactForce, f"{_name}_split", globals()[_name], devices=split_devices, pgs_mode="split"
    )
    # The propagation response solves these contacts as body-space rows.
    add_function_test(
        TestFeatherPGSContactForce, f"{_name}_propagation", globals()[_name], devices=devices, response="propagation"
    )
# The fused response only differs from the immediate one with an articulated (dense-path) body.
add_function_test(
    TestFeatherPGSContactForce,
    "test_free_body_contact_response_includes_armature_propagation_fused",
    test_free_body_contact_response_includes_armature,
    devices=devices,
    response="propagation-fused",
)


if __name__ == "__main__":
    unittest.main()
