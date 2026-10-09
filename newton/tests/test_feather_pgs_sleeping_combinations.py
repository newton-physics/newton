# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental FeatherPGS sleeping combined with PGS drive rows and with contact compliance."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 0.005
SLEEP = {"enable_sleeping": True, "sleep_quiet_time": 0.05}


def _drive_scene(device, sleeping):
    """A position-driven fixed-base hinge next to an independent box on the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.2, hy=0.02, hz=0.02)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([hinge])
    box = builder.add_body(xform=wp.transform((1.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        drive_mode="physx_pgs",
        pgs_iterations=16,
        friction_anchor_beta=0.0,
        **(SLEEP if sleeping else {}),
    )
    control = model.control()
    control.joint_target_q.assign(np.array([0.5], dtype=np.float32))
    return model, newton.CollisionPipeline(model, rigid_contact_max=64), solver, control, link, box


def _compliance_scene(device, sleeping):
    """Two independent spheres resting on implicit spring-damper contacts."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.005
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    for x in (0.0, 1.0):
        body = builder.add_body(xform=wp.transform((x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(
            body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.3 / (4 / 3 * np.pi * 0.05**3), mu=0.5)
        )
    model = builder.finalize(device=device)
    model.rigid_contact_max = 32
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = pipeline.contacts()
    for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        setattr(contacts, name, wp.zeros(32, dtype=float, device=device))
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        contact_compliance=True,
        friction_anchor_beta=0.0,
        pgs_iterations=8,
        pgs_beta=0.05,
        dense_max_constraints=32,
        mf_max_constraints=32,
        **(SLEEP if sleeping else {}),
    )
    return model, pipeline, contacts, solver


def _run(model, pipeline, solver, control, states, steps, *, contacts=None, force=None, stiffness=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        states[0].clear_forces()
        if force is not None:
            states[0].body_f.assign(force)
        pipeline.collide(states[0], contacts)
        if stiffness is not None:
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(1.0)
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()


def test_drive_rows_stay_awake_while_a_box_sleeps(test, device):
    """Keep a drive-row articulation awake and tracking while an independent box sleeps and wakes."""
    model, pipeline, solver, control, link, box = _drive_scene(device, sleeping=True)
    reference_model, reference_pipeline, reference, reference_control, _, _ = _drive_scene(device, sleeping=False)
    states = [model.state(), model.state()]
    reference_states = [reference_model.state(), reference_model.state()]
    _run(model, pipeline, solver, control, states, 300)
    _run(reference_model, reference_pipeline, reference, reference_control, reference_states, 300)
    awake = solver.sleeping.body_awake.numpy()
    test.assertEqual(int(awake[link]), 1)
    test.assertEqual(int(awake[box]), 0)
    # The driven hinge reaches its target, exactly as without sleeping.
    test.assertAlmostEqual(float(states[0].joint_q.numpy()[0]), 0.5, delta=0.01)
    np.testing.assert_allclose(states[0].joint_q.numpy()[:1], reference_states[0].joint_q.numpy()[:1], atol=1.0e-5)
    frozen = states[0].body_q.numpy()[box].copy()
    _run(model, pipeline, solver, control, states, 10)
    np.testing.assert_array_equal(states[0].body_q.numpy()[box], frozen)
    # A push wakes the box, and the drive keeps holding its target.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[box, 0] = 300.0
    _run(model, pipeline, solver, control, states, 5, force=force)
    test.assertEqual(int(solver.sleeping.body_awake.numpy()[box]), 1)
    test.assertGreater(float(states[0].body_q.numpy()[box, 0]), float(frozen[0]))
    test.assertAlmostEqual(float(states[0].joint_q.numpy()[0]), 0.5, delta=0.01)


def test_compliant_contacts_sleep_and_wake(test, device):
    """Sleep spheres resting on compliant contacts at their spring rest depth and wake one by a push."""
    stiffness = 3000.0
    model, pipeline, contacts, solver = _compliance_scene(device, sleeping=True)
    reference_model, reference_pipeline, reference_contacts, reference = _compliance_scene(device, sleeping=False)
    control = model.control()
    states = [model.state(), model.state()]
    reference_states = [reference_model.state(), reference_model.state()]
    _run(model, pipeline, solver, control, states, 400, contacts=contacts, stiffness=stiffness)
    _run(
        reference_model,
        reference_pipeline,
        reference,
        reference_model.control(),
        reference_states,
        400,
        contacts=reference_contacts,
        stiffness=stiffness,
    )
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    # Frozen at the same compliant rest height as the solver without sleeping.
    np.testing.assert_allclose(states[0].body_q.numpy()[:, 2], reference_states[0].body_q.numpy()[:, 2], atol=1.0e-4)
    depth = 0.05 - float(states[0].body_q.numpy()[0, 2])
    expected = float(model.body_mass.numpy()[0]) * 9.81 / stiffness
    test.assertAlmostEqual(depth, expected, delta=0.25 * expected)
    frozen = states[0].body_q.numpy().copy()
    _run(model, pipeline, solver, control, states, 10, contacts=contacts, stiffness=stiffness)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 5.0
    _run(model, pipeline, solver, control, states, 5, contacts=contacts, force=force, stiffness=stiffness)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(float(states[0].body_q.numpy()[0, 0]), float(frozen[0, 0]))
    np.testing.assert_array_equal(states[0].body_q.numpy()[1], frozen[1])


class TestFeatherPGSSleepingCombinations(unittest.TestCase):
    pass


for _name, _func in (
    ("test_drive_rows_stay_awake_while_a_box_sleeps", test_drive_rows_stay_awake_while_a_box_sleeps),
    ("test_compliant_contacts_sleep_and_wake", test_compliant_contacts_sleep_and_wake),
):
    add_function_test(TestFeatherPGSSleepingCombinations, _name, _func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
