# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental passive-island sleeping of SolverFeatherPGS: freezing and waking."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 0.005


def _scene(device, *, enabled=True, ground=True, stack=False):
    """Two boxes on the ground, side by side or stacked."""
    builder = newton.ModelBuilder()
    if ground:
        builder.add_ground_plane()
    for index, x in enumerate((0.0, 0.5)):
        position = wp.vec3(0.0, 0.0, 0.1 + index * 0.2) if stack else wp.vec3(x, 0.0, 0.1)
        body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        enable_sleeping=enabled,
        sleep_quiet_time=0.05,
        pgs_iterations=16,
        friction_anchor_beta=0.0,
    )
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance(pipeline, solver, states, control, steps, *, clear=True):
    contacts = pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()


def test_disabled_allocates_no_state(test, device):
    """Keep sleep state absent on the default solver path."""
    _model, _pipeline, solver, _states, _control = _scene(device, enabled=False)
    test.assertIsNone(solver.sleeping)


def test_separated_boxes_sleep_and_force_wakes_one(test, device):
    """Sleep independent boxes and wake only the forced island."""
    _model, pipeline, solver, states, control = _scene(device)
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    test.assertNotEqual(*solver.sleeping.body_island.numpy())
    before = states[0].body_q.numpy().copy()
    _advance(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_q.numpy(), before)
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    states[0].body_f.assign(force)
    _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(states[0].body_qd.numpy()[0, 0], 0.0)


def test_nonfinite_force_wakes_and_propagates(test, device):
    """A NaN external force on a sleeping body wakes its island instead of publishing a frozen state."""
    _model, pipeline, solver, states, control = _scene(device)
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = np.nan
    states[0].body_f.assign(force)
    _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
    test.assertFalse(np.all(np.isfinite(states[0].body_qd.numpy()[0])))


def test_reset_and_gravity_change_wake(test, device):
    """Wake every island on reset and on a gravity change."""
    model, pipeline, solver, states, control = _scene(device)
    _advance(pipeline, solver, states, control, 120)
    solver.reset(states[0])
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
    _advance(pipeline, solver, states, control, 120)
    model.set_gravity((0.0, 0.0, 9.81))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    _advance(pipeline, solver, states, control, 1)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
    test.assertGreater(states[0].body_qd.numpy()[0, 2], 0.0)


def test_free_flight_cannot_sleep(test, device):
    """Keep unsupported free bodies awake at zero speed."""
    model, pipeline, solver, states, control = _scene(device, ground=False)
    model.set_gravity((0.0, 0.0, 0.0))
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])


def test_invalid_options(test, device):
    """Reject unsupported sleep settings with the sleeping guard's own error."""
    model, *_ = _scene(device, enabled=False)
    for kwargs, message in (
        ({"sleep_quiet_time": 0.0}, "finite and positive"),
        ({"sleep_linear_threshold": float("nan")}, "finite and positive"),
        ({"sleep_angular_threshold": -1.0}, "finite and positive"),
        ({"pgs_velocity_iterations": 1}, "sleeping requires"),
        ({"pgs_warmstart": True}, "sleeping requires"),
    ):
        with test.subTest(kwargs=kwargs), test.assertRaisesRegex(ValueError, message):
            SolverFeatherPGS(model, pgs_mode="matrix_free", enable_sleeping=True, friction_anchor_beta=0.0, **kwargs)
    # The same options are valid without sleeping, so the errors above come from the sleep guard.
    SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_velocity_iterations=1, friction_anchor_beta=0.0)
    SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_warmstart=True, friction_anchor_beta=0.0)


def test_default_and_explicit_disabled_match(test, device):
    """Keep the trajectory bitwise unchanged with sleeping explicitly disabled."""
    model, pipeline, explicit, states, control = _scene(device, enabled=False)
    default = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, friction_anchor_beta=0.0)
    reference = [model.state(), model.state()]
    _advance(pipeline, explicit, states, control, 30)
    _advance(pipeline, default, reference, control, 30)
    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_array_equal(getattr(states[0], field).numpy(), getattr(reference[0], field).numpy())


def test_stack_wakes_together(test, device):
    """Wake the whole contacting stack when its top box is forced."""
    _model, pipeline, solver, states, control = _scene(device, stack=True)
    _advance(pipeline, solver, states, control, 200)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    islands = solver.sleeping.body_island.numpy()
    test.assertEqual(islands[0], islands[1])
    force = np.zeros((2, 6), dtype=np.float32)
    force[1, 0] = 100.0
    states[0].body_f.assign(force)
    _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])


def test_cuda_graph_sleep_and_wake(test, device):
    """Advance sleep timers and wake on a captured force during graph replay."""
    model, pipeline, solver, states, control = _scene(device)
    contacts = pipeline.contacts()
    for _ in range(2):
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, DT)
            states.reverse()
    for _ in range(100):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    forces = np.zeros((2, 6), dtype=np.float32)
    forces[0, 0] = 100.0
    states[0].body_f.assign(forces)
    states[1].body_f.assign(forces)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])


def test_sleeping_body_keeps_its_parent_wrench(test, device):
    """Publish the last awake joint wrench of a sleeping body instead of a skipped dynamics result."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    model.request_state_attributes("body_parent_f")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    solver = SolverFeatherPGS(
        model, pgs_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, friction_anchor_beta=0.0
    )
    reference = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_anchor_beta=0.0)
    states, reference_states = [model.state(), model.state()], [model.state(), model.state()]
    test.assertIsNotNone(states[0].body_parent_f)
    control = model.control()
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    asleep = states[0].body_parent_f.numpy().copy()
    _advance(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), asleep)
    np.testing.assert_array_equal(states[1].body_parent_f.numpy(), asleep)
    # The frozen value is the resting value an always-awake solver publishes.
    _advance(pipeline, reference, reference_states, control, 130)
    np.testing.assert_allclose(asleep, reference_states[0].body_parent_f.numpy(), rtol=0.0, atol=1.0e-3)


devices = get_cuda_test_devices()


class TestFeatherPGSSleeping(unittest.TestCase):
    pass


for _name in (
    "test_disabled_allocates_no_state",
    "test_separated_boxes_sleep_and_force_wakes_one",
    "test_nonfinite_force_wakes_and_propagates",
    "test_reset_and_gravity_change_wake",
    "test_free_flight_cannot_sleep",
    "test_invalid_options",
    "test_default_and_explicit_disabled_match",
    "test_stack_wakes_together",
    "test_cuda_graph_sleep_and_wake",
    "test_sleeping_body_keeps_its_parent_wrench",
):
    add_function_test(TestFeatherPGSSleeping, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
