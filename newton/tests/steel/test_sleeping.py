# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental passive-island sleeping of SolverSteel."""

import functools
import inspect
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction_patches import _FLOOD_MIN_CONTACTS
from newton._src.solvers.steel.sleeping import _SleepState
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import PROFILE, _advance, _articulations, _parent_wrench_scene
from newton.tests.unittest_utils import add_function_test, capture_with_cuda_driver, get_cuda_test_devices


def _response(name: str) -> dict:
    """Return the constructor options of a response label (``"immediate"``, ``"propagation"`` or ``"colored"``)."""
    if name == "colored":
        return {"solve_mode": "colored", "articulated_contact_response": "propagation"}
    return {"articulated_contact_response": name}


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
    solver = SolverSteel(
        model,
        enable_sleeping=enabled,
        sleep_quiet_time=0.05,
        iterations=16,
        friction_anchor_beta=0.0,
    )
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance_boxes(pipeline, solver, states, control, steps, *, clear=True, observables=None):
    contacts = pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()


def test_disabled_allocates_no_state(test, device):
    """Keep sleep state absent on the default solver path."""
    _model, _pipeline, solver, _states, _control = _scene(device, enabled=False)
    test.assertIsNone(solver._sleeping)
    test.assertNotIn(SolverSteel.ObservableKind.BODY_AWAKE, solver.supported_observables)
    with test.assertRaises(ValueError):
        solver.observables({SolverSteel.ObservableKind.BODY_AWAKE})


def test_separated_boxes_sleep_and_force_wakes_one(test, device):
    """Sleep independent boxes, wake only the forced island, and report both through ``BODY_AWAKE``."""
    _model, pipeline, solver, states, control = _scene(device)
    observables = solver.observables({SolverSteel.ObservableKind.BODY_AWAKE})
    body_awake = observables.body_awake
    test.assertEqual((body_awake.dtype, body_awake.shape), (wp.bool, (2,)))
    np.testing.assert_array_equal(body_awake.numpy(), [True, True])
    _advance_boxes(pipeline, solver, states, control, 120, observables=observables)
    np.testing.assert_array_equal(body_awake.numpy(), [False, False])
    test.assertNotEqual(*solver._sleeping.body_island.numpy())
    before = states[0].body_q.numpy().copy()
    _advance_boxes(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_q.numpy(), before)
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    states[0].body_f.assign(force)
    _advance_boxes(pipeline, solver, states, control, 1, clear=False, observables=observables)
    np.testing.assert_array_equal(body_awake.numpy(), [True, False])
    test.assertGreater(states[0].body_qd.numpy()[0, 0], 0.0)
    solver.reset(states[0])
    np.testing.assert_array_equal(body_awake.numpy(), [True, False])
    _advance_boxes(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(body_awake.numpy(), [True, True])


def test_nonfinite_force_wakes_and_propagates(test, device):
    """Wake the island of a sleeping body hit by a NaN external force instead of publishing a frozen state."""
    _model, pipeline, solver, states, control = _scene(device)
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = np.nan
    states[0].body_f.assign(force)
    _advance_boxes(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
    test.assertFalse(np.all(np.isfinite(states[0].body_qd.numpy()[0])))


def test_reset_and_gravity_change_wake(test, device):
    """Wake every island on reset and on a gravity change."""
    model, pipeline, solver, states, control = _scene(device)
    _advance_boxes(pipeline, solver, states, control, 120)
    solver.reset(states[0])
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
    _advance_boxes(pipeline, solver, states, control, 120)
    model.set_gravity((0.0, 0.0, 9.81))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    _advance_boxes(pipeline, solver, states, control, 1)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
    test.assertGreater(states[0].body_qd.numpy()[0, 2], 0.0)


def test_free_flight_cannot_sleep(test, device):
    """Keep unsupported free bodies awake at zero speed."""
    model, pipeline, solver, states, control = _scene(device, ground=False)
    model.set_gravity((0.0, 0.0, 0.0))
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])


def test_invalid_options(test, device):
    """Reject unsupported sleep settings with the sleeping guard's own error."""
    model, *_ = _scene(device, enabled=False)
    for kwargs, message in (
        ({"sleep_quiet_time": 0.0}, "finite and positive"),
        ({"sleep_linear_threshold": float("nan")}, "finite and positive"),
        ({"sleep_angular_threshold": -1.0}, "finite and positive"),
        ({"velocity_iterations": 1}, "sleeping requires"),
        ({"contact_warmstart": True}, "sleeping requires"),
    ):
        with test.subTest(kwargs=kwargs), test.assertRaisesRegex(ValueError, message):
            SolverSteel(model, enable_sleeping=True, friction_anchor_beta=0.0, **kwargs)
    # The same options are valid without sleeping, so the errors above come from the sleep guard.
    SolverSteel(model, velocity_iterations=1, friction_anchor_beta=0.0)
    SolverSteel(model, contact_warmstart=True, friction_anchor_beta=0.0)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    empty = builder.finalize(device=device)
    SolverSteel(empty, friction_anchor_beta=0.0)
    with test.assertRaisesRegex(ValueError, "at least one articulation"):
        SolverSteel(empty, enable_sleeping=True, friction_anchor_beta=0.0)


def test_default_and_explicit_disabled_match(test, device):
    """Keep the trajectory bitwise unchanged with sleeping explicitly disabled."""
    model, pipeline, explicit, states, control = _scene(device, enabled=False)
    default = SolverSteel(model, iterations=16, friction_anchor_beta=0.0)
    reference = [model.state(), model.state()]
    _advance_boxes(pipeline, explicit, states, control, 30)
    _advance_boxes(pipeline, default, reference, control, 30)
    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_array_equal(getattr(states[0], field).numpy(), getattr(reference[0], field).numpy())


def test_stack_wakes_together(test, device):
    """Wake the whole contacting stack when its top box is forced."""
    _model, pipeline, solver, states, control = _scene(device, stack=True)
    _advance_boxes(pipeline, solver, states, control, 200)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    islands = solver._sleeping.body_island.numpy()
    test.assertEqual(islands[0], islands[1])
    force = np.zeros((2, 6), dtype=np.float32)
    force[1, 0] = 100.0
    states[0].body_f.assign(force)
    _advance_boxes(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])


def test_cuda_graph_sleep_and_wake(test, device):
    """Advance sleep timers and wake on a captured force during graph replay."""
    model, pipeline, solver, states, control = _scene(device)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverSteel.ObservableKind.BODY_AWAKE})
    for _ in range(2):
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, DT, observables=observables)
            states.reverse()
    for _ in range(100):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(observables.body_awake.numpy(), [False, False])
    forces = np.zeros((2, 6), dtype=np.float32)
    forces[0, 0] = 100.0
    states[0].body_f.assign(forces)
    states[1].body_f.assign(forces)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(observables.body_awake.numpy(), [True, False])


def _assert_resting_wrench(test, model, wrench):
    """Free roots transmit nothing; each weld carries its top box's weight."""
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(wrench[0::2], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(wrench[1::2, 2], weight[1::2], rtol=1.0e-4)
    np.testing.assert_allclose(wrench[1::2, [0, 1, 3, 4, 5]], 0.0, atol=1.0e-3)


def test_sleeping_body_keeps_its_parent_wrench(test, device):
    """Publish the last awake joint wrench of a sleeping body instead of a skipped dynamics result."""
    model, pipeline = _parent_wrench_scene(device)
    solver = SolverSteel(
        model,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    reference = SolverSteel(model, friction_anchor_beta=0.0, articulation_constraint_max=128)
    kinds = {newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F}
    observables, reference_observables = solver.observables(kinds), reference.observables(kinds)
    states, reference_states = [model.state(), model.state()], [model.state(), model.state()]
    control = model.control()
    _advance_boxes(pipeline, solver, states, control, 120, observables=observables)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = observables.body_parent_f.numpy().copy()
    _advance_boxes(pipeline, solver, states, control, 10, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), asleep)
    # The frozen value is the resting value an always-awake solver publishes.
    _advance_boxes(pipeline, reference, reference_states, control, 130, observables=reference_observables)
    np.testing.assert_allclose(asleep, reference_observables.body_parent_f.numpy(), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)


def test_sleeping_body_keeps_its_legacy_parent_wrench(test, device):
    """Freeze the deprecated State.body_parent_f output of a sleeping body as well."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver = SolverSteel(
        model,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    states = [model.state(), model.state()]
    test.assertIsNotNone(states[0].body_parent_f)
    control = model.control()
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = states[0].body_parent_f.numpy().copy()
    _advance_boxes(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), asleep)
    np.testing.assert_array_equal(states[1].body_parent_f.numpy(), asleep)


def _sleeping_wrench_solvers(model):
    solver = SolverSteel(
        model,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    reference = SolverSteel(model, friction_anchor_beta=0.0, articulation_constraint_max=128)
    return solver, reference


def _resting_wrench(model, pipeline, reference, control):
    kinds = {newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F}
    observables = reference.observables(kinds)
    _advance_boxes(pipeline, reference, [model.state(), model.state()], control, 130, observables=observables)
    return observables.body_parent_f.numpy()


def test_sleeping_parent_wrench_is_independent_of_the_output(test, device):
    """Report the frozen wrench in a newly allocated or cleared output, in eager and captured steps."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    kinds = {newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F}
    first = solver.observables(kinds)
    states, control = [model.state(), model.state()], model.control()
    _advance_boxes(pipeline, solver, states, control, 120, observables=first)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = first.body_parent_f.numpy().copy()
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)

    second = solver.observables(kinds)
    _advance_boxes(pipeline, solver, states, control, 1, observables=second)
    np.testing.assert_array_equal(second.body_parent_f.numpy(), asleep)
    first.body_parent_f.fill_(wp.spatial_vector(-1.0))
    _advance_boxes(pipeline, solver, states, control, 1, observables=first)
    np.testing.assert_array_equal(first.body_parent_f.numpy(), asleep)

    contacts = pipeline.contacts()

    def step(observables):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT, observables=observables)

    step(second)
    third = solver.observables(kinds)
    with wp.ScopedCapture(device) as capture:
        step(third)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(third.body_parent_f.numpy(), asleep)


def test_parent_wrench_first_requested_after_sleep(test, device):
    """Report the resting wrench when the first output is requested after the bodies fell asleep."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    observables = solver.observables({newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    _advance_boxes(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    resting = _resting_wrench(model, pipeline, reference, control)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)
    # A reset wakes the bodies, which then publish freshly computed wrenches.
    solver.reset(states[0])
    _advance_boxes(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)


def test_unrequested_parent_wrench_history_skips_moving_articulations(test, device):
    """Keep the unrequested wrench history only for articulations that can fall asleep by the next step."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    # Slide the second pair along the ground so it never turns quiet.
    push = np.zeros((model.body_count, 6), dtype=np.float32)
    push[2, 0] = 500.0
    for state in states:
        state.body_f.assign(push)
    _advance_boxes(pipeline, solver, states, control, 20, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 1, 1])
    history = solver._sleeping.parent_wrenches.numpy()
    np.testing.assert_array_equal(history[2:], 0.0)
    np.testing.assert_allclose(history[0], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(history[1], [0.0, 0.0, model.body_mass.numpy()[1] * 9.81, 0.0, 0.0, 0.0], atol=1.0e-3)

    # With a quiet time of one step an island can sleep two steps after it turns quiet.
    solver = SolverSteel(
        model,
        enable_sleeping=True,
        sleep_quiet_time=DT,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    states = [model.state(), model.state()]
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    observables = solver.observables({newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    _advance_boxes(pipeline, solver, states, control, 1, observables=observables)
    resting = _resting_wrench(model, pipeline, reference, control)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)

    # A gravity change wakes the pairs. Short steps run until both turn quiet, which skips their wrenches, and one
    # long step puts them back to sleep.
    solver = SolverSteel(
        model,
        enable_sleeping=True,
        sleep_quiet_time=0.01,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    states = [model.state(), model.state()]
    contacts = pipeline.contacts()

    def step(dt, observables=None):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, dt, observables=observables)
        states.reverse()

    for _ in range(120):
        step(0.004)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    model.set_gravity((0.0, 0.0, -4.905))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    for _ in range(200):
        step(0.004)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1, 1, 1])
        if (solver._sleeping.quiet_age.numpy() > 0.0).all():
            break
    test.assertTrue(np.isclose(solver._sleeping.quiet_age.numpy(), 0.004).any())
    step(0.007)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    observables = solver.observables({newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    step(0.004, observables)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), 0.5 * resting, rtol=0.0, atol=1.0e-3)


def test_sleeping_parent_wrench_from_legacy_to_observable(test, device):
    """Report the same frozen wrench in an observable after sleeping with only the legacy output."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance_boxes(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    legacy = states[0].body_parent_f.numpy().copy()
    observables = solver.observables({newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    _advance_boxes(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), legacy)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), legacy)


devices = get_cuda_test_devices()


class TestSteelSleeping(unittest.TestCase):
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
    "test_sleeping_body_keeps_its_legacy_parent_wrench",
    "test_sleeping_parent_wrench_is_independent_of_the_output",
    "test_parent_wrench_first_requested_after_sleep",
    "test_unrequested_parent_wrench_history_skips_moving_articulations",
    "test_sleeping_parent_wrench_from_legacy_to_observable",
):
    add_function_test(TestSteelSleeping, _name, globals()[_name], devices=devices)


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
    solver = SolverSteel(
        model,
        drive_mode="rows",
        iterations=16,
        friction_anchor_beta=0.0,
        **(SLEEP if sleeping else {}),
    )
    control = model.control()
    control.joint_target_q.assign(np.array([0.5], dtype=np.float32))
    return model, newton.CollisionPipeline(model, rigid_contact_max=64), solver, control, link, box


def _compliance_scene(device, sleeping, response="immediate"):
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
    solver = SolverSteel(
        model,
        contact_compliance=True,
        friction_anchor_beta=0.0,
        iterations=8,
        baumgarte=0.05,
        articulation_constraint_max=32,
        free_body_constraint_max=32,
        **_response(response),
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
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
    awake = solver._sleeping.body_awake.numpy()
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
    test.assertEqual(int(solver._sleeping.body_awake.numpy()[box]), 1)
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
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
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
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(float(states[0].body_q.numpy()[0, 0]), float(frozen[0, 0]))
    np.testing.assert_array_equal(states[0].body_q.numpy()[1], frozen[1])


class TestSteelSleepingCombinations(unittest.TestCase):
    pass


for _name, _func in (
    ("test_drive_rows_stay_awake_while_a_box_sleeps", test_drive_rows_stay_awake_while_a_box_sleeps),
    ("test_compliant_contacts_sleep_and_wake", test_compliant_contacts_sleep_and_wake),
):
    add_function_test(TestSteelSleepingCombinations, _name, _func, devices=get_cuda_test_devices())


_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


# Hard point friction: branched articulations of one topology select sparse mass factors.
_SPARSE_PROFILE = {
    "friction_anchor_beta": 0.0,
    "contact_regularization": 0.0,
    "contact_friction_gap_max": float("inf"),
}


def _compare(
    test,
    device,
    *,
    interval,
    velocity_limits,
    graph,
    arm=True,
    expect_sparse=False,
    settle_steps=100,
    resettle_steps=150,
    **options,
):
    runs = [_Run(device, skip, interval, velocity_limits, graph, arm, options) for skip in (False, True)]
    test.assertTrue(runs[1].solver._sleep_skips_dynamics)
    test.assertFalse(runs[0].solver._sleep_skips_dynamics)
    test.assertEqual(runs[1].solver._sparse_mass_matrix_size is not None, expect_sparse)
    latch = _MismatchLatch(device)
    for phase, steps, force in (("settle", settle_steps, 0.0), ("push", 6, 20.0), ("resettle", resettle_steps, 0.0)):
        latch.begin_phase()
        for step in range(steps):
            for run in runs:
                run.advance(force)
            latch.record(runs[1], runs[0], step, sleep=phase == "settle", wake=phase == "push")
        latch.check(test, phase)
    test.assertTrue(latch.slept_and_woke())
    # The driven arm never sleeps, and everything else settles again.
    expected = [0, 0, 0, 1] if arm else [0, 0, 0]
    np.testing.assert_array_equal(runs[1].solver._sleeping.art_awake.numpy(), expected)
    test.assertFalse(runs[1].solver._dynamics_art_active.numpy()[:3].any())


def test_eager_trajectories_match(test, device):
    """Settle, sleep, force-wake and resettle identically with and without skipped dynamics."""
    for interval in (1, 3):
        for velocity_limits in (False, True):
            # Islands sleep within the quiet time; one configuration also covers a long sleeping horizon.
            horizon = {"settle_steps": 400, "resettle_steps": 600} if interval == 1 and not velocity_limits else {}
            with test.subTest(interval=interval, velocity_limits=velocity_limits):
                _compare(test, device, interval=interval, velocity_limits=velocity_limits, graph=False, **horizon)


def test_alternate_kernel_paths_match(test, device):
    """Cover tiled triangular solves, cooperative tree traversal and sparse mass factors."""
    for options in (
        {"kernel_overrides": {"trisolve_kernel": "tiled"}},
        {"parallel_tree": True},
        {"arm": False, "expect_sparse": True, **_SPARSE_PROFILE},
        {"arm": False, "expect_sparse": True, "parallel_tree": True, **_SPARSE_PROFILE},
    ):
        with test.subTest(options=options):
            _compare(test, device, interval=3, velocity_limits=False, graph=False, **options)


def test_graph_trajectories_match(test, device):
    """Replay captured steps identically with and without skipped dynamics."""
    for interval in (1, 2):
        with test.subTest(interval=interval):
            _compare(test, device, interval=interval, velocity_limits=True, graph=True)


def test_transient_divergence_is_reported(test, device):
    """Report the step of a published mismatch that is gone again one step later."""
    advance = _Run.advance
    glitch = {"calls": 0, "saved": None}

    def advance_with_glitch(run, force):
        if not run.solver._sleeping.skip_dynamics:
            advance(run, force)
            return
        if glitch["saved"] is not None:
            wp.copy(run.states[0].joint_q, glitch["saved"])
            glitch["saved"] = None
        advance(run, force)
        if glitch["calls"] == 7:
            joint_q = run.states[0].joint_q
            glitch["saved"] = wp.clone(joint_q)
            values = joint_q.numpy()
            values[0] += 1.0e-3
            joint_q.assign(values)
        glitch["calls"] += 1

    with patch.object(_Run, "advance", advance_with_glitch):
        with test.assertRaisesRegex(AssertionError, "settle step 7 joint_q differs"):
            _compare(test, device, interval=1, velocity_limits=False, graph=False)


class _Run:
    def __init__(self, device, skip, interval, velocity_limits, graph, arm, options):
        self.model = _dynamics_scene(device, arm)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=512)
        options = dict(options)
        overrides = options.pop("kernel_overrides", {})
        with kernel_overrides(overrides):
            self.solver = newton.solvers.SolverSteel(
                self.model,
                **{
                    **PROFILE,
                    "update_mass_matrix_interval": interval,
                    "enable_joint_velocity_limits": velocity_limits,
                    **options,
                },
            )
        self.solver._sleeping.skip_dynamics = skip
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.graph = None
        if graph:
            self._step()
            self._step()
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step()
                self._step()
            self.graph = capture.graph
        self.forces = {}

    def advance(self, force):
        if force not in self.forces:
            body_f = np.zeros((self.model.body_count, 6), dtype=np.float32)
            body_f[0, 0] = force
            self.forces[force] = wp.array(body_f, dtype=wp.spatial_vector, device=self.model.device)
        for state in self.states:
            wp.copy(state.body_f, self.forces[force])
        if self.graph is None:
            self._step()
        else:
            wp.capture_launch(self.graph)

    def _step(self):
        self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, 0.005)
        self.states.reverse()

    def field(self, name):
        return getattr(self.states[0], name).view(float).flatten()


class _MismatchLatch:
    """Latch the first mismatching step of each published field on the device and read it once per phase."""

    def __init__(self, device):
        self.device = device
        self.first_step = wp.empty(len(_FIELDS), dtype=int, device=device)
        self.max_diff = wp.empty(len(_FIELDS), dtype=float, device=device)
        # First settle step with both branching articulations asleep, first push step with the pushed one awake.
        self.transitions = wp.full(2, _NO_STEP, dtype=int, device=device)

    def begin_phase(self):
        self.first_step.fill_(_NO_STEP)
        self.max_diff.zero_()

    def record(self, actual, expected, step, *, sleep, wake):
        for index, name in enumerate(_FIELDS):
            values = actual.field(name)
            wp.launch(
                _latch_field_mismatch,
                dim=values.size,
                inputs=[values, expected.field(name), step, index],
                outputs=[self.first_step, self.max_diff],
                device=self.device,
            )
        wp.launch(
            _latch_sleep_transitions,
            dim=1,
            inputs=[actual.solver._sleeping.art_awake, step, int(sleep), int(wake)],
            outputs=[self.transitions],
            device=self.device,
        )

    def check(self, test, phase):
        first_step = self.first_step.numpy()
        step = int(first_step.min())
        if step != _NO_STEP:
            index = int(np.argmax(first_step == step))
            diff = self.max_diff.numpy()[index]
            test.fail(f"{phase} step {step} {_FIELDS[index]} differs (max abs diff over the phase {diff})")

    def slept_and_woke(self):
        return bool((self.transitions.numpy() != _NO_STEP).all())


_NO_STEP = 2**31 - 1


@wp.kernel(enable_backward=False)
def _latch_field_mismatch(
    actual: wp.array[float],
    expected: wp.array[float],
    step: int,
    field: int,
    first_step: wp.array[int],
    max_diff: wp.array[float],
):
    i = wp.tid()
    a = actual[i]
    e = expected[i]
    if a == e or (wp.isnan(a) and wp.isnan(e)):
        return
    wp.atomic_min(first_step, field, step)
    diff = wp.abs(a - e)
    if wp.isnan(diff):
        diff = wp.inf
    wp.atomic_max(max_diff, field, diff)


@wp.kernel(enable_backward=False)
def _latch_sleep_transitions(awake: wp.array[int], step: int, sleep: int, wake: int, transitions: wp.array[int]):
    if sleep != 0 and awake[0] == 0 and awake[1] == 0:
        wp.atomic_min(transitions, 0, step)
    if wake != 0 and awake[0] != 0:
        wp.atomic_min(transitions, 1, step)


def _dynamics_scene(device, arm):
    """Two resting branching articulations, a resting box and, with ``arm``, a driven fixed-base arm."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        joints = [builder.add_joint_free(child=base)]
        # Two tips make the articulation a branching tree.
        for side in (-1.0, 1.0):
            tip = builder.add_link(xform=wp.transform((x + side * 0.3, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
            hinge = wp.transform((side * 0.3, 0.0, 0.0), wp.quat_identity())
            joints.append(builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)))
        builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform((3.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    if arm:
        root = builder.add_link(xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
        joints = [
            builder.add_joint_fixed(
                parent=-1, child=root, parent_xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity())
            )
        ]
        parent = root
        for k in range(3):
            link = builder.add_link(xform=wp.transform((5.0, 0.0, 0.3 - 0.2 * k), wp.quat_identity()))
            builder.add_shape_box(link, hx=0.03, hy=0.03, hz=0.08)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    parent_xform=wp.transform((0.0, 0.0, -0.2), wp.quat_identity()),
                    axis=(0.0, 1.0, 0.0),
                    target_ke=100.0,
                    target_kd=10.0,
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = link
        builder.add_articulation(joints)
    builder.joint_velocity_limit[:] = [5.0] * len(builder.joint_velocity_limit)
    return builder.finalize(device=device)


class TestSteelSleepingDynamics(unittest.TestCase):
    pass


for _name in (
    "test_eager_trajectories_match",
    "test_alternate_kernel_paths_match",
    "test_graph_trajectories_match",
    "test_transient_divergence_is_reported",
):
    add_function_test(TestSteelSleepingDynamics, _name, globals()[_name], devices=devices)


def test_anchored_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore the anchored rows of a force-woken articulation."""
    model, pipeline, solver, states, control = _articulations(device)
    _advance(pipeline, solver, states, control, 5)
    awake_rows = int(solver._constraint_count.numpy()[0])
    test.assertGreater(awake_rows, 0)
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
    np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
    frozen = states[0].body_q.numpy().copy()
    _advance(pipeline, solver, states, control, 20)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

    # Pushing the first articulation restores its rows and leaves the second asleep.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 20.0
    for _ in range(5):
        states[0].body_f.assign(force)
        _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1, 0, 0])
    test.assertEqual(int(solver._constraint_count.numpy()[0]), awake_rows // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_sleeps_and_wakes(test, device):
    """Sleep inside a captured graph and wake on an explicit wake before the next replay."""
    model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 2, contacts=contacts)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
    solver._sleeping.wake()
    wp.capture_launch(capture.graph)
    test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)


def test_masked_reset_wakes_only_selected_worlds(test, device):
    """Wake only the worlds a masked reset selects, including none for an empty mask."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    # The final mask entry selects global articulations; these boxes belong to worlds 0 and 1.
    for mask, expected in (
        ([False, False, False], [0, 0]),
        ([False, False, True], [0, 0]),
        ([True, False, False], [1, 0]),
    ):
        solver.reset(states[0], world_mask=wp.array(mask, dtype=wp.bool, device=device))
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), expected)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])


def test_masked_reset_restarts_only_selected_history(test, device):
    """Restart the selected world's state-change history from the reset state and keep the other world's."""
    _model, _pipeline, solver, states = _two_world_boxes(device)
    sleeping = solver._sleeping
    for name in _HISTORY:
        history = getattr(sleeping, name)
        history.assign(np.full_like(history.numpy(), 7.0))
    sleeping.asleep_steps.assign(np.array([11, 11], dtype=np.int32))
    sleeping.quiet_age.assign(np.array([3.0, 3.0], dtype=np.float32))
    sleeping.art_awake.assign(np.array([0, 0], dtype=np.int32))
    sleeping.valid.fill_(1)
    previous_root = sleeping.previous_root.numpy().copy()
    state = states[0]
    solver.reset(state, world_mask=wp.array([True, False, False], dtype=wp.bool, device=device))
    sources = {"last_q": state.joint_q, "last_qd": state.joint_qd, "last_body_q": state.body_q}
    sources["last_body_qd"] = state.body_qd
    for name in _HISTORY:
        history = getattr(sleeping, name).numpy()
        half = history.shape[0] // 2
        np.testing.assert_array_equal(history[:half], sources[name].numpy()[:half], err_msg=name)
        np.testing.assert_array_equal(history[half:], np.full_like(history[half:], 7.0), err_msg=name)
    np.testing.assert_array_equal(sleeping.asleep_steps.numpy(), [0, 11])
    np.testing.assert_array_equal(sleeping.quiet_age.numpy(), [0.0, 3.0])
    np.testing.assert_array_equal(sleeping.art_awake.numpy(), [1, 0])
    np.testing.assert_array_equal(sleeping.valid.numpy(), [1])
    np.testing.assert_array_equal(sleeping.previous_root.numpy(), previous_root)


def test_masked_reset_restarts_state_change_history(test, device):
    """Continue bitwise like the world whose sleeping state another world was given, after a masked reset of both."""
    model, pipeline, solver, states = _two_world_boxes(device)
    q = model.joint_q.numpy()
    q[model.joint_q_start.numpy()[1] + 2] = 0.3
    model.joint_q.assign(q)
    for state in states:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    # World 1 takes world 0's state; only world 0's history matches it.
    for state in states:
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            values = getattr(state, name).numpy()
            half = values.shape[0] // 2
            values[half:] = values[:half]
            getattr(state, name).assign(values)
    solver.reset(states[0], world_mask=wp.array([True, True, False], dtype=wp.bool, device=device))
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[:, 2] = 30.0
    contacts = pipeline.contacts()
    for step in range(150):
        states[0].clear_forces()
        if step == 60:
            states[0].body_f.assign(force)
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()
    body_q = states[0].body_q.numpy()
    np.testing.assert_array_equal(body_q[0], body_q[1])


def test_global_reset_keeps_shared_support_history(test, device):
    """Wake two worlds' sleepers whose global kinematic support moved before a reset of only the global entry."""
    builder = newton.ModelBuilder()
    support = _add_slab(builder)
    box = newton.ModelBuilder()
    _add_box(box)
    builder.replicate(box, 2)
    _expect_support_move_wakes(builder, support, [1, 2], [False, False, True], device)


def test_world_reset_keeps_shared_support_history(test, device):
    """Wake a global sleeper whose world's kinematic support moved before a reset of only that world."""
    builder = newton.ModelBuilder()
    _add_box(builder)
    slab = newton.ModelBuilder()
    _add_slab(slab)
    builder.replicate(slab, 1)
    _expect_support_move_wakes(builder, 1, [0], [True, False], device)


def test_world_reset_keeps_support_history_read_through_another_world_shape(test, device):
    """Wake a world-1 sleeper on a fixed world-0 support carrying a world-1 slab, after a world-0 reset."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    support = _add_fixed_support(builder)
    builder.end_world()
    builder.begin_world()
    builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.05)
    _add_box(builder)
    builder.end_world()
    _expect_support_move_wakes(builder, support, [1], [True, False, False], device)


def test_world_reset_keeps_support_history_read_through_a_global_shape(test, device):
    """Wake a world-1 sleeper on a fixed world-0 support carrying a global slab, after a world-0 reset."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    support = _add_fixed_support(builder)
    builder.end_world()
    builder.begin_world()
    _add_box(builder)
    builder.end_world()
    builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.05)
    _expect_support_move_wakes(builder, support, [1], [True, False, False], device)


def test_world_reset_keeps_support_history_read_by_a_global_shape(test, device):
    """Wake a world-1 sleeper whose box shape is global on a fixed world-0 support, after a world-0 reset."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    support = _add_fixed_support(builder)
    builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.05)
    builder.end_world()
    builder.begin_world()
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    builder.end_world()
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    _expect_support_move_wakes(builder, support, [box], [True, False, False], device)


def test_property_notification_wakes_only_changed_islands(test, device):
    """Wake the island whose mass changed, and nothing for a notification without a change."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0, 0])
    mass = model.body_mass.numpy()
    mass[0] *= 2.0
    model.body_mass.assign(mass)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])


def test_frozen_patch_carry_matches_rebuild(test, device):
    """Match rebuilt friction patches by carrying frozen ones through sleep, wake and resettle."""
    _check_carry(test, device, tiles=False)


def test_frozen_patch_carry_matches_rebuild_for_flooded_pairs(test, device):
    """Skip the warp flood for large pairs while frozen and rebuild them identically on wake."""
    _check_carry(test, device, tiles=True)


def _check_carry(test, device, tiles):
    trajectories, flood_pairs, histories = [], [], []
    for carry in (False, True):
        model, pipeline, solver, states, control = _articulations(device, tiles=tiles)
        solver._sleeping.carry_frozen_patches = carry
        contacts = pipeline.contacts()
        frozen = 0
        trajectory = []
        for step in range(900):
            states[0].clear_forces()
            if 400 <= step < 405:
                force = np.zeros((model.body_count, 6), dtype=np.float32)
                force[0, 0] = 20.0
                states[0].body_f.assign(force)
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
            frozen += int(solver._sleeping.frozen_bodies.numpy().sum())
            if step == 399:
                flood_pairs.append(int(solver._friction_patches._flood_pair_count.numpy()[0]))
                # The anchor history a sleeping pair hands to its next awake step.
                current = solver._friction_patches.current
                count = int(contacts.rigid_contact_count.numpy()[0])
                valid = current.valid.numpy()[:count] != 0
                histories.append(
                    (
                        valid,
                        current.displacement.numpy()[:count][valid],
                        current.tangent_impulse.numpy()[:count][valid],
                    )
                )
            trajectory.append(np.concatenate((states[0].body_q.numpy().ravel(), states[0].body_qd.numpy().ravel())))
        test.assertGreater(frozen, 0)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0, 0])
        trajectories.append(np.array(trajectory))
    if not tiles:
        np.testing.assert_array_equal(histories[1][0], histories[0][0])
        test.assertGreater(int(histories[0][0].sum()), 0)
        test.assertGreater(float(np.abs(histories[0][1]).max()), 0.0)
        for carried, rebuilt in zip(histories[1][1:], histories[0][1:], strict=True):
            np.testing.assert_allclose(carried, rebuilt, rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_array_equal(trajectories[1][:400], trajectories[0][:400])
        # Roundoff after wake can flip a near-tied patch anchor, so the resettled trajectory gets a physical bound.
        np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=2.0e-3)
        return
    # Frozen pairs skip the warp flood; bodies built from many shapes are not bitwise reproducible.
    test.assertGreater(flood_pairs[0], 0)
    test.assertEqual(flood_pairs[1], 0)
    positions = [t[-1][: 7 * 4].reshape(4, 7)[:, :3] for t in trajectories]
    np.testing.assert_allclose(positions[1], positions[0], rtol=0.0, atol=2.0e-3)


def test_frozen_patch_carry_copies_the_previous_history(test, device):
    """Take a sleeping pair's anchor history from the previous step, not from the current frame's storage."""
    _model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 400, contacts=contacts)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertTrue(np.all(solver._sleeping.frozen_bodies.numpy() == 1))
    current = solver._friction_patches.current
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("valid", "displacement", "tangent_impulse", "anchor_a", "anchor_b", "owner")
    before = {name: getattr(current, name).numpy()[:count].copy() for name in names}
    test.assertGreater(int(before["valid"].sum()), 0)
    # Overwrite the current frame; the carry must restore every field from the stored history.
    for name in names:
        getattr(current, name).fill_(7)
    _advance(pipeline, solver, states, control, 1, contacts=contacts)
    for name in names:
        np.testing.assert_array_equal(getattr(current, name).numpy()[:count], before[name], err_msg=name)


def _frozen_plates(device):
    """Settled plates of box tiles on the ground, each pair large enough for the warp flood."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    tile = 0.03
    for level in range(3):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01 + 0.0205 * level), wp.quat_identity()))
        for ix in range(4):
            for iy in range(4):
                offset = wp.vec3((ix - 1.5) * 2 * tile, (iy - 1.5) * 2 * tile, 0.0)
                builder.add_shape_box(body, xform=wp.transform(offset, wp.quat_identity()), hx=tile, hy=tile, hz=0.01)
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1024
    solver = SolverSteel(model, iterations=8, free_body_constraint_max=2048)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=1024)
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    for _ in range(10):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    pipeline.collide(state_0, contacts)
    model.body_q.assign(state_0.body_q)
    return model, contacts, solver


_CONTACT_FIELDS = ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1")


def test_frozen_pair_floods_on_device_when_its_carry_fails(test, device):
    """A frozen pair whose contacts no longer repeat floods on a warp and matches the serial flood exactly.

    Reversing the contact order breaks the carry, as an unordered collision pipeline does every step.
    """
    model, contacts, solver = _frozen_plates(device)
    patches = solver._friction_patches
    count = int(contacts.rigid_contact_count.numpy()[0])
    original = {name: getattr(contacts, "rigid_contact_" + name).numpy().copy() for name in _CONTACT_FIELDS}
    reversed_order = {}
    for name, values in original.items():
        reversed_order[name] = values.copy()
        reversed_order[name][:count] = values[:count][::-1]
    state = model.state()
    frozen = wp.ones(model.body_count, dtype=int, device=model.device)

    def run(device_flood):
        patches.device_flood = device_flood
        patches.previous.valid.zero_()
        patches.build(model, state, contacts, shape_material_mu=solver._shape_mu)
        patches.store(state)
        for name, values in reversed_order.items():
            getattr(contacts, "rigid_contact_" + name).assign(values)
        patches.build(model, state, contacts, shape_material_mu=solver._shape_mu, frozen_bodies=frozen)
        listed = int(patches._flood_pair_count.numpy()[0])
        for name, values in original.items():
            getattr(contacts, "rigid_contact_" + name).assign(values)
        outputs = {"frame." + name: getattr(patches.current, name) for name in _PATCH_FRAME_OUTPUTS}
        outputs.update({"view." + name: getattr(patches.view, name) for name in _PATCH_VIEW_OUTPUTS})
        return listed, {name: array.numpy().copy() for name, array in outputs.items()}

    _, serial = run(False)
    listed, flooded = run(True)
    _, sizes = np.unique(patches.current.keys.numpy()[:count], return_counts=True)
    large_pairs = int(np.count_nonzero(sizes > _FLOOD_MIN_CONTACTS))
    test.assertGreater(large_pairs, 0, "no body pair was large enough for the warp flood")
    test.assertEqual(listed, large_pairs, "a frozen pair skipped the warp flood")
    source = flooded["frame.source"][:count]
    test.assertLess(int(np.count_nonzero(source >= 0)), count, "the frozen pairs carried their history")
    for name, value in serial.items():
        np.testing.assert_array_equal(flooded[name], value, err_msg=f"{name} differs between the floods")


_PATCH_FRAME_OUTPUTS = ("owner", "members", "valid", "source", "anchor_a", "anchor_b", "displacement")
_PATCH_VIEW_OUTPUTS = ("weight", "next_contact", "point_a", "point_b", "phi")


def test_frozen_pair_carry_maps_regions_across_frames(test, device):
    """Carry each frozen contact's source and region owner after its buffer slot and sorted position move.

    The pairs' buffer blocks are reversed, and the first sorted pair loses a contact, so it rebuilds and
    every later pair starts one sorted position earlier than in the previous frame.
    """
    model, contacts, solver = _frozen_plates(device)
    patches = solver._friction_patches
    count = int(contacts.rigid_contact_count.numpy()[0])
    state = model.state()
    patches.previous.valid.zero_()
    patches.build(model, state, contacts, shape_material_mu=solver._shape_mu)
    patches.store(state)
    previous_owner = patches.previous.owner.numpy()[:count].copy()
    keys = patches.previous.keys.numpy()[:count]
    _, starts = np.unique(keys, return_index=True)
    pairs = np.split(patches.previous.indices.numpy()[:count], starts[1:])
    # New slot i holds the previous frame's contact perm[i]; each pair keeps its contacts' relative order.
    perm = np.concatenate([*pairs[:0:-1], pairs[0][:-1]])
    new_count = count - 1
    for name in _CONTACT_FIELDS:
        array = getattr(contacts, "rigid_contact_" + name)
        values = array.numpy().copy()
        values[:new_count] = values[perm]
        array.assign(values)
    contacts.rigid_contact_count.fill_(new_count)
    frozen = wp.ones(model.body_count, dtype=int, device=model.device)
    patches.build(model, state, contacts, shape_material_mu=solver._shape_mu, frozen_bodies=frozen)
    test.assertEqual(int(patches._flood_pair_count.numpy()[0]), int(len(pairs[0]) - 1 > _FLOOD_MIN_CONTACTS))
    slot = np.full(count, -1)
    slot[perm] = np.arange(new_count)
    carried = np.arange(new_count - (len(pairs[0]) - 1))
    source = patches.current.source.numpy()[carried]
    owner = patches.current.owner.numpy()[carried]
    np.testing.assert_array_equal(source, perm[carried])
    np.testing.assert_array_equal(owner, slot[previous_owner[perm[carried]]])
    regions = np.unique(owner)
    test.assertGreater(len(regions), 4, "too few friction regions to exercise the carry")
    test.assertLess(len(regions), len(carried), "no region spans several contacts")


def test_successive_notifications_keep_every_wake(test, device):
    """Keep the first notification's wake through a second notification before the next step."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    mass = model.body_mass.numpy()
    mass[0] *= 2.0
    model.body_mass.assign(mass)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    mu = model.shape_material_mu.numpy()
    mu[model.shape_body.numpy() == 1] *= 0.5
    model.shape_material_mu.assign(mu)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 1])


def test_replaced_property_array_wakes_its_island(test, device):
    """Wake the island changed by a model array that is replaced rather than assigned in place."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    com = model.body_com.numpy()
    com[0, 0] = 0.3
    model.body_com = wp.array(com, dtype=wp.vec3, device=model.device)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])


def test_resized_property_array_is_rejected(test, device):
    """Reject a model array replaced with a different entity count, which cannot be diffed."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 1)
    model.body_mass = wp.zeros(model.body_count + 1, dtype=float, device=model.device)
    with test.assertRaisesRegex(ValueError, "body_mass"):
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


def test_coincident_entity_counts_keep_property_owners(test, device):
    """Diff a body array only as a body property when body and joint counts coincide."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    bodies = []
    for x in (0.0, 0.5):
        body = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        bodies.append(body)
    # Reversed joint order gives body_to_articulation [1, 0] against joint_articulation [0, 1].
    for body in reversed(bodies):
        builder.add_articulation([builder.add_joint_free(child=body)])
    model = builder.finalize(device=device)
    test.assertEqual(model.body_count, model.joint_count)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverSteel(model, **PROFILE)
    states = [model.state(), model.state()]
    np.testing.assert_array_equal(solver._body_to_articulation.numpy(), [1, 0])
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    mass = model.body_mass.numpy()
    mass[0] *= 2.0
    model.body_mass.assign(mass)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])


def test_solver_body_attribute_notification_wakes_its_island(test, device):
    """Diff a Steel custom body attribute assigned to the model like a core body property."""
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    newton.solvers.SolverSteel.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverSteel(model, **PROFILE)
    states = [model.state(), model.state()]
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    depenetration = model.steel.rigid_body_max_depenetration_velocity.numpy()
    depenetration[0] = 1.0
    model.steel.rigid_body_max_depenetration_velocity.assign(depenetration)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])


def test_mimic_articulation_stays_awake(test, device):
    """Keep an articulation with mimic rows awake while an independent box sleeps."""
    for legacy in (False, True):
        with test.subTest(legacy_constraint_mimic=legacy):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            base = builder.add_link(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
            builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.05)
            joints = [builder.add_joint_free(child=base)]
            for side in (-1.0, 1.0):
                finger = builder.add_link(xform=wp.transform((side * 0.15, 0.0, 0.15), wp.quat_identity()))
                builder.add_shape_box(finger, hx=0.02, hy=0.02, hz=0.05)
                joints.append(
                    builder.add_joint_revolute(
                        parent=base,
                        child=finger,
                        parent_xform=wp.transform((side * 0.15, 0.0, 0.1), wp.quat_identity()),
                    )
                )
            builder.add_articulation(joints)
            if legacy:
                _expect_one_warning(
                    test,
                    newton.exceptions.NewtonDeprecationWarning,
                    r"add_constraint_mimic\(\) is deprecated",
                    functools.partial(builder.add_constraint_mimic, joint0=joints[2], joint1=joints[1]),
                )
            else:
                builder.set_joint_mimic(joints[2], joints[1])
            box = builder.add_body(xform=wp.transform((2.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverSteel(model, **PROFILE)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
            _advance(pipeline, solver, [model.state(), model.state()], model.control(), 400)
            awake = solver._sleeping.art_awake.numpy()
            test.assertEqual(int(awake[solver._body_to_articulation.numpy()[base]]), 1)
            test.assertEqual(int(awake[solver._body_to_articulation.numpy()[box]]), 0)


def _two_world_boxes(device):
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, deterministic=True)
    solver = newton.solvers.SolverSteel(model, **PROFILE)
    return model, pipeline, solver, [model.state(), model.state()]


def test_stiffness_without_a_spring_lets_articulations_sleep(test, device):
    """Sleep and step bitwise like zero stiffness when only FREE and BALL DOFs, which carry no spring, have stiffness."""
    runs = []
    for stiffness in (0.0, 50.0):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (0.0, 1.0):
            base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
            tip = builder.add_link(xform=wp.transform((x + 0.3, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
            hinge = wp.transform((0.3, 0.0, 0.0), wp.quat_identity())
            builder.add_articulation(
                [builder.add_joint_free(child=base), builder.add_joint_ball(parent=base, child=tip, parent_xform=hinge)]
            )
        builder.joint_stiffness = [stiffness] * builder.joint_dof_count
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
        solver = newton.solvers.SolverSteel(model, **PROFILE)
        states = [model.state(), model.state()]
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
        runs.append(states[0].body_q.numpy())
    np.testing.assert_array_equal(runs[1], runs[0])


def _add_box(builder):
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)


def _add_slab(builder):
    slab = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(slab, hx=1.0, hy=1.0, hz=0.05)
    return slab


def _add_fixed_support(builder):
    """Add a prescribed link welded to the world, without a shape."""
    xform = wp.transform((0.0, 0.0, -0.05), wp.quat_identity())
    support = builder.add_link(xform=xform)
    builder.add_articulation([builder.add_joint_fixed(-1, support, parent_xform=xform)])
    return support


def _expect_support_move_wakes(builder, support, sleepers, world_mask, device):
    """Let ``sleepers`` sleep on ``support``, slide it, reset with ``world_mask``, and require them awake a step later."""
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, deterministic=True)
    solver = newton.solvers.SolverSteel(model, **PROFILE)
    states = [model.state(), model.state()]
    for state in states:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy()[sleepers], 0)
    # The contact stays closed under a slide, so only the support's pose history shows the move.
    q_start = model.joint_q_start.numpy()
    joints = np.flatnonzero(model.joint_child.numpy() == support)
    coords = [q_start[j] for j in joints if q_start[j + 1] > q_start[j]]
    for state in states:
        body_q, joint_q = state.body_q.numpy(), state.joint_q.numpy()
        body_q[support, 0] += 0.05
        joint_q[coords] += 0.05
        state.body_q.assign(body_q)
        state.joint_q.assign(joint_q)
    solver.reset(states[0], world_mask=wp.array(world_mask, dtype=wp.bool, device=device))
    _advance(pipeline, solver, states, control, 1)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy()[sleepers], 1)


_HISTORY = ("last_q", "last_qd", "last_body_q", "last_body_qd")


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


class TestSteelSleepingProduction(unittest.TestCase):
    pass


for _name in (
    "test_anchored_articulations_sleep_and_wake",
    "test_graph_replay_sleeps_and_wakes",
    "test_masked_reset_wakes_only_selected_worlds",
    "test_masked_reset_restarts_only_selected_history",
    "test_masked_reset_restarts_state_change_history",
    "test_global_reset_keeps_shared_support_history",
    "test_world_reset_keeps_shared_support_history",
    "test_world_reset_keeps_support_history_read_through_another_world_shape",
    "test_world_reset_keeps_support_history_read_through_a_global_shape",
    "test_world_reset_keeps_support_history_read_by_a_global_shape",
    "test_property_notification_wakes_only_changed_islands",
    "test_frozen_patch_carry_matches_rebuild",
    "test_frozen_patch_carry_matches_rebuild_for_flooded_pairs",
    "test_frozen_patch_carry_copies_the_previous_history",
    "test_frozen_pair_floods_on_device_when_its_carry_fails",
    "test_frozen_pair_carry_maps_regions_across_frames",
    "test_successive_notifications_keep_every_wake",
    "test_replaced_property_array_wakes_its_island",
    "test_resized_property_array_is_rejected",
    "test_coincident_entity_counts_keep_property_owners",
    "test_solver_body_attribute_notification_wakes_its_island",
    "test_mimic_articulation_stays_awake",
    "test_stiffness_without_a_spring_lets_articulations_sleep",
):
    add_function_test(TestSteelSleepingProduction, _name, globals()[_name], devices=devices)


def test_fixed_table_does_not_merge_dynamic_islands(test, device):
    """Treat a zero-response table as a support boundary between free boxes."""
    for fixed_articulation in (False, True):
        with test.subTest(fixed_articulation=fixed_articulation):
            builder = newton.ModelBuilder()
            table = -1
            if fixed_articulation:
                table = builder.add_link()
                root = builder.add_joint_fixed(parent=-1, child=table)
                builder.add_articulation([root])
            builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.2)
            boxes = []
            for x in (-0.5, 0.5):
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.3), wp.quat_identity()))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                boxes.append(body)
            model = builder.finalize(device=device)
            pipeline, solver, state_in, _state_out, control = _runtime(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape_body = model.shape_body.numpy()
            body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
            body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
            for body in boxes:
                test.assertTrue(np.any(((body0 == body) & (body1 == table)) | ((body1 == body) & (body0 == table))))
            body_art = solver._body_to_articulation.numpy()
            if fixed_articulation:
                test.assertEqual(int(solver._model_plan.response_dof_count[body_art[table]]), 0)

            solver._sleeping.begin(state_in, control, contacts)

            roots = solver._sleeping.parent.numpy()[body_art[boxes]]
            test.assertNotEqual(int(roots[0]), int(roots[1]), f"Table joined box islands: {roots}")
            np.testing.assert_array_equal(solver._sleeping.root_supported.numpy()[roots], [1, 1])
            np.testing.assert_array_equal(solver._sleeping.root_veto.numpy()[roots], [0, 0])


def test_lost_support_wakes_below_velocity_threshold(test, device):
    """Release an unsupported former island member despite a small gravity kick."""
    model = _boxes(2, device)
    pipeline, solver, state_in, state_out, control = _runtime(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    body0, body1 = shape_body[shape0], shape_body[shape1]
    keep = np.flatnonzero(((body0 == 0) & (body1 < 0)) | ((body1 == 0) & (body0 < 0)))
    test.assertGreater(len(keep), 0)
    test.assertTrue(np.any((body0 == 1) | (body1 == 1)))
    # Retain real ground contacts only for A; B belonged to A's sleeping island.
    for suffix in (
        "shape0",
        "shape1",
        "point0",
        "point1",
        "offset0",
        "offset1",
        "normal",
        "margin0",
        "margin1",
        "tids",
    ):
        array = getattr(contacts, "rigid_contact_" + suffix)
        values = array.numpy()
        values[: len(keep)] = values[keep].copy()
        array.assign(values)
    contacts.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
    _seed_sleep(solver, state_in, [0, 0])
    dt = 0.001
    kick = abs(float(model.gravity.numpy()[0, 2])) * dt
    test.assertGreater(kick, 0.0)
    test.assertLess(kick, solver._sleeping.linear_threshold)
    before = state_in.body_q.numpy()[1, 2]

    solver.step(state_in, state_out, control, contacts, dt)

    test.assertEqual(int(solver._sleeping.body_awake.numpy()[1]), 1)
    test.assertLess(state_out.body_q.numpy()[1, 2], before)
    test.assertLess(state_out.body_qd.numpy()[1, 2], 0.0)
    test.assertLess(abs(state_out.body_qd.numpy()[1, 2]), solver._sleeping.linear_threshold)


def test_wake_crosses_alternating_old_and_current_islands(test, device):
    """Propagate a wake through alternating historical and current edges only."""
    model = _boxes(7, device)
    pipeline, solver, state_in, _state_out, control = _runtime(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertLessEqual(count + 2, contacts.rigid_contact_max)
    shape_body = model.shape_body.numpy()
    shapes = [int(np.flatnonzero(shape_body == body)[0]) for body in range(7)]
    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
    for body in range(7):
        test.assertTrue(np.any(((body0 == body) & (body1 < 0)) | ((body1 == body) & (body0 < 0))))
    # Old edges: 0--1, 2--3, 4--5. Current edges: 1--2, 3--4. Body 6 is isolated.
    for name, endpoints in (("shape0", [shapes[1], shapes[3]]), ("shape1", [shapes[2], shapes[4]])):
        array = getattr(contacts, "rigid_contact_" + name)
        values = array.numpy()
        values[count : count + 2] = endpoints
        array.assign(values)
    for suffix in ("point0", "point1", "normal", "margin0", "margin1"):
        array = getattr(contacts, "rigid_contact_" + suffix)
        values = array.numpy()
        values[count : count + 2] = 0
        array.assign(values)
    contacts.rigid_contact_count.assign(np.array([count + 2], dtype=np.int32))
    _seed_sleep(solver, state_in, [0, 0, 2, 2, 4, 4, 6])

    solver._sleeping.begin(state_in, control, contacts)

    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), np.zeros(7))
    _seed_sleep(solver, state_in, [0, 0, 2, 2, 4, 4, 6])
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 1.0
    state_in.body_f.assign(force)

    solver._sleeping.begin(state_in, control, contacts)

    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy()[:6], np.zeros(6))


def test_nonfinite_trial_output_is_not_hidden(test, device):
    """Preserve evidence of invalid trial outputs instead of freezing over it."""
    for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
        with test.subTest(field=field):
            model = _boxes(1, device)
            pipeline, solver, state_in, state_out, control = _runtime(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            _seed_sleep(solver, state_in, [0])
            # Only computed dynamics produce trial outputs; skipped ones publish the frozen state.
            solver._sleeping.skip_dynamics = False
            # Joint state is written by the integrator, body state by the kinematics publication.
            stage = "_integrate" if field.startswith("joint") else "_stage7_update_kinematics"
            original = getattr(solver, stage)

            def inject_nonfinite(*args, original=original, state_out=state_out, field=field):
                original(*args)
                array = getattr(state_out, field)
                values = array.numpy()
                values.flat[0] = np.nan
                array.assign(values)

            with patch.object(solver, stage, side_effect=inject_nonfinite):
                solver.step(state_in, state_out, control, contacts, 0.001)

            test.assertFalse(np.isfinite(getattr(state_out, field).numpy()).all())
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])


def test_aliased_states_are_rejected(test, device):
    """Reject aliased state allocations before changing physical state."""
    for kind, field in (("same_state", ""), ("alias", "joint_q"), ("alias", "body_q")):
        with test.subTest(kind=kind, field=field):
            model = _boxes(1, device)
            pipeline, solver, state_in, state_out, control = _runtime(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            if kind == "same_state":
                state_out = state_in
            else:
                setattr(state_out, field, getattr(state_in, field))
            before = state_in.joint_q.numpy().copy()

            with test.assertRaises(ValueError):
                solver.step(state_in, state_out, control, contacts, 0.001)

            np.testing.assert_array_equal(state_in.joint_q.numpy(), before)


def test_extended_state_outputs_are_accepted(test, device):
    """Sleep with requested acceleration and joint-wrench outputs, which Steel leaves untouched."""
    model = _boxes(1, device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_qdd", "body_parent_f")
    pipeline, solver, state_in, state_out, control = _runtime(model)
    test.assertIsNotNone(state_in.body_parent_f)
    contacts = pipeline.contacts()
    states = [state_in, state_out]
    for _ in range(400):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0])


def test_invalid_timestep_is_rejected(test, device):
    """Reject nonpositive and nonfinite timesteps before changing physical state."""
    model = _boxes(1, device)
    pipeline, solver, state_in, state_out, control = _runtime(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    before = state_in.joint_q.numpy().copy()
    for dt in (0.0, -0.001, float("nan"), float("inf")):
        with test.subTest(dt=dt), test.assertRaises(ValueError):
            solver.step(state_in, state_out, control, contacts, dt)
        np.testing.assert_array_equal(state_in.joint_q.numpy(), before)


def test_reduction_overflow_cannot_authorize_sleep(test, device):
    """Keep an island awake when its contact producer reports missing contacts."""
    model = _boxes(1, device)
    pipeline, solver, state_in, state_out, control = _runtime(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    _seed_sleep(solver, state_in, [0])
    contacts._reduction_overflow.fill_(1)

    solver.step(state_in, state_out, control, contacts, 0.001)

    np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
    np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy(), [0.0])
    # The loss wakes the island before the step, so its dynamics and rows run.
    np.testing.assert_array_equal(solver._dynamics_art_active.numpy(), [1])


def test_notifications_reset_quiet_age_and_preserve_state(test, device):
    """Wake sleeping state on model notification or reset without moving bodies."""
    model = _boxes(2, device)
    _pipeline, solver, state_in, _state_out, _control = _runtime(model)
    before = state_in.joint_q.numpy().copy()
    for event in ("model", "reset"):
        with test.subTest(event=event):
            _seed_sleep(solver, state_in, [0, 0])
            if event == "model":
                solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            else:
                solver.reset(state_in)
            np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
            np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy(), [0.0, 0.0])
            np.testing.assert_array_equal(state_in.joint_q.numpy(), before)


def _boxes(count, device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for index in range(count):
        body = builder.add_body(xform=wp.transform(wp.vec3(index * 0.5, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def _runtime(model):
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
    solver = newton.solvers.SolverSteel(
        model,
        enable_sleeping=True,
        friction_anchor_beta=0.0,
        sleep_quiet_time=0.05,
        iterations=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    state_in.clear_forces()
    return pipeline, solver, state_in, state_out, model.control()


def _seed_sleep(solver, state, previous_roots):
    sleeping = solver._sleeping
    sleeping.art_awake.zero_()
    sleeping.body_awake.zero_()
    sleeping.previous_root.assign(np.asarray(previous_roots, dtype=np.int32))
    sleeping.quiet_age.fill_(sleeping.quiet_time)
    for destination, source in (
        (sleeping.last_q, state.joint_q),
        (sleeping.last_qd, state.joint_qd),
        (sleeping.last_body_q, state.body_q),
        (sleeping.last_body_qd, state.body_qd),
    ):
        wp.copy(destination, source)
    sleeping.valid.fill_(1)


class TestSteelSleepingSafety(unittest.TestCase):
    pass


for _name in (
    "test_fixed_table_does_not_merge_dynamic_islands",
    "test_lost_support_wakes_below_velocity_threshold",
    "test_wake_crosses_alternating_old_and_current_islands",
    "test_nonfinite_trial_output_is_not_hidden",
    "test_aliased_states_are_rejected",
    "test_extended_state_outputs_are_accepted",
    "test_invalid_timestep_is_rejected",
    "test_reduction_overflow_cannot_authorize_sleep",
    "test_notifications_reset_quiet_age_and_preserve_state",
):
    add_function_test(TestSteelSleepingSafety, _name, globals()[_name], devices=devices)


TORSION_PROFILE = {**PROFILE, "contact_torsion_device": True}


def _torsion_rows(solver):
    from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_TORSION  # noqa: PLC0415

    row_types = solver._row_type.numpy()[0, : int(solver._constraint_count.numpy()[0])]
    return int(np.count_nonzero(row_types == PGS_CONSTRAINT_TYPE_TORSION))


def test_anchored_torsion_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore anchored torsion rows on a force wake."""
    model, pipeline, solver, states, control = _articulations(device, profile=TORSION_PROFILE, torsion_radius=0.01)
    _advance(pipeline, solver, states, control, 5)
    awake_rows = int(solver._constraint_count.numpy()[0])
    awake_torsion = _torsion_rows(solver)
    test.assertGreater(awake_torsion, 0)
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
    np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
    frozen = states[0].body_q.numpy().copy()
    _advance(pipeline, solver, states, control, 20)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

    # Pushing the first articulation restores its rows and leaves the second asleep.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 20.0
    for _ in range(5):
        states[0].body_f.assign(force)
        _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1, 0, 0])
    test.assertEqual(int(solver._constraint_count.numpy()[0]), awake_rows // 2)
    test.assertEqual(_torsion_rows(solver), awake_torsion // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_validates_torsion(test, device):
    """Sleep and wake inside a captured graph with device torsion, checking its status after replays."""
    model, pipeline, solver, states, control = _articulations(device, profile=TORSION_PROFILE, torsion_radius=0.01)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 2, contacts=contacts)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    solver._sleeping.wake()
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    test.assertGreater(_torsion_rows(solver), 0)


class TestSteelSleepingTorsion(unittest.TestCase):
    pass


for _name in ("test_anchored_torsion_articulations_sleep_and_wake", "test_graph_replay_validates_torsion"):
    add_function_test(TestSteelSleepingTorsion, _name, globals()[_name], devices=devices)


def test_ant_joint_limits_skip_and_restore(test, device):
    """Skip a sleeping ant's limit rows and restore them on captured force wake."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(Path(newton.__file__).parent / "examples/assets/nv_ant.xml"),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        parse_sites=False,
    )
    builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
    builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    builder.joint_act[:] = [0.0] * len(builder.joint_act)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    for skip in (False, True):
        with test.subTest(skip_constraints=skip):
            solver = newton.solvers.SolverSteel(
                model,
                enable_sleeping=True,
                sleep_skip_constraints=skip,
                sleep_linear_threshold=0.05,
                sleep_angular_threshold=0.15,
                sleep_quiet_time=0.5,
                enable_joint_limits=True,
                iterations=8,
                friction_anchor_beta=0.0,
                articulation_constraint_max=256,
                free_body_constraint_max=256,
            )
            original_indices = solver._joint_limit_q_index.numpy().copy()
            test.assertEqual(int(np.count_nonzero(original_indices >= 0)), 8)
            test.assertTrue(solver._joint_limit_warp_kernels)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
            contacts = pipeline.contacts()
            states = [model.state(), model.state()]
            control = model.control()
            for _ in range(2):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                states.reverse()
            with wp.ScopedCapture(device=model.device) as capture:
                for _ in range(2):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                    states.reverse()
            for _ in range(1200):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0])
            if skip:
                np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
                np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
                test.assertTrue(np.all(solver._sleeping.limit_q_index.numpy() < 0))
            else:
                test.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[0, 0] = 100.0
            for state in states:
                state.body_f.assign(force)

            wp.capture_launch(capture.graph)

            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
            test.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
            np.testing.assert_array_equal(solver._sleeping.limit_q_index.numpy(), original_indices)
            test.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
            test.assertTrue(np.isfinite(states[0].body_qd.numpy()).all())
            test.assertFalse(solver.constraint_overflow.numpy().any())


def test_settled_rows_drop_without_changing_contacts(test, device):
    """Remove solver rows while retaining complete collision contact inputs."""
    scene = _Scene(device)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    before = _contact_inputs(scene.contacts)
    test.assertGreater(int(before["count"][0]), 0)
    pose = scene.states[0].body_q.numpy().copy()

    scene.step(collide=False)

    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
    for name, values in before.items():
        np.testing.assert_array_equal(_contact_inputs(scene.contacts)[name], values, err_msg=name)


def test_explicit_full_solve_retains_rows(test, device):
    """Compare settled row counts against explicit contact-skipping disablement."""
    full, skipped = _Scene(device, skip=False), _Scene(device)
    for scene in (full, skipped):
        scene.settle()
        scene.step()
        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
        test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertGreater(int(full.row_counts().sum()), 0)
    np.testing.assert_array_equal(skipped.row_counts(), [0, 0])
    np.testing.assert_allclose(full.states[0].body_q.numpy(), skipped.states[0].body_q.numpy(), atol=1.0e-6, rtol=0)


def test_force_restores_only_independent_ground_island_rows(test, device):
    """Restore forced-body rows while another ground-supported island sleeps."""
    scene = _Scene(device)
    scene.step()
    active_rows = int(scene.row_counts().sum())
    test.assertGreater(active_rows, 0)
    scene.settle()
    test.assertNotEqual(*scene.solver._sleeping.body_island.numpy())
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    resting_pose = scene.states[0].body_q.numpy()[1].copy()
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    scene.states[0].body_f.assign(force)

    scene.step(clear=False)

    rows = int(scene.row_counts().sum())
    test.assertGreater(rows, 0)
    test.assertLess(rows, active_rows)
    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
    np.testing.assert_array_equal(scene.states[0].body_q.numpy()[1], resting_pose)


def test_support_loss_wakes_before_skipped_step(test, device):
    """Let a newly unsupported sleeper fall during the same small timestep."""
    scene = _Scene(device)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    shape_body = scene.model.shape_body.numpy()
    body0 = shape_body[scene.contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[scene.contacts.rigid_contact_shape1.numpy()[:count]]
    keep = np.flatnonzero(((body0 == 0) & (body1 < 0)) | ((body1 == 0) & (body0 < 0)))
    test.assertGreater(len(keep), 0)
    test.assertTrue(np.any((body0 == 1) | (body1 == 1)))
    for suffix in _CONTACT_FIELDS:
        array = getattr(scene.contacts, "rigid_contact_" + suffix)
        values = array.numpy()
        values[: len(keep)] = values[keep].copy()
        array.assign(values)
    scene.contacts.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
    before = float(scene.states[0].body_q.numpy()[1, 2])

    scene.step(collide=False, dt=0.001)

    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 1])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    test.assertLess(float(scene.states[0].body_q.numpy()[1, 2]), before)
    velocity = float(scene.states[0].body_qd.numpy()[1, 2])
    test.assertLess(velocity, 0.0)
    test.assertLess(abs(velocity), scene.solver._sleeping.linear_threshold)


def test_fixed_table_supports_sleep_and_independent_force_wake(test, device):
    """Sleep both table-supported boxes and restore only the forced box's rows."""
    scene = _Scene(device, fixed_table=True)
    observables = scene.solver.observables({SolverSteel.ObservableKind.BODY_AWAKE})
    scene.step(observables=observables)
    active_rows = int(scene.row_counts().sum())
    test.assertGreater(active_rows, 0)
    scene.settle(observables=observables)
    np.testing.assert_array_equal(observables.body_awake.numpy(), [True, False, False])
    islands = scene.solver._sleeping.body_island.numpy()[scene.bodies]
    test.assertNotEqual(*islands)
    np.testing.assert_array_equal(scene.solver._sleeping.root_supported.numpy()[islands], [1, 1])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    resting_pose = scene.states[0].body_q.numpy()[scene.bodies[1]].copy()
    force = np.zeros((scene.model.body_count, 6), dtype=np.float32)
    force[scene.bodies[0], 0] = 100.0
    scene.states[0].body_f.assign(force)

    scene.step(clear=False, observables=observables)

    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy()[scene.bodies], [1, 0])
    np.testing.assert_array_equal(observables.body_awake.numpy(), [True, True, False])
    rows = int(scene.row_counts().sum())
    test.assertGreater(rows, 0)
    test.assertLess(rows, active_rows)
    test.assertGreater(float(scene.states[0].body_qd.numpy()[scene.bodies[0], 0]), 0.0)
    np.testing.assert_array_equal(scene.states[0].body_q.numpy()[scene.bodies[1]], resting_pose)


def test_graph_replay_restores_rows_on_force(test, device):
    """Restore matrix-free contact rows on a captured sleeping-to-awake transition."""
    scene = _Scene(device)
    scene.step()
    test.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
    scene.step()
    with wp.ScopedCapture(device=scene.model.device) as capture:
        for _ in range(2):
            scene.step(clear=False)
    for _ in range(100):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    for state in scene.states:
        state.body_f.assign(force)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
    test.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
    test.assertTrue(np.isfinite(scene.states[0].body_q.numpy()).all())
    test.assertTrue(np.isfinite(scene.states[0].body_qd.numpy()).all())
    test.assertFalse(scene.solver.constraint_overflow.numpy().any())


def test_nonfinite_prescribed_velocity_wakes_before_allocation(test, device):
    """Wake supported sleepers when their fixed table has invalid velocity."""
    scene = _Scene(device, fixed_table=True)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    pose = scene.states[0].body_q.numpy().copy()
    velocity = scene.states[0].body_qd.numpy()
    velocity[0, 0] = np.nan
    scene.states[0].body_qd.assign(velocity)

    scene.solver._sleeping.begin(scene.states[0], scene.control, scene.contacts)

    np.testing.assert_array_equal(scene.solver._sleeping.art_awake.numpy()[scene.bodies], [1, 1])
    np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
    test.assertTrue(np.isnan(scene.states[0].body_qd.numpy()[0, 0]))


class _Scene:
    def __init__(self, device, *, skip=True, fixed_table=False):
        builder = newton.ModelBuilder()
        if fixed_table:
            table = builder.add_link()
            root = builder.add_joint_fixed(parent=-1, child=table)
            builder.add_articulation([root])
            builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.1)
        else:
            builder.add_ground_plane()
        self.bodies = []
        for x in (0.0, 0.5):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.2 if fixed_table else 0.1), wp.quat_identity())
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            self.bodies.append(body)
        self.model = builder.finalize(device=device)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64)
        self.contacts = self.pipeline.contacts()
        options = {} if skip else {"sleep_skip_constraints": False}
        self.solver = newton.solvers.SolverSteel(
            self.model,
            enable_sleeping=True,
            sleep_quiet_time=0.05,
            friction_anchor_beta=0.0,
            iterations=16,
            **options,
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()

    def step(self, *, clear=True, collide=True, dt=0.005, observables=None):
        if clear:
            self.states[0].clear_forces()
        if collide:
            self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, dt, observables=observables)
        self.states.reverse()

    def settle(self, *, observables=None):
        for _ in range(120):
            self.step(observables=observables)
        np.testing.assert_array_equal(self.solver._sleeping.body_awake.numpy()[self.bodies], [0, 0])
        if self.solver.constraint_overflow.numpy().any():
            raise AssertionError("Contact capacity exceeded")

    def row_counts(self):
        return np.array([self.solver._constraint_count.numpy().sum(), self.solver._mf_constraint_count.numpy().sum()])


_CONTACT_FIELDS = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1", "tids")


def _contact_inputs(contacts):
    values = {name: getattr(contacts, "rigid_contact_" + name).numpy().copy() for name in _CONTACT_FIELDS}
    values["count"] = contacts.rigid_contact_count.numpy().copy()
    return values


class TestSteelSleepWorkSkipping(unittest.TestCase):
    pass


for _name in (
    "test_ant_joint_limits_skip_and_restore",
    "test_settled_rows_drop_without_changing_contacts",
    "test_explicit_full_solve_retains_rows",
    "test_force_restores_only_independent_ground_island_rows",
    "test_support_loss_wakes_before_skipped_step",
    "test_fixed_table_supports_sleep_and_independent_force_wake",
    "test_graph_replay_restores_rows_on_force",
    "test_nonfinite_prescribed_velocity_wakes_before_allocation",
):
    add_function_test(TestSteelSleepWorkSkipping, _name, globals()[_name], devices=devices)


COLORED_STEPS = 240


def test_colored_separated_boxes_sleep_and_wake_like_immediate(test, device):
    """Sleep, force-wake and contact-wake separated boxes at the same steps as immediate, with units only while awake."""
    traces = {}
    for response in ("immediate", "colored"):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (0.0, 0.5):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        model.rigid_contact_max = 64
        scene = _Heap(device, response, model=model, sleep_quiet_time=0.05, friction_anchor_beta=0.0)
        awake, poses, units = [], [], []
        # One step of this force slides box 0 into box 1, which no force or reset touches.
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = 8000.0
        for k in range(240):
            if k == 120:
                scene.state_in.body_f.assign(force)
            scene.step(clear=k != 120)
            awake.append(scene.solver._sleeping.body_awake.numpy().copy())
            poses.append(scene.state_in.body_q.numpy().copy())
            if response == "colored":
                _contact, body_a, body_b = scene.units()
                units.append([int(np.sum((body_a == b) | (body_b == b))) for b in (0, 1)])
        traces[response] = (np.array(awake), np.array(poses), np.array(units))
    immediate, colored = traces["immediate"], traces["colored"]
    awake, poses, units = colored
    slept = int(np.argmax(awake.sum(axis=1) == 0))
    test.assertGreater(slept, 0, "both boxes must fall asleep before the force")
    test.assertLess(slept, 120)
    hit = 120 + int(np.argmax(awake[120:, 1] == 1))
    test.assertGreater(hit, 120, "box 1 must wake after the force")
    np.testing.assert_array_equal(awake[: hit + 1], immediate[0][: hit + 1])
    np.testing.assert_allclose(poses[:slept], immediate[1][:slept], atol=1.0e-4, rtol=0)
    # Units exist only for a box that was awake during the step, and never for one asleep before and after it.
    test.assertTrue(np.all(units[:slept] > 0))
    np.testing.assert_array_equal(units[slept + 1 : 120], 0)
    np.testing.assert_array_equal(awake[120:hit], [[1, 0]] * (hit - 120))
    test.assertGreater(units[120, 0], 0)
    np.testing.assert_array_equal(units[120:hit, 1], 0)
    test.assertGreater(units[hit + 1, 1], 0)
    test.assertGreater(poses[hit - 1, 0, 0], poses[119, 0, 0] + 0.05)
    test.assertGreater(poses[-1, 1, 0], poses[hit, 1, 0] + 0.05)
    np.testing.assert_array_equal(awake[-1], [0, 0])
    np.testing.assert_array_equal(immediate[0][-1], [0, 0])


def test_colored_compliant_contacts_sleep_and_wake_like_immediate(test, device):
    """Sleep and push-wake spheres on compliant contacts at the steps immediate does, eager and captured."""
    stiffness = 3000.0
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 5.0
    traces = {}
    for response, graph in (("immediate", False), ("colored", False), ("colored", True)):
        model, pipeline, contacts, solver = _compliance_scene(device, sleeping=True, response=response)
        states = [model.state(), model.state()]
        control = model.control()
        pushes = [wp.zeros_like(states[0].body_f), wp.array(force, dtype=wp.spatial_vector, device=device)]

        def step(push, pipeline=pipeline, contacts=contacts, solver=solver, states=states, control=control):
            # Fixed buffers, so the same work can be captured.
            wp.copy(states[0].body_f, push)
            pipeline.collide(states[0], contacts)
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(1.0)
            solver.step(states[0], states[1], control, contacts, DT)
            for name in _FIELDS:
                wp.copy(getattr(states[0], name), getattr(states[1], name))

        if graph:
            graphs = []
            for push in pushes:
                with wp.ScopedCapture(device=model.device) as capture:
                    step(push)
                graphs.append(capture.graph)

            def run(push, graphs=graphs):
                wp.capture_launch(graphs[push])
        else:

            def run(push, step=step, pushes=pushes):
                step(pushes[push])

        awake, poses = [], []
        for k in range(420):
            run(1 if 400 <= k < 405 else 0)
            awake.append(solver._sleeping.body_awake.numpy().copy())
            poses.append(states[0].body_q.numpy().copy())
        solver.check_constraint_capacity()
        traces[response, graph] = (np.array(awake), np.array(poses))
    awake, poses = traces["colored", False]
    slept = int(np.argmax(awake.sum(axis=1) == 0))
    test.assertGreater(slept, 0, "both spheres must fall asleep before the push")
    test.assertLess(slept, 400)
    np.testing.assert_array_equal(awake, traces["immediate", False][0])
    np.testing.assert_allclose(poses, traces["immediate", False][1], atol=1.0e-4, rtol=0)
    # Asleep, the spheres stay frozen; the push wakes and moves sphere 0 only.
    np.testing.assert_array_equal(poses[slept:400], np.broadcast_to(poses[slept], poses[slept:400].shape))
    np.testing.assert_array_equal(awake[404], [1, 0])
    test.assertGreater(float(poses[-1, 0, 0]), float(poses[399, 0, 0]))
    np.testing.assert_array_equal(poses[400:, 1], np.broadcast_to(poses[399, 1], poses[400:, 1].shape))
    # Captured replay matches eager stepping bitwise.
    np.testing.assert_array_equal(traces["colored", True][0], awake)
    np.testing.assert_array_equal(traces["colored", True][1], poses)


def test_colored_units_omit_exactly_the_sleeping_islands(test, device):
    """Color every contact of an awake island and none of a skipped one; keep all of them with skipping off."""
    for skip in (True, False):
        with test.subTest(skip=skip):
            scene = _Heap(device, sleep_skip_constraints=skip)
            scene.step(COLORED_STEPS)
            asleep = scene.asleep_this_step()
            body0, body1 = scene.contact_bodies()
            sleeping_contacts = _touches(body0, asleep) | _touches(body1, asleep)
            test.assertGreater(int(sleeping_contacts.sum()), 0, "the heap must have sleeping contacts")
            test.assertGreater(int((~sleeping_contacts).sum()), 0, "the arm must keep awake contacts")
            contact, body_a, body_b = scene.units()
            test.assertEqual(len(contact), len(set(contact.tolist())))
            path = scene.solver._contact_path.numpy()[: len(body0)]
            np.testing.assert_array_equal(np.sort(contact), np.flatnonzero(path == 2))
            expected = np.flatnonzero(~sleeping_contacts) if skip else np.arange(len(body0))
            np.testing.assert_array_equal(np.sort(contact), expected)
            rows_touch = _touches(scene.propagation_row_bodies(), asleep)
            units_touch = _touches(body_a, asleep) | _touches(body_b, asleep)
            if skip:
                test.assertFalse(units_touch.any())
                test.assertFalse(rows_touch.any())
            else:
                test.assertEqual(int(units_touch.sum()), int(sleeping_contacts.sum()))
                test.assertTrue(rows_touch.any())


def test_colored_propagation_overflow_vetoes_sleep(test, device):
    """Keep worlds whose propagation rows overflowed awake; the same scene sleeps when the veto ignores it."""
    module = inspect.getmodule(_SleepState)
    launch = wp.launch
    asleep = {}
    for ignore_overflow in (False, True):
        # The settled arm and heap need more rows than this; once they sleep they need fewer.
        scene = _Heap(device, propagation_constraint_max=300, warn_constraint_overflow=False)
        clean = wp.zeros_like(scene.solver.constraint_overflow)

        def veto_without_overflow(kernel, dim, inputs=None, *args, scene=scene, clean=clean, **kwargs):
            if kernel in (module._veto_unsafe_islands, module._quiet_components):
                inputs = [clean if value is scene.solver.constraint_overflow else value for value in inputs]
            return launch(kernel, dim, inputs, *args, **kwargs)

        with patch.object(module.wp, "launch", veto_without_overflow if ignore_overflow else launch):
            scene.step(COLORED_STEPS)
        marks = scene.solver.constraint_row_watermarks()
        test.assertGreater(marks["propagation_overflow_world_steps"], 0, "the propagation rows must overflow")
        test.assertEqual(marks["articulation_overflow_world_steps"], 0)
        test.assertEqual(marks["free_body_overflow_world_steps"], 0)
        test.assertTrue(scene.solver.constraint_overflow.numpy()[: scene.model.world_count].all())
        asleep[ignore_overflow] = int(np.sum(scene.solver._sleeping.art_awake.numpy() == 0))
    test.assertEqual(asleep[False], 0)
    test.assertGreater(asleep[True], 0)


def test_colored_force_reset_and_notification_wake_islands(test, device):
    """Wake a forced island, a masked-reset world and a notified body's island, and color their contacts again."""
    scene = _Heap(device)
    scene.step(COLORED_STEPS)
    sleeping = scene.solver._sleeping
    worlds = scene.model.world_count
    bodies_per_world = scene.model.body_count // worlds
    body_world = np.arange(scene.model.body_count) // bodies_per_world
    islands = sleeping.body_island.numpy()
    asleep = np.flatnonzero(sleeping.body_awake.numpy() == 0)
    test.assertEqual(len(asleep) % worlds, 0)
    test.assertGreater(len(asleep), 0)

    def colored_bodies():
        _contact, body_a, body_b = scene.units()
        return set(body_a.tolist()) | set(body_b.tolist())

    # Force: the forced body's island wakes alone, in world 1.
    target = int(asleep[body_world[asleep] == 1][0])
    force = np.zeros((scene.model.body_count, 6), dtype=np.float32)
    force[target, 2] = 5.0
    scene.state_in.body_f.assign(force)
    before = sleeping.body_awake.numpy().copy()
    scene.step(clear=False)
    awake = sleeping.body_awake.numpy()
    woken = np.flatnonzero((before == 0) & (awake == 1))
    test.assertIn(target, woken)
    test.assertTrue(np.all(body_world[woken] == 1))
    test.assertIn(target, colored_bodies())
    np.testing.assert_array_equal(islands[woken], islands[target])

    # Masked reset: world 2 wakes, other worlds keep their sleepers.
    before = sleeping.body_awake.numpy().copy()
    mask = np.zeros(worlds + 1, dtype=bool)
    mask[2] = True
    scene.solver.reset(scene.state_in, world_mask=wp.array(mask, dtype=wp.bool, device=device))
    after = sleeping.body_awake.numpy()
    test.assertTrue(np.all(after[body_world == 2] == 1))
    np.testing.assert_array_equal(after[body_world != 2], before[body_world != 2])
    scene.step()
    test.assertFalse(scene.asleep_this_step()[body_world == 2].any())
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    contact_world = scene.solver._contact_world.numpy()[:count]
    # Contacts beyond the gap limit get no rows and are never colored.
    routed = scene.solver._contact_path.numpy()[:count] == 2
    contact, _body_a, _body_b = scene.units()
    np.testing.assert_array_equal(
        np.sort(contact[contact_world[contact] == 2]), np.flatnonzero((contact_world == 2) & routed)
    )
    test.assertTrue(set(np.flatnonzero((body_world == 2) & (before == 0)).tolist()) <= colored_bodies())

    # Notification: changing one sleeping body's mass wakes its island in world 3 only.
    before = sleeping.body_awake.numpy().copy()
    target = int(np.flatnonzero((before == 0) & (body_world == 3))[0])
    mass = scene.model.body_mass.numpy()
    mass[target] *= 2.0
    scene.model.body_mass.assign(mass)
    scene.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    scene.step()
    awake = sleeping.body_awake.numpy()
    woken = np.flatnonzero((before == 0) & (awake == 1))
    test.assertIn(target, woken)
    test.assertTrue(np.all(body_world[woken] == 3))
    test.assertIn(target, colored_bodies())


def test_colored_sleeping_rollout_is_independent_of_world_order(test, device):
    """Give each world the same trajectory and sleep pattern whichever position it takes in the model."""
    offsets = (0.0, 0.05)
    results = []
    for order in ((0, 1), (1, 0)):
        scene = _Heap(device, worlds=2)
        model = scene.model
        coord_world = model.joint_q.numpy().size // 2
        q = scene.state_in.joint_q.numpy()
        arm = np.arange(6)
        for position, variant in enumerate(order):
            q[position * coord_world + arm] += offsets[variant]
        scene.state_in.joint_q.assign(q)
        newton.eval_fk(model, scene.state_in.joint_q, scene.state_in.joint_qd, scene.state_in)
        scene.step(COLORED_STEPS)
        per_variant = {}
        for position, variant in enumerate(order):
            per_variant[variant] = [
                values.reshape(2, -1, *values.shape[1:])[position]
                for values in (
                    scene.state_in.body_q.numpy(),
                    scene.state_in.body_qd.numpy(),
                    scene.state_in.joint_q.numpy(),
                    scene.solver._sleeping.art_awake.numpy(),
                )
            ]
        results.append(per_variant)
    for variant in (0, 1):
        for a, b in zip(results[0][variant], results[1][variant], strict=True):
            np.testing.assert_array_equal(a, b, err_msg=f"offset {offsets[variant]}")
        test.assertGreater(int(np.sum(results[0][variant][3] == 0)), 0)
    test.assertFalse(np.array_equal(results[0][0][0], results[0][1][0]), "the two worlds must differ")


def test_colored_captured_sleep_and_wake_match_eager(test, device):
    """Cold-capture one step, then match an eager twin bitwise, with identical worlds, through sleeping and a force wake.

    The joint wrenches match to rounding as well.
    """
    captured, eager = _Heap(device), _Heap(device)
    for scene in (captured, eager):
        scene.observables = scene.solver.observables({SolverBase.ObservableKind.BODY_PARENT_F})
    with wp.ScopedCapture(device) as capture:
        captured.step(clear=False)
    worlds = eager.model.world_count
    force = np.zeros((captured.model.body_count, 6), dtype=np.float32)
    for k in range(COLORED_STEPS + 20):
        if k == COLORED_STEPS:
            asleep = np.flatnonzero(eager.solver._sleeping.body_awake.numpy() == 0)
            test.assertGreater(len(asleep), 0)
            force[asleep[0], 2] = 5.0
        for scene in (captured, eager):
            scene.state_in.body_f.assign(force)
        wp.capture_launch(capture.graph)
        eager.step(clear=False)
        a, b = captured.snapshot(), eager.snapshot()
        for name in _FIELDS:
            np.testing.assert_array_equal(a[name], b[name], err_msg=f"replay {k} {name}")
            if k < COLORED_STEPS:
                per_world = b[name].reshape(worlds, -1, *b[name].shape[1:])
                for world in range(1, worlds):
                    np.testing.assert_array_equal(per_world[world], per_world[0], err_msg=f"world {world} {name} {k}")
        np.testing.assert_array_equal(
            captured.solver._sleeping.art_awake.numpy(), eager.solver._sleeping.art_awake.numpy()
        )
        wrench = eager.observables.body_parent_f.numpy()
        # The wrench sums are not ordered, so replay and eager agree only to rounding.
        np.testing.assert_allclose(
            captured.observables.body_parent_f.numpy(), wrench, rtol=1.0e-5, atol=1.0e-4, err_msg=f"replay {k}"
        )
        if k == COLORED_STEPS - 1:
            # The arm's joints carry its weight while the heap sleeps, so the match is not trivial.
            test.assertGreater(np.abs(wrench).max(), 1.0)
        force[:] = 0.0
    test.assertEqual(int(captured.solver._sleeping.body_awake.numpy()[asleep[0]]), 1)


def test_colored_separated_stacks_sleep_like_immediate(test, device):
    """Separated box stacks sleep within a few steps of immediate and leave no colored units."""
    scene_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(4):
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    body = world.add_body(
                        xform=wp.transform(wp.vec3(0.3 * ix, 0.3 * iy, 0.051 + 0.102 * iz), wp.quat_identity())
                    )
                    world.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        scene_builder.add_world(world)
    scene_builder.add_ground_plane()
    model = scene_builder.finalize(device=device)
    model.rigid_contact_max = 8192
    first_sleep, final_asleep = {}, {}
    for response in ("immediate", "colored"):
        scene = _Heap(device, response, model=model)
        first = np.full(model.articulation_count, -1)
        for k in range(COLORED_STEPS):
            scene.step()
            asleep = scene.solver._sleeping.art_awake.numpy() == 0
            first[(first < 0) & asleep] = k
            if response == "colored" and k == 0:
                test.assertGreater(len(scene.units()[0]), 0)
        first_sleep[response], final_asleep[response] = first, asleep
    test.assertEqual(len(scene.units()[0]), 0)
    test.assertTrue(final_asleep["immediate"].all())
    test.assertTrue(final_asleep["colored"].all())
    delta = np.abs(first_sleep["colored"] - first_sleep["immediate"])
    test.assertLessEqual(int(delta.max()), 6)


def _dumbbells_model(device, count=2, welded=False):
    """Two-link floating chains resting apart on a ground plane, so their trees go through propagation.

    ``welded`` adds a box welded to the world away from the chains, an articulation that never sleeps.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    for i in range(count):
        x = 0.6 * i
        a = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.04), wp.quat_identity()))
        builder.add_shape_box(a, hx=0.08, hy=0.04, hz=0.04)
        b = builder.add_link(xform=wp.transform(wp.vec3(x + 0.16, 0.0, 0.04), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.08, hy=0.04, hz=0.04)
        root = builder.add_joint_free(child=a, parent_xform=wp.transform(wp.vec3(x, 0.0, 0.04), wp.quat_identity()))
        hinge = builder.add_joint_revolute(
            parent=a,
            child=b,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root, hinge])
    if welded:
        pose = wp.transform(wp.vec3(-1.0, 0.0, 0.2), wp.quat_identity())
        post = builder.add_link(xform=pose)
        builder.add_shape_box(post, hx=0.05, hy=0.05, hz=0.05)
        builder.add_articulation([builder.add_joint_fixed(-1, post, parent_xform=pose)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 128
    return model


def test_colored_sleeping_skips_tree_and_solve_work(test, device):
    """Leave sleeping trees unfactored, and skip the captured propagation solve while all sleep, bitwise like eager.

    With contact compliance the propagation setup still runs while all sleep, since compliance replaces its rows.
    A welded box, which never sleeps, does not keep the solve running.
    """
    for tree_kernel, compliant, welded in (
        ("auto", False, False),
        ("generic", False, False),
        ("auto", True, False),
        ("auto", False, True),
    ):
        with (
            test.subTest(tree_kernel=tree_kernel, compliant=compliant, welded=welded),
            kernel_overrides({"propagation_tree_kernel": tree_kernel}),
        ):
            _check_colored_sleeping_skips_work(test, device, compliant, welded)


def _check_colored_sleeping_skips_work(test, device, compliant, welded):
    options = {"sleep_quiet_time": 0.05, "friction_anchor_beta": 0.0}
    if compliant:
        options.update(contact_compliance=True, contact_regularization=0.0, compliance_stiffness=2.0e4)
    captured, eager = (_Heap(device, model=_dumbbells_model(device, welded=welded), **options) for _ in range(2))
    test.assertTrue(captured.solver._propagation_tree_sizes)
    with wp.ScopedCapture(device) as capture:
        captured.step(clear=False)
    gated = wp.is_conditional_graph_supported()
    joint_art = captured.model.joint_articulation.numpy()
    sleepable = captured.solver._sleeping.art_skippable.numpy()
    test.assertEqual(sleepable.tolist(), [1, 1, 0] if welded else [1, 1])
    body_art = captured.solver._body_to_articulation.numpy()
    tree_bodies = np.flatnonzero(sleepable[body_art] == 1)
    force = np.zeros((captured.model.body_count, 6), dtype=np.float32)
    all_asleep = 0
    for k in range(60):
        force[:] = 0.0
        if k == 40:
            test.assertGreater(all_asleep, 0, "both chains must sleep before the force")
            force[0, 2] = 5.0
        for scene in (captured, eager):
            scene.state_in.body_f.assign(force)
            for name in ("_propagation_tree_D_inv", "_propagation_body_com_rel", "_propagation_body_response"):
                getattr(scene.solver, name).fill_(np.nan)
            scene.solver._propagation_body_qd.fill_(np.nan)
        wp.capture_launch(capture.graph)
        eager.step(clear=False)
        a, b = captured.snapshot(), eager.snapshot()
        for name in _FIELDS:
            np.testing.assert_array_equal(a[name], b[name], err_msg=f"replay {k} {name}")
        active = eager.solver._dynamics_art_active.numpy()
        np.testing.assert_array_equal(captured.solver._dynamics_art_active.numpy(), active)
        awake = active * sleepable
        for scene in (captured, eager):
            factored = ~np.isnan(scene.solver._propagation_tree_D_inv.numpy()[:, 0, 0])
            np.testing.assert_array_equal(factored, awake[joint_art] == 1, err_msg=f"step {k}")
            # The response, twist refresh and impulse propagation kernels leave sleeping trees untouched.
            asleep = tree_bodies[awake[body_art[tree_bodies]] == 0]
            for name in ("_propagation_body_response", "_propagation_body_qd"):
                values = getattr(scene.solver, name).numpy()[asleep]
                test.assertTrue(np.isnan(values).all(), f"step {k} {name}")
        moving = tree_bodies[awake[body_art[tree_bodies]] == 1]
        for name in ("_propagation_body_response", "_propagation_body_qd"):
            test.assertTrue(np.isfinite(getattr(eager.solver, name).numpy()[moving]).all(), f"step {k} {name}")
        solved = ~np.isnan(captured.solver._propagation_body_com_rel.numpy()[tree_bodies]).any(axis=1)
        test.assertEqual(bool(solved.all()), bool(awake.any() or not gated or compliant), f"step {k}")
        test.assertTrue(bool(solved.all()) or not solved.any())
        if k == 40:
            test.assertEqual(awake[:2].tolist(), [1, 0], "the force must wake its chain only")
        all_asleep += int(not awake.any())


def test_colored_sleeping_runs_under_external_and_warp_capture(test, device):
    """Replay a colored sleeping step captured by Warp or directly through CUDA, and sleep and wake as eager steps do."""
    # The RUN_TO_RUN row ordering fails under a capture Warp does not track; this test covers the sleeping solve.
    options = {
        "sleep_quiet_time": 0.05,
        "friction_anchor_beta": 0.0,
        "deterministic": wp.DeterministicMode.NOT_GUARANTEED,
    }
    for mode in ("warp", "external"):
        with test.subTest(mode=mode):
            if mode == "external" and not sys.platform.startswith("linux"):
                test.skipTest("the CUDA driver capture loads libcuda.so.1")
            captured, eager = (_Heap(device, model=_dumbbells_model(device), **options) for _ in range(2))
            replay, release = _capture_step(captured, mode)
            try:
                force = np.zeros((captured.model.body_count, 6), dtype=np.float32)
                slept = False
                for k in range(60):
                    force[:] = 0.0
                    if k == 40:
                        test.assertTrue(slept, "both chains must sleep before the force")
                        force[0, 2] = 5.0
                    for scene in (captured, eager):
                        scene.state_in.body_f.assign(force)
                    replay()
                    eager.step(clear=False)
                    awake = captured.solver._sleeping.art_awake.numpy()
                    np.testing.assert_array_equal(
                        awake, eager.solver._sleeping.art_awake.numpy(), err_msg=f"{mode} {k}"
                    )
                    slept |= not awake.any()
                    if k == 40:
                        np.testing.assert_array_equal(awake, [1, 0])
                a, b = captured.snapshot(), eager.snapshot()
                for name in _FIELDS:
                    np.testing.assert_allclose(a[name], b[name], rtol=0.0, atol=1.0e-4, err_msg=f"{mode} {name}")
            finally:
                release()


def _capture_step(scene, mode):
    """Capture one solver step of ``scene`` with Warp or with the CUDA driver alone; return a replay function.

    Collision runs eagerly before each replay, so only the solver step is captured. Call the returned
    release function when done; the CUDA driver capture loads ``libcuda.so.1`` and runs only on Linux.
    """
    device = scene.model.device

    def solve():
        scene.solver.step(scene.state_in, scene.state_out, scene.control, scene.contacts, 1.0 / 240.0)
        for name in _FIELDS:
            wp.copy(getattr(scene.state_in, name), getattr(scene.state_out, name))

    scene.pipeline.collide(scene.state_in, scene.contacts)
    if mode == "warp":
        with wp.ScopedCapture(device) as capture:
            solve()

        def replay():
            scene.pipeline.collide(scene.state_in, scene.contacts)
            wp.capture_launch(capture.graph)

        return replay, lambda: None
    # Capture that Warp does not track, as another framework's graph capture would start.
    # The captured step allocates stream-ordered scratch, which the graph frees on each relaunch.
    launch, release = capture_with_cuda_driver(device, solve, auto_free_on_launch=True)

    def replay():
        scene.pipeline.collide(scene.state_in, scene.contacts)
        launch()

    return replay, release


class TestSteelSleepingColoredPropagation(unittest.TestCase):
    pass


for _name in (
    "test_colored_separated_boxes_sleep_and_wake_like_immediate",
    "test_colored_units_omit_exactly_the_sleeping_islands",
    "test_colored_propagation_overflow_vetoes_sleep",
    "test_colored_force_reset_and_notification_wake_islands",
    "test_colored_sleeping_rollout_is_independent_of_world_order",
    "test_colored_captured_sleep_and_wake_match_eager",
    "test_colored_separated_stacks_sleep_like_immediate",
    "test_colored_sleeping_skips_tree_and_solve_work",
    "test_colored_sleeping_runs_under_external_and_warp_capture",
    "test_colored_compliant_contacts_sleep_and_wake_like_immediate",
):
    add_function_test(TestSteelSleepingColoredPropagation, _name, globals()[_name], devices=devices)


class _Heap:
    """Colored-propagation solver on arm-on-heap worlds, stepped in place so steps can be captured."""

    def __init__(
        self,
        device,
        response="colored",
        *,
        model=None,
        worlds=4,
        heap=(5, 2, 2),
        compliance_stiffness=None,
        **solver_kwargs,
    ):
        self.model = model if model is not None else _arm_heap_model(device, worlds=worlds, heap=heap)
        args = {
            **_response(response),
            "deterministic": wp.DeterministicMode.RUN_TO_RUN,
            "iterations": 16,
            "contact_regularization": 0.01,
            "free_body_constraint_max": 1024,
            "articulation_constraint_max": 128,
            "enable_joint_limits": True,
            "row_watermark": True,
            "enable_sleeping": True,
        }
        args.update(solver_kwargs)
        self.solver = SolverSteel(self.model, **args)
        # Tests that compare joint wrenches set this to the solver's observables.
        self.observables = None
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=self.model.rigid_contact_max, deterministic=True
        )
        self.contacts = self.pipeline.contacts()
        self.compliance_stiffness = compliance_stiffness
        if compliance_stiffness is not None:
            for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
                setattr(self.contacts, name, wp.zeros(self.model.rigid_contact_max, dtype=float, device=device))
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.control = self.model.control()

    def step(self, steps=1, *, clear=True):
        for _ in range(steps):
            if clear:
                self.state_in.clear_forces()
            self.pipeline.collide(self.state_in, self.contacts)
            if self.compliance_stiffness is not None:
                self.contacts.rigid_contact_stiffness.fill_(self.compliance_stiffness)
                self.contacts.rigid_contact_damping.fill_(200.0)
                self.contacts.rigid_contact_friction.fill_(1.0)
            self.solver.step(
                self.state_in, self.state_out, self.control, self.contacts, 1.0 / 240.0, observables=self.observables
            )
            for name in _FIELDS:
                wp.copy(getattr(self.state_in, name), getattr(self.state_out, name))

    def snapshot(self):
        return {name: getattr(self.state_in, name).numpy().copy() for name in _FIELDS}

    def asleep_this_step(self):
        """Bodies of the islands the last step skipped."""
        nodes = self.solver._sleeping.body_nodes.numpy()
        skipped = self.solver._sleeping.step_asleep.numpy()
        return (nodes >= 0) & (skipped[np.maximum(nodes, 0)] == 1)

    def contact_bodies(self):
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        shape_body = self.model.shape_body.numpy()
        bodies = []
        for shapes in (self.contacts.rigid_contact_shape0, self.contacts.rigid_contact_shape1):
            shape = shapes.numpy()[:count]
            bodies.append(np.where(shape >= 0, shape_body[np.maximum(shape, 0)], -1))
        return bodies

    def units(self):
        """Contact index and bodies of every colored unit of the last step."""
        solver = self.solver
        capacity = solver._propagation_max_constraints
        cursor = np.minimum(solver._color_world_unit_cursor.numpy(), capacity)
        index = np.concatenate([w * capacity + np.arange(n) for w, n in enumerate(cursor)]).astype(int)
        return (
            solver._color_unit_contact.numpy()[index],
            solver._color_unit_body_a.numpy()[index],
            solver._color_unit_body_b.numpy()[index],
        )

    def propagation_row_bodies(self):
        counts = self.solver._propagation_constraint_count.numpy()
        rows = [
            np.concatenate(
                [table.numpy()[w, :n] for table in (self.solver._propagation_body_a, self.solver._propagation_body_b)]
            )
            for w, n in enumerate(counts)
        ]
        return np.concatenate(rows)


def _touches(bodies, mask):
    return (bodies >= 0) & mask[np.maximum(bodies, 0)]


def _arm_heap_model(device, worlds=4, heap=(5, 2, 2)):
    """Worlds of a fixed-base six-link arm drooping onto a box heap; spheres on links 0, 3 and 5."""
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
    scene.rigid_gap = 0.005
    for _ in range(worlds):
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        world.rigid_gap = 0.005
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        no_collision = world.default_shape_cfg.copy()
        no_collision.has_shape_collision = False
        no_collision.collision_group = 0
        prev, joints = -1, []
        for i in range(6):
            link = world.add_link()
            world.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
            if i in (0, 3, 5):
                world.add_shape_sphere(link, radius=0.08)
            parent_xform = wp.transform(
                wp.vec3(0.0, 0.0, 0.32) if prev == -1 else wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()
            )
            joints.append(
                world.add_joint_revolute(
                    parent=prev,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=parent_xform,
                    child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                )
            )
            prev = link
        world.add_articulation(joints)
        nx, ny, nz = heap
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    body = world.add_body(
                        xform=wp.transform(
                            wp.vec3(0.15 + 0.216 * ix, 0.083 * (iy - (ny - 1) / 2), 0.041 + 0.082 * iz),
                            wp.quat_rpy(0.0, 0.0, 0.05 * (ix + iy + iz)),
                        )
                    )
                    world.add_shape_box(body, hx=0.04, hy=0.04, hz=0.04)
        scene.add_world(world)
    scene.add_ground_plane()
    model = scene.finalize(device=device)
    model.rigid_contact_max = 4096 * worlds
    return model


if __name__ == "__main__":
    unittest.main()
