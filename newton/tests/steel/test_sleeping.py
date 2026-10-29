# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental passive-island sleeping of SolverSteel."""

import functools
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import PROFILE, _advance, _articulations
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
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
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
            SolverSteel(model, solve_mode="matrix_free", enable_sleeping=True, friction_anchor_beta=0.0, **kwargs)
    # The same options are valid without sleeping, so the errors above come from the sleep guard.
    SolverSteel(model, solve_mode="matrix_free", velocity_iterations=1, friction_anchor_beta=0.0)
    SolverSteel(model, solve_mode="matrix_free", contact_warmstart=True, friction_anchor_beta=0.0)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    empty = builder.finalize(device=device)
    SolverSteel(empty, solve_mode="matrix_free", friction_anchor_beta=0.0)
    with test.assertRaisesRegex(ValueError, "at least one articulation"):
        SolverSteel(empty, solve_mode="matrix_free", enable_sleeping=True, friction_anchor_beta=0.0)


def test_default_and_explicit_disabled_match(test, device):
    """Keep the trajectory bitwise unchanged with sleeping explicitly disabled."""
    model, pipeline, explicit, states, control = _scene(device, enabled=False)
    default = SolverSteel(model, solve_mode="matrix_free", iterations=16, friction_anchor_beta=0.0)
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


def _parent_wrench_scene(device):
    """Two free boxes resting on the ground, each carrying a smaller box welded on top."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        base = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
        top = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.05, hy=0.05, hz=0.05)
        free = builder.add_joint_free(base)
        weld = builder.add_joint_fixed(
            base, top, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())
        )
        builder.add_articulation([free, weld])
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


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
        solve_mode="matrix_free",
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    reference = SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.0, articulation_constraint_max=128)
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
        solve_mode="matrix_free",
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
        solve_mode="matrix_free",
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    reference = SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.0, articulation_constraint_max=128)
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
        solve_mode="matrix_free",
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
        solve_mode="matrix_free",
        drive_mode="rows",
        iterations=16,
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
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        contact_compliance=True,
        friction_anchor_beta=0.0,
        iterations=8,
        baumgarte=0.05,
        articulation_constraint_max=32,
        free_body_constraint_max=32,
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
                solve_mode="matrix_free",
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
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **PROFILE)
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
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **PROFILE)
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
            solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **PROFILE)
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
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **PROFILE)
    return model, pipeline, solver, [model.state(), model.state()]


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
    "test_property_notification_wakes_only_changed_islands",
    "test_frozen_patch_carry_matches_rebuild",
    "test_frozen_patch_carry_matches_rebuild_for_flooded_pairs",
    "test_frozen_patch_carry_copies_the_previous_history",
    "test_successive_notifications_keep_every_wake",
    "test_replaced_property_array_wakes_its_island",
    "test_resized_property_array_is_rejected",
    "test_coincident_entity_counts_keep_property_owners",
    "test_solver_body_attribute_notification_wakes_its_island",
    "test_mimic_articulation_stays_awake",
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
        solve_mode="matrix_free",
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
                solve_mode="matrix_free",
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
            solve_mode="matrix_free",
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


if __name__ == "__main__":
    unittest.main()
