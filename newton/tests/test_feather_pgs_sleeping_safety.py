# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adversarial boundaries of experimental SolverFeatherPGS sleeping."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


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
            body_art = solver.body_to_articulation.numpy()
            if fixed_articulation:
                test.assertEqual(int(solver._model_plan.response_dof_count[body_art[table]]), 0)

            solver.sleeping.begin(state_in, control, contacts)

            roots = solver.sleeping.parent.numpy()[body_art[boxes]]
            test.assertNotEqual(int(roots[0]), int(roots[1]), f"Table joined box islands: {roots}")
            np.testing.assert_array_equal(solver.sleeping.root_supported.numpy()[roots], [1, 1])
            np.testing.assert_array_equal(solver.sleeping.root_veto.numpy()[roots], [0, 0])


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
    test.assertLess(kick, solver.sleeping.linear_threshold)
    before = state_in.body_q.numpy()[1, 2]

    solver.step(state_in, state_out, control, contacts, dt)

    test.assertEqual(int(solver.sleeping.body_awake.numpy()[1]), 1)
    test.assertLess(state_out.body_q.numpy()[1, 2], before)
    test.assertLess(state_out.body_qd.numpy()[1, 2], 0.0)
    test.assertLess(abs(state_out.body_qd.numpy()[1, 2]), solver.sleeping.linear_threshold)


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

    solver.sleeping.begin(state_in, control, contacts)

    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), np.zeros(7))
    _seed_sleep(solver, state_in, [0, 0, 2, 2, 4, 4, 6])
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 1.0
    state_in.body_f.assign(force)

    solver.sleeping.begin(state_in, control, contacts)

    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(solver.sleeping.quiet_age.numpy()[:6], np.zeros(6))


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
            solver.sleeping.skip_dynamics = False
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
            np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1])


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
    """Sleep with requested acceleration and joint-wrench outputs, which FeatherPGS leaves untouched."""
    model = _boxes(1, device)
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
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0])


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

    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1])
    np.testing.assert_array_equal(solver.sleeping.quiet_age.numpy(), [0.0])
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
            np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
            np.testing.assert_array_equal(solver.sleeping.quiet_age.numpy(), [0.0, 0.0])
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
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        enable_sleeping=True,
        friction_anchor_beta=0.0,
        sleep_quiet_time=0.05,
        pgs_iterations=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    state_in.clear_forces()
    return pipeline, solver, state_in, state_out, model.control()


def _seed_sleep(solver, state, previous_roots):
    sleeping = solver.sleeping
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


devices = get_cuda_test_devices()


class TestFeatherPGSSleepingSafety(unittest.TestCase):
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
    add_function_test(TestFeatherPGSSleepingSafety, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
