# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise experimental FPGS passive-island state freezing."""

import unittest

import numpy as np
import warp as wp

import newton


class TestSleeping(unittest.TestCase):
    def test_disabled_allocates_no_state(self):
        """Keep sleep state absent on the default solver path."""
        model, pipeline, solver, states, control = _scene(enabled=False)
        self.assertIsNone(solver.sleeping)

    def test_separated_boxes_sleep_and_force_wakes_one(self):
        """Sleep independent boxes and wake only the forced island."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        self.assertNotEqual(*solver.sleeping.body_island.numpy())
        before = states[0].body_q.numpy().copy()
        _advance(model, pipeline, solver, states, control, 10)
        np.testing.assert_array_equal(states[0].body_q.numpy(), before)
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = 100
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
        self.assertGreater(states[0].body_qd.numpy()[0, 0], 0)

    def test_nonfinite_force_wakes_and_propagates(self):
        """A NaN external force on a sleeping body wakes its island instead of publishing a frozen state."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = np.nan
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
        self.assertFalse(np.all(np.isfinite(states[0].body_qd.numpy()[0])))

    def test_reset_and_gravity_change_wake(self):
        """Wake state after reset and model-property notifications."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        solver.reset(states[0])
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
        _advance(model, pipeline, solver, states, control, 120)
        model.set_gravity((0, 0, 9.81))
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        _advance(model, pipeline, solver, states, control, 1)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
        self.assertGreater(states[0].body_qd.numpy()[0, 2], 0)

    def test_free_flight_cannot_sleep(self):
        """Keep unsupported free bodies awake at zero speed."""
        model, pipeline, solver, states, control = _scene(ground=False)
        model.set_gravity((0, 0, 0))
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])

    def test_invalid_options(self):
        """Reject unsupported sleep settings explicitly."""
        model, *_ = _scene(enabled=False)
        for kwargs in (
            {"sleep_quiet_time": 0},
            {"sleep_linear_threshold": float("nan")},
            {"pgs_velocity_iterations": 1},
            {"pgs_warmstart": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, enable_sleeping=True, friction_anchor_beta=0, **kwargs)

    def test_default_and_explicit_disabled_match(self):
        """Preserve the existing trajectory with sleeping explicitly disabled."""
        model, pipeline, explicit, states, control = _scene(enabled=False)
        default = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, pgs_mode="split", friction_anchor_beta=0)
        reference = [model.state(), model.state()]
        _advance(model, pipeline, explicit, states, control, 30)
        _advance(model, pipeline, default, reference, control, 30)
        np.testing.assert_array_equal(states[0].joint_q.numpy(), reference[0].joint_q.numpy())
        np.testing.assert_array_equal(states[0].joint_qd.numpy(), reference[0].joint_qd.numpy())

    def test_stack_wakes_together(self):
        """Wake the whole contacting stack when its top box is forced."""
        model, pipeline, solver, states, control = _scene(stack=True)
        _advance(model, pipeline, solver, states, control, 200)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        islands = solver.sleeping.body_island.numpy()
        self.assertEqual(islands[0], islands[1])
        force = np.zeros((2, 6), dtype=np.float32)
        force[1, 0] = 100
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])

    def test_cuda_graph_sleep_and_wake(self):
        """Advance sleep timers and force waking on captured graph replay."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        model, pipeline, solver, states, control = _scene(device="cuda:0")
        contacts = pipeline.contacts()
        for _ in range(2):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.seed_double_buffer_events()
            for _ in range(2):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
        for _ in range(100):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[0, 0] = 100
        states[0].body_f.assign(forces)
        states[1].body_f.assign(forces)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])


def _scene(*, enabled=True, ground=True, device="cpu", stack=False):
    builder = newton.ModelBuilder()
    if ground:
        builder.add_ground_plane()
    for index, x in enumerate((0.0, 0.5)):
        position = wp.vec3(0, 0, 0.1 + index * 0.2) if stack else wp.vec3(x, 0, 0.1)
        body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        enable_sleeping=enabled,
        sleep_quiet_time=0.05,
        pgs_iterations=16,
        friction_anchor_beta=0,
        pgs_mode="matrix_free" if model.device.is_cuda else "split",
    )
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance(model, pipeline, solver, states, control, steps, *, clear=True):
    contacts = pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


if __name__ == "__main__":
    unittest.main()
