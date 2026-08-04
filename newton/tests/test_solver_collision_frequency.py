# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.solvers import SolverBase
from newton.tests.unittest_utils import add_function_test, get_test_devices

Frequency = SolverBase.CollisionFrequencyType


class _StubSolver(SolverBase):
    """Minimal owning solver: runs the rigid pass per the resolved slot type."""

    supports_collision_pipeline = True

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.collide_calls = 0

    def step(self, state_in, state_out, control, contacts, dt):
        contacts = self._resolve_step_contacts(contacts)
        if self.pipeline is not None and self._resolved_collision_frequency_type(0) != Frequency.NONE:
            self._run_rigid_collision(state_in)
            self.collide_calls += 1


def _build_model(device):
    builder = newton.ModelBuilder()
    # box half-extent 0.5 at z=0.45 -> penetrates the ground plane, guaranteeing contacts
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_frequency_toggle_drives_detection(test, device):
    """Verify toggling the rigid slot NONE <-> PRE_INIT controls per-step detection.

    The solver holds no cross-step counters; every-N detection is expressed by
    the caller changing the setting between steps.
    """
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = _StubSolver(model, pipeline=pipeline)
    state_a, state_b = model.state(), model.state()

    test.assertIsNotNone(solver.contacts)

    for i in range(6):
        solver.set_collision_frequency(
            collision_frequency_type=[Frequency.PRE_INIT if i % 3 == 0 else Frequency.NONE, Frequency.NONE]
        )
        solver.step(state_a, state_b, None, None, 1e-3)
    test.assertEqual(solver.collide_calls, 2)

    # AUTO resolves to PRE_INIT for the rigid slot of an owning solver.
    solver.set_collision_frequency(collision_frequency_type=[Frequency.AUTO, Frequency.AUTO])
    solver.step(state_a, state_b, None, None, 1e-3)
    test.assertEqual(solver.collide_calls, 3)
    test.assertGreater(int(solver.contacts.rigid_contact_count.numpy()[0]), 0)


def test_frequency_validation_and_ownership(test, device):
    """Verify constructor/setter validation and pipeline-ownership error paths."""
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    class _NonOwning(SolverBase):
        def step(self, state_in, state_out, control, contacts, dt):
            pass

    # Pipeline passed to a solver that does not opt in.
    with test.assertRaises(ValueError):
        _NonOwning(model, pipeline=pipeline)

    # Non-owning solver: contacts property is None, external contacts pass through.
    plain = _NonOwning(model)
    test.assertIsNone(plain.contacts)
    external = pipeline.contacts()
    test.assertIs(plain._resolve_step_contacts(external), external)

    # Active rigid slot without a pipeline.
    with test.assertRaises(ValueError):
        _NonOwning(model, collision_frequency_type=[Frequency.PRE_INIT, Frequency.NONE])

    # PRE_POST_INIT is meaningless for the rigid slot.
    with test.assertRaises(ValueError):
        _StubSolver(model, pipeline=pipeline, collision_frequency_type=[Frequency.PRE_POST_INIT, Frequency.NONE])

    # List shape and range validation.
    with test.assertRaises(ValueError):
        _StubSolver(model, pipeline=pipeline, collision_frequency=[1])
    with test.assertRaises(ValueError):
        _StubSolver(model, pipeline=pipeline, collision_frequency=[0, 1])
    with test.assertRaises(ValueError):
        _StubSolver(model, pipeline=pipeline, collision_frequency_type=[Frequency.PRE_INIT])

    solver = _StubSolver(model, pipeline=pipeline, collision_frequency=[1, 2])
    # Partial update keeps the other setting.
    solver.set_collision_frequency(collision_frequency_type=[Frequency.NONE, Frequency.ITERATIONS])
    test.assertEqual(solver.collision_frequency, [1, 2])
    test.assertEqual(solver.collision_frequency_type, [Frequency.NONE, Frequency.ITERATIONS])

    # With an owned pipeline, step() must receive contacts=None.
    with test.assertRaises(ValueError):
        solver.step(model.state(), model.state(), None, pipeline.contacts(), 1e-3)


devices = get_test_devices()


class TestSolverCollisionFrequency(unittest.TestCase):
    pass


add_function_test(
    TestSolverCollisionFrequency,
    "test_frequency_toggle_drives_detection",
    test_frequency_toggle_drives_detection,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_frequency_validation_and_ownership",
    test_frequency_validation_and_ownership,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main()
