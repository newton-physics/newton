# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping of SolverFeatherPGS with device-prepared contact torsion.

These tests need the experimental contact torsion options and are skipped by a solver without them.
"""

import inspect
import unittest

import numpy as np
import warp as wp

from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_sleeping_production import PROFILE, _advance, _articulations
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

TORSION_PROFILE = {**PROFILE, "contact_torsion_radius": 0.01, "contact_torsion_device": True}


def _torsion_available():
    return "contact_torsion_radius" in inspect.signature(SolverFeatherPGS.__init__).parameters


def _torsion_rows(solver):
    from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_TORSION  # noqa: PLC0415

    row_types = solver.row_type.numpy()[0, : int(solver.constraint_count.numpy()[0])]
    return int(np.count_nonzero(row_types == PGS_CONSTRAINT_TYPE_TORSION))


def test_anchored_torsion_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore anchored torsion rows on a force wake."""
    if not _torsion_available():
        test.skipTest("this solver has no contact torsion")
    model, pipeline, solver, states, control = _articulations(device, profile=TORSION_PROFILE)
    _advance(pipeline, solver, states, control, 5)
    awake_rows = int(solver.constraint_count.numpy()[0])
    awake_torsion = _torsion_rows(solver)
    test.assertGreater(awake_torsion, 0)
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
    np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), [0])
    frozen = states[0].body_q.numpy().copy()
    _advance(pipeline, solver, states, control, 20)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

    # Pushing the first articulation restores its rows and leaves the second asleep.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 20.0
    for _ in range(5):
        states[0].body_f.assign(force)
        _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1, 0, 0])
    test.assertEqual(int(solver.constraint_count.numpy()[0]), awake_rows // 2)
    test.assertEqual(_torsion_rows(solver), awake_torsion // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_validates_torsion(test, device):
    """Sleep and wake inside a captured graph with mandatory torsion validation."""
    if not _torsion_available():
        test.skipTest("this solver has no contact torsion")
    model, pipeline, solver, states, control = _articulations(device, profile=TORSION_PROFILE)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 2, contacts=contacts)
    solver.prepare_contact_torsion_capture(states[0], states[1])
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    solver.validate_contact_torsion()
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    solver.sleeping.wake()
    wp.capture_launch(capture.graph)
    solver.validate_contact_torsion()
    test.assertGreater(_torsion_rows(solver), 0)


devices = get_cuda_test_devices()


class TestFeatherPGSSleepingTorsion(unittest.TestCase):
    pass


for _name in ("test_anchored_torsion_articulations_sleep_and_wake", "test_graph_replay_validates_torsion"):
    add_function_test(TestFeatherPGSSleepingTorsion, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
