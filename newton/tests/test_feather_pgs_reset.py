# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reset-mask contract of SolverFeatherPGS: ``(world_count + 1,)`` masks with a global slot."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _build_two_worlds_with_global_body(device):
    """Two worlds with one driven slider each, plus one global (world -1) free body."""
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_prismatic(-1, body, axis=newton.Axis.X, target_ke=100.0, target_kd=1.0)
    template.add_articulation([joint])

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_world(template)
    builder.add_world(template)
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(model.articulation_world.numpy(), [0, 1, -1])
    return model


def _mask(values, device):
    return wp.array(values, dtype=wp.bool, device=device)


def test_reset_validates_mask_length(test, device, pgs_mode="matrix_free"):
    """Accept ``(world_count + 1,)`` masks and reject every other length, including the legacy ``(world_count,)``."""
    model = _build_two_worlds_with_global_body(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state = model.state()
    solver.impulses.fill_(17.0)
    before = solver.impulses.numpy().copy()

    # Two worlds: the valid mask has three entries, the last one selecting global entities.
    solver.reset(state, _mask([True, False, True], device))
    for length in (1, 2, 4):
        with test.subTest(length=length):
            with test.assertRaisesRegex(ValueError, r"model\.world_count \+ 1 \(3\)"):
                solver.reset(state, wp.ones(length, dtype=wp.bool, device=device))
    with test.assertRaisesRegex(TypeError, "dtype bool"):
        solver.reset(state, wp.ones(3, dtype=wp.int32, device=device))
    # The solver keeps no impulse history, so a reset never touches the impulse scratch.
    np.testing.assert_array_equal(solver.impulses.numpy(), before)


def test_reset_isolates_global_slot(test, device, pgs_mode="matrix_free"):
    """Select local worlds and global entities independently through the final mask entry."""
    model = _build_two_worlds_with_global_body(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
    state, output = model.state(), model.state()
    control = model.control()
    solver.step(state, output, control, None, 0.01)

    # The status has the mask's layout, so every entry is cleared only by its own mask entry.
    cases = (
        ([False, False, True], [0, 0, 1], [True, True, False]),
        ([True, False, False], [1, 0, 0], [False, True, True]),
        ([False, True, False], [0, 1, 0], [True, False, True]),
        (None, [1, 1, 1], [False, False, False]),
    )
    for mask, expected_refresh, expected_overflow in cases:
        with test.subTest(mask=mask):
            solver.constraint_overflow.fill_(True)
            solver.reset(state, None if mask is None else _mask(mask, device))
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), expected_overflow)
            solver.step(state, output, control, None, 0.01)
            np.testing.assert_array_equal(solver.mass_update_mask.numpy(), expected_refresh)
            # The request is consumed by the first step after the reset.
            solver.step(state, output, control, None, 0.01)
            np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [0, 0, 0])


class TestFeatherPGSReset(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSReset, "test_reset_validates_mask_length", test_reset_validates_mask_length, devices=devices
)
add_function_test(
    TestFeatherPGSReset, "test_reset_isolates_global_slot", test_reset_isolates_global_slot, devices=devices
)
for _name, _func in (
    ("test_reset_validates_mask_length", test_reset_validates_mask_length),
    ("test_reset_isolates_global_slot", test_reset_isolates_global_slot),
):
    add_function_test(TestFeatherPGSReset, f"{_name}_split", _func, devices=get_test_devices(), pgs_mode="split")


if __name__ == "__main__":
    unittest.main()
