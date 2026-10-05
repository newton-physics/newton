# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint-capacity status of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _box_on_ground(device, worlds=1):
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_capacity_failure_is_observable(test, device, pgs_mode="matrix_free"):
    """Flag dropped free-body contact rows until reset, without optional telemetry."""
    model = _box_on_ground(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, mf_max_constraints=1, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_constraint_capacity()
    test.assertTrue(solver.constraint_overflow.numpy()[0])
    test.assertGreater(int(solver._row_dropped_mf.numpy()[0]), 0)
    solver.reset(state_out)
    solver.check_constraint_capacity()


def test_dense_capacity_failure_is_world_local(test, device, pgs_mode="matrix_free"):
    """Flag only the world whose dense rows overflow."""
    template = newton.ModelBuilder()
    link = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(link, hx=0.3, hy=0.3, hz=0.1)
    joint = template.add_joint_prismatic(-1, link, axis=newton.Axis.Z)
    template.add_articulation([joint])
    # The second world's slider starts high above the ground: no contacts there.
    lifted = newton.ModelBuilder()
    lifted_link = lifted.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
    lifted.add_shape_box(lifted_link, hx=0.3, hy=0.3, hz=0.1)
    lifted.add_articulation([lifted.add_joint_prismatic(-1, lifted_link, axis=newton.Axis.Z)])
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(lifted)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, dense_max_constraints=3, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, False, False])
    test.assertLessEqual(int(solver.constraint_count.numpy()[0]), 3)


def test_overflow_warning_is_printed_once(test, device, pgs_mode="matrix_free", response="immediate"):
    """Print one device-side warning per overflowing row family."""
    model = _box_on_ground(device, worlds=2)
    if response == "propagation":
        # The propagation rows hold every contact, in mf_max_constraints + dense_max_constraints rows.
        solver = SolverFeatherPGS(
            model, mf_max_constraints=1, dense_max_constraints=1, articulated_contact_response=response
        )
    else:
        solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, mf_max_constraints=1)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    expected = [0, 0, 0, 1] if response == "propagation" else [0, 1, 0, 0]
    np.testing.assert_array_equal(solver._row_overflow_warning_emitted.numpy(), expected)


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSCapacityStatus,
    "test_capacity_failure_is_observable",
    test_capacity_failure_is_observable,
    devices=devices,
)
add_function_test(
    TestFeatherPGSCapacityStatus,
    "test_dense_capacity_failure_is_world_local",
    test_dense_capacity_failure_is_world_local,
    devices=devices,
)
add_function_test(
    TestFeatherPGSCapacityStatus,
    "test_overflow_warning_is_printed_once",
    test_overflow_warning_is_printed_once,
    devices=devices,
    check_output=False,
)
for _name in (
    "test_capacity_failure_is_observable",
    "test_dense_capacity_failure_is_world_local",
    "test_overflow_warning_is_printed_once",
):
    add_function_test(
        TestFeatherPGSCapacityStatus,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        check_output=_name != "test_overflow_warning_is_printed_once",
        pgs_mode="split",
    )
add_function_test(
    TestFeatherPGSCapacityStatus,
    "test_overflow_warning_is_printed_once_propagation",
    test_overflow_warning_is_printed_once,
    devices=devices,
    check_output=False,
    response="propagation",
)


if __name__ == "__main__":
    unittest.main()
