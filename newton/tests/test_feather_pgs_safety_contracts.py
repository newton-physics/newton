# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Selected-world reset, per-world dense row storage and the contact identity consumed by warm starting."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _driven_model(device, worlds=2):
    """One driven slider per world plus one global (world -1) free body, without gravity."""
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_prismatic(
        -1, body, axis=newton.Axis.X, target_ke=10000.0, target_kd=0.0, armature=0.0, damping=0.0
    )
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for _ in range(worlds):
        builder.add_world(template)
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    return builder.finalize(device=device)


def test_partial_reset_refreshes_only_selected_mass_factors(test, device, pgs_mode="matrix_free"):
    """Refresh only the selected world's (or the global slot's) factors, eagerly and under graph replay."""
    model = _driven_model(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
    state, output = model.state(), model.state()
    control = model.control()
    solver.step(state, output, control, None, 0.01)

    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, mask)
    solver.step(state, output, control, None, 0.01)
    np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [1, 0, 0])
    if not wp.get_device(device).is_cuda:
        return

    first_mask = wp.empty_like(solver.mass_update_mask)
    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, mask)
        solver.step(state, output, control, None, 0.01)
        wp.copy(first_mask, solver.mass_update_mask)
        solver.step(state, output, control, None, 0.01)

    for selected, expected in (([False, True, False], [0, 1, 0]), ([False, False, True], [0, 0, 1])):
        with test.subTest(mask=selected):
            mask.assign(np.array(selected))
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(first_mask.numpy(), expected)
            # The request is consumed by the first step; the second reuses every factor.
            np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [0, 0, 0])


def test_partial_warp_capacity_preserves_world_boundaries(test, device, pgs_kernel="tiled"):
    """Keep the dense split-mode solve's loads and stores inside each world's row storage.

    Capacities that are not a multiple of the warp width leave partial warps in the native
    kernel; an inactive world must stay untouched.
    """
    for capacity in (33, 35, 40):
        with test.subTest(capacity=capacity):
            SolverFeatherPGS._kernel_overrides = {"pgs_kernel": pgs_kernel}
            try:
                solver = SolverFeatherPGS(
                    _driven_model(device, worlds=3), pgs_mode="split", dense_max_constraints=capacity
                )
            finally:
                SolverFeatherPGS._kernel_overrides = {}
            test.assertEqual(
                solver._pgs_solve_tiled_row_kernel is not None, pgs_kernel == "tiled" and wp.get_device(device).is_cuda
            )
            initial = np.zeros((3, capacity), dtype=np.float32)
            initial[2] = 1234.0
            solver.constraint_count.assign(np.array([capacity, 1, 0], dtype=np.int32))
            solver.impulses.assign(initial)
            solver.diag.fill_(2.0)
            solver.C.assign(np.tile(2.0 * np.eye(capacity, dtype=np.float32), (3, 1, 1)))
            solver.rhs.fill_(-1.0)
            solver.row_type.zero_()
            solver.row_parent.zero_()
            solver.row_mu.zero_()
            solver._dense_pgs_solve(1)
            expected = initial.copy()
            expected[0] = 0.5
            expected[1, 0] = 0.5
            np.testing.assert_array_equal(solver.impulses.numpy(), expected)


def _cylinder_foot(builder, pos, num_cyl=7, radius=0.02, half_height=0.015):
    """Add one free body whose collision is ``num_cyl`` small cylinders in a row."""
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=1.0)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), radius=radius, half_height=half_height
        )
    return body


def test_unmatched_pipeline_clears_stale_contact_identity(test, device):
    """Cold-start a matched buffer when its current producer disables matching.

    Warm starting reads ``rigid_contact_match_index``; a stale index left by an earlier
    matching producer would seed impulses into unrelated contacts.
    """
    builder = newton.ModelBuilder()
    _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    matched = newton.CollisionPipeline(model, contact_matching="latest")
    unmatched = newton.CollisionPipeline(model, deterministic=True)
    state, contacts = model.state(), matched.contacts()
    matched.collide(state, contacts)
    matched.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertTrue(np.any(contacts.rigid_contact_match_index.numpy()[:count] >= 0))
    unmatched.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] == -1))


class TestFeatherPGSSafety(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSSafety,
    "test_partial_reset_refreshes_only_selected_mass_factors",
    test_partial_reset_refreshes_only_selected_mass_factors,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSSafety,
    "test_partial_reset_refreshes_only_selected_mass_factors_split",
    test_partial_reset_refreshes_only_selected_mass_factors,
    devices=get_test_devices(),
    pgs_mode="split",
)
add_function_test(
    TestFeatherPGSSafety,
    "test_partial_warp_capacity_preserves_world_boundaries",
    test_partial_warp_capacity_preserves_world_boundaries,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSSafety,
    "test_partial_warp_capacity_preserves_world_boundaries_scalar",
    test_partial_warp_capacity_preserves_world_boundaries,
    devices=get_test_devices(),
    pgs_kernel="loop",
)
add_function_test(
    TestFeatherPGSSafety,
    "test_unmatched_pipeline_clears_stale_contact_identity",
    test_unmatched_pipeline_clears_stale_contact_identity,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
