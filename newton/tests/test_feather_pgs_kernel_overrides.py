# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernel-selection test hooks of SolverFeatherPGS (``_kernel_overrides``)."""

import contextlib
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@contextlib.contextmanager
def _overrides(**overrides):
    SolverFeatherPGS._kernel_overrides = overrides
    try:
        yield
    finally:
        SolverFeatherPGS._kernel_overrides = {}


def _legged_base(device, worlds=2):
    """A floating base with four hinged legs dropped on the ground: articulated (dense) contact rows only."""
    template = newton.ModelBuilder()
    template.default_shape_cfg.mu = 0.6
    root = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_rpy(0.05, 0.02, 0.0)))
    template.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [template.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = template.add_link()
            template.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                template.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    template.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _trajectory(model, steps=90, **solver_kwargs):
    solver = SolverFeatherPGS(model, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd[0:6:2] = 0.4
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_contact_kernels_match_the_scalar_split_solve(test, device):
    """Converge the split solve's contact-only Gauss-Seidel kernels to the scalar loop's solution.

    They sweep whole contacts as 3x3 blocks, so they agree with the row-by-row loop at convergence.
    """
    model = _legged_base(device)
    options = {"pgs_mode": "split", "dense_max_constraints": 48, "pgs_iterations": 64}
    with _overrides(pgs_kernel="loop"):
        _, reference = _trajectory(model, steps=30, **options)
    for overrides in (
        {"pgs_kernel": "tiled_contact"},
        {"pgs_kernel": "streaming"},
        {"pgs_kernel": "streaming", "pgs_chunk_size": 4},
    ):
        with test.subTest(**overrides):
            with _overrides(**overrides):
                solver, joint_q = _trajectory(model, steps=30, **options)
            test.assertIsNotNone(solver._pgs_solve_contact_kernel)
            test.assertGreater(int(solver.constraint_count.numpy().max()), 0)
            np.testing.assert_allclose(joint_q, reference, rtol=0.0, atol=1.0e-5)


def test_contact_kernels_reject_non_contact_rows(test, device):
    """Reject the contact-only kernels with joint-limit rows and normal-only contact rows."""
    model = _legged_base(device)
    for kernel in ("tiled_contact", "streaming"):
        for options in ({"enable_joint_limits": True}, {"contact_friction_gap_threshold": 0.01}):
            with test.subTest(pgs_kernel=kernel, **options), _overrides(pgs_kernel=kernel):
                with test.assertRaisesRegex(ValueError, "contact rows only"):
                    SolverFeatherPGS(model, pgs_mode="split", **options)
    with _overrides(pgs_kernel="tiled_contact"), test.assertRaisesRegex(ValueError, "shared memory"):
        SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=768)


def test_matrix_free_ignores_the_dense_pgs_kernel(test, device):
    """Resolve the dense Gauss-Seidel kernel to the scalar loop in the matrix-free solve; keep friction patches."""
    model = _legged_base(device)
    for kernel in ("tiled_contact", "streaming", "tiled"):
        with test.subTest(pgs_kernel=kernel), _overrides(pgs_kernel=kernel), warnings.catch_warnings():
            warnings.simplefilter("error")
            solver = SolverFeatherPGS(model, enable_joint_limits=True)
            test.assertEqual(solver.pgs_kernel, "loop")
            test.assertTrue(solver._friction_anchors_enabled)
            test.assertAlmostEqual(solver.friction_anchor_beta, 0.2)


def test_launch_widths_step_identically(test, device, pgs_mode="matrix_free"):
    """Step identically with non-default tile and serial-kernel launch widths."""
    model = _legged_base(device)
    options = {"pgs_mode": pgs_mode, "dense_max_constraints": 48, "friction_anchor_beta": 0.0}
    with _overrides(sparse_mass_matrix=False, cholesky_kernel="tiled", trisolve_kernel="tiled", hinv_jt_kernel="tiled"):
        default, reference = _trajectory(model, **options)
    with _overrides(
        cholesky_kernel="tiled",
        trisolve_kernel="tiled",
        hinv_jt_kernel="tiled",
        tile_threads=128,
        serial_kernel_block_dim=64,
    ):
        solver, joint_q = _trajectory(model, **options)
    test.assertEqual((default._tile_threads, default._serial_kernel_block_dim), (64, 256))
    test.assertEqual((solver._tile_threads, solver._serial_kernel_block_dim), (128, 64))
    size = solver.size_groups[0]
    test.assertIsNot(solver._cholesky_kernels_by_size[size], default._cholesky_kernels_by_size[size])
    np.testing.assert_allclose(joint_q, reference, rtol=0.0, atol=1.0e-4)


def test_launch_width_overrides_are_validated(test, device):
    """Reject invalid launch widths and chunk sizes."""
    model = _legged_base(device, worlds=1)
    for key, value in (
        ("tile_threads", 48),
        ("serial_kernel_block_dim", 48),
        ("serial_kernel_block_dim", 0),
        ("pgs_chunk_size", 0),
        ("pgs_kernel", "bad"),
    ):
        with test.subTest(**{key: value}), _overrides(**{key: value}):
            with test.assertRaisesRegex(ValueError, key):
                SolverFeatherPGS(model, pgs_mode="split")


class TestFeatherPGSKernelOverrides(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (
    test_contact_kernels_match_the_scalar_split_solve,
    test_contact_kernels_reject_non_contact_rows,
    test_matrix_free_ignores_the_dense_pgs_kernel,
    test_launch_widths_step_identically,
):
    add_function_test(TestFeatherPGSKernelOverrides, _fn.__name__, _fn, devices=cuda_devices)
add_function_test(
    TestFeatherPGSKernelOverrides,
    "test_launch_widths_step_identically_split",
    test_launch_widths_step_identically,
    devices=cuda_devices,
    pgs_mode="split",
)
add_function_test(
    TestFeatherPGSKernelOverrides,
    "test_launch_width_overrides_are_validated",
    test_launch_width_overrides_are_validated,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
