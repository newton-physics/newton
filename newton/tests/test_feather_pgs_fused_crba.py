# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fused mass-matrix assembly and factorization of SolverFeatherPGS."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _tree(device, links, *, branched=False, worlds=2, loop=False):
    """Driven revolute chains (or a two-branch tree) hanging above the ground."""
    template = newton.ModelBuilder()
    joints = []
    bodies = []
    for i in range(links):
        parent = -1 if i == 0 else bodies[(i - 1) // 2 if branched else i - 1]
        link = template.add_link(mass=1.0 + 0.1 * i, inertia=wp.mat33(np.eye(3) * (0.01 + 0.002 * i)))
        template.add_shape_capsule(link, radius=0.03, half_height=0.08)
        side = 1.0 if i % 2 else -1.0
        offset = wp.vec3(0.0, 0.0, 1.5) if parent < 0 else wp.vec3(0.05 * side if branched else 0.0, 0.0, -0.2)
        joints.append(
            template.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0) if i % 3 else wp.vec3(1.0, 0.0, 0.0),
                parent_xform=wp.transform(offset, wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                target_ke=20.0 + i,
                target_kd=1.0,
                target_pos=0.2,
                armature=0.01,
            )
        )
        bodies.append(link)
    template.add_articulation(joints)
    if loop:
        template.add_joint_ball(
            bodies[0], bodies[-1], child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity())
        )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


def _trajectory(model, steps=40, **solver_kwargs):
    solver = SolverFeatherPGS(model, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    qd = np.linspace(-0.5, 0.5, model.joint_dof_count).astype(np.float32)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy(), state_0.joint_qd.numpy()


def test_fused_assembly_matches_the_separate_pipeline(test, device, pgs_mode="matrix_free"):
    """Match separate assembly and factorization with the warp-per-articulation and tiled kernels."""
    for links, branched, kernel in ((3, False, "warp"), (6, True, "warp"), (14, False, "tiled"), (15, True, "tiled")):
        with test.subTest(links=links, branched=branched):
            model = _tree(device, links, branched=branched)
            fused, q, qd = _trajectory(model, pgs_mode=pgs_mode, use_parallel_streams=True)
            reference, q_ref, qd_ref = _trajectory(model, pgs_mode=pgs_mode)
            size = fused.size_groups[0]
            selected = {
                "warp": fused._crba_cholesky_warp_kernels_by_size[size],
                "tiled": fused._crba_cholesky_kernels_by_size[size],
            }
            test.assertIsNotNone(selected[kernel])
            test.assertIsNone(reference._crba_cholesky_warp_kernels_by_size[size])
            test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
            np.testing.assert_allclose(fused.L_by_size[size].numpy(), reference.L_by_size[size].numpy(), atol=2.0e-5)
            np.testing.assert_allclose(q, q_ref, rtol=0.0, atol=2.0e-5)
            np.testing.assert_allclose(qd, qd_ref, rtol=0.0, atol=2.0e-4)


def test_fused_assembly_covers_articulations_wider_than_the_block(test, device):
    """Match separate assembly when the tiled block has fewer threads than the articulation has DOFs."""
    for links in (33, 64):
        model = _tree(device, links, worlds=1)
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"sparse_mass_matrix": False}):
            reference, q_ref, qd_ref = _trajectory(model, steps=2)
        size = reference.size_groups[0]
        test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
        for tile_threads in (32, 64):
            with test.subTest(links=links, tile_threads=tile_threads):
                overrides = {"tile_threads": tile_threads, "sparse_mass_matrix": False}
                with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", overrides):
                    fused, q, qd = _trajectory(model, steps=2, use_parallel_streams=True)
                test.assertIsNotNone(fused._crba_cholesky_kernels_by_size[size])
                np.testing.assert_allclose(
                    fused.L_by_size[size].numpy(), reference.L_by_size[size].numpy(), atol=2.0e-5
                )
                np.testing.assert_allclose(q, q_ref, rtol=1.0e-5, atol=2.0e-5)
                np.testing.assert_allclose(qd, qd_ref, rtol=1.0e-5, atol=2.0e-4)


def test_fused_assembly_follows_the_mass_update_interval(test, device):
    """Refresh the fused factors on the interval and after an inertia notification only."""
    model = _tree(device, 3)
    results = []
    for streams in (True, False):
        solver, q, _ = _trajectory(model, steps=7, use_parallel_streams=streams, update_mass_matrix_interval=3)
        body_mass = model.body_mass.numpy()
        model.body_mass.assign(body_mass * 1.5)
        model.body_inv_mass.assign(1.0 / (body_mass * 1.5))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        model.body_mass.assign(body_mass)
        model.body_inv_mass.assign(1.0 / body_mass)
        results.append((q, solver.L_by_size[solver.size_groups[0]].numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0.0, atol=2.0e-5)


def test_fused_assembly_is_selected_only_where_supported(test, device):
    """Keep the separate pipeline without streams, with loop joints, PGS drive rows or on CPU."""
    is_cuda = wp.get_device(device).is_cuda
    pgs_mode = "matrix_free" if is_cuda else "split"
    solver = SolverFeatherPGS(_tree(device, 3), pgs_mode=pgs_mode, use_parallel_streams=True)
    test.assertEqual(bool(solver._crba_source_dof_by_size), is_cuda)
    if not is_cuda:
        return
    for model, options in (
        (_tree(device, 3), {}),
        (_tree(device, 3), {"use_parallel_streams": True, "drive_mode": "physx_pgs"}),
        (_tree(device, 3, loop=True), {"use_parallel_streams": True}),
    ):
        with test.subTest(**options):
            solver = SolverFeatherPGS(model, **options)
            test.assertFalse(solver._crba_source_dof_by_size)
            test.assertIsNone(solver._crba_cholesky_warp_kernels_by_size[solver.size_groups[0]])


class TestFeatherPGSFusedCrba(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_matches_the_separate_pipeline",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_matches_the_separate_pipeline_split",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
    pgs_mode="split",
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_covers_articulations_wider_than_the_block",
    test_fused_assembly_covers_articulations_wider_than_the_block,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_follows_the_mass_update_interval",
    test_fused_assembly_follows_the_mass_update_interval,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_is_selected_only_where_supported",
    test_fused_assembly_is_selected_only_where_supported,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
