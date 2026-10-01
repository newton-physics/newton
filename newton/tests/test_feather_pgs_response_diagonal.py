# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tiled ``H^-1 J^T`` response and row diagonal of SolverFeatherPGS against a dense reference."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import accumulate_group_diag_worlds
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_hinv_jt_kernel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_tiled_response_diagonal_matches_dense_reference(test, device):
    """Match the tiled world response and accumulated diagonal to a dense solve, with two articulations per world."""
    device = wp.get_device(device)
    rng = np.random.default_rng(41)
    num_dofs = 23
    max_constraints = 16
    num_articulations = 4
    world_constraint_count_np = np.array((13, 7), dtype=np.int32)
    articulation_world_np = np.array((0, 0, 1, 1), dtype=np.int32)
    articulation_world_dof_offset_np = np.array((0, num_dofs, 0, num_dofs), dtype=np.int32)

    factors = rng.normal(size=(num_articulations, num_dofs, num_dofs)).astype(np.float32)
    mass = factors @ np.transpose(factors, (0, 2, 1))
    mass += 2.0 * np.eye(num_dofs, dtype=np.float32)[None, :, :]
    cholesky = wp.array(np.linalg.cholesky(mass).astype(np.float32), device=device)

    jacobian_np = np.zeros((num_articulations, max_constraints, num_dofs), dtype=np.float32)
    for articulation, world in enumerate(articulation_world_np):
        count = int(world_constraint_count_np[world])
        jacobian_np[articulation, :count] = rng.normal(size=(count, num_dofs))
    jacobian = wp.array(jacobian_np, device=device)

    group_to_art = wp.array(np.arange(num_articulations, dtype=np.int32), device=device)
    art_to_world = wp.array(articulation_world_np, device=device)
    articulation_world_dof_offset = wp.array(articulation_world_dof_offset_np, device=device)
    constraint_count = wp.array(world_constraint_count_np, device=device)
    response = wp.zeros_like(jacobian)
    world_jacobian = wp.zeros((2, max_constraints, 2 * num_dofs), dtype=wp.float32, device=device)
    world_response = wp.zeros_like(world_jacobian)
    group_diag = wp.zeros((num_articulations, max_constraints), dtype=wp.float32, device=device)

    kernel = _get_hinv_jt_kernel(
        num_dofs,
        max_constraints,
        str(device.arch),
        constraint_chunk_size=8,
        write_world=True,
        write_group=False,
        compute_diag=True,
    )
    wp.launch_tiled(
        kernel,
        dim=(num_articulations, 2),
        inputs=[
            cholesky,
            jacobian,
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            constraint_count,
        ],
        outputs=[response, world_jacobian, world_response, group_diag],
        block_dim=64,
        device=device,
    )

    world_diag = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
    wp.launch(
        accumulate_group_diag_worlds,
        dim=2 * max_constraints,
        inputs=[
            group_diag,
            wp.array((0, 2, 4), dtype=wp.int32, device=device),
            group_to_art,
            group_to_art,
            constraint_count,
            max_constraints,
        ],
        outputs=[world_diag],
        device=device,
    )
    wp.synchronize_device(device)

    response_ref = np.zeros_like(jacobian_np)
    group_diag_ref = np.zeros((num_articulations, max_constraints), dtype=np.float32)
    world_jacobian_ref = np.zeros((2, max_constraints, 2 * num_dofs), dtype=np.float32)
    world_response_ref = np.zeros_like(world_jacobian_ref)
    for articulation, world in enumerate(articulation_world_np):
        count = int(world_constraint_count_np[world])
        response_ref[articulation, :count] = np.linalg.solve(mass[articulation], jacobian_np[articulation, :count].T).T
        group_diag_ref[articulation, :count] = np.sum(
            jacobian_np[articulation, :count] * response_ref[articulation, :count], axis=1
        )
        dof_offset = articulation_world_dof_offset_np[articulation]
        world_jacobian_ref[world, :count, dof_offset : dof_offset + num_dofs] = jacobian_np[articulation, :count]
        world_response_ref[world, :count, dof_offset : dof_offset + num_dofs] = response_ref[articulation, :count]
    world_diag_ref = np.stack((group_diag_ref[:2].sum(axis=0), group_diag_ref[2:].sum(axis=0)))

    np.testing.assert_array_equal(response.numpy(), np.zeros_like(response_ref))
    np.testing.assert_allclose(world_jacobian.numpy(), world_jacobian_ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(world_response.numpy(), world_response_ref, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(group_diag.numpy(), group_diag_ref, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(world_diag.numpy(), world_diag_ref, rtol=2.0e-5, atol=2.0e-6)


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSResponseDiagonal,
    "test_tiled_response_diagonal_matches_dense_reference",
    test_tiled_response_diagonal_matches_dense_reference,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
