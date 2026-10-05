# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import build_joint_limit_rows_for_size
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_joint_limit_warp_kernel


def _outputs(articulation_count: int, max_constraints: int, size: int, device):
    return (
        wp.zeros(articulation_count, dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints, size), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
    )


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel joint-limit assembly requires CUDA")
    def test_warp_builder_matches_scalar_row_order_and_values(self):
        device = wp.get_cuda_device()
        articulation_count = 257
        size = 6
        max_constraints = 2 * size
        dof_count = articulation_count * size
        rng = np.random.default_rng(42)

        articulation_start = wp.array(np.arange(articulation_count + 1), dtype=wp.int32, device=device)
        articulation_dof_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_type = wp.array(np.full(articulation_count, JointType.D6), dtype=wp.int32, device=device)
        joint_q_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_qd_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_dof_dim = wp.array(np.tile([3, 3], (articulation_count, 1)), dtype=wp.int32, device=device)
        lower = wp.array(np.full(dof_count, -1.0, dtype=np.float32), dtype=wp.float32, device=device)
        upper = wp.array(np.full(dof_count, 1.0, dtype=np.float32), dtype=wp.float32, device=device)
        joint_q = wp.array(rng.uniform(-1.2, 1.2, dof_count).astype(np.float32), dtype=wp.float32, device=device)
        art_to_world = wp.array(np.arange(articulation_count), dtype=wp.int32, device=device)
        group_to_art = wp.array(np.arange(articulation_count), dtype=wp.int32, device=device)
        scalar = _outputs(articulation_count, max_constraints, size, device)

        wp.launch(
            build_joint_limit_rows_for_size,
            dim=articulation_count,
            inputs=[
                articulation_start,
                articulation_dof_start,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_dof_dim,
                lower,
                upper,
                joint_q,
                0.25,
                art_to_world,
                group_to_art,
                max_constraints,
                0.2,
                1.0e-6,
            ],
            outputs=list(scalar),
            device=device,
        )

        parallel = _outputs(articulation_count, max_constraints, size, device)
        warps_per_block = 4
        kernel = _get_joint_limit_warp_kernel(size, device.arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[(articulation_count + warps_per_block - 1) // warps_per_block],
            inputs=[
                articulation_count,
                articulation_dof_start,
                art_to_world,
                group_to_art,
                wp.array(np.arange(dof_count), dtype=wp.int32, device=device),
                lower,
                upper,
                joint_q,
                0.25,
                max_constraints,
                0.2,
                1.0e-6,
            ],
            outputs=list(parallel),
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        for parallel_array, scalar_array in zip(parallel, scalar, strict=True):
            np.testing.assert_array_equal(parallel_array.numpy(), scalar_array.numpy())


if __name__ == "__main__":
    unittest.main()
