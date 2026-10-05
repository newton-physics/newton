# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import compute_composite_inertia
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_composite_inertia_warp_kernel


class TestFeatherPGSCompositeInertia(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel composite inertia requires CUDA")
    def test_warp_reduction_matches_scalar_branched_trees(self):
        device = wp.get_cuda_device()
        articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
        joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
        joint_child = wp.array(np.arange(7, dtype=np.int32), dtype=wp.int32, device=device)
        body_inertia = wp.array(
            np.random.default_rng(42).normal(size=(7, 6, 6)).astype(np.float32),
            dtype=wp.spatial_matrix,
            device=device,
        )

        scalar = wp.empty_like(body_inertia)
        wp.launch(
            compute_composite_inertia,
            dim=2,
            inputs=[
                articulation_start,
                articulation_joint_end,
                wp.ones(2, dtype=wp.int32, device=device),
                joint_ancestor,
                joint_child,
                body_inertia,
            ],
            outputs=[scalar],
            device=device,
        )

        parallel = wp.empty_like(body_inertia)
        warps_per_block = 4
        kernel = _get_composite_inertia_warp_kernel(device.arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                2,
                wp.array([0, 1], dtype=wp.int32, device=device),
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                body_inertia,
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            outputs=[parallel],
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(parallel.numpy(), scalar.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel composite inertia requires CUDA")
    def test_compact_reduction_matches_assembled_inertia(self):
        device = wp.get_cuda_device()
        articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
        joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
        joint_child = wp.array(np.arange(7, dtype=np.int32), dtype=wp.int32, device=device)

        rng = np.random.default_rng(43)
        mass = rng.uniform(0.1, 10.0, size=7).astype(np.float32)
        com = rng.normal(size=(7, 3)).astype(np.float32)
        inertia_origin = rng.normal(size=(7, 3, 3)).astype(np.float32)
        terms = np.concatenate((com, inertia_origin.reshape(7, 9)), axis=1)
        assembled = np.zeros((7, 6, 6), dtype=np.float32)
        for body in range(7):
            com_cross = np.array(
                [
                    [0.0, -com[body, 2], com[body, 1]],
                    [com[body, 2], 0.0, -com[body, 0]],
                    [-com[body, 1], com[body, 0], 0.0],
                ],
                dtype=np.float32,
            )
            assembled[body, :3, :3] = mass[body] * np.eye(3, dtype=np.float32)
            assembled[body, :3, 3:] = -mass[body] * com_cross
            assembled[body, 3:, :3] = mass[body] * com_cross
            assembled[body, 3:, 3:] = inertia_origin[body]

        scalar = wp.empty((7,), dtype=wp.spatial_matrix, device=device)
        wp.launch(
            compute_composite_inertia,
            dim=2,
            inputs=[
                articulation_start,
                articulation_joint_end,
                wp.ones(2, dtype=wp.int32, device=device),
                joint_ancestor,
                joint_child,
                wp.array(assembled, dtype=wp.spatial_matrix, device=device),
            ],
            outputs=[scalar],
            device=device,
        )

        parallel = wp.empty_like(scalar)
        warps_per_block = 4
        kernel = _get_composite_inertia_warp_kernel(device.arch, warps_per_block, compact_terms=True)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                2,
                wp.array([0, 1], dtype=wp.int32, device=device),
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                wp.array(mass, dtype=wp.float32, device=device),
                wp.array(terms, dtype=wp.float32, device=device),
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            outputs=[parallel],
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(parallel.numpy(), scalar.numpy(), rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
