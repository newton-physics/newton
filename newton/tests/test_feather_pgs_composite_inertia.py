# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-parallel composite-inertia reduction of SolverFeatherPGS against the scalar reduction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import compute_composite_inertia
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_composite_inertia_warp_kernel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_warp_reduction_matches_scalar_branched_trees(test, device):
    """Reduce two branched trees, joints not ordered by body, identically to the scalar kernel."""
    articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
    articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
    joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
    joint_child = wp.array(np.random.default_rng(41).permutation(7).astype(np.int32), dtype=wp.int32, device=device)
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
        ],
        outputs=[parallel],
        block_dim=32 * warps_per_block,
        device=device,
    )
    np.testing.assert_array_equal(parallel.numpy(), scalar.numpy())

    # Independent oracle: each composite inertia is the sum over the subtree.
    ancestor = joint_ancestor.numpy()
    child = joint_child.numpy()
    inertia = body_inertia.numpy().astype(np.float64)
    for joint in range(7):
        expected = np.zeros((6, 6))
        for other in range(7):
            node = other
            while node >= 0 and node != joint:
                node = ancestor[node]
            if node == joint:
                expected += inertia[child[other]]
        np.testing.assert_allclose(parallel.numpy()[child[joint]], expected, rtol=1.0e-5, atol=1.0e-5)


class TestFeatherPGSCompositeInertia(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSCompositeInertia,
    "test_warp_reduction_matches_scalar_branched_trees",
    test_warp_reduction_matches_scalar_branched_trees,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
