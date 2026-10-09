# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-limit row assembly of SolverFeatherPGS: the warp-parallel CUDA builder and the scalar builder."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_LIMIT, build_joint_limit_rows
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_joint_limit_warp_kernel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def test_warp_builder_matches_reference_row_order_and_values(test, device, scalar_builder=False):
    """Emit active limit rows in DOF order with the lower row before the upper row of each DOF.

    ``scalar_builder`` checks the one-thread-per-articulation builder of the CPU path instead.
    """
    articulation_count = 257
    size = 6
    max_constraints = 2 * size
    dof_count = articulation_count * size
    gap = 0.25
    rng = np.random.default_rng(42)
    q = rng.uniform(-1.2, 1.2, dof_count).astype(np.float32)
    lower = np.full(dof_count, -1.0, dtype=np.float32)
    upper = np.full(dof_count, 1.0, dtype=np.float32)

    outputs = (
        wp.zeros(articulation_count, dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints, size), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
    )
    inputs = [
        wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device),
        wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
        wp.array(np.arange(articulation_count), dtype=wp.int32, device=device),
        wp.array(np.arange(dof_count), dtype=wp.int32, device=device),
        wp.array(lower, dtype=wp.float32, device=device),
        wp.array(upper, dtype=wp.float32, device=device),
        wp.array(q, dtype=wp.float32, device=device),
        gap,
        max_constraints,
    ]
    if scalar_builder:
        wp.launch(
            build_joint_limit_rows,
            dim=articulation_count,
            inputs=[*inputs, size],
            outputs=list(outputs),
            device=device,
        )
    else:
        warps_per_block = 4
        kernel = _get_joint_limit_warp_kernel(size, wp.get_device(device).arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[(articulation_count + warps_per_block - 1) // warps_per_block],
            inputs=[articulation_count, *inputs],
            outputs=list(outputs),
            block_dim=32 * warps_per_block,
            device=device,
        )
    counter, J_group, row_type, row_parent, row_mu, phi, target = (array.numpy() for array in outputs)

    for art in range(articulation_count):
        expected_J = np.zeros((max_constraints, size), dtype=np.float32)
        expected_phi = []
        for local in range(size):
            dof = art * size + local
            for sign, value, active in (
                (1.0, q[dof] - lower[dof], q[dof] <= lower[dof] + gap),
                (-1.0, upper[dof] - q[dof], q[dof] >= upper[dof] - gap),
            ):
                if active:
                    expected_J[len(expected_phi), local] = sign
                    expected_phi.append(value)
        count = len(expected_phi)
        test.assertEqual(int(counter[art]), count)
        np.testing.assert_array_equal(J_group[art], expected_J)
        np.testing.assert_array_equal(row_type[art, :count], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        np.testing.assert_array_equal(row_parent[art, :count], -1)
        np.testing.assert_array_equal(row_mu[art, :count], 0.0)
        np.testing.assert_array_equal(target[art, :count], 0.0)
        np.testing.assert_allclose(phi[art, :count], np.asarray(expected_phi, dtype=np.float32), rtol=0.0, atol=1.0e-7)


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSJointLimitWarp,
    "test_warp_builder_matches_reference_row_order_and_values",
    test_warp_builder_matches_reference_row_order_and_values,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSJointLimitWarp,
    "test_scalar_builder_matches_reference_row_order_and_values",
    test_warp_builder_matches_reference_row_order_and_values,
    devices=get_test_devices(),
    scalar_builder=True,
)


if __name__ == "__main__":
    unittest.main()
