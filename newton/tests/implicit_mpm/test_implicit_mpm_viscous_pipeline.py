# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm.rheology_solver_kernels import (
    StrainMatrixRows,
    YieldParamVec,
    make_solve_strain_rows,
    mat55,
    mat66,
    vec6,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_pipeline_test_kernel(enabled):
    solve_rows = make_solve_strain_rows(True, False, False, True, False, False, -1, enabled)

    @fem.cache.dynamic_kernel(suffix=enabled, kernel_options={"fast_math": True})
    def solve(
        rows: StrainMatrixRows,
        params: wp.array[YieldParamVec],
        volume: wp.array[float],
        diagonal: wp.array[vec6],
        rotation: wp.array[mat55],
        inverse_mass: wp.array[float],
        rhs: wp.array[vec6],
        empty_indices: wp.array[int],
        empty_values: wp.array[mat66],
        velocity: wp.array[wp.vec3],
        stress: wp.array[vec6],
        delta: wp.array[vec6],
    ):
        wp.static(solve_rows)(
            0,
            params.shape[0],
            rows,
            params,
            volume,
            empty_indices,
            empty_indices,
            empty_values,
            diagonal,
            rotation,
            inverse_mass,
            rhs,
            velocity,
            stress,
            delta,
        )

    return solve


def test_viscous_row_pipeline(test, device, capture=False):
    """Match scalar rows across pipeline tails, repeated columns, and empty volumes."""
    rng = np.random.default_rng(723)
    lengths = np.array([0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    count = len(lengths)
    nvel = 96
    with wp.ScopedDevice(device):
        rows = StrainMatrixRows()
        rows.offsets = wp.array(offsets, dtype=int)
        # Repeated columns ensure scatter keeps sequential read-modify-write semantics.
        rows.columns = wp.array(rng.integers(0, nvel, offsets[-1], dtype=np.int32), dtype=int)
        rows.values = wp.array(rng.normal(0, 0.01, (offsets[-1], 3)), dtype=wp.vec3)
        params = wp.array(np.tile([2, 0.1, 0.2, 0.6, 0, 0.3], (count, 1)), dtype=YieldParamVec)
        volume_values = np.ones(count, dtype=np.float32)
        volume_values[1] = 0.0
        volume = wp.array(volume_values, dtype=float)
        diagonal = wp.array(rng.uniform(0.5, 2, (count, 6)), dtype=vec6)
        rotation = wp.array(np.tile(np.eye(5), (count, 1, 1)), dtype=mat55)
        inverse_mass = wp.array(rng.uniform(0.1, 2, nvel), dtype=float)
        rhs = wp.array(rng.normal(0, 0.1, (count, 6)), dtype=vec6)
        empty_indices = wp.empty(0, dtype=int)
        empty_values = wp.empty(0, dtype=mat66)
        initial_velocity = rng.normal(0, 0.01, (nvel, 3))
        initial_stress = rng.normal(0, 0.001, (count, 6))
        results = []
        for enabled in (False, True):
            velocity = wp.array(initial_velocity, dtype=wp.vec3)
            stress = wp.array(initial_stress, dtype=vec6)
            delta = wp.full(count, vec6(float("nan")), dtype=vec6)
            kernel = _make_pipeline_test_kernel(enabled)
            inputs = [rows, params, volume, diagonal, rotation, inverse_mass, rhs, empty_indices, empty_values]
            outputs = [velocity, stress, delta]
            wp.load_module(device=device)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs)
                wp.capture_launch(graph.graph)
            else:
                wp.launch(kernel, dim=1, inputs=inputs, outputs=outputs)
            results.append([array.numpy() for array in outputs])
        for actual, expected in zip(results[1], results[0], strict=True):
            test.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(results[1][2][0], np.zeros(6))


class TestViscousPipeline(unittest.TestCase):
    pass


add_function_test(
    TestViscousPipeline, "test_viscous_row_pipeline", test_viscous_row_pipeline, devices=get_test_devices()
)
add_function_test(
    TestViscousPipeline,
    "test_viscous_row_pipeline_graph",
    test_viscous_row_pipeline,
    devices=[device for device in get_test_devices() if device.is_cuda],
    capture=True,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
