# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `core/materials.py` module."""

import logging
import unittest

import numpy as np
import warp as wp

import newton

# Module to be tested
from newton._src.solvers.kamino._src.core.materials import (
    MaterialMixMode,
    make_get_mixed_material_pair_property,
)

# Test utilities
from newton.tests.kamino import setup_test_output, setup_tests, test_context
from newton.tests.unittest_utils import NewtonTestCase

logger = logging.getLogger(__name__)

###
# Utilities
###


def make_mix_kernel(mixmode: MaterialMixMode):
    mix_func = make_get_mixed_material_pair_property(mixmode)

    @wp.kernel(module="unique")
    def _mix_kernel(
        values_0: wp.array[wp.float32],
        values_1: wp.array[wp.float32],
        mixed: wp.array[wp.float32],
    ):
        tid = wp.tid()
        mixed[tid] = mix_func(values_0[tid], values_1[tid])

    return _mix_kernel


###
# Tests
###


class TestMaterials(NewtonTestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output

        setup_test_output(self, verbose=self.verbose)

    def test_05_material_mix_modes(self):
        """Apply each mixing mode to zero, equal, and distinct coefficients."""
        values_0 = np.array([0.0, 0.5, 0.2, 1.0], dtype=np.float32)
        values_1 = np.array([0.8, 0.5, 0.8, 0.25], dtype=np.float32)
        expected = {
            MaterialMixMode.AVERAGE: 0.5 * (values_0 + values_1),
            MaterialMixMode.MULTIPLY: values_0 * values_1,
            MaterialMixMode.MAX: np.maximum(values_0, values_1),
            MaterialMixMode.MIN: np.minimum(values_0, values_1),
            MaterialMixMode.GEOMETRIC_AVERAGE: np.array([0.0, 0.5, 0.4, 0.5], dtype=np.float32),
        }
        self.assertEqual(set(expected), set(MaterialMixMode))

        device = test_context.device
        v0 = wp.array(values_0, dtype=wp.float32, device=device)
        v1 = wp.array(values_1, dtype=wp.float32, device=device)
        for mixmode, expected_values in expected.items():
            with self.subTest(mixmode=mixmode):
                self.assertEqual(MaterialMixMode.from_string(mixmode.name.lower()), mixmode)
                mixed = wp.zeros(len(values_0), dtype=wp.float32, device=device)
                wp.launch(make_mix_kernel(mixmode), dim=len(values_0), inputs=[v0, v1, mixed], device=device)
                np.testing.assert_allclose(mixed.numpy(), expected_values, rtol=1e-6)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    newton.utils.setup_console_logging()
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
