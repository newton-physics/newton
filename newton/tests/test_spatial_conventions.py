# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for spatial vector convention utilities."""

import unittest

import numpy as np
import warp as wp

import newton.utils as nu


class TestSpatialConventions(unittest.TestCase):
    """Test spatial vector convention conversion utilities."""

    def test_swap_spatial_halves_symmetry(self):
        """Verify double-swap returns original vector."""

        # Create a Warp kernel that swaps twice
        @wp.kernel
        def double_swap_kernel(input: wp.array(dtype=wp.spatial_vector), output: wp.array(dtype=wp.spatial_vector)):
            tid = wp.tid()
            x = input[tid]
            # Swap twice - should return original
            swapped_once = nu.swap_spatial_halves(x)
            swapped_twice = nu.swap_spatial_halves(swapped_once)
            output[tid] = swapped_twice

        # Test data
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=np.float32)

        # Create Warp arrays
        input_wp = wp.array(input_data, dtype=wp.spatial_vector)
        output_wp = wp.empty(2, dtype=wp.spatial_vector)

        # Run kernel
        wp.launch(double_swap_kernel, dim=2, inputs=[input_wp, output_wp])
        wp.synchronize()

        # Convert back to numpy
        output_data = output_wp.numpy()

        # Verify double-swap returns original
        np.testing.assert_allclose(output_data, input_data, rtol=1e-6)

    def test_swap_warp_kernel(self):
        """Test swap_spatial_halves in Warp kernel."""

        @wp.kernel
        def swap_kernel(input: wp.array(dtype=wp.spatial_vector), output: wp.array(dtype=wp.spatial_vector)):
            tid = wp.tid()
            x = input[tid]
            output[tid] = nu.swap_spatial_halves(x)

        # Input: [linear, angular]
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

        # Create Warp arrays
        input_wp = wp.array(input_data, dtype=wp.spatial_vector)
        output_wp = wp.empty(1, dtype=wp.spatial_vector)

        # Run kernel
        wp.launch(swap_kernel, dim=1, inputs=[input_wp, output_wp])
        wp.synchronize()

        # Expected: [angular, linear] = [4, 5, 6, 1, 2, 3]
        expected = np.array([[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        output_data = output_wp.numpy()

        np.testing.assert_allclose(output_data, expected, rtol=1e-6)

    def test_convert_batch_numpy(self):
        """Test NumPy batch conversion."""
        # Input: Newton [linear, angular]
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=np.float32)

        # Convert to MuJoCo [angular, linear]
        output = nu.convert_spatial_vector_batch(
            input_data, from_form=nu.SpatialVectorForm.LINEAR_ANGULAR, to_form=nu.SpatialVectorForm.ANGULAR_LINEAR
        )

        # Expected: swap halves
        expected = np.array([[4.0, 5.0, 6.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 7.0, 8.0, 9.0]], dtype=np.float32)

        np.testing.assert_allclose(output, expected, rtol=1e-6)

    def test_convert_batch_no_op(self):
        """Test conversion with same form is a no-op."""
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

        # Same form - should return copy of input
        output = nu.convert_spatial_vector_batch(
            input_data, from_form=nu.SpatialVectorForm.LINEAR_ANGULAR, to_form=nu.SpatialVectorForm.LINEAR_ANGULAR
        )

        np.testing.assert_allclose(output, input_data, rtol=1e-6)

    def test_convert_batch_warp_array(self):
        """Test conversion from Warp array to NumPy."""
        # Create Warp array
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        input_wp = wp.array(input_data, dtype=wp.spatial_vector)

        # Convert
        output = nu.convert_spatial_vector_batch(
            input_wp, from_form=nu.SpatialVectorForm.LINEAR_ANGULAR, to_form=nu.SpatialVectorForm.ANGULAR_LINEAR
        )

        # Expected: swap halves
        expected = np.array([[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]], dtype=np.float32)

        # Output should be NumPy array
        self.assertIsInstance(output, np.ndarray)
        np.testing.assert_allclose(output, expected, rtol=1e-6)

    def test_conventions_match_mujoco(self):
        """Verify MuJoCo convention handling."""
        # Simulate MuJoCo cfrc_int: [angular, linear]
        mujoco_wrench = np.array([[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]], dtype=np.float32)  # [τx, τy, τz, fx, fy, fz]

        # Convert to Newton [linear, angular]
        newton_wrench = nu.convert_spatial_vector_batch(
            mujoco_wrench, from_form=nu.SpatialVectorForm.ANGULAR_LINEAR, to_form=nu.SpatialVectorForm.LINEAR_ANGULAR
        )

        # Expected: [fx, fy, fz, τx, τy, τz]
        expected = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], dtype=np.float32)

        np.testing.assert_allclose(newton_wrench, expected, rtol=1e-6)

    def test_conventions_match_newton(self):
        """Verify Newton convention is preserved."""
        # Newton/Warp wrench: [linear, angular]
        newton_wrench = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], dtype=np.float32)  # [fx, fy, fz, τx, τy, τz]

        # No-op conversion
        same = nu.convert_spatial_vector_batch(
            newton_wrench, from_form=nu.SpatialVectorForm.LINEAR_ANGULAR, to_form=nu.SpatialVectorForm.LINEAR_ANGULAR
        )

        np.testing.assert_allclose(same, newton_wrench, rtol=1e-6)

    def test_multi_dimensional_batch(self):
        """Test conversion with multi-dimensional batches."""
        # Shape: [nworlds, nbodies, 6]
        rng = np.random.default_rng(42)
        input_data = rng.random((3, 4, 6), dtype=np.float32)

        # Convert and back
        converted = nu.convert_spatial_vector_batch(
            input_data, from_form=nu.SpatialVectorForm.LINEAR_ANGULAR, to_form=nu.SpatialVectorForm.ANGULAR_LINEAR
        )
        back = nu.convert_spatial_vector_batch(
            converted, from_form=nu.SpatialVectorForm.ANGULAR_LINEAR, to_form=nu.SpatialVectorForm.LINEAR_ANGULAR
        )

        # Should recover original
        np.testing.assert_allclose(back, input_data, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
