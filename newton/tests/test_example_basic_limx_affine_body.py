# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestBasicLimxAffineBodyExample(unittest.TestCase):
    def test_relaxes_sheared_body_during_free_fall(self):
        """Relax visible affine shear while the reconstructed body falls."""
        module = importlib.import_module("newton.examples.basic.example_basic_limx_affine_body")
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=100), None)
            initial_vertices = example.state_0.particle_q.numpy()
            initial_center_height = float(initial_vertices[:, 2].mean())
            initial_matrix = example.solver.q.numpy()[0, 3:].reshape(3, 3)
            initial_error = float(np.linalg.norm(np.linalg.svd(initial_matrix, compute_uv=False) - 1.0))

            for _ in range(100):
                example.step()
                example.test_post_step()

            final_vertices = example.state_0.particle_q.numpy()
            final_matrix = example.solver.q.numpy()[0, 3:].reshape(3, 3)
            final_error = float(np.linalg.norm(np.linalg.svd(final_matrix, compute_uv=False) - 1.0))

            self.assertTrue(np.isfinite(final_vertices).all())
            self.assertGreater(float(np.linalg.det(final_matrix)), 0.0)
            self.assertLess(float(final_vertices[:, 2].mean()), initial_center_height)
            self.assertLess(final_error, initial_error)
            example.test_final()


if __name__ == "__main__":
    unittest.main()
