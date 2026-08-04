# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerNull


def _load_example_module(test_case: unittest.TestCase):
    module_name = "newton.examples.cloth.example_cloth_limx_three_tshirts_box"
    if "cloth_limx_three_tshirts_box" not in newton.examples.get_examples():
        test_case.fail(f"Missing example module {module_name}")
    return importlib.import_module(module_name)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxThreeTshirtsBox(unittest.TestCase):
    def test_cuda_graph_step_is_finite_and_configured(self):
        """Build three garments and keep one captured CUDA step finite."""
        module = _load_example_module(self)
        self.assertEqual(
            newton.examples.get_examples()["cloth_limx_three_tshirts_box"],
            "newton.examples.cloth.example_cloth_limx_three_tshirts_box",
        )

        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), None)
            example.step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()
            example.test_post_step()
            example.test_final()

        self.assertEqual(example.garment_count, 3)
        self.assertEqual(example.model.particle_count, 3 * example.garment_vertex_count)
        self.assertEqual(example.model.tri_count, 3 * example.garment_triangle_count)
        self.assertEqual(len(example.box_contacts), 5)
        self.assertEqual(example.self_collision.max_contacts, 393216)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())

    def test_cuda_rollout_stays_contained_without_contact_overflow(self):
        """Keep the initial three-garment rollout finite, contained, and within contact capacity."""
        module = _load_example_module(self)
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=300), None)
            overflow_counts = np.zeros(3, dtype=np.int32)
            for _ in range(300):
                example.step()
                example.test_post_step()
                overflow_counts = np.maximum(
                    overflow_counts,
                    np.asarray(
                        [
                            example.self_collision.vertex_face_contacts.overflow_count.numpy()[0],
                            example.self_collision.edge_edge_contacts.overflow_count.numpy()[0],
                            example.self_collision.edge_face_contacts.overflow_count.numpy()[0],
                        ],
                        dtype=np.int32,
                    ),
                )

        np.testing.assert_array_equal(overflow_counts, np.zeros(3, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
