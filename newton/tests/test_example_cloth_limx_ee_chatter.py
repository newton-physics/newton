# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

import newton.examples
from newton.viewer import ViewerNull


def _load_example_module(test_case: unittest.TestCase):
    module_name = "newton.examples.cloth.example_cloth_limx_ee_chatter"
    if "cloth_limx_ee_chatter" not in newton.examples.get_examples():
        test_case.fail(f"Missing example module {module_name}")
    return importlib.import_module(module_name)


class TestClothLimxEeChatterConfiguration(unittest.TestCase):
    def test_rejects_cpu_execution(self):
        """Reject running the CUDA-only characterization example on CPU."""
        module = _load_example_module(self)

        with wp.ScopedDevice("cpu"):
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
                module.Example(ViewerNull(num_frames=1), None)

    def test_keeps_stored_patch_data_immutable(self):
        """Keep the embedded diagnostic snapshot immutable after import."""
        module = _load_example_module(self)

        for name in (
            "_REST_POSITIONS",
            "_INITIAL_POSITIONS",
            "_TRIANGLE_INDICES",
            "_MASSES",
            "_BOUNDARY_INDICES",
        ):
            patch_data = getattr(module, name)
            with self.assertRaisesRegex(ValueError, "read-only", msg=name):
                patch_data.flat[0] = patch_data.flat[0]


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxEeChatter(unittest.TestCase):
    def test_cuda_graph_step_preserves_the_two_patch_configuration(self):
        """Keep one captured step finite with collision enabled only on the reproduction patch."""
        module = _load_example_module(self)

        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), None)
            example.step()
            example.test_post_step()
            example.test_final()
            control_positions = example.control_patch.state_0.particle_q.numpy()
            control_velocities = example.control_patch.state_0.particle_qd.numpy()
            collision_positions = example.collision_patch.state_0.particle_q.numpy()
            collision_velocities = example.collision_patch.state_0.particle_qd.numpy()

        self.assertEqual(example.patch_vertex_count, 74)
        self.assertEqual(example.patch_triangle_count, 112)
        self.assertEqual(example.boundary_vertex_count, 34)
        self.assertEqual(example.control_patch.model.particle_count, 74)
        self.assertEqual(example.collision_patch.model.particle_count, 74)
        self.assertEqual(example.control_patch.model.tri_count, 112)
        self.assertEqual(example.collision_patch.model.tri_count, 112)
        self.assertEqual(len(example.control_patch.solver.constraints[-1].indices), 34)
        self.assertEqual(len(example.collision_patch.solver.constraints[-1].indices), 34)
        self.assertIsNone(example.control_patch.self_collision)
        self.assertIsNotNone(example.collision_patch.self_collision)
        self.assertEqual(example.collision_patch.self_collision.max_contacts, 4096)
        self.assertEqual(example.collision_patch.self_collision.geometry_radius_scale, 0.25)
        radii = example.collision_patch.self_collision.particle_radii.numpy()
        self.assertEqual(radii.shape, (74,))
        self.assertTrue(np.isfinite(radii).all())
        self.assertTrue(np.all(radii > 0.0))
        self.assertTrue(np.all(radii <= 0.003))
        self.assertLess(float(np.min(radii)), 0.003)
        self.assertEqual(example.control_patch.solver.nonlinear_iterations, 1)
        self.assertEqual(example.collision_patch.solver.nonlinear_iterations, 1)
        self.assertEqual(example.control_patch.solver.linear_iterations, 50)
        self.assertEqual(example.collision_patch.solver.linear_iterations, 50)
        self.assertEqual(example.control_patch.solver.velocity_damping, 1.0)
        self.assertEqual(example.collision_patch.solver.velocity_damping, 1.0)
        self.assertTrue(np.isfinite(control_positions).all())
        self.assertTrue(np.isfinite(control_velocities).all())
        self.assertTrue(np.isfinite(collision_positions).all())
        self.assertTrue(np.isfinite(collision_velocities).all())

    def test_geometry_aware_collision_settles_without_contact_churn(self):
        """Settle the irregular patch without persistent EE active-set churn."""
        module = _load_example_module(self)
        frame_count = 1400
        sample_start = 1000

        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=frame_count), None)
            control_rms_speeds = []
            collision_rms_speeds = []
            previous_ee_ids = None
            ee_births = 0
            ee_deaths = 0
            maximum_overflow = np.zeros(3, dtype=np.int32)
            for frame in range(frame_count):
                example.step()
                self_collision = example.collision_patch.self_collision
                maximum_overflow = np.maximum(
                    maximum_overflow,
                    np.asarray(
                        [
                            self_collision.vertex_face_contacts.overflow_count.numpy()[0],
                            self_collision.edge_edge_contacts.overflow_count.numpy()[0],
                            self_collision.edge_face_contacts.overflow_count.numpy()[0],
                        ],
                        dtype=np.int32,
                    ),
                )
                if frame < sample_start:
                    continue

                interior_indices = example.interior_indices
                control_velocities = example.control_patch.state_0.particle_qd.numpy()[interior_indices]
                collision_velocities = example.collision_patch.state_0.particle_qd.numpy()[interior_indices]
                control_rms_speeds.append(float(np.sqrt(np.mean(control_velocities * control_velocities))))
                collision_rms_speeds.append(float(np.sqrt(np.mean(collision_velocities * collision_velocities))))

                contacts = self_collision.edge_edge_contacts
                contact_count = min(int(contacts.count.numpy()[0]), contacts.capacity)
                current_ee_ids = {tuple(map(int, ids)) for ids in contacts.ids[:contact_count].numpy()}
                if previous_ee_ids is not None:
                    ee_births += len(current_ee_ids - previous_ee_ids)
                    ee_deaths += len(previous_ee_ids - current_ee_ids)
                previous_ee_ids = current_ee_ids

        control_rms_mean = float(np.mean(control_rms_speeds))
        collision_rms_mean = float(np.mean(collision_rms_speeds))
        total_ee_churn = ee_births + ee_deaths
        summary = (
            f"control_rms={control_rms_mean:.8f}, collision_rms={collision_rms_mean:.8f}, "
            f"EE_births={ee_births}, EE_deaths={ee_deaths}"
        )
        self.assertLess(control_rms_mean, 1.0e-6, summary)
        self.assertLess(collision_rms_mean, 1.0e-5, summary)
        self.assertLessEqual(total_ee_churn, 10, summary)
        np.testing.assert_array_equal(maximum_overflow, np.zeros(3, dtype=np.int32), err_msg=summary)


if __name__ == "__main__":
    unittest.main()
