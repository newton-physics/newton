# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


@wp.kernel
def _record_particle_diagnostics(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    table_top: float,
    frame: int,
    minimum_gaps: wp.array[float],
    speed_sums: wp.array[float],
):
    particle = wp.tid()
    wp.atomic_min(minimum_gaps, frame, positions[particle][2] - table_top)
    wp.atomic_add(speed_sums, frame, wp.length(velocities[particle]))


@wp.kernel
def _accumulate_particle_speed_squared(
    velocities: wp.array[wp.vec3],
    speed_squared_sums: wp.array[float],
):
    particle = wp.tid()
    wp.atomic_add(speed_squared_sums, particle, wp.length_sq(velocities[particle]))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxTshirtTable(unittest.TestCase):
    def test_cuda_graph_step_is_finite(self):
        """Keep one captured CUDA simulation step finite and correctly configured."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_tshirt_table")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), None)
            example.step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()
            example.test_post_step()
            example.test_final()

        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.sim_substeps, 1)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(example.model.particle_count, 6436)
        self.assertEqual(example.model.tri_count, 12736)
        bending = next(
            constraint
            for constraint in example.solver.constraints
            if isinstance(constraint, newton.solvers.ConstraintDihedralBending)
        )
        self.assertEqual(bending.stiffness, 1.0e-4)
        self.assertIsNone(example.self_collision.stiffness)
        self.assertIsNone(example.self_collision.untangle_stiffness)
        self.assertEqual(example.self_collision.stiffness_factors, (0.5, 0.1, 1.5))
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())

    def test_settles_on_table(self):
        """Settle every garment particle on the table without localized jitter."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_tshirt_table")
        frame_count = 3000
        settling_window = 500
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=frame_count), None)
            minimum_gaps = wp.full(frame_count, 1.0e6, dtype=float, device="cuda:0")
            speed_sums = wp.zeros(frame_count, dtype=float, device="cuda:0")
            particle_speed_squared_sums = wp.zeros(example.model.particle_count, dtype=float, device="cuda:0")
            for frame in range(frame_count):
                example.step()
                wp.launch(
                    _record_particle_diagnostics,
                    dim=example.model.particle_count,
                    inputs=[
                        example.state_0.particle_q,
                        example.state_0.particle_qd,
                        example.table_top,
                        frame,
                    ],
                    outputs=[minimum_gaps, speed_sums],
                    device="cuda:0",
                )
                if frame >= frame_count - settling_window:
                    wp.launch(
                        _accumulate_particle_speed_squared,
                        dim=example.model.particle_count,
                        inputs=[example.state_0.particle_qd],
                        outputs=[particle_speed_squared_sums],
                        device="cuda:0",
                    )

            gaps = minimum_gaps.numpy()
            mean_speeds = speed_sums.numpy() / example.model.particle_count
            particle_rms_speeds = np.sqrt(particle_speed_squared_sums.numpy() / settling_window)

        self.assertTrue(np.isfinite(gaps).all())
        self.assertTrue(np.isfinite(mean_speeds).all())
        self.assertGreaterEqual(float(gaps.min()), -0.008)
        self.assertLess(float(mean_speeds[-50:].mean()), 0.02)
        persistent_particle_count = int(np.count_nonzero(particle_rms_speeds >= 0.02))
        settling_summary = (
            f"persistent={persistent_particle_count}, "
            f"p99={np.percentile(particle_rms_speeds, 99.0):.6f}, "
            f"max={particle_rms_speeds.max():.6f}"
        )
        self.assertEqual(persistent_particle_count, 0, settling_summary)
        self.assertLess(float(particle_rms_speeds.max()), 0.02, settling_summary)


if __name__ == "__main__":
    unittest.main()
