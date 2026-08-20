# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gaussian visual payloads embedded in deformable bodies."""

import unittest

import numpy as np
import warp as wp

import newton


def _soft_builder():
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    return builder


class TestDeformableVisualGaussianBuilder(unittest.TestCase):
    """Public builder and finalized model behavior."""

    def test_add_tet_bound_gaussian_visual(self):
        """Create a stable model record for a tetrahedron-bound Gaussian field."""
        builder = _soft_builder()
        positions = np.array(
            [[0.2, 0.2, 0.2], [0.4, 0.2, 0.2], [0.2, 0.4, 0.2], [0.2, 0.2, 0.4]], dtype=np.float32
        )
        gaussian = newton.Gaussian(positions=positions, scales=np.full((4, 3), 0.05, dtype=np.float32))
        index = builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=np.zeros(4, dtype=np.int32),
            weights=np.full((4, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        model = builder.finalize()

        self.assertEqual(index, 0)
        self.assertEqual(model.deformable_visual_gaussian_count, 1)
        visual = model.deformable_visual_gaussians[index]
        self.assertIsInstance(visual, newton.DeformableVisualGaussian)
        self.assertIs(visual.gaussian, gaussian)
        self.assertEqual(visual.kind, newton.DeformableVisualBinding.Kind.TET)
        self.assertEqual(visual.count, 4)
        self.assertEqual(visual.label, "soft_splats")
        self.assertEqual(visual.index, index)

    def test_replicate_offsets_gaussian_drivers(self):
        """Replicate Gaussian bindings into distinct worlds without copying rest appearance."""
        source = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, scales=np.full((2, 3), 0.05, dtype=np.float32))
        source.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, source.tet_count),
            parent=np.array([0, 1], dtype=np.int32),
            weights=np.full((2, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 2)
        self.assertEqual([visual.world for visual in model.deformable_visual_gaussians], [0, 1])
        np.testing.assert_array_equal(model.deformable_visual_gaussians[0].parent.numpy(), [0, 1])
        np.testing.assert_array_equal(
            model.deformable_visual_gaussians[1].parent.numpy(), np.array([0, 1]) + source.tet_count
        )
        self.assertIs(model.deformable_visual_gaussians[0].gaussian, gaussian)
        self.assertIs(model.deformable_visual_gaussians[1].gaussian, gaussian)

    def test_rejects_invalid_gaussian_visual_data(self):
        """Reject malformed appearance and unsupported bindings before finalization."""
        builder = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        bad_sh = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, sh_coeffs=bad_sh)

        with self.assertRaisesRegex(ValueError, "Gaussian data must be finite"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="tet",
                tet_range=(0, builder.tet_count),
                parent=[0],
                weights=np.full((1, 4), 0.25, dtype=np.float32),
            )

        gaussian = newton.Gaussian(positions=positions)
        with self.assertRaisesRegex(ValueError, "supports only kind='tet'"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="particle",
                tet_range=(0, builder.tet_count),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
