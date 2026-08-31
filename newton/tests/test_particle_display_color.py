# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


class _ViewerFallbackProbe(ViewerNull):
    """Capture the colors the shared viewer path sends for particles and surfaces."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.point_colors = None
        self.mesh_colors = None

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.point_colors = colors

    def log_mesh(self, _name, _points, _indices, *args, colors=None, **kwargs):
        self.mesh_colors = colors


class TestParticleDisplayColor(unittest.TestCase):
    def test_high_level_particle_builders_author_display_color(self):
        """Propagate a uniform display color through every particle-producing helper."""
        color = (0.2, 0.4, 0.6)

        def assert_authored(builder, expected_count):
            self.assertEqual(builder.particle_count, expected_count)
            np.testing.assert_allclose(
                np.asarray(builder.particle_display_color, dtype=np.float32),
                np.tile(np.asarray(color, dtype=np.float32), (expected_count, 1)),
            )
            model = builder.finalize(device="cpu")
            np.testing.assert_allclose(
                model.particle_display_color.numpy(),
                np.tile(np.asarray(color, dtype=np.float32), (expected_count, 1)),
            )

        with self.subTest(helper="add_particle_grid"):
            builder = newton.ModelBuilder()
            builder.add_particle_grid(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=1,
                dim_y=1,
                dim_z=2,
                cell_x=1.0,
                cell_y=1.0,
                cell_z=1.0,
                mass=1.0,
                jitter=0.0,
                color=color,
            )
            assert_authored(builder, 2)

        with self.subTest(helper="add_cloth_grid"):
            builder = newton.ModelBuilder()
            builder.add_cloth_grid(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=1,
                dim_y=1,
                cell_x=1.0,
                cell_y=1.0,
                mass=1.0,
                color=color,
            )
            assert_authored(builder, 4)

        with self.subTest(helper="add_cloth_mesh"):
            builder = newton.ModelBuilder()
            builder.add_cloth_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                indices=[0, 1, 2],
                density=1.0,
                color=color,
            )
            assert_authored(builder, 3)

        with self.subTest(helper="add_soft_grid"):
            builder = newton.ModelBuilder()
            builder.add_soft_grid(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=1,
                dim_y=1,
                dim_z=1,
                cell_x=1.0,
                cell_y=1.0,
                cell_z=1.0,
                density=1.0,
                k_mu=1.0,
                k_lambda=1.0,
                k_damp=0.0,
                add_surface_mesh_edges=False,
                color=color,
            )
            assert_authored(builder, 8)

        with self.subTest(helper="add_soft_mesh"):
            builder = newton.ModelBuilder()
            builder.add_soft_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=[
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ],
                indices=[0, 1, 2, 3],
                density=1.0,
                k_mu=1.0,
                k_lambda=1.0,
                k_damp=0.0,
                add_surface_mesh_edges=False,
                color=color,
            )
            assert_authored(builder, 4)

    def test_finalize_preserves_opt_in_particle_display_colors(self):
        """Preserve a null model field until at least one particle authors a display color."""
        uncolored = newton.ModelBuilder()
        uncolored.add_particle((0.0, 0.0, 0.0), wp.vec3(), 1.0)

        self.assertEqual(uncolored.particle_display_color, [None])
        self.assertIsNone(uncolored.finalize(device="cpu").particle_display_color)

        colored = newton.ModelBuilder()
        colored.add_particle((0.0, 0.0, 0.0), wp.vec3(), 1.0, color=(0.1, 0.2, 0.3))
        colored.add_particle((1.0, 0.0, 0.0), wp.vec3(), 1.0)
        colored.add_particles(
            pos=[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
            vel=[wp.vec3(), wp.vec3()],
            mass=[1.0, 1.0],
            colors=[(0.4, 0.5, 0.6), None],
        )

        model = colored.finalize(device="cpu")
        self.assertIsNotNone(model.particle_display_color)
        np.testing.assert_allclose(
            model.particle_display_color.numpy(),
            np.asarray(
                [
                    (0.1, 0.2, 0.3),
                    (1.0, 1.0, 1.0),
                    (0.4, 0.5, 0.6),
                    (1.0, 1.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(model.particle_colors.dtype, wp.int32)

    def test_add_particles_rejects_mismatched_display_colors(self):
        """Reject bulk display colors that do not match the particle count."""
        builder = newton.ModelBuilder()
        builder.add_particle((0.0, 0.0, 0.0), wp.vec3(), 1.0, color=(0.1, 0.2, 0.3))
        before = list(builder.particle_display_color)

        with self.assertRaisesRegex(ValueError, r"colors.*2.*1"):
            builder.add_particles(
                pos=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                vel=[wp.vec3(), wp.vec3()],
                mass=[1.0, 1.0],
                colors=[(0.4, 0.5, 0.6)],
            )

        self.assertEqual(builder.particle_count, 1)
        self.assertEqual(builder.particle_display_color, before)

    def test_add_builder_preserves_particle_display_colors(self):
        """Preserve authored and unspecified display colors when composing builders."""
        source = newton.ModelBuilder()
        source.add_particles(
            pos=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            vel=[wp.vec3(), wp.vec3()],
            mass=[1.0, 1.0],
            colors=[(0.25, 0.5, 0.75), None],
        )

        destination = newton.ModelBuilder()
        destination.add_particle((-1.0, 0.0, 0.0), wp.vec3(), 1.0)
        destination.add_builder(source)

        self.assertEqual(destination.particle_display_color[0], None)
        np.testing.assert_allclose(destination.particle_display_color[1], (0.25, 0.5, 0.75))
        self.assertEqual(destination.particle_display_color[2], None)

        model = destination.finalize(device="cpu")
        np.testing.assert_allclose(
            model.particle_display_color.numpy(),
            np.asarray(
                [
                    (1.0, 1.0, 1.0),
                    (0.25, 0.5, 0.75),
                    (1.0, 1.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )

        replicated = newton.ModelBuilder()
        replicated.replicate(source, 2)
        replicated_model = replicated.finalize(device="cpu")
        np.testing.assert_allclose(
            replicated_model.particle_display_color.numpy(),
            np.asarray(
                [
                    (0.25, 0.5, 0.75),
                    (1.0, 1.0, 1.0),
                    (0.25, 0.5, 0.75),
                    (1.0, 1.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )

    @staticmethod
    def _build_surface(colors):
        """Build one triangle whose three particles carry the given display colors."""
        builder = newton.ModelBuilder()
        builder.add_particles(
            pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
            vel=[wp.vec3()] * 3,
            mass=[1.0] * 3,
            colors=colors,
        )
        builder.add_triangle(0, 1, 2)
        return builder.finalize(device="cpu")

    def test_partial_authoring_replaces_viewer_fallbacks(self):
        """Send the model array for every particle once any particle is colored.

        Authoring a single particle materializes the whole array, so the viewer
        stops substituting its own default color for the unauthored remainder.
        This pins the documented cost of the dense representation.
        """
        uncolored = self._build_surface(None)
        self.assertIsNone(uncolored.particle_display_color)

        probe = _ViewerFallbackProbe()
        probe.set_model(uncolored)
        probe.begin_frame(0.0)
        probe.log_state(uncolored.state())

        self.assertIsNone(probe.mesh_colors)
        np.testing.assert_allclose(
            probe.point_colors.numpy(),
            np.tile(np.asarray((0.7, 0.6, 0.4), dtype=np.float32), (3, 1)),
        )

        partial = self._build_surface([(1.0, 0.0, 0.0), None, None])
        probe = _ViewerFallbackProbe()
        probe.set_model(partial)
        probe.begin_frame(0.0)
        probe.log_state(partial.state())

        expected = np.asarray(
            [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(probe.point_colors.numpy(), expected)
        np.testing.assert_allclose(probe.mesh_colors.numpy(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
