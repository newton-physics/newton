# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton.viewer import ViewerNull


class _LogPointsProbe(ViewerNull):
    """Captures args passed to ``log_points`` so tests can inspect them."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.logged_points = None
        self.logged_radii = None
        self.logged_hidden = None
        self.log_points_called = False

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.log_points_called = True
        self.logged_points = points
        self.logged_radii = radii
        self.logged_hidden = hidden


class TestViewerParticleFlags(unittest.TestCase):
    """Verify _log_particles filters out inactive particles."""

    @staticmethod
    def _build_model(flags_list):
        """Build a model with particles at known positions and given flags."""
        builder = newton.ModelBuilder()
        for i, flag in enumerate(flags_list):
            builder.add_particle(
                pos=(float(i), 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.1,
                flags=flag,
            )
        model = builder.finalize(device="cpu")
        return model

    def test_all_active_renders_all(self):
        """When all particles are ACTIVE, all are passed to log_points."""
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active, active, active])
        state = model.state()

        viewer = _LogPointsProbe()
        viewer.set_model(model)
        viewer._log_particles(state)

        self.assertTrue(viewer.log_points_called)
        self.assertEqual(len(viewer.logged_points), 3)

    def test_mixed_active_inactive_filters(self):
        """Only ACTIVE particles should be passed to log_points."""
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active, 0, active, 0, active])
        state = model.state()

        viewer = _LogPointsProbe()
        viewer.set_model(model)
        viewer._log_particles(state)

        self.assertTrue(viewer.log_points_called)
        self.assertEqual(len(viewer.logged_points), 3)
        points_np = viewer.logged_points.numpy()
        np.testing.assert_allclose(points_np[:, 0], [0.0, 2.0, 4.0], atol=1e-6)

    def test_all_inactive_clears_particles(self):
        """When no particles are ACTIVE, log_points should clear the point cloud."""
        model = self._build_model([0, 0, 0])
        state = model.state()

        viewer = _LogPointsProbe()
        viewer.set_model(model)
        viewer._log_particles(state)

        self.assertTrue(viewer.log_points_called)
        self.assertIsNone(viewer.logged_points)
        self.assertTrue(viewer.logged_hidden)

    def test_no_flags_renders_all(self):
        """When particle_flags is None, all particles should be rendered."""
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active, active])
        state = model.state()
        model.particle_flags = None

        viewer = _LogPointsProbe()
        viewer.set_model(model)
        viewer._log_particles(state)

        self.assertTrue(viewer.log_points_called)
        self.assertEqual(len(viewer.logged_points), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
