# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer_viser import ViewerViser


class TestViewerViserPlaneFlicker(unittest.TestCase):
    """Regression tests for plane-grid handle churn in ``ViewerViser`` (issue #2099)."""

    def _make_viser_viewer(self):
        captured = {"add_grid_calls": 0}

        def add_grid(name, **kwargs):
            captured["add_grid_calls"] += 1
            handle = Mock()
            handle.name = name
            return handle

        scene = Mock()
        scene.add_grid = add_grid
        scene.add_light_ambient = Mock()
        scene.configure_environment_map = Mock()

        server = Mock()
        server.scene = scene
        server.on_client_connect = Mock()
        server.on_client_disconnect = Mock()
        server.get_scene_serializer = Mock(return_value=None)
        server.stop = Mock()

        fake_viser = Mock()
        fake_viser.ViserServer = Mock(return_value=server)

        patches = [
            patch.object(ViewerViser, "_get_viser", return_value=fake_viser),
            patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        viewer = ViewerViser(verbose=False)
        self.addCleanup(viewer.close)
        return viewer, captured

    def _log_plane(self, viewer, position):
        """Log one plane instance at ``position`` with identity orientation."""
        xform = wp.array([wp.transform(wp.vec3(*position), wp.quat_identity())], dtype=wp.transform)
        viewer.log_geo("/ground", int(newton.GeoType.PLANE), (10.0, 10.0), 0.0, True)
        viewer.log_instances("/ground_instances", "/ground", xform, None, None, None)

    def test_unchanged_plane_pose_reuses_grid_handle(self):
        """Logging the same static plane every frame should not tear down and rebuild its grid."""
        viewer, captured = self._make_viser_viewer()

        self._log_plane(viewer, (0.0, 0.0, 0.0))
        self.assertEqual(captured["add_grid_calls"], 1)
        handle = viewer._plane_handles["/ground_instances"][0]

        for _ in range(4):
            self._log_plane(viewer, (0.0, 0.0, 0.0))

        # Same static pose logged repeatedly: exactly one add_grid call ever, same handle.
        self.assertEqual(captured["add_grid_calls"], 1)
        self.assertIs(viewer._plane_handles["/ground_instances"][0], handle)

    def test_moving_plane_updates_handle_position_in_place(self):
        """A plane whose pose changes each frame should move the existing handle, not recreate it."""
        viewer, captured = self._make_viser_viewer()

        self._log_plane(viewer, (0.0, 0.0, 0.0))
        handle = viewer._plane_handles["/ground_instances"][0]

        self._log_plane(viewer, (1.0, 2.0, 3.0))

        self.assertEqual(captured["add_grid_calls"], 1)
        self.assertIs(viewer._plane_handles["/ground_instances"][0], handle)
        np.testing.assert_array_equal(np.asarray(handle.position), np.array([1.0, 2.0, 3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
