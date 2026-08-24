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

        def add_grid(
            name,
            width,
            height,
            plane,
            cell_color,
            section_color,
            cell_size,
            section_size,
            position,
            wxyz,
        ):
            captured["add_grid_calls"] += 1
            captured["last_kwargs"] = {
                "width": width,
                "height": height,
                "plane": plane,
                "cell_color": cell_color,
                "section_color": section_color,
                "cell_size": cell_size,
                "section_size": section_size,
                "position": position,
                "wxyz": wxyz,
            }
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

    def _log_plane_with_scale(self, viewer, mesh_name, base_extents, scale):
        """Log one plane instance under ``mesh_name`` with the given base extents and scale."""
        xform = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform)
        scales = wp.array([wp.vec3(*scale)], dtype=wp.vec3)
        viewer.log_geo(mesh_name, int(newton.GeoType.PLANE), base_extents, 0.0, True)
        viewer.log_instances("/ground_instances", mesh_name, xform, scales, None, None)

    def test_unchanged_plane_pose_reuses_grid_handle(self):
        """Reuse the grid handle instead of rebuilding it when a static plane is logged unchanged."""
        viewer, captured = self._make_viser_viewer()

        self._log_plane(viewer, (0.0, 0.0, 0.0))
        self.assertEqual(captured["add_grid_calls"], 1)
        handle = viewer._plane_handles["/ground_instances"][0]

        for _ in range(4):
            self._log_plane(viewer, (0.0, 0.0, 0.0))

        self.assertEqual(captured["add_grid_calls"], 1)
        self.assertIs(viewer._plane_handles["/ground_instances"][0], handle)

    def test_moving_plane_updates_handle_position_in_place(self):
        """Move the grid handle in place instead of recreating it when a plane's pose changes."""
        viewer, captured = self._make_viser_viewer()

        self._log_plane(viewer, (0.0, 0.0, 0.0))
        handle = viewer._plane_handles["/ground_instances"][0]

        self._log_plane(viewer, (1.0, 2.0, 3.0))

        self.assertEqual(captured["add_grid_calls"], 1)
        self.assertIs(viewer._plane_handles["/ground_instances"][0], handle)
        np.testing.assert_array_equal(np.asarray(handle.position), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_hidden_plane_toggles_visibility_without_removing_handle(self):
        """Hide the grid handle in place instead of removing it, so re-showing it is cheap."""
        viewer, captured = self._make_viser_viewer()

        self._log_plane(viewer, (0.0, 0.0, 0.0))
        handle = viewer._plane_handles["/ground_instances"][0]

        xform = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform)
        viewer.log_instances("/ground_instances", "/ground", xform, None, None, None, hidden=True)

        self.assertFalse(handle.visible)
        handle.remove.assert_not_called()
        self.assertIn("/ground_instances", viewer._plane_handles)

        self._log_plane(viewer, (0.0, 0.0, 0.0))

        self.assertEqual(captured["add_grid_calls"], 1)
        self.assertIs(viewer._plane_handles["/ground_instances"][0], handle)
        self.assertTrue(handle.visible)

    def test_matching_extents_with_different_cell_size_rebuilds_grid(self):
        """Rebuild the grid when a new plane's cell size differs even though its final extents match."""
        viewer, captured = self._make_viser_viewer()

        narrow_base_extents = (10.0, 1.0)
        narrow_base_scale = (1.0, 10.0, 1.0)
        self._log_plane_with_scale(viewer, "/ground_a", narrow_base_extents, narrow_base_scale)
        self.assertEqual(captured["add_grid_calls"], 1)
        narrow_base_cell_size = captured["last_kwargs"]["cell_size"]

        square_base_extents = (10.0, 10.0)
        square_base_scale = (1.0, 1.0, 1.0)
        self._log_plane_with_scale(viewer, "/ground_b", square_base_extents, square_base_scale)

        self.assertEqual(captured["add_grid_calls"], 2)
        self.assertNotEqual(captured["last_kwargs"]["cell_size"], narrow_base_cell_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
