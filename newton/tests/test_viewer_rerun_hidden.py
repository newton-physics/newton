# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np

# ruff: noqa: PLC0415


class TestViewerRerunHidden(unittest.TestCase):
    """Regression tests for the hidden parameter in ViewerRerun log_mesh/log_instances."""

    def _create_viewer(self):
        """Create a ViewerRerun with mocked rerun backend."""
        self.mock_rr = Mock()
        self.mock_rr.init = Mock()
        self.mock_rr.spawn = Mock()
        self.mock_rr.connect_grpc = Mock()
        self.mock_rr.set_time = Mock()
        self.mock_rr.save = Mock()
        self.mock_rr.log = Mock()
        self.mock_rr.Clear = Mock(return_value=Mock())
        self.mock_rr.Mesh3D = Mock(return_value=Mock())
        self.mock_rr.InstancePoses3D = Mock(return_value=Mock())

        self.mock_rrb = Mock()
        self.mock_rrb.Blueprint = Mock(return_value=Mock())
        self.mock_rrb.Horizontal = Mock(return_value=Mock())
        self.mock_rrb.Spatial3DView = Mock(return_value=Mock())
        self.mock_rrb.TimePanel = Mock(return_value=Mock())
        self.mock_rrb.TimeSeriesView = Mock(return_value=Mock())

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(serve_web_viewer=False)

        return viewer

    def _make_mock_wp_array(self, data):
        """Create a mock warp array that behaves enough for ViewerRerun."""
        arr = Mock()
        np_data = np.array(data, dtype=np.float32)
        arr.numpy.return_value = np_data
        arr.dtype = Mock()
        arr.device = "cpu"
        arr.shape = np_data.shape
        arr.__len__ = lambda self_: len(np_data)
        return arr

    def test_log_mesh_hidden_skips_registration(self):
        """log_mesh(hidden=True) should not store the mesh in _meshes."""
        viewer = self._create_viewer()

        points = self._make_mock_wp_array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        indices = self._make_mock_wp_array([0, 1, 2])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_mesh("hidden_mesh", points, indices, hidden=True)

        self.assertNotIn("hidden_mesh", viewer._meshes)

    def test_log_instances_hidden_clears_entity(self):
        """log_instances(hidden=True) should clear a previously visible entity."""
        viewer = self._create_viewer()

        # Manually register a mesh and instance so log_instances sees them
        viewer._meshes["my_mesh"] = {
            "points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "indices": np.array([[0, 1, 2]], dtype=np.uint32),
            "normals": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
            "uvs": None,
            "texture_image": None,
            "texture_buffer": None,
            "texture_format": None,
        }
        viewer._instances["my_instance"] = Mock()

        xforms = self._make_mock_wp_array([[0, 0, 0, 0, 0, 0, 1]])
        scales = self._make_mock_wp_array([[1, 1, 1]])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_instances("my_instance", "my_mesh", xforms, scales, colors=None, materials=None, hidden=True)

        # Verify rr.Clear was constructed and logged
        self.mock_rr.Clear.assert_called_once_with(recursive=False)
        self.mock_rr.log.assert_called_once_with("my_instance", self.mock_rr.Clear.return_value)

    def test_log_instances_hidden_noop_when_not_created(self):
        """log_instances(hidden=True) for a never-visible entity should not crash or log."""
        viewer = self._create_viewer()

        # Register a mesh but do NOT create any instances
        viewer._meshes["my_mesh"] = {
            "points": np.array([[0, 0, 0]], dtype=np.float32),
            "indices": np.array([[0, 0, 0]], dtype=np.uint32),
            "normals": np.array([[0, 0, 1]], dtype=np.float32),
            "uvs": None,
            "texture_image": None,
            "texture_buffer": None,
            "texture_format": None,
        }

        xforms = self._make_mock_wp_array([[0, 0, 0, 0, 0, 0, 1]])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            # Reset mock to track only calls from this point
            self.mock_rr.log.reset_mock()
            viewer.log_instances(
                "new_instance", "my_mesh", xforms, scales=None, colors=None, materials=None, hidden=True
            )

        # No rr.log call should have been made
        self.mock_rr.log.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
