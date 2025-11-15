# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

# ruff: noqa: PLC0415


class TestViewerRerunInitArgs(unittest.TestCase):
    """Unit tests for ViewerRerun initialization parameters."""

    def setUp(self):
        """Create a fresh mock rerun object for each test."""
        self.mock_rr = Mock()
        self.mock_rr.init = Mock()
        self.mock_rr.spawn = Mock()
        self.mock_rr.connect_grpc = Mock()
        self.mock_rr.set_time = Mock()
        self.mock_rr.save = Mock()

        # Mock blueprint module and components
        self.mock_rrb = Mock()
        self.mock_blueprint = Mock()
        self.mock_rrb.Blueprint = Mock(return_value=self.mock_blueprint)
        self.mock_rrb.Horizontal = Mock(return_value=Mock())
        self.mock_rrb.Spatial3DView = Mock(return_value=Mock())
        self.mock_rrb.TimePanel = Mock(return_value=Mock())
        self.mock_rrb.TimeSeriesView = Mock(return_value=Mock())

    def test_default_spawns_viewer(self):
        """Test that ViewerRerun() with no arguments spawns a viewer."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    # Suppress deprecation warnings for cleaner test output
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun()

                    # Verify rr.init was called with app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with("newton-viewer", default_blueprint=ANY)

                    # Verify rr.spawn() was called
                    self.mock_rr.spawn.assert_called_once()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()

    def test_explicit_address_none_spawns_viewer(self):
        """Test that ViewerRerun(address=None) explicitly spawns a viewer."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(address=None)

                    # Verify rr.spawn() was called when address is None
                    self.mock_rr.spawn.assert_called_once()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()

    def test_custom_address_connects_grpc(self):
        """Test that ViewerRerun(address='...') connects via gRPC."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_address = "localhost:9876"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(address=test_address)

                    # Verify rr.init was called with app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with("newton-viewer", default_blueprint=ANY)

                    # Verify rr.connect_grpc() was called with the address
                    self.mock_rr.connect_grpc.assert_called_once_with(test_address)

                    # Verify rr.spawn() was NOT called
                    self.mock_rr.spawn.assert_not_called()

    def test_custom_address_connects_grpc_in_jupyter(self):
        """Test that ViewerRerun(address='...') connects via gRPC even in Jupyter notebooks."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=True):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_address = "localhost:9876"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(address=test_address)

                    # Verify viewer detected Jupyter environment
                    self.assertTrue(viewer.is_jupyter_notebook)

                    # Verify rr.connect_grpc() was called with the address even in Jupyter
                    self.mock_rr.connect_grpc.assert_called_once_with(test_address)

                    # Verify rr.spawn() was NOT called
                    self.mock_rr.spawn.assert_not_called()

    def test_custom_app_id_used(self):
        """Test that custom app_id is passed to rr.init."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    custom_app_id = "my-simulation-123"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(app_id=custom_app_id)

                    # Verify rr.init was called with custom app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with(custom_app_id, default_blueprint=ANY)

                    # Verify the viewer stored the app_id correctly
                    self.assertEqual(viewer.app_id, custom_app_id)

    def test_blueprint_passed_to_init(self):
        """Test that blueprint is created and passed to rr.init()."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun()

                    # Verify blueprint components were created
                    self.mock_rrb.Blueprint.assert_called_once()
                    self.mock_rrb.Spatial3DView.assert_called()
                    self.mock_rrb.TimePanel.assert_called()

                    # Verify blueprint was passed to rr.init
                    call_args = self.mock_rr.init.call_args
                    self.assertIn("default_blueprint", call_args[1])
                    self.assertEqual(call_args[1]["default_blueprint"], self.mock_blueprint)

    def test_record_to_rrd_calls_save(self):
        """Test that providing record_to_rrd calls rr.save() with blueprint."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_path = "test_recording.rrd"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(record_to_rrd=test_path)

                    # Verify rr.save was called
                    self.mock_rr.save.assert_called_once()
                    call_args = self.mock_rr.save.call_args
                    self.assertEqual(call_args[0][0], test_path)
                    self.assertIn("default_blueprint", call_args[1])
                    self.assertEqual(call_args[1]["default_blueprint"], self.mock_blueprint)

    def test_jupyter_notebook_skips_spawn(self):
        """Test that viewer is not spawned in Jupyter notebook environment."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=True):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun()

                    # Verify viewer detected Jupyter environment
                    self.assertTrue(viewer.is_jupyter_notebook)

                    # Verify rr.spawn() was NOT called in Jupyter
                    self.mock_rr.spawn.assert_not_called()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()

    def test_non_jupyter_spawns_viewer(self):
        """Test that viewer is spawned in non-Jupyter environment."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun()

                    # Verify viewer detected non-Jupyter environment
                    self.assertFalse(viewer.is_jupyter_notebook)

                    # Verify rr.spawn() WAS called in non-Jupyter
                    self.mock_rr.spawn.assert_called_once()

    def test_server_parameter_deprecated(self):
        """Test that server parameter triggers deprecation warning."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    # Check that warning is raised
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        _ = ViewerRerun(server=True)

                        # Verify warning was raised
                        self.assertEqual(len(w), 1)
                        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
                        self.assertIn("'server' parameter is deprecated", str(w[0].message))

    def test_launch_viewer_parameter_deprecated(self):
        """Test that launch_viewer parameter triggers deprecation warning."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    # Check that warning is raised
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        _ = ViewerRerun(launch_viewer=True)

                        # Verify warning was raised
                        self.assertEqual(len(w), 1)
                        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
                        self.assertIn("'launch_viewer' parameter is deprecated", str(w[0].message))

    def test_keep_historical_data_stored(self):
        """Test that keep_historical_data parameter is stored correctly."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer_true = ViewerRerun(keep_historical_data=True)
                        viewer_false = ViewerRerun(keep_historical_data=False)

                    # Verify parameters were stored correctly
                    self.assertTrue(viewer_true.keep_historical_data)
                    self.assertFalse(viewer_false.keep_historical_data)

    def test_keep_scalar_history_stored(self):
        """Test that keep_scalar_history parameter is stored correctly."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun._is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer_true = ViewerRerun(keep_scalar_history=True)
                        viewer_false = ViewerRerun(keep_scalar_history=False)

                    # Verify parameters were stored correctly
                    self.assertTrue(viewer_true.keep_scalar_history)
                    self.assertFalse(viewer_false.keep_scalar_history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
