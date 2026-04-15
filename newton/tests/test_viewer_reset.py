# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer import ViewerBase


class _StubViewer(ViewerBase):
    """Minimal concrete subclass of ViewerBase for testing."""

    def end_frame(self):
        pass

    def log_mesh(self, name, vertices, indices, colors=None, smooth_shading=True):
        pass

    def log_instances(self, name, mesh_name, positions, rotations, colors=None, scalings=None):
        pass

    def log_lines(self, name, vertices_start, vertices_end, colors=None, radius=0.001):
        pass

    def log_points(self, name, positions, colors=None, radii=None, radius=0.01):
        pass

    def log_array(self, name, array):
        pass

    def log_scalar(self, name, value):
        pass

    def apply_forces(self, state):
        pass

    def close(self):
        pass


class TestViewerResetSignal(unittest.TestCase):
    """Tests for the ViewerBase reset signal API."""

    def test_initial_state_not_requested(self):
        """Fresh viewer has no reset requested."""
        viewer = _StubViewer()
        self.assertFalse(viewer.is_reset_requested())

    def test_request_and_query(self):
        """Setting _reset_requested is visible via is_reset_requested."""
        viewer = _StubViewer()
        viewer._reset_requested = True
        self.assertTrue(viewer.is_reset_requested())

    def test_clear_reset_request(self):
        """clear_reset_request resets the flag to False."""
        viewer = _StubViewer()
        viewer._reset_requested = True
        viewer.clear_reset_request()
        self.assertFalse(viewer.is_reset_requested())

    def test_clear_model_resets_flag(self):
        """clear_model() should clear the reset flag."""
        viewer = _StubViewer()
        viewer._reset_requested = True
        viewer.clear_model()
        self.assertFalse(viewer.is_reset_requested())


if __name__ == "__main__":
    unittest.main()
