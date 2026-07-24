# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global Show Ground visualization toggle."""

import unittest

from newton import GeoType, ShapeFlags
from newton.viewer import ViewerNull


class TestShowGroundToggle(unittest.TestCase):
    def test_ground_toggle_hides_only_plane_shapes(self):
        """``show_ground`` gates plane shapes without affecting other geometry."""
        viewer = ViewerNull()
        flags = int(ShapeFlags.VISIBLE) | int(ShapeFlags.COLLIDE_SHAPES)

        self.assertTrue(viewer.show_ground, "ground is shown by default")
        self.assertTrue(viewer._should_show_shape(flags, True, int(GeoType.PLANE)))

        viewer.show_ground = False
        self.assertFalse(viewer._should_show_shape(flags, True, int(GeoType.PLANE)))
        # Non-plane geometry stays visible regardless of the ground toggle.
        self.assertTrue(viewer._should_show_shape(flags, False, int(GeoType.BOX)))

    def test_ground_gate_ignored_without_geo_type(self):
        """A ``None`` geo type falls back to the normal visibility logic."""
        viewer = ViewerNull()
        viewer.show_ground = False
        self.assertTrue(viewer._should_show_shape(int(ShapeFlags.VISIBLE), False, None))


if __name__ == "__main__":
    unittest.main()
