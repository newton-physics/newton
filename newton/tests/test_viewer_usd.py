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

import os
import tempfile
import unittest

import numpy as np
import warp as wp

from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerUSD

if USD_AVAILABLE:
    from pxr import UsdGeom


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerUSD(unittest.TestCase):
    def _make_viewer(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))
        viewer = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer.close)
        self.addCleanup(lambda: setattr(viewer, "output_path", ""))
        return viewer

    def test_log_points_keeps_per_point_wp_vec3_colors_for_three_points(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        colors = wp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=wp.vec3,
        )

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_per_point", points, radii=0.01, colors=colors)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.vertex)
        np.testing.assert_allclose(display_color, colors.numpy(), atol=1e-6)

    def test_reuses_existing_layer_for_same_output_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))

        # Create first viewer and write some data into the stage.
        viewer1 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer1.close)
        self.addCleanup(lambda: setattr(viewer1, "output_path", ""))

        viewer1.begin_frame(0.0)
        points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
        colors = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3)
        path = viewer1.log_points("/points_from_viewer1", points, radii=0.01, colors=colors)

        # Ensure the prim written by viewer1 is present before creating viewer2.
        prim_before = UsdGeom.Points.Get(viewer1.stage, path).GetPrim()
        self.assertTrue(prim_before.IsValid())

        # Create second viewer for the same output path; this should reuse the same
        # underlying layer and clear any previous contents.
        viewer2 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer2.close)
        self.addCleanup(lambda: setattr(viewer2, "output_path", ""))

        # Verify that the stage/layer reuse actually occurred.
        self.assertIsNotNone(viewer2.stage)
        self.assertIs(viewer1.stage.GetRootLayer(), viewer2.stage.GetRootLayer())

        # Verify that viewer2 cleared/overwrote viewer1's data.
        prim_after = UsdGeom.Points.Get(viewer2.stage, path).GetPrim()
        self.assertFalse(prim_after.IsValid())
        self.assertTrue(os.path.exists(temp_file.name))

    def test_log_points_treats_wp_float_triplet_as_single_constant_color(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        color_triplet = wp.array([0.25, 0.5, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_constant", points, radii=0.01, colors=color_triplet)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.constant)
        np.testing.assert_allclose(display_color, np.array([[0.25, 0.5, 0.75]], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
