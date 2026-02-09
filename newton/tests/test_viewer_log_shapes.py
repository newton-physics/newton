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

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import assert_np_equal
from newton.viewer import ViewerNull


class _LogShapesProbe(ViewerNull):
    """Captures args passed to ``log_instances`` so tests can inspect them."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.last_colors = None
        self.last_materials = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.last_colors = colors
        self.last_materials = materials


class TestLogShapesBroadcast(unittest.TestCase):
    """Regression tests for broadcasting length-1 warp arrays in ``log_shapes`` (issue #1417)."""

    def test_length_one_color_and_material_broadcast(self):
        """A single-element color/material warp array should be broadcast to match num_instances."""
        viewer = _LogShapesProbe()

        num_instances = 3
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )
        color = wp.array([wp.vec3(0.9, 0.1, 0.1)], dtype=wp.vec3)
        material = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)

        viewer.log_shapes(
            "/test_sphere",
            newton.GeoType.SPHERE,
            0.5,
            xforms,
            colors=color,
            materials=material,
        )

        # colors and materials should have been broadcast to num_instances
        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

        # every row should match the original single element
        assert_np_equal(
            viewer.last_colors.numpy(),
            np.tile([0.9, 0.1, 0.1], (num_instances, 1)).astype(np.float32),
        )
        assert_np_equal(
            viewer.last_materials.numpy(),
            np.tile([0.0, 0.7, 0.0, 0.0], (num_instances, 1)).astype(np.float32),
        )

    def test_full_length_arrays_pass_through(self):
        """Arrays already matching num_instances should be passed through unchanged."""
        viewer = _LogShapesProbe()

        num_instances = 2
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )
        colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
            dtype=wp.vec3,
        )
        materials = wp.array(
            [wp.vec4(0.1, 0.2, 0.0, 0.0), wp.vec4(0.3, 0.4, 0.0, 0.0)],
            dtype=wp.vec4,
        )

        viewer.log_shapes(
            "/test_box",
            newton.GeoType.BOX,
            (0.5, 0.3, 0.8),
            xforms,
            colors=colors,
            materials=materials,
        )

        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

    def test_none_colors_and_materials_use_defaults(self):
        """Passing None for colors/materials should produce default values."""
        viewer = _LogShapesProbe()

        num_instances = 2
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )

        viewer.log_shapes(
            "/test_capsule",
            newton.GeoType.CAPSULE,
            (0.3, 1.0),
            xforms,
        )

        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

        # default color is (0.3, 0.8, 0.9)
        assert_np_equal(
            viewer.last_colors.numpy(),
            np.tile([0.3, 0.8, 0.9], (num_instances, 1)).astype(np.float32),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
