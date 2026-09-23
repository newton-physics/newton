# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test RTX runtime mesh updates without starting OVRTX."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton.viewer import ViewerRTX


class TestViewerRTX(unittest.TestCase):
    def _make_runtime_viewer(self):
        """Capture mesh attribute writes at the OVRTX boundary."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._phase = viewer._PHASE_RENDER
        viewer._qualify = mock.Mock(side_effect=lambda name: name)
        viewer._mesh_prim_paths = {"/mesh": "/root/mesh"}
        viewer._pending_mesh_points = {}
        viewer._pending_mesh_normals = {}
        viewer._pending_mesh_topology = {}
        viewer._pending_mesh_visibility = {}
        viewer._rtx = mock.Mock()
        # Avoid the optional OVRTX DLPack adapter, but retain the arrays sent to it.
        viewer._make_point3f_dltensor = mock.Mock(side_effect=np.copy)
        attributes = {}

        def write_array_attribute(prim_paths, attribute_name, tensors):
            self.assertEqual(prim_paths, ["/root/mesh"])
            self.assertEqual(len(tensors), 1)
            attributes[attribute_name] = np.array(tensors[0], copy=True)

        viewer._rtx.write_array_attribute.side_effect = write_array_attribute
        return viewer, attributes

    def test_dynamic_mesh_generates_runtime_normals_after_topology_change(self):
        """Send fresh smooth or sharp normals to RTX after replacing mesh topology."""
        for index_dtype in (wp.int32, wp.uint32):
            with self.subTest(index_dtype=index_dtype):
                viewer, attributes = self._make_runtime_viewer()
                points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu")
                indices = wp.array([0, 1, 2], dtype=index_dtype, device="cpu")
                normals = wp.array([[1, 0, 0]] * 3, dtype=wp.vec3, device="cpu")
                viewer.log_mesh("/mesh", points, indices, normals=normals, dynamic=True)
                viewer._update_ovrtx_mesh_points()
                np.testing.assert_array_equal(attributes["normals"], normals.numpy())

                folded_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                folded_indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.int32)
                for split in (False, True):
                    with self.subTest(split=split):
                        if split:
                            vertices = folded_points[folded_indices]
                            triangles = np.arange(6, dtype=np.int32)
                            expected_normals = [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3
                        else:
                            vertices, triangles = folded_points, folded_indices
                            expected_normals = [[0, 2**-0.5, 2**-0.5]] * 2 + [[0, 0, 1], [0, 1, 0]]
                        viewer.log_mesh(
                            "/mesh",
                            wp.array(vertices, dtype=wp.vec3, device="cpu"),
                            wp.array(triangles, dtype=index_dtype, device="cpu"),
                            dynamic=True,
                        )
                        viewer._update_ovrtx_mesh_points()
                        np.testing.assert_allclose(attributes["normals"], expected_normals, atol=1e-6)
                        np.testing.assert_array_equal(attributes["points"], vertices)
                        np.testing.assert_array_equal(attributes["faceVertexIndices"], triangles)
                        np.testing.assert_array_equal(attributes["faceVertexCounts"], [3, 3])

                viewer.log_mesh(
                    "/mesh",
                    wp.empty(0, dtype=wp.vec3, device="cpu"),
                    wp.empty(0, dtype=index_dtype, device="cpu"),
                    dynamic=True,
                )
                viewer._update_ovrtx_mesh_points()
                self.assertEqual(attributes["normals"].shape, (0, 3))
                self.assertEqual(attributes["points"].shape, (0, 3))
                self.assertEqual(attributes["faceVertexIndices"].size, 0)

    def test_deforming_mesh_recomputes_runtime_normals(self):
        """Refresh omitted normals when points change without a topology update."""
        viewer, attributes = self._make_runtime_viewer()
        indices = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
        for vertices, normal in (
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 1]),
            ([[0, 0, 0], [0, 0, 1], [1, 0, 0]], [0, 1, 0]),
        ):
            with self.subTest(normal=normal):
                viewer.log_mesh("/mesh", wp.array(vertices, dtype=wp.vec3, device="cpu"), indices)
                viewer._update_ovrtx_mesh_points()
                self.assertIn("normals", attributes)
                np.testing.assert_allclose(attributes["normals"], [normal] * 3, atol=1e-6)
                self.assertNotIn("faceVertexIndices", attributes)


if __name__ == "__main__":
    unittest.main()
