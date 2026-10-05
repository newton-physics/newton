# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test RTX frame capture and runtime mesh updates."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerRTXGetFrame(unittest.TestCase):
    def test_headless_frame_capture(self):
        """Capture the latest moving scene in sync and async modes and save screenshots."""
        if importlib.util.find_spec("ovrtx") is None:
            self.skipTest("Requires ovrtx")
        if not wp.is_cuda_available():
            self.skipTest("Requires an NVIDIA RTX-capable GPU")

        from PIL import Image

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25, color=(1.0, 0.0, 0.0))
        model = builder.finalize()
        state = model.state()
        for async_rendering in (False, True):
            with self.subTest(async_rendering=async_rendering):
                viewer = ViewerRTX(width=64, height=48, headless=True, async_rendering=async_rendering)
                try:
                    viewer.set_model(model)
                    viewer.set_camera(pos=wp.vec3(2.0, 0.0, 0.0), pitch=0.0, yaw=180.0)
                    for frame_index, y in enumerate((-0.5, 0.5, -0.5)):
                        state.body_q.assign([wp.transform((0.0, y, 0.0), wp.quat_identity())])
                        viewer.begin_frame(frame_index / 60)
                        viewer.log_state(state)
                        viewer.end_frame()
                        frame = viewer.get_frame()
                        self.assertEqual(frame.shape, (48, 64, 3))
                        self.assertEqual(frame.dtype, wp.uint8)
                        self.assertEqual(frame.device, model.device)
                        rgb = frame.numpy()
                        red_pixels = (rgb[:, :, 0] > 32) & (rgb[:, :, 1] < rgb[:, :, 0] // 2)
                        _, columns = np.nonzero(red_pixels)
                        self.assertGreater(columns.size, 0)
                        # The box must appear on the side logged in this frame.
                        self.assertGreater(y * (columns.mean() - 32), 0)

                    target = wp.empty_like(frame)
                    self.assertIs(viewer.get_frame(target_image=target), target)
                    np.testing.assert_array_equal(target.numpy(), rgb)

                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "screenshot.png"
                        with self.assertWarnsRegex(DeprecationWarning, "get_frame"):
                            viewer.save_screenshot(str(path))
                        with Image.open(path) as screenshot:
                            np.testing.assert_array_equal(np.asarray(screenshot.convert("RGB")), rgb)
                finally:
                    viewer.close()


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
