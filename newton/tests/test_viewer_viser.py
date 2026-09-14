# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import unittest

import numpy as np

from newton.viewer import ViewerViser


@unittest.skipUnless(importlib.util.find_spec("trimesh") is not None, "Requires trimesh")
@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Requires Pillow")
class TestViewerViser(unittest.TestCase):
    def _roundtrip_textured_mesh(self, channels):
        import trimesh

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        texture = np.arange(2 * 2 * channels, dtype=np.uint8).reshape(2, 2, channels) * 16
        mesh = ViewerViser._build_trimesh_mesh(points, indices, uvs, texture)
        self.assertIsNotNone(mesh)

        # Exercise the actual glTF export/import used by Viser, including glTF defaults.
        scene = trimesh.load_scene(io.BytesIO(mesh.export(file_type="glb")), file_type="glb", process=False)
        loaded = next(iter(scene.geometry.values()))
        np.testing.assert_array_equal(loaded.visual.material.baseColorTexture, texture)
        np.testing.assert_allclose(loaded.visual.uv, uvs)
        return loaded.visual.material

    def test_textured_mesh_preserves_texture_brightness(self):
        """Export RGB and RGBA textures without an unintended gray multiplier."""
        for channels in (3, 4):
            with self.subTest(channels=channels):
                material = self._roundtrip_textured_mesh(channels)
                np.testing.assert_array_equal(material.baseColorFactor, [255, 255, 255, 255])

    def test_textured_mesh_is_nonmetallic(self):
        """Keep textured meshes nonmetallic after glTF applies material defaults."""
        material = self._roundtrip_textured_mesh(3)
        self.assertEqual(material.metallicFactor, 0.0)


if __name__ == "__main__":
    unittest.main()
